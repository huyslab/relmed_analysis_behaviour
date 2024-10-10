### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ 63e8c382-8560-11ef-1246-0923653e81d2
begin
	cd("/home/jovyan")
	import Pkg
	
	# activate the shared project environment
    Pkg.activate("$(pwd())/relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	using CairoMakie, Random, DataFrames, Distributions, StatsBase,
		ForwardDiff, LinearAlgebra, JLD2, FileIO, CSV, Dates, JSON, RCall, Turing, Printf, Combinatorics, JuMP, HiGHS
	using LogExpFunctions: logistic, logit

	Turing.setprogress!(false)

	include("$(pwd())/PILT_models.jl")
	include("$(pwd())/sample_utils.jl")
	include("$(pwd())/stats_utils.jl")
	include("$(pwd())/plotting_utils.jl")
	include("$(pwd())/single_p_QL.jl")
	include("$(pwd())/fetch_preprocess_data.jl")
	include("$(pwd())/generate_experimental_sequences/sequence_utils.jl")
	nothing
end

# ╔═╡ 1d93b32c-bd95-4aae-83c6-f6acb5bd6d52
# General attributes of pilot PILT structure
begin

	# PILT Parameters
	set_sizes = 1:3
	block_per_set = 8 # Including reward and punishment
	base_blocks_per_set = 6 # Last two are extra for EEG quality
	trials_per_pair = 10

	# Total number of blocks
	n_total_blocks = length(set_sizes) * block_per_set

	# Total number of pairs
	n_total_pairs = sum(set_sizes .* block_per_set)

	# Shaping protocol
	n_confusing = vcat([0, 1, 1], fill(2, n_total_blocks - 3)) # Per block

	# Initial Q value for simulation - average outcome
	aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])

	# Post-PILT test parameters
	test_n_blocks = 2
end

# ╔═╡ 9a12e584-eff1-482a-b7cc-9daa5321d8de
begin
	# Set theme
	inter_bold = assetpath(pwd() * "/fonts/Inter/Inter-Bold.ttf")
	
	th = Theme(
		font = "Helvetica",
		fontsize = 16,
		Axis = (
			xgridvisible = false,
			ygridvisible = false,
			rightspinevisible = false,
			topspinevisible = false,
			xticklabelsize = 14,
			yticklabelsize = 14,
			spinewidth = 1.5,
			xtickwidth = 1.5,
			ytickwidth = 1.5
		)
	)
	
	set_theme!(th)
end

# ╔═╡ 0685f415-66b5-4ef5-aa4f-6bbdddf79c4a
md"""
# PILT

## Legend for output csv files (Columns that are important for task code in bold.):

- **block: Block number. Integer.**
- n_pairs: Number of stimulus pairs in block (set size). Integer.
- valence: Reward = 1, punishment = -1 block.
- n_confusing: Number of confusing trials in block. Integer.
- **pair: Pair number for trial (1-3).**
- **cpair: Pair number for trial, across all blocks (1-48).**
- **appearance: Apperance number for pair. Integer.**
- **trial: Trial number in block. Integer.**
- feedback_common: Whether this is a common feedback trial (True) or a confusing trial (False).
- variable_magnitude: Feedback magnitude for the stimulus that can have 50pence as feedback. Used in optimization.
- fifty_high: Whether the high magnitude stimulus has the 50pence (True / False). Balanced across pairs.
- feedback_high: High magnitude feedback for the trial.
- feedback_low: Low magnitude feedback for the trial.
- feedback_optimal: Feedback for the optimal stimulus on this trial.
- feedback_suboptimal: Feedback for the suboptimal stimulus on this trial.
- session: Session number. Integer.
- stimulus_A: Image filename for stimulus "A". Used for randomization.
- stimulus_B: Image filename for stimulus "B". Used for randomization.
- optimal_A: Whether stimulus A is the optimal stimulus for the block. Output of randomization.
- A\_on\_right: Whether stimulus A is displayed on righ. Output of randomization.
- **stimulus_right: Image filename for the stimulus on the right.**
- **stimulus_left: Image filename for the stimulus on the left.**
- **optimal_right: Whether the stimulus on the right is the optimal one (True /False).**
- **feedback_right: Feedback for the stimulus on the right for this trial.**
- **feedback_left: Feedback for the stimulus on the right for this trial.**
"""

# ╔═╡ 4c09fe35-8015-43d8-a38f-2434318e74fe
# Assign valence and set size per block
valence_set_size = let random_seed = 0
	
	# # All combinations of set sizes and valence
	@assert iseven(block_per_set) # Requisite for code below

	n_repetitions = div(block_per_set, 2) # How many repetitions of each cell
	valence_set_size = DataFrame(
		n_pairs = repeat(set_sizes, inner = block_per_set),
		valence = repeat([1, -1], inner = n_repetitions, outer = div(n_total_blocks, block_per_set)),
		super_block = repeat(1:n_repetitions, outer = div(n_total_blocks, n_repetitions))
	)

	# Shuffle set size and valence, making sure set size rises gradually and valence is varied in first three blocks, and positive in the first
	rng = Xoshiro(random_seed)
	
	while (valence_set_size[1:3, :n_pairs] != [1, 2, 3]) | 
		allequal(valence_set_size[1:3, :valence]) | 
		(valence_set_size.valence[1] == -1)

		DataFrames.transform!(
			groupby(valence_set_size, :super_block),
			:super_block => (x -> shuffle(rng, 1:length(x))) => :order
		)
		
		sort!(valence_set_size, [:super_block, :order])
	end

	# Add n_confusing
	valence_set_size.n_confusing = n_confusing

	# Add block variable
	valence_set_size.block = 1:nrow(valence_set_size)

	# Return
	valence_set_size
end

# ╔═╡ 2018073a-656d-4723-8384-07c9d533245f
# Create feedback sequences per pair
pair_sequences, common_per_pos, EV_per_pos = let random_seed = 321
	# Compute n_confusing per each pair
	n_confusing_pair = vcat([fill(nc, np) 
		for (nc, np) in zip(valence_set_size.n_confusing, valence_set_size.n_pairs)]...)

	# Compute how much we need of each sequence category
	n_confusing_pair_sum = countmap(n_confusing_pair)
	
	n_confusing_vals = sort([kv[1] for kv in n_confusing_pair_sum])
	
	n_wanted = vcat([[div(kv[2], 2), div(kv[2], 2) + rem(kv[2], 2)] for kv in sort(collect(n_confusing_pair_sum), by = first)]...)[2:end]

	# Generate all sequences and compute FI
	FI_seqs = [compute_save_FIs_for_all_seqs(;
		n_trials = 10,
		n_confusing = nc,
		fifty_high = fh,
		initV = aao
	) for nc in filter(x -> x > 0, n_confusing_vals) 
		for fh in [true, false]]

	# Add zero confusing sequences
	zero_seq = compute_save_FIs_for_all_seqs(;
			n_trials = 10,
			n_confusing = 0,
			fifty_high = true,
			initV = aao
		)

	# Uncpack FI and sequence arrays
	FIs = [x[1] for x in FI_seqs]
	pushfirst!(FIs, zero_seq[1])

	common_seqs = [x[2] for x in FI_seqs]
	pushfirst!(common_seqs, zero_seq[2])

	magn_seqs = [x[3] for x in FI_seqs]
	pushfirst!(magn_seqs, zero_seq[3])

	# Choose sequences optimizing FI under contraints
	chosen_idx, common_per_pos, EV_per_pos = optimize_FI_distribution(
		n_wanted = n_wanted,
		FIs = FIs,
		common_seqs = common_seqs,
		magn_seqs = magn_seqs,
		ω_FI = 0.1,
		filename = "results/exp_sequences/eeg_pilot_FI_opt.jld2"
	)

	@assert length(vcat(chosen_idx...)) == sum(valence_set_size.n_pairs) "Number of saved optimize sequences does not match number of sequences needed. Delete file and rerun."

	# Unpack chosen sequences
	chosen_common = [[common_seqs[s][idx[1]] for idx in chosen_idx[s]]
		for s in eachindex(common_seqs)]

	chosen_magn = [[magn_seqs[s][idx[2]] for idx in chosen_idx[s]]
		for s in eachindex(magn_seqs)]

	# Repack into DataFrame	
	task = DataFrame(
		appearance = repeat(1:trials_per_pair, n_total_pairs),
		cpair = repeat(1:n_total_pairs, inner = trials_per_pair),
		feedback_common = vcat(vcat(chosen_common...)...),
		variable_magnitude = vcat(vcat(chosen_magn...)...)
	)

	# Compute n_confusing and fifty_high per block
	DataFrames.transform!(
		groupby(task, :cpair),
		:feedback_common => (x -> trials_per_pair - sum(x)) => :n_confusing,
		:variable_magnitude => (x -> 1. in x) => :fifty_high
	)

	@assert mean(task.fifty_high) == 0.5 "Proportion of blocks with 50 pence in high magnitude option expected to be 0.5"

	rng = Xoshiro(random_seed)

	# Shuffle within n_confusing
	DataFrames.transform!(
		groupby(task, [:n_confusing, :cpair]),
		:n_confusing => (x -> x .* 10 .+ rand(rng)) => :random_pair
	)

	sort!(task, [:n_confusing, :random_pair, :appearance])

	task.cpair = repeat(1:n_total_pairs, inner = trials_per_pair)

	select!(task, Not(:random_pair))

	task, common_per_pos, EV_per_pos
end

# ╔═╡ 424aaf3d-f773-4ce3-a21c-eabd449e4105
# Shape into per-block sequence
feedback_sequence = let random_seed = 0

	# Create pair list
	task = combine(
		groupby(valence_set_size, [:block, :n_pairs, :valence, :n_confusing]),
		:n_pairs => (x -> repeat(vcat([1:xi for xi in x]...), inner = trials_per_pair)) => :pair
	)

	# Shuffle pair appearance
	rng = Xoshiro(random_seed)
	
	DataFrames.transform!(
		groupby(task, :block),
		:pair => (x -> shuffle(rng, x)) => :pair
	)

	# Add cumulative pair variable
	pairs = sort(unique(task[!, [:block, :pair]]), [:block, :pair])
	pairs.cpair = 1:nrow(pairs)

	task = innerjoin(
		task,
		pairs,
		on = [:block, :pair],
		order = :left
	)

	# Add apperance count variable
	DataFrames.transform!(
		groupby(task, [:block, :pair]),
		:block => (x -> 1:length(x)) => :appearance
	)

	# Add trial variable
	DataFrames.transform!(
		groupby(task, :block),
		:block => (x -> 1:length(x)) => :trial
	)

	# Join with pair sequences
	task = innerjoin(
		task,
		pair_sequences,
		on = [:cpair, :appearance],
		order = :left,
		makeunique = true
	)

	# Check and delete excess column
	@assert task.n_confusing == task.n_confusing_1 "Problem in join"

	select!(task, Not(:n_confusing_1))

	# Compute low and high feedback
	task.feedback_high = ifelse.(
		task.valence .> 0,
		ifelse.(
			task.fifty_high,
			task.variable_magnitude,
			fill(1., nrow(task))
		),
		ifelse.(
			task.fifty_high,
			fill(-0.01, nrow(task)),
			.- task.variable_magnitude
		)
	)

	task.feedback_low = ifelse.(
		task.valence .> 0,
		ifelse.(
			.!task.fifty_high,
			task.variable_magnitude,
			fill(0.01, nrow(task))
		),
		ifelse.(
			.!task.fifty_high,
			fill(-1, nrow(task)),
			.- task.variable_magnitude
		)
	)

	# Compute feedback optimal and suboptimal
	task.feedback_optimal = ifelse.(
		task.feedback_common,
		task.feedback_high,
		task.feedback_low
	)

	task.feedback_suboptimal = ifelse.(
		.!task.feedback_common,
		task.feedback_high,
		task.feedback_low
	)

	task

end

# ╔═╡ 56bf5285-75e3-46cc-8b35-389ae7281ce3
# Assign stimulus images
stimuli = let random_seed = 0
	# Load stimulus names
	categories = shuffle(unique([replace(s, ".png" => "")[1:(end-1)] for s in 
		readlines("generate_experimental_sequences/eeg_stim_list.txt")]))

	# Assign stimulus pairs
	stimuli = assign_stimuli_and_optimality(;
		n_phases = 1,
		n_pairs = valence_set_size.n_pairs,
		categories = categories,
		random_seed = random_seed
	)

	rename!(stimuli, :phase => :session) # For compatibility with multi-phase sessions
end

# ╔═╡ bbdd062e-49e4-4a03-bb8f-9ad97b08288a
# Add stimulus assignments to sequences DataFrame, and assign right / left
task = let random_seed = 0

	# Join stimuli and sequences
	task = innerjoin(
		feedback_sequence,
		stimuli,
		on = [:block, :pair],
		order = :left
	)

	@assert nrow(task) == nrow(feedback_sequence) "Problem in join operation"

	# Assign right / left, equal proportions within each pair
	rng = Xoshiro(random_seed)

	DataFrames.transform!(
		groupby(task, [:block, :cpair]),
		:block => 
			(x -> shuffled_fill([true, false], length(x); random_seed = random_seed)) =>
			:A_on_right
	)

	# Create stimulus_right and stimulus_left variables
	task.stimulus_right = ifelse.(
		task.A_on_right,
		task.stimulus_A,
		task.stimulus_B
	)

	task.stimulus_left = ifelse.(
		.!task.A_on_right,
		task.stimulus_A,
		task.stimulus_B
	)

	# Create optimal_right variable
	task.optimal_right = (task.A_on_right .& task.optimal_A) .| (.!task.A_on_right .& .!task.optimal_A)

	# Create feedback_right and feedback_left variables
	task.feedback_right = ifelse.(
		task.optimal_right,
		task.feedback_optimal,
		task.feedback_suboptimal
	)

	task.feedback_left = ifelse.(
		.!task.optimal_right,
		task.feedback_optimal,
		task.feedback_suboptimal
	)

	# Mark which blocks are extra
	task.extra = task.block .> base_blocks_per_set * length(set_sizes)

	task
end

# ╔═╡ 263f4366-b6f4-47fd-b76a-fbeffcc07f14
# Validate task DataFrame
let
	@assert maximum(task.block) == length(unique(task.block)) "Error in block numbering"

	@assert issorted(task.block) "Task structure not sorted by block"

	@assert all(combine(groupby(task, [:session, :block]), 
		:trial => issorted => :sorted).sorted) "Task structure not sorted by trial number"

	@assert all(sign.(task.valence) == sign.(task.feedback_right)) "Valence doesn't match feedback sign"

	@assert all(sign.(task.valence) == sign.(task.feedback_left)) "Valence doesn't match feedback sign"

	@assert sum(unique(task[!, [:session, :block, :valence]]).valence) == 0 "Number of reward and punishment blocks not equal"

	@info "Overall proportion of common feedback: $(round(mean(task.feedback_common), digits = 2))"

	@assert all(combine(groupby(task, :cpair),
		:appearance => maximum => :max_appear
	).max_appear .== 10) "Didn't find exactly 10 apperances per pair"

	@assert all(combine(groupby(task, :appearance), :feedback_common => sum => :common_per_pos).common_per_pos .== common_per_pos) "Number of common feedback per position doesn't match output from optimization function"

	@assert all(combine(groupby(task, :appearance), :variable_magnitude => mean => :EV_per_pos).EV_per_pos .≈ EV_per_pos) "EV per position doesn't match output from optimization function"

	@assert all((task.variable_magnitude .== abs.(task.feedback_right)) .| 
		(task.variable_magnitude .== abs.(task.feedback_left))) ":variable_magnitude, which is used for sequnece optimization, doesn't match end result column :feedback_right no :feedback_left"

	# Count losses to allocate coins in to safe for beginning of task
	worst_loss = filter(x -> x.valence == -1, task) |> 
		df -> ifelse.(
			df.feedback_right < df.feedback_left, 
			df.feedback_right, 
			df.feedback_left) |> 
		countmap

	@info "Worst possible loss in this task is of these coin numbers: $worst_loss"

end

# ╔═╡ c68c334e-54f9-4410-92b1-91c0e78a2dc9
let
	save_to_JSON(task, "results/eeg_pilot.json")
	CSV.write("results/eeg_pilot.csv", task)
end

# ╔═╡ 59bc1127-32b4-4c68-84e9-1892c10a6c45
# Visualize PILT seuqnce
let

	# Proportion of confusing by trial number
	confusing_location = combine(
		groupby(task, :trial),
		:feedback_common => (x -> mean(.!x)) => :feedback_confusing
	)

	# Plot
	f = Figure(size = (700, 500))

	ax_prob = Axis(
		f[1,1][1,1],
		xlabel = "Trial #",
		ylabel = "Prop. confusing feedback"
	)

	scatter!(
		ax_prob,
		confusing_location.trial,
		confusing_location.feedback_confusing
	)

	# Proportion of confusing by aeppearance
	confusing_app = combine(
		groupby(task, :appearance),
		:feedback_common => (x -> mean(.!x)) => :feedback_confusing
	)

	# Plot
	ax_app = Axis(
		f[1,1][2,1],
		xlabel = "Pair appearance #",
		ylabel = "Prop. confusing feedback",
		limits = (nothing, nothing, 0., maximum(confusing_app.feedback_confusing) + 0.01)
	)

	scatter!(
		ax_app,
		confusing_app.appearance,
		confusing_app.feedback_confusing
	)

	# Plot confusing trials by block
	ax_heatmap = Axis(
		f[1, 2],
		xlabel = "Trial #",
		ylabel = "Block",
		yreversed = true,
		subtitle = "Green - reward, yellow - punishment,\ndark - confusing"
	)

	heatmap!(
		task.trial,
		task.block,
		ifelse.(
			.!task.feedback_common,
			fill(0., nrow(task)),
			ifelse.(
				task.valence .> 0,
				fill(1., nrow(task)),
				fill(2., nrow(task))
			)
		)
	)

	save("results/eeg_trial_plan.png", f, pt_per_unit = 1)

	f

end

# ╔═╡ 96f2d1b5-248b-4c43-8e87-fc727c9ea6f0
md"""
# Post-PILT test
"""

# ╔═╡ 6155f43d-be4c-40c9-9d81-fc42b1197e21
test_pairs_wide = let random_seed = 0

	rng = Xoshiro(random_seed)

	# Extract stimuli and their common feedback from task structure
	stimuli = vcat([rename(
		task[task.feedback_common, [:session, :n_pairs, :block, :cpair, Symbol("stimulus_$s"), Symbol("feedback_$s")]],
		Symbol("stimulus_$s") => :stimulus,
		Symbol("feedback_$s") => :feedback
	) for s in ["right", "left"]]...)

	# Summarize magnitude per stimulus
	stimuli = combine(
		groupby(stimuli, [:session, :n_pairs, :block, :cpair, :stimulus]),
		:feedback => (x -> mean(unique(x))) => :magnitude
	)

	println(first(stimuli[!, [:block, :cpair, :stimulus, :magnitude]], 10))


	# # Function that creates a list of pairs from DataFrame
	# create_pair_list(d) = [filter(x -> x.cpair == p, d).stimulus 
	# 	for p in unique(stimuli.cpair)]

	# # Function to check whether pair is novel
	# check_novel(p) = !(p in used_pairs) && !(reverse(p) in used_pairs)

	# # Pairs used in PILT - will add post-PILT test blocks into this as we go
	# used_pairs = create_pair_list(stimuli)


	# # Intialize DataFrame and summary stats for checking
	# test_pairs_wide = DataFrame()
	# prop_same_original = 0.
	# prop_same_valence = 1.
	# same_magnitude_n = 10
	# diff_magnitude_min = 0
	# dif_magnitude_sd = 3.
	# all_different_category = false

	# # Make sure with have exactly 1/3 pairs that were previously in same block
	# # and 1/2 that were of the same valence
	# while diff_magnitude_min < 1 || same_magnitude_n > 10 || dif_magnitude_sd > 2. ||
	# 	!all_different_category

	# 	# Initialize long format DataFrame
	# 	test_pairs = DataFrame()

	# 	# Run over neede block number
	# 	for bl in 1:test_n_blocks

	# 		# Variable to record whether suggested pairs are novel
	# 		all_within_novel = false

	# 		# Intialize DataFrame for pairs that were in the same block
	# 		block_within_pairs = DataFrame()

	# 		# Create within-block pairs
	# 		while !all_within_novel
				
	# 			# Select blocks with n_pairs > 1
	# 			multi_pair_blocks = groupby(
	# 				filter(x -> x.n_pairs > 1, stimuli), 
	# 				[:session, :block]
	# 			)

	# 			# For each block, pick one stimulus from each of two pairs
	# 			for (i, gdf) in enumerate(multi_pair_blocks)
			
	# 				if gdf.n_pairs[1] == 3
	# 					chosen_pairs = sample(rng, unique(gdf.cpair), 2, replace = false)
	# 					tdf = filter(x -> x.cpair in chosen_pairs, gdf)
	# 				else
	# 					tdf = copy(gdf)
	# 				end
			
	# 				stim = vcat([sample(rng, filter(x -> x.cpair == c, tdf).stimulus, 1) 
	# 					for c in unique(tdf.cpair)]...)
	
	# 				append!(block_within_pairs, DataFrame(
	# 					block = fill(bl, 2),
	# 					cpair = fill(i, 2),
	# 					stimulus = stim
	# 				))
	# 			end

	# 			# Check that picked combinations are novel
	# 			pair_list = create_pair_list(block_within_pairs)
	
	# 			all_within_novel = all(check_novel.(pair_list))
	
	# 		end

	# 		# Add to block list
	# 		block_pairs = block_within_pairs

	# 		# Stimuli not included previously
	# 		remaining_stimuli = 
	# 			filter(x -> !(x in block_pairs.stimulus), stimuli.stimulus)

	# 		# Variable for checking whether remaining pairs are novel
	# 		all_between_novel = false
	
	# 		block_between_pairs = DataFrame()

	# 		# Add across-block pairs
	# 		while !all_between_novel

	# 			# Assign pairs
	# 			block_between_pairs = DataFrame(
	# 				block = fill(bl, length(remaining_stimuli)),
	# 				cpair = repeat(
	# 					(maximum(block_pairs.cpair) + 1):maximum(stimuli.cpair), 
	# 					inner = 2
	# 				),
	# 				stimulus = shuffle(rng, remaining_stimuli)
	# 			)

	# 			# Check novelty
	# 			pair_list = create_pair_list(block_between_pairs)
		
	# 			all_between_novel = all(check_novel.(pair_list))
	# 		end

	# 		# Add to lists of pairs for checking future block against
	# 		append!(block_pairs, block_between_pairs)
	
	# 		append!(test_pairs, block_pairs)
			
	# 	end

	# 	# Add magnitude from PILT
	# 	test_pairs = innerjoin(test_pairs, 
	# 		rename(stimuli, :block => :original_block)[!, Not(:cpair)],
	# 		on = :stimulus
	# 	)

	# 	# Compute EV difference and match in block and valence
	# 	sort!(test_pairs, [:session, :block, :cpair, :stimulus])
		
	# 	test_pairs_wide = combine(
	# 		groupby(test_pairs, [:session, :block, :cpair]),
	# 		:stimulus => maximum => :stimulus_A,
	# 		:stimulus => minimum => :stimulus_B,
	# 		[:magnitude, :stimulus] => ((m, s) -> m[argmax(s)]) => :magnitude_A,
	# 		[:magnitude, :stimulus] => ((m, s) -> m[argmin(s)]) => :magnitude_B,
	# 		:magnitude => maximum => :magnitude_high,
	# 		:magnitude => minimum => :magnitude_low,
	# 		:magnitude => (x -> sign(x[1]) == sign(x[2])) => :same_valence,
	# 		:original_block => (x -> x[1] == x[2]) => :same_block
	# 	)

	# 	# Compute conditions for accepting this shuffle
	# 	same_magnitude_n, diff_magnitude_min, diff_magnitude_sd = 
	# 		test_squence_evaluate_magnitudes(test_pairs_wide)
		
	# 	prop_same_original = mean(test_pairs_wide.same_block)
	
	# 	prop_same_valence = mean(test_pairs_wide.same_valence)

	# 	all_different_category = 
	# 		all((x -> x.stimulus_A[1:(end-5)] != x.stimulus_B[1:(end-5)]).(eachrow(test_pairs_wide)))

	# end

	# @info prop_same_original
	# @info prop_same_valence

	# @info test_squence_evaluate_magnitudes(test_pairs_wide)

	# test_pairs_wide

end

# ╔═╡ 47379c5a-67c1-4f44-9e31-efa34a6a525a
function test_squence_evaluate_magnitudes(test_pairs_wide::DataFrame)

	# Summarize current proposal
	magnitude_count = combine(
		groupby(test_pairs_wide, [:magnitude_low, :magnitude_high]),
		:magnitude_low => length => :n
	)

	sort!(magnitude_count, [:magnitude_low, :magnitude_high])

	# Compute all possible options
	mags = vcat([0.01, 0.255, 0.75, 1.], .- [0.01, 0.255, 0.75, 1.])

	couples = DataFrame(
		unique([(x1, x2) for x1 in mags for x2 in mags if x2 >= x1]),
		[:magnitude_low, :magnitude_high]
	)

	# Combine to have missing options
	magnitude_count = leftjoin(
		couples,
		magnitude_count,
		on = [:magnitude_low, :magnitude_high]
	)

	# Replace missing with zero
	magnitude_count.n = ifelse.(
		ismissing.(magnitude_count.n),
		fill(0, nrow(magnitude_count)),
		magnitude_count.n
	)

	# Compute how many pairs have the same magnitude
	same_magnitude_n = sum(
		filter(x -> x.magnitude_low == x.magnitude_high, magnitude_count).n
	)

	# Compute the range of different magnitude pair numbers
	diff_magnitude = 
		filter(x -> x.magnitude_low != x.magnitude_high, magnitude_count).n
	diff_magnitude_min = minimum(diff_magnitude)

	diff_magnitude_sd = std(diff_magnitude)

	return same_magnitude_n, diff_magnitude_min, diff_magnitude_sd
	
end

# ╔═╡ Cell order:
# ╠═63e8c382-8560-11ef-1246-0923653e81d2
# ╠═1d93b32c-bd95-4aae-83c6-f6acb5bd6d52
# ╠═9a12e584-eff1-482a-b7cc-9daa5321d8de
# ╟─0685f415-66b5-4ef5-aa4f-6bbdddf79c4a
# ╠═4c09fe35-8015-43d8-a38f-2434318e74fe
# ╠═2018073a-656d-4723-8384-07c9d533245f
# ╠═424aaf3d-f773-4ce3-a21c-eabd449e4105
# ╠═56bf5285-75e3-46cc-8b35-389ae7281ce3
# ╠═bbdd062e-49e4-4a03-bb8f-9ad97b08288a
# ╠═263f4366-b6f4-47fd-b76a-fbeffcc07f14
# ╠═c68c334e-54f9-4410-92b1-91c0e78a2dc9
# ╠═59bc1127-32b4-4c68-84e9-1892c10a6c45
# ╟─96f2d1b5-248b-4c43-8e87-fc727c9ea6f0
# ╠═6155f43d-be4c-40c9-9d81-fc42b1197e21
# ╠═47379c5a-67c1-4f44-9e31-efa34a6a525a
