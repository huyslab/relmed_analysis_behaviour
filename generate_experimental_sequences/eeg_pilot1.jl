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

# ╔═╡ 6631654e-7368-4228-8694-df35c607b1a3
function create_test_sequence(
	pilt_task::DataFrame;
	random_seed::Int64, 
	same_weight::Float64 = 6.5
) 
	
	rng = Xoshiro(random_seed)

	# Extract stimuli and their common feedback from task structure
	stimuli = vcat([rename(
		pilt_task[pilt_task.feedback_common, [:session, :n_pairs, :block, :cpair, Symbol("stimulus_$s"), Symbol("feedback_$s")]],
		Symbol("stimulus_$s") => :stimulus,
		Symbol("feedback_$s") => :feedback
	) for s in ["right", "left"]]...)

	# Summarize magnitude per stimulus
	stimuli = combine(
		groupby(stimuli, [:session, :n_pairs, :block, :cpair, :stimulus]),
		:feedback => (x -> mean(unique(x))) => :magnitude
	)

	# Step 1: Identify unique stimuli
	unique_stimuli = unique(stimuli.stimulus)

	# Step 2: Define existing pairs
	create_pair_list(d) = [filter(x -> x.cpair == p, d).stimulus 
		for p in unique(stimuli.cpair)]

	existing_pairs = create_pair_list(stimuli)

	# Step 3: Generate all possible pairs
	all_possible_pairs = unique(sort.(collect(combinations(unique_stimuli, 2))))

	# Step 6: Select pairs ensuring each stimulus is used once and magnitudes are balanced
	final_pairs = []
	used_stimuli = Set{String}()

	# Create a priority queue for balanced selection based on pair counts
	pair_counts = Dict{Vector{Float64}, Int}()

	# Function to retrieve attribute of stimulus
	stim_attr(s, attr) = stimuli[stimuli.stimulus .== s, :][!, attr][1]

	for b in 1:test_n_blocks

		# Step 4: Filter valid pairs: were not paired in PILT, ano same category
		valid_pairs = 
			filter(pair -> 
				!(pair in existing_pairs) && 
				!(reverse(pair) in existing_pairs) && 
				(pair[1][1:(end-5)] != pair[2][1:(end-5)]), 
			all_possible_pairs)
	
		# Step 5: Create a mapping of pairs to their magnitudes
		magnitude_pairs = Dict{Vector{Float64}, Vector{Vector{String}}}()
		
		for pair in valid_pairs
		    mag1 = stimuli[stimuli.stimulus .== pair[1], :].magnitude[1]
		    mag2 = stimuli[stimuli.stimulus .== pair[2], :].magnitude[1]
		    key = sort([mag1, mag2])
		    if !haskey(magnitude_pairs, key)
		        magnitude_pairs[key] = []
		    end
		    push!(magnitude_pairs[key], pair)
		end
	
		@assert sum(length(vec) for vec in values(magnitude_pairs)) == length(valid_pairs)
	
		# Step 5.5 - Shuffle order within each magnitude
		for (k, v) in magnitude_pairs
			magnitude_pairs[k] = shuffle(rng, v)
		end

		# Initialize counts
		if b == 1
			for key in keys(magnitude_pairs)
			    pair_counts[key] = 0
			end
		end
		
		block_pairs = []
		
		while true
		    found_pair = false
	
		    # Select pairs while balancing magnitudes
		    for key in sort(collect(keys(magnitude_pairs)), by = x -> pair_counts[x] + same_weight * (x[1] == x[2])) # Sort by count, putting equal magnitude las
		        pairs = magnitude_pairs[key]
	
				# First try to find a same block pair
		        for pair in pairs
		            if !(pair[1] in used_stimuli) && !(pair[2] in used_stimuli)  && 
						stim_attr(pair[1], "block") == stim_attr(pair[2], "block")
					
		                push!(block_pairs, pair)
		                push!(used_stimuli, pair[1])
		                push!(used_stimuli, pair[2])
		                pair_counts[key] += 1
		                found_pair = true
		                break  # Stop going over pairs
		            end
		        end
	
				# Then try different block pair
		        for pair in pairs
		            if !found_pair &&!(pair[1] in used_stimuli) && 
						!(pair[2] in used_stimuli) 
					
		                push!(block_pairs, pair)
		                push!(used_stimuli, pair[1])
		                push!(used_stimuli, pair[2])
		                pair_counts[key] += 1
		                found_pair = true
		                break  # Stop going over pairs
		            end
		        end
		        
		        if found_pair
		            break  # Restart the outer loop if a pair was found
		        end
		    end

			# Alert bad seed
			if !found_pair
				return DataFrame(), NaN, NaN, NaN
			end
		
		    if length(used_stimuli) == length(unique_stimuli)
		        break  # Exit if all stimuli are used or no valid pairs remain
		    end
		end

		# Step 7 - Shuffle pair order
		shuffle!(rng, block_pairs)

		# Add block pairs to final pairs
		append!(final_pairs, block_pairs)

		# Add block pairs to existing pairs
		append!(existing_pairs, block_pairs)

		# Empty used_stimuli
		used_stimuli = []
	end

	# Shuffle order within each pair
	shuffle!.(rng, final_pairs)

	# Step 8 - Form DataFrame
	pairs_df = DataFrame(
		block = repeat(1:test_n_blocks, inner = length(unique_stimuli) ÷ 2),
		trial = repeat(1:(length(unique_stimuli) ÷ 2) * test_n_blocks),
		stimulus_right = [p[2] for p in final_pairs],
		stimulus_left = [p[1] for p in final_pairs],
		magnitude_right = [stimuli[stimuli.stimulus .== p[2], :].magnitude[1] for p in final_pairs],
		magnitude_left = [stimuli[stimuli.stimulus .== p[1], :].magnitude[1] for p in final_pairs],
		original_block_right = [stimuli[stimuli.stimulus .== p[2], :].block[1] for p in final_pairs],
		original_block_left = [stimuli[stimuli.stimulus .== p[1], :].block[1] for p in final_pairs]
	)

	# Same / different block variable
	pairs_df.same_block = pairs_df.original_block_right .== pairs_df.original_block_left

	# Valence variables
	pairs_df.valence_left = sign.(pairs_df.magnitude_left)
	pairs_df.valence_right = sign.(pairs_df.magnitude_right)
	pairs_df.same_valence = pairs_df.valence_left .== pairs_df.valence_right

	# Compute sequence stats
	prop_same_block = (mean(pairs_df.same_block)) 
	prop_same_valence = (mean(pairs_df.same_valence))
	n_same_magnitude = sum(pairs_df.magnitude_right .== pairs_df.magnitude_left)
	
	pairs_df, prop_same_block, prop_same_valence, n_same_magnitude
end

# ╔═╡ 81754bac-950c-4f0e-a51b-14feab30e0e1
# Choose test sequence with best stats
let n_seeds = 100, same_weight = 4.1

	prop_block = []
	prop_valence = []
	n_magnitude = []

	for s in 1:n_seeds
		_, pb, pv, nm = create_test_sequence(filter(x -> !x.extra, task), random_seed = s, same_weight = same_weight)

		push!(prop_block, pb)
		push!(prop_valence, pv)
		push!(n_magnitude, nm)
	end

	pass_magnitude = (1:n_seeds)[n_magnitude .== 
		minimum(filter(x -> !isnan(x), n_magnitude))]

	@assert !isempty(pass_magnitude)

	prop_block = prop_block[pass_magnitude]
	prop_valence = prop_valence[pass_magnitude]

	dev_block = abs.(prop_block .- 1/3)
	dev_valence = abs.(prop_block .- 0.5)

	chosen = pass_magnitude[argmin(dev_block .+ dev_valence)]

	create_test_sequence(filter(x -> !x.extra, task), random_seed = chosen, same_weight = same_weight)
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
# ╠═6631654e-7368-4228-8694-df35c607b1a3
# ╠═81754bac-950c-4f0e-a51b-14feab30e0e1
# ╠═47379c5a-67c1-4f44-9e31-efa34a6a525a
