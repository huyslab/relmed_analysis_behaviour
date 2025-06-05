### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ 99c994e4-9c36-11ef-2c8f-d5829be639eb
begin
	cd("/home/jovyan")
	import Pkg
	
	# activate the shared project environment
    Pkg.activate("$(pwd())/relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	using CairoMakie, Random, DataFrames, Distributions, StatsBase,
		ForwardDiff, LinearAlgebra, JLD2, FileIO, CSV, Dates, JSON, RCall, Turing, Printf, Combinatorics, JuMP, HiGHS, AlgebraOfGraphics
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

# ╔═╡ 65ec8b8f-9eba-467b-bb19-9f0c72b8933e
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

# ╔═╡ 3a5cf6e9-1cee-4fd3-b0db-8fc02b7747fb
categories = let
	categories = (s -> replace(s, ".jpg" => "")[1:(end-1)]).(readdir("generate_experimental_sequences/pilot6_stims"))

	# Keep only categories where we have two files exactly
	keeps = filter(x -> last(x) == 2, countmap(categories))

	filter(x -> x in keys(keeps), unique(categories))
end

# ╔═╡ ea917db6-ec27-454f-8b4e-9df65d65064b
md"""## PILT"""

# ╔═╡ 381e61e2-7d51-4070-8ad1-ce9e63015eb6
# PILT parameters
begin
	# PILT Parameters
	PILT_blocks_per_valence = 10
	PILT_trials_per_block = 10
	
	PILT_total_blocks = PILT_blocks_per_valence * 2
	PILT_n_confusing = vcat([0, 0, 1, 1], fill(2, PILT_total_blocks ÷ 2 - 4)) # Per valence
		
	# Post-PILT test parameters
	PILT_test_n_blocks = 2
end

# ╔═╡ 687f5ae6-86c6-449f-86f5-5ed359e6d580
function has_consecutive_repeats(vec::Vector, n::Int = 3)
    count = 1
    for i in 2:length(vec)
        if vec[i] == vec[i - 1]
            count += 1
            if count > n
                return true
            end
        else
            count = 1
        end
    end
    return false
end

# ╔═╡ 31128edd-5d2d-49e9-8f65-842bb42639f9
# Assign valence and set size per block
PILT_block_attr = let random_seed = 5
	
	# # All combinations of set sizes and valence
	block_attr = DataFrame(
		block = repeat(1:PILT_total_blocks),
		valence = repeat([1, -1], inner = PILT_blocks_per_valence),
		fifty_high = repeat([true, false], outer = PILT_blocks_per_valence)
	)

	# Shuffle set size and valence, making sure valence is varied, and positive in the first block and any time noise is introduced, and shaping doesn't extend too far into the task
	rng = Xoshiro(random_seed)

	first_three_same = true
	first_block_punishement = true
	too_many_repeats = true
	first_confusing_punishment = true
	shaping_too_long = true
	while first_three_same || first_block_punishement || too_many_repeats ||
		first_confusing_punishment || shaping_too_long

		DataFrames.transform!(
			block_attr,
			:block => (x -> shuffle(rng, x)) => :block
		)
		
		sort!(block_attr, :block)

		# Add n_confusing
		DataFrames.transform!(
			groupby(block_attr, :valence),
			:block => (x -> PILT_n_confusing) => :n_confusing
		)

		# Compute criterion variables
		first_three_same = allequal(block_attr[1:3, :valence])
		
		first_block_punishement = block_attr.valence[1] == -1

		too_many_repeats = has_consecutive_repeats(block_attr.valence)

		first_confusing_punishment = 
			(block_attr.valence[findfirst(block_attr.n_confusing .== 1)] == -1) |
			(block_attr.valence[findfirst(block_attr.n_confusing .== 2)] == -1)

		shaping_too_long = 
			!all(block_attr.n_confusing[11:end] .== maximum(PILT_n_confusing))
	end

	# Return
	block_attr
end

# ╔═╡ c7c5b78e-ad76-4877-8c80-af9151105544
# Create feedback sequences per pair
PILT_sequences, common_per_pos, EV_per_pos = 
	let random_seed = 2
	
	# Compute how much we need of each sequence category
	n_confusing_wanted = combine(
		groupby(PILT_block_attr, [:n_confusing, :fifty_high]),
		:block => length => :n
	)
	
	# Generate all sequences and compute FI
	FI_seqs = [compute_save_FIs_for_all_seqs(;
		n_trials = 10,
		n_confusing = r.n_confusing,
		fifty_high = r.fifty_high,
		initV = nothing
	) for r in eachrow(n_confusing_wanted)]

	# Unpack results
	common_seqs = [x[2] for x in FI_seqs]
	magn_seqs = [x[3] for x in FI_seqs]

	# Choose sequences optimizing FI under contraints
	chosen_idx, common_per_pos, EV_per_pos = optimize_FI_distribution(
		n_wanted = n_confusing_wanted.n,
		FIs = [x[1] for x in FI_seqs],
		common_seqs = common_seqs,
		magn_seqs = magn_seqs,
		ω_FI = 0.08,
		filename = "results/exp_sequences/pilot6_opt.jld2"
	)

	@assert length(vcat(chosen_idx...)) == nrow(PILT_block_attr) "Number of saved optimize sequences does not match number of sequences needed. Delete file and rerun."

	# Shuffle chosen sequences
	rng = Xoshiro(random_seed)
	shuffle!.(rng, chosen_idx)

	# Unpack chosen sequences
	chosen_common = [[common_seqs[s][idx[1]] for idx in chosen_idx[s]]
		for s in eachindex(common_seqs)]

	chosen_magn = [[magn_seqs[s][idx[2]] for idx in chosen_idx[s]]
		for s in eachindex(magn_seqs)]

	# Repack into DataFrame	
	n_sequences = sum(length.(chosen_common))
	task = DataFrame(
		idx = repeat(1:n_sequences, inner = PILT_trials_per_block),
		sequence = repeat(vcat([1:length(x) for x in chosen_common]...), 
			inner = PILT_trials_per_block),
		trial = repeat(1:PILT_trials_per_block, n_sequences),
		feedback_common = vcat(vcat(chosen_common...)...),
		variable_magnitude = vcat(vcat(chosen_magn...)...)
	)

	# Create n_confusing and fifty_high varaibles
	DataFrames.transform!(
		groupby(task, :idx),
		:feedback_common => (x -> PILT_trials_per_block - sum(x)) => :n_confusing,
		:variable_magnitude => (x -> 1. in x) => :fifty_high
	)

	# Add sequnces variable to PILT_block_attr
	DataFrames.transform!(
		groupby(PILT_block_attr, [:n_confusing, :fifty_high]),
		:block => (x -> shuffle(rng, 1:length(x))) => :sequence
	)


	# Combine with block attributes
	task = innerjoin(
		task,
		PILT_block_attr,
		on = [:n_confusing, :fifty_high, :sequence],
		order = :left
	)


	@assert nrow(task) == length(vcat(vcat(chosen_common...)...)) "Problem with join operation"
	@assert nrow(unique(task[!, [:block]])) == PILT_total_blocks "Problem with join operation"
		
	@assert mean(task.fifty_high) == 0.5 "Proportion of blocks with 50 pence in high magnitude option expected to be 0.5"

	# Sort by block
	sort!(task, [:block, :trial])

	# Remove auxillary variables
	select!(task, Not([:sequence, :idx]))

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

	task, common_per_pos, EV_per_pos
end

# ╔═╡ e1b66454-7700-4d79-9df2-59e13bd031ee
# Assign stimulus images
PILT_stimuli = let random_seed = 0

	# Shuffle categories
	shuffle!(Xoshiro(random_seed), categories)

	# Assign stimulus pairs
	stimuli = assign_stimuli_and_optimality(;
		n_phases = 2,
		n_pairs = fill(1, PILT_total_blocks),
		categories = categories,
		random_seed = random_seed,
		ext = "jpg"
	)

	@info "Proportion of blocks on which the novel category is optimal : $(mean(stimuli.optimal_A))"

	rename!(stimuli, :phase => :session) # For compatibility with multi-phase sessions

	stimuli
end

# ╔═╡ 1cedb080-d46b-45a2-b781-aadb1d9a48d0
# Add stimulus assignments to sequences DataFrame, and assign right / left
PILT_task = let random_seed = 1

	# Join stimuli and sequences
	task = innerjoin(
		vcat(
			insertcols(PILT_sequences, 1, :session => 1), 
			insertcols(PILT_sequences, 1, :session => 2)),
		PILT_stimuli,
		on = [:session, :block],
		order = :left
	)

	@assert nrow(task) == nrow(PILT_sequences) * 2 "Problem in join operation"

	# Assign right / left, equal proportions within each pair
	rng = Xoshiro(random_seed)

	DataFrames.transform!(
		groupby(task, [:session, :block]),
		:block => 
			(x -> shuffled_fill([true, false], length(x); rng = rng)) =>
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

	# Add empty variables needed for experiment script
	insertcols!(
		task,
		:n_stimuli => 2,
		:n_groups => 1,
		:stimulus_group => 1,
		:stimulus_group_id => task.block,
		:stimulus_middle => "",
		:feedback_middle => "",
		:optimal_side => "",
		:present_pavlovian => true
	)

	task
end

# ╔═╡ 6507d118-6977-4023-b43c-0d483a720f96
# Validate task DataFrame
let task = PILT_task
	@assert maximum(task.block) == length(unique(task.block)) "Error in block numbering"

	@assert all(combine(groupby(task, :session), 
		:block => issorted => :sorted).sorted) "Task structure not sorted by block"

	@assert all(combine(groupby(task, [:session, :block]), 
		:trial => issorted => :sorted).sorted) "Task structure not sorted by trial number"

	@assert all(sign.(task.valence) == sign.(task.feedback_right)) "Valence doesn't match feedback sign"

	@assert all(sign.(task.valence) == sign.(task.feedback_left)) "Valence doesn't match feedback sign"

	@assert sum(unique(task[!, [:session, :block, :valence]]).valence) == 0 "Number of reward and punishment blocks not equal"

	@info "Overall proportion of common feedback: $(round(mean(task.feedback_common), digits = 2))"

	@assert all((task.variable_magnitude .== abs.(task.feedback_right)) .| 
		(task.variable_magnitude .== abs.(task.feedback_left))) ":variable_magnitude, which is used for sequnece optimization, doesn't match end result column :feedback_right no :feedback_left"

	# Count losses to allocate coins in to safe for beginning of task
	worst_loss = filter(x -> x.valence == -1, task) |> 
		df -> ifelse.(
			df.feedback_right .< df.feedback_left, 
			df.feedback_right, 
			df.feedback_left) |> 
		countmap

	@info "Worst possible loss in this task is of these coin numbers: $worst_loss"

end

# ╔═╡ a64f05e8-9eb6-435c-850e-b69c04c6721b
let
	save_to_JSON(PILT_task, "results/pilot6_PILT.json")
	CSV.write("results/pilot6_PILT.csv", PILT_task)
end

# ╔═╡ b05f81e5-837d-4a7d-8b6a-73628568e106
# Visualize PILT seuqnce
let task = PILT_task

	f = Figure(size = (700, 300))

	# Proportion of confusing by trial number
	confusing_location = combine(
		groupby(task, [:session, :trial]),
		:feedback_common => (x -> mean(.!x)) => :feedback_confusing
	)

	mp1 = data(confusing_location) * mapping(
		:trial => "Trial", 
		:feedback_confusing => "Prop. confusing feedback",
		color = :session => nonnumeric => "Session",
		group = :session => nonnumeric
	) * visual(ScatterLines)

	plt1 = draw!(f[1,1], mp1)

	legend!(
		f[1,1], 
		plt1,
		tellwidth = false,
		tellheight = false,
		valign = 1.2,
		halign = 0.,
		framevisible = false
	)

	# Plot confusing trials by block
	fp = insertcols(
		task,
		:color => ifelse.(
			task.feedback_common,
			(task.valence .+ 1) .÷ 2,
			fill(3, nrow(task))
		)
	)

	for (i, s) in enumerate(unique(fp.session))
		mp = data(filter(x -> x.session == s, fp)) * mapping(
			:trial => "Trial",
			:block => "Block",
			:color
		) * visual(Heatmap)

		draw!(f[1,i+1], mp, axis = (; yreversed = true, subtitle = "Session $i"))
	end
	


	save("results/pilot6_pilt_trial_plan.png", f, pt_per_unit = 1)

	f

end

# ╔═╡ f3391ced-7620-40b9-9c05-341e2c7106d2
md"""## Post-PILT test"""

# ╔═╡ 3268d6e0-9dd6-46b6-8234-c04bd63a48ef
function create_test_sequence(
	pilt_task::DataFrame;
	random_seed::Int64, 
	same_weight::Float64 = 6.5,
	test_n_blocks::Int64 = PILT_test_n_blocks
) 
	
	rng = Xoshiro(random_seed)

	# Extract stimuli and their common feedback from task structure
	stimuli = vcat([rename(
		pilt_task[pilt_task.feedback_common, [:session, :block, Symbol("stimulus_$s"), Symbol("feedback_$s")]],
		Symbol("stimulus_$s") => :stimulus,
		Symbol("feedback_$s") => :feedback
	) for s in ["right", "left"]]...)

	# Summarize magnitude per stimulus
	stimuli = combine(
		groupby(stimuli, [:session, :block, :stimulus]),
		:feedback => (x -> mean(unique(x))) => :magnitude
	)

	# Step 1: Identify unique stimuli
	unique_stimuli = unique(stimuli.stimulus)

	# Step 2: Define existing pairs
	create_pair_list(d) = [filter(x -> x.block == p, d).stimulus 
		for p in unique(stimuli.block)]

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

# ╔═╡ 160bb7c6-cdd4-4c42-87e6-6ccf987d3f7f
# Choose test sequence with best stats
function find_best_test_sequence(
	task::DataFrame; # PILT task structure
	n_seeds::Int64 = 100, # Number of random seeds to try
	same_weight::Float64 = 4.1 # Weight reducing the number of same magntiude pairs
) 

	# Initialize stats variables
	prop_block = []
	prop_valence = []
	n_magnitude = []

	# Run over seeds
	for s in 1:n_seeds
		_, pb, pv, nm = create_test_sequence(task, random_seed = s, same_weight = same_weight)

		push!(prop_block, pb)
		push!(prop_valence, pv)
		push!(n_magnitude, nm)
	end

	# First, choose a sequence with the minimal number of same-magnitude pairs
	pass_magnitude = (1:n_seeds)[n_magnitude .== 
		minimum(filter(x -> !isnan(x), n_magnitude))]

	@assert !isempty(pass_magnitude)

	# Apply magnitude selection
	prop_block = prop_block[pass_magnitude]
	prop_valence = prop_valence[pass_magnitude]

	# Compute deviation from goal
	dev_block = abs.(prop_block .- 1/3)
	dev_valence = abs.(prop_block .- 0.5)

	# Choose best sequence
	chosen = pass_magnitude[argmin(dev_block .+ dev_valence)]

	# Return sequence and stats
	return create_test_sequence(task, random_seed = chosen, same_weight = same_weight)
end

# ╔═╡ 2dd93d65-9b8f-4d6f-8a96-d9dd0c2ea345
PILT_test = let task = PILT_task
	
	# Find test sequence for each session
	PILT_test = []
	for s in 1:2
		test, pb, pv, nm = find_best_test_sequence(
			filter(x -> x.session == s, task),
			n_seeds = 100, # Number of random seeds to try
			same_weight = 1.1 # Weight reducing the number of same magntiude pairs
		) 

		# Add session variable
		insertcols!(test, 1, :session => s)

		push!(PILT_test, test)
	
		@info "Session $s: proportion of same block pairs: $pb"
		@info "Session $s: proportion of same valence pairs: $pv"
		@info "Session $s: number of same magnitude pairs: $nm"
	end

	# Concatenate together
	PILT_test = vcat(PILT_test...)

	# Create magnitude_pair variable
	PILT_test.magnitude_pair = [sort([r.magnitude_left, r.magnitude_right]) for r in eachrow(PILT_test)]

	@assert all([diff(collect(extrema(values(countmap(filter(x -> x.session == s, PILT_test).magnitude_pair)))))[1] == 1 for s in 1:2]) "Distribution across magnitude pairs could be more equal"

	@assert all([length(unique(filter(x -> x.session == 2, PILT_test).magnitude_pair)) for s in 1:2] .== binomial(length(unique(PILT_test.magnitude_left)), 2)) "Wrong number of magnitude paris"

	PILT_test
end

# ╔═╡ edca0f2a-dba2-4882-944c-3b28e6c14a90
let
	save_to_JSON(PILT_test, "results/pilot6_PILT_test.json")
	CSV.write("results/pilot6_PILT_test.csv", PILT_test)
end

# ╔═╡ 663bb9a3-1a1e-4171-96bb-1c699b8dfb9c
md"""## WM task"""

# ╔═╡ c9861999-aa41-4fd3-8b15-afe380be0483
# WM Parameters
begin

	WM_set_sizes = [1, 7]
	WM_blocks_per_set = [6, 4] # Including reward and punishment
	WM_trials_per_triplet = 7

	# Total number of blocks
	WM_n_total_blocks = sum(WM_blocks_per_set)

	# Total number of triplets
	WM_n_total_tiplets = sum(WM_set_sizes .* WM_blocks_per_set)

	# Full deterministic task
	WM_n_confusing = fill(0, WM_n_total_blocks) # Per block

	# For uniform shuffling of triplets in block
	WM_triplet_mini_block_size = 2

end

# ╔═╡ 9f9cc1b2-50b3-4919-a785-31fe4a45be81
# Assign valence and set size per block
WM_valence_set_size = let random_seed = 2
	
	# # All combinations of set sizes and valence
	@assert all(iseven.(WM_blocks_per_set)) # Requisite for code below

	valence_set_size = DataFrame(
		n_triplets = vcat([fill(x, n) for (x, n) in zip(WM_set_sizes, WM_blocks_per_set)]...)
	)
		
	valence_set_size.valence = collect(Iterators.take(Iterators.cycle([1, -1]), 
		nrow(valence_set_size)))

	# Shuffle set size and valence, making sure set size rises gradually and valence is positive in the first two blocks
	rng = Xoshiro(random_seed)
	
	while (valence_set_size[1:length(WM_set_sizes), :n_triplets] != 
		WM_set_sizes) ||
		any(valence_set_size.valence[1:2] .== -1)

		DataFrames.transform!(
			valence_set_size,
			:n_triplets => (x -> shuffle(rng, 1:length(x))) => :block
		)
		
		sort!(valence_set_size, :block)
	end

	# Add n_confusing
	valence_set_size.n_confusing = WM_n_confusing


	# Order columns
	select!(valence_set_size, :block, Not(:block))

	# Return
	valence_set_size
end

# ╔═╡ 29c4c572-4080-4264-a09c-86d1f416b956
# # Create trial sequence
WM_feedback_sequence = let random_seed = 1,
	valence_set_size = WM_valence_set_size

	rng = Xoshiro(random_seed)

	# Create pair list
	task = combine(
		groupby(valence_set_size, 
			[:block, :n_triplets, :valence, :n_confusing]),
		:n_triplets => (x -> repeat(vcat([1:xi for xi in x]...), inner = WM_trials_per_triplet)) => :triplet
	)

	# Add cumulative pair variable
	triplets = sort(unique(task[!, [:block, :triplet, :n_triplets, :valence]]), [:block, :triplet])
	triplets.ctriplet = 1:nrow(triplets)
	

	# Join into task
	task = innerjoin(
		task,
		triplets[!, Not(:valence, :n_triplets)],
		on = [:block, :triplet],
		order = :left
	)

	# Add apperance count variable
	DataFrames.transform!(
		groupby(task, [:block, :triplet]),
		:block => (x -> 1:length(x)) => :appearance
	)

	sort!(task, [:block, :appearance, :triplet])

	task.mini_block = (task.appearance .÷ WM_triplet_mini_block_size) .+ 1

	# Shuffle pair appearance	
	DataFrames.transform!(
		groupby(task, [:block, :mini_block]),
		:triplet => (x -> shuffle(rng, 1:length(x))) => :order_var
	)

	sort!(task, [:block, :mini_block, :order_var])

	# Reorder apperance
	DataFrames.transform!(
		groupby(task, [:block, :triplet]),
		:block => (x -> 1:length(x)) => :appearance
	)

	# Trial counter
	DataFrames.transform!(
		groupby(task, :block),
		:block => (x -> 1:length(x)) => :trial
	)

	# Create deterministic sequence
	task[!, :feedback_common] .= true

	# Compute optimal and suboptimal feedback
	task.feedback_optimal = ifelse.(
		task.valence .> 0,
		fill(1., nrow(task)),
		fill(-0.01, nrow(task))
	)

	task.feedback_suboptimal = ifelse.(
		task.valence .> 0,
		fill(0.01, nrow(task)),
		fill(-1, nrow(task)),
	)

	task

end

# ╔═╡ c9d4cb13-bdab-4c7c-87d8-6ba022f1c4fd
function assign_triplet_stimuli_and_optimality(;
	n_phases::Int64,
	n_groups::Vector{Int64}, # Number of groups in each block. Assume same for all phases
	categories::Vector{String} = [('A':'Z')[div(i - 1, 26) + 1] * ('a':'z')[rem(i - 1, 26)+1] 
		for i in 1:(sum(n_pairs) * 2 * n_phases + n_phases)],
	random_seed::Int64 = 1
	)

	total_n_groups = sum(n_groups) # Number of pairs needed
	
	@assert rem(length(n_groups), 2) == 0 "Code only works for even number of blocks per sesion"

	# Compute how many repeating categories we will have
	n_repeating = sum(min.(n_groups[2:end], n_groups[1:end-1]))

	@info "Determined that $n_repeating pairs per phase will have a repating category, out of $(sum(n_groups)) groups."

	rng = Xoshiro(random_seed)

	# Assign whether repeating is optimal and shuffle
	repeating_optimal = vcat([shuffled_fill([true, false, false], n_repeating, rng = rng) for p in 1:n_phases]...)

	# Assign whether categories that cannot repeat are optimal
	rest_optimal = vcat([shuffled_fill([true, false, false], total_n_groups - n_repeating, rng = rng) for p in 1:n_phases]...)

	# Initialize vectors for stimuli. A novel to be repeated, B just novel, C may be repeating
	stimulus_A = []
	stimulus_B = []
	stimulus_C = []
	optimal_C = []
	
	for j in 1:n_phases
		for (i, p) in enumerate(n_groups)

	
			# Choose repeating categories for this block
			n_repeating = ((i > 1) && minimum([p, n_groups[i - 1]])) * 1
			append!(
				stimulus_C,
				stimulus_A[(end - n_repeating + 1):end]
			)
	
			# Fill up stimulus_repeating with novel categories if not enough to repeat
			if (p - n_repeating) > 0
				for _ in 1:(p - n_repeating)
					push!(
						stimulus_C,
						popfirst!(categories)
					)
				end
			end
			
			# Choose novel categories for this block
			for _ in 1:p
				push!(
					stimulus_A,
					popfirst!(categories)
				)

				push!(
					stimulus_B,
					popfirst!(categories)
				)
			end

			# Populate who is optimal vector
			for _ in 1:(n_repeating)
				push!(
					optimal_C,
					popfirst!(repeating_optimal)
				)
			end

			if (p - n_repeating) > 0
				for _ in 1:(p - n_repeating)
					push!(
						optimal_C,
						popfirst!(rest_optimal)
					)
				end
			end
		end
	end

	stimulus_A = (x -> x * "1.jpg").(stimulus_A)
	stimulus_B = (x -> x * "1.jpg").(stimulus_B)
	stimulus_C = (x -> x * "2.jpg").(stimulus_C)

	optimal_stimulus = ifelse.(
		optimal_C,
		"C",
		"X"
	)

	optimal_stimulus[optimal_stimulus .== "X"] = shuffled_fill(["A","B"], sum(.!optimal_C), rng = rng)

	return DataFrame(
		phase = repeat(1:n_phases, inner = total_n_groups),
		block = repeat(
			vcat([fill(i, p) for (i, p) in enumerate(n_groups)]...), n_phases),
		triplet = repeat(
			vcat([1:p for p in n_groups]...), n_phases),
		stimulus_A = stimulus_A,
		stimulus_B = stimulus_B,
		stimulus_C = stimulus_C,
		optimal_stimulus = optimal_stimulus
	)

end


# ╔═╡ 7185295b-0f6f-44cc-b5c4-6425cbff7e2f
# Assign stimulus images
WM_stimuli = let random_seed = 0

	shuffle!(Xoshiro(random_seed), categories)

	# Filter out used stimulus categories
	remaining_categories = filter(x -> x ∉ unique((s -> replace(s, ".jpg" => "")[1:(end-1)]).(PILT_task.stimulus_left)), categories)

	# Assign stimulus pairs
	stimuli = assign_triplet_stimuli_and_optimality(;
		n_phases = 2,
		n_groups = WM_valence_set_size.n_triplets,
		categories = remaining_categories,
		random_seed = random_seed
	)

	rename!(stimuli, :phase => :session) # For compatibility with multi-phase sessions
end

# ╔═╡ 1b612ce4-557c-450c-af2f-721c975970db
# Add stimulus assignments to sequences DataFrame, and assign right / left
WM_task = let random_seed = 0

	# Join stimuli and sequences
	task = innerjoin(
		vcat(
			insertcols(WM_feedback_sequence, 1, :session => 1), 
			insertcols(WM_feedback_sequence, 1, :session => 2)
		),
		WM_stimuli,
		on = [:session, :block, :triplet],
		order = :left
	)

	@assert nrow(task) == nrow(WM_feedback_sequence) * 2 "Problem in join operation"

	# Assign location, equal proportions within each pair
	rng = Xoshiro(random_seed)

	DataFrames.transform!(
		groupby(task, [:block, :ctriplet]),
		:block => 
			(x -> shuffled_fill(collect(permutations(["A", "B", "C"])), length(x); random_seed = random_seed)) =>
			:stimulus_locations
	)

	# Create stimulus_right, stimulus_middle, and stimulus_left variables
	task.stimulus_left = [task[i, Symbol("stimulus_$(x[1])")] 
		for (i,x) in enumerate(task.stimulus_locations)]
	
	task.stimulus_middle = [task[i, Symbol("stimulus_$(x[2])")] 
		for (i,x) in enumerate(task.stimulus_locations)]

	
	task.stimulus_right = [task[i, Symbol("stimulus_$(x[3])")] 
		for (i,x) in enumerate(task.stimulus_locations)]

	# Create optimal_side variable
	task.optimal_side = [["left", "middle", "right"][findfirst(task.stimulus_locations[i] .== x)] for (i,x) in enumerate(task.optimal_stimulus)]

	# Create feedback_right, feedback_middle, feedback_left variables
	task.feedback_left = ifelse.(
		task.optimal_side .== "left",
		task.feedback_optimal,
		task.feedback_suboptimal
	)

	task.feedback_middle = ifelse.(
		task.optimal_side .== "middle",
		task.feedback_optimal,
		task.feedback_suboptimal
	)

	task.feedback_right = ifelse.(
		task.optimal_side .== "right",
		task.feedback_optimal,
		task.feedback_suboptimal
	)

	# Add variables needed for experiment code
	insertcols!(
		task,
		:n_stimuli => 3,
		:optimal_right => "",
		:present_pavlovian => false
	)

	rename!(
		task,
		:n_triplets => :n_groups,
		:triplet => :stimulus_group,
		:ctriplet => :stimulus_group_id
	)


	task
end

# ╔═╡ 5f481788-392e-4467-bd49-ea10c2805fa0
# Validate task DataFrame
let task = WM_task
	@assert maximum(task.block) == length(unique(task.block)) "Error in block numbering"

	@assert all(combine(groupby(task, [:session]), 
		:block => issorted => :sorted).sorted) "Task structure not sorted by block"

	@assert all(combine(groupby(task, [:session, :block]), 
		:trial => issorted => :sorted).sorted) "Task structure not sorted by trial number"

	@assert all(sign.(task.valence) == sign.(task.feedback_right)) "Valence doesn't match feedback sign"

	@assert all(sign.(task.valence) == sign.(task.feedback_left)) "Valence doesn't match feedback sign"

	@assert sum(unique(task[!, [:session, :block, :valence]]).valence) == 0 "Number of reward and punishment blocks not equal"

	@info "Overall proportion of common feedback: $(round(mean(task.feedback_common), digits = 2))"

	@assert all(combine(groupby(task, :stimulus_group_id),
		:appearance => maximum => :max_appear
	).max_appear .== WM_trials_per_triplet) "Didn't find exactly $WM_trials_per_triplet apperances per pair"

	# Count losses to allocate coins in to safe for beginning of task
	worst_loss = filter(x -> x.valence < 0, task) |> 
		df -> ifelse.(
			df.feedback_right .< df.feedback_left, 
			df.feedback_right, 
			df.feedback_left) |> 
		countmap

	@info "Worst possible loss in this task is of these coin numbers: $worst_loss"

end

# ╔═╡ 3a2b86f4-389d-4e98-ab40-1d68b3006fd9
let
	save_to_JSON(WM_task, "results/pilot6_WM.json")
	CSV.write("results/pilot6_WM.csv", WM_task)
end

# ╔═╡ 9c324f0b-9d94-4623-9006-61518cb897fc
let

	used_categories = unique((s -> replace(s, ".jpg" => "")[1:(end-1)]).(unique(
		vcat(
			WM_task.stimulus_right,
			PILT_task.stimulus_right
		)
	)))

	filter(x -> x ∉ used_categories, categories)


end

# ╔═╡ db145989-72d6-4390-ae15-ccad606e36c1
md"""## Reversal task"""

# ╔═╡ b2def05e-0044-4942-b8e5-38ac335cf25d
# Reversal task parameters
begin
	rev_n_blocks = 30
	rev_n_trials = 80
	rev_prop_confusing = vcat([0, 0.1, 0.1, 0.2, 0.2], fill(0.3, rev_n_blocks - 5))
	rev_criterion = vcat(
		[8, 7, 6, 6, 5], 
		shuffled_fill(
			3:8, 
			rev_n_blocks - 5; 
			rng = Xoshiro(0)
		)
	)
end

# ╔═╡ e34841e2-c400-4f99-a4b0-57bf6e38646a
function find_lcm_denominators(props::Vector{Float64})
	# Convert to rational numbers
	rational_props = rationalize.(props)
	
	# Extract denominators
	denominators = [denominator(r) for r in rational_props]
	
	# Compute the LCM of all denominators
	smallest_int = foldl(lcm, denominators)
end

# ╔═╡ 03fe7c82-fbfe-463c-8c45-21ad9b8886eb
# Reversal task structure
rev_feedback_optimal = let random_seed = 1

	# Compute minimal mini block length to accomodate proportions
	mini_block_length = find_lcm_denominators(rev_prop_confusing)
	@info "Randomizing in miniblocks of $mini_block_length trials"

	# Function to create high magnitude values for miniblock
	mini_block_high_mag(p, rng) = shuffle(rng, vcat(
		fill(1., round(Int64, mini_block_length * (1-p))),
		fill(0.01, round(Int64, mini_block_length * p))
	))

	# Function to create high magntidue values for block
	block_high_mag(p, rng) = 
		vcat(
			[mini_block_high_mag(p, rng) 
				for _ in 1:(div(rev_n_trials, mini_block_length))]...)

	# Set random seed
	rng = Xoshiro(random_seed)

	# Initialize
	feedback_optimal = Vector{Vector{Float64}}()

	# Make sure first sixs blocks don't start with confusing feedback on first trial
	dist_diff = 11
	while isempty(feedback_optimal) || 
		!all([bl[1] != 0.01 for bl in feedback_optimal[1:6]]) || dist_diff > 2

		# Assign blocks
		feedback_optimal = [block_high_mag(p, rng) for p in rev_prop_confusing]

		# Check distribution of confusing feedback
		dist = (x -> permutedims(reshape(x, div(length(x), 10), 10), (2,1))).(feedback_optimal)

		dist = vcat(dist...)

		dist = vec(sum(dist .== 1., dims = 1))
		dist_diff = maximum(abs.(diff(dist)))
	end

	# Function to compute feedback_suboptimal from feedback_optimal
	inverter(x) = 1 ./ (100 * x)

	# Create timeline variables
	timeline = [[Dict(
		:block => bl,
		:trial => t,
		:feedback_left => isodd(bl) ? feedback_optimal[bl][t] : 
			inverter(feedback_optimal[bl][t]),
		:feedback_right => iseven(bl) ? feedback_optimal[bl][t] : 
			inverter(feedback_optimal[bl][t]),
		:optimal_right => iseven(bl),
		:criterion => rev_criterion[bl]
	) for t in 1:rev_n_trials] for bl in 1:rev_n_blocks]
	
	# Convert to JSON String
	json_string = JSON.json(timeline)

	# Add JS variable assignment
	json_string = "const reversal_json = '$json_string';"
		
	# Write the JSON string to the file
	open("results/pilot6_reversal_sequence.js", "w") do file
	    write(file, json_string)
	end

	feedback_optimal
end

# ╔═╡ 47947f68-6a0b-4a27-b58c-5b28a220dad5
let

	f = Figure(size = (700, 600))

	mp1 = data(
		DataFrame(
			block = repeat(1:rev_n_blocks, 2),
			prop = vcat(rev_prop_confusing, 1. .- rev_prop_confusing),
			feedback_type = repeat(["Confusing", "Common"], inner = rev_n_blocks)
		)
	) * mapping(
		:block => "Block", 
		:prop => "Proportion of trials", 
		color = :feedback_type => "", 
		stack = :feedback_type) * visual(BarPlot)

	plt1 = draw!(f[1,1], mp1, axis = (; yticks = [0., 0.5, 0.7, 0.8, 0.9, 1.]))

	legend!(f[1,1], plt1, 
		valign = 1.18,
		tellheight = false, 
		framevisible = false,
		orientation = :horizontal,
		labelsize = 14
	)

	rowgap!(f[1,1].layout, 0)

	rev_confusing = DataFrame(
		block = repeat(1:rev_n_blocks, inner = rev_n_trials),
		trial = repeat(1:rev_n_trials, outer = rev_n_blocks),
		feedback_common = vcat(rev_feedback_optimal...) .== 1.
	)

	mp2 = data(rev_confusing) * 
		mapping(:trial => "Trial", :block => "Block", :feedback_common) *
		visual(Heatmap)

	draw!(f[1,2], mp2, 
		axis = (; 
			yreversed = true, 
			yticks = [1, 10, 20, 30],
			subtitle = "Confusing Feedback"
		)
	)

	insertcols!(
		rev_confusing,
		:rel_trial => rev_confusing.trial .- div.(rev_confusing.trial, 10) .* 10 .+ 1
	)
	
	mp3 = data(
		combine(
			groupby(rev_confusing, :rel_trial), 
			:feedback_common => (x -> mean(.!x)) => :feedback_confusing
		)
	) * mapping(
		:rel_trial => "Trial", 
		:feedback_confusing => "Prop. confusing feedback"
	) * visual(ScatterLines)

	draw!(f[2,1], mp3)

	mp4 = mapping(1:rev_n_blocks => "Block", rev_criterion => "# optimal choices)") * visual(ScatterLines)

	draw!(f[2,2], mp4, axis = (; 
		yticks = 3:8, 
		xticks = [1, 10, 20, 30], 
		subtitle = "Reversal criterion")
	)

	save("results/pilot6_reversal_sequence.png", f, pt_per_unit = 1)

	f

end

# ╔═╡ Cell order:
# ╠═99c994e4-9c36-11ef-2c8f-d5829be639eb
# ╠═65ec8b8f-9eba-467b-bb19-9f0c72b8933e
# ╠═3a5cf6e9-1cee-4fd3-b0db-8fc02b7747fb
# ╟─ea917db6-ec27-454f-8b4e-9df65d65064b
# ╠═381e61e2-7d51-4070-8ad1-ce9e63015eb6
# ╠═687f5ae6-86c6-449f-86f5-5ed359e6d580
# ╠═31128edd-5d2d-49e9-8f65-842bb42639f9
# ╠═c7c5b78e-ad76-4877-8c80-af9151105544
# ╠═e1b66454-7700-4d79-9df2-59e13bd031ee
# ╠═1cedb080-d46b-45a2-b781-aadb1d9a48d0
# ╠═6507d118-6977-4023-b43c-0d483a720f96
# ╠═a64f05e8-9eb6-435c-850e-b69c04c6721b
# ╠═b05f81e5-837d-4a7d-8b6a-73628568e106
# ╟─f3391ced-7620-40b9-9c05-341e2c7106d2
# ╠═2dd93d65-9b8f-4d6f-8a96-d9dd0c2ea345
# ╠═edca0f2a-dba2-4882-944c-3b28e6c14a90
# ╠═3268d6e0-9dd6-46b6-8234-c04bd63a48ef
# ╠═160bb7c6-cdd4-4c42-87e6-6ccf987d3f7f
# ╟─663bb9a3-1a1e-4171-96bb-1c699b8dfb9c
# ╠═c9861999-aa41-4fd3-8b15-afe380be0483
# ╠═9f9cc1b2-50b3-4919-a785-31fe4a45be81
# ╠═29c4c572-4080-4264-a09c-86d1f416b956
# ╠═7185295b-0f6f-44cc-b5c4-6425cbff7e2f
# ╠═1b612ce4-557c-450c-af2f-721c975970db
# ╠═5f481788-392e-4467-bd49-ea10c2805fa0
# ╠═3a2b86f4-389d-4e98-ab40-1d68b3006fd9
# ╠═9c324f0b-9d94-4623-9006-61518cb897fc
# ╠═c9d4cb13-bdab-4c7c-87d8-6ba022f1c4fd
# ╠═db145989-72d6-4390-ae15-ccad606e36c1
# ╠═b2def05e-0044-4942-b8e5-38ac335cf25d
# ╠═e34841e2-c400-4f99-a4b0-57bf6e38646a
# ╠═03fe7c82-fbfe-463c-8c45-21ad9b8886eb
# ╠═47947f68-6a0b-4a27-b58c-5b28a220dad5
