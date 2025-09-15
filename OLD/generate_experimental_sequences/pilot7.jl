### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ 2d7211b4-b31e-11ef-3c0b-e979f01c47ae
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
	include("$(pwd())/model_utils.jl")
	include("$(pwd())/plotting_utils.jl")
	include("$(pwd())/fetch_preprocess_data.jl")
	include("$(pwd())/generate_experimental_sequences/sequence_utils.jl")
	nothing
end

# ╔═╡ 114f2671-1888-4b11-aab1-9ad718ababe6
begin
	# Set theme	
	th = merge(theme_minimal(), Theme(
		font = "Helvetica",
		fontsize = 16,
		Axis = (
			xticklabelsize = 14,
			yticklabelsize = 14,
			spinewidth = 1.5,
			xtickwidth = 1.5,
			ytickwidth = 1.5
		)
	))
	
	set_theme!(th)
end

# ╔═╡ de74293f-a452-4292-b5e5-b4419fb70feb
categories = let
	categories = (s -> replace(s, ".jpg" => "")[1:(end-2)]).(readdir("generate_experimental_sequences/pilot7_stims"))

	# Keep only categories where we have two files exactly
	keeps = filter(x -> last(x) == 2, countmap(categories))

	categories = filter(x -> x in keys(keeps), unique(categories))

	@info "Found $(length(categories)) categories"

	categories
end

# ╔═╡ ffe06202-d829-4145-ae26-4a95449d64e6
md"""# RLWM"""

# ╔═╡ 05f25eb8-3a48-4d16-9837-84d1fdf5c806
triplet_order = let
	triplet_order = DataFrame(CSV.File(
		"generate_experimental_sequences/pilot7_wm_stimulus_sequence_longer.csv"))

	select!(
		triplet_order, 
		:stimset => :stimulus_group,
		:delay
	)
end

# ╔═╡ c05d90b6-61a7-4f9e-a03e-3e11791da6d0
# RLWM Parameters
begin
	RLWM_prop_fifty = 0.2
	RLWM_shaping_n = 20
	WM_test_n_blocks = 2
end

# ╔═╡ 6eadaa9d-5ed0-429b-b446-9cc7fbfb52dc
length(categories)

# ╔═╡ 3fa8c293-ac47-4acd-bdb7-9313286ee464
function assign_triplet_stimuli_RLWM(
	categories::AbstractVector,
	n_triplets::Int64;
	rng::AbstractRNG = Xoshiro(0)
)

	dicts = [
		Dict(
			:stimulus_group => i,
			:stimulus_A => popat!(
				categories, 
				rand(rng, 1:length(categories))
			) * "_1.jpg",
			:stimulus_B => popat!(
				categories, 
				rand(rng, 1:length(categories))
			) * "_1.jpg",
			:stimulus_C => popat!(
					categories, 
					rand(rng, 1:length(categories))
			) * "_1.jpg"
		)
		for i in 1:n_triplets
	]

	return select(
		DataFrame(dicts),
		:stimulus_group,
		:stimulus_A,
		:stimulus_B,
		:stimulus_C
	)
	
end

# ╔═╡ f89e88c9-ebfc-404f-964d-acff5c7f8985
function integer_allocation(p::Vector{Float64}, n::Int)
    i = floor.(Int, p * n)  # Floor to ensure sum does not exceed n
    diff = n - sum(i)       # Remaining to distribute
    indices = sortperm(p * n .- i, rev=true)  # Sort by largest remainder
    i[indices[1:diff]] .+= 1  # Distribute the remainder
    return i
end

# ╔═╡ 68873d3e-054d-4ab4-9d89-73586bb0370e
function prop_fill_shuffle(
	values::AbstractVector,
	props::Vector{Float64},
	n::Int64;
	rng::AbstractRNG = Xoshiro(1)
)
	# Find integers
	ints = integer_allocation(props, n)
	
	# Fill
	res = [fill(v, i) for (v, i) in zip(values, ints)]
	
	# Return shuffled
	shuffle(rng, vcat(res...))
end

# ╔═╡ f5916a9f-ddcc-4c03-9328-7dd76c4c74b2
# Create deterministic block
RLWM_det_block = let rng = Xoshiro(0)
	n_trials = nrow(triplet_order)

	# Basic variables
	det_block = DataFrame(
		block = fill(1, n_trials),
		trial = 1:n_trials,
		stimulus_group = triplet_order.stimulus_group,
		delay = triplet_order.delay
	)

	# Draw optimal feedback
	DataFrames.transform!(
		groupby(det_block, :stimulus_group),
		:trial => (x -> prop_fill_shuffle(
			[1., 0.5],
			[1 - RLWM_prop_fifty, RLWM_prop_fifty],
			length(x),
			rng = rng
			)
		) => :feedback_optimal
	) 

	@info "Proportion fifty pence: $(mean(det_block.feedback_optimal .== 0.5))"

	# Assign stimuli categories
	stimuli = assign_triplet_stimuli_RLWM((categories),
		maximum(det_block.stimulus_group);
		rng = rng
	)

	# Merge with trial structure
	det_block = innerjoin(
		det_block,
		stimuli,
		on = :stimulus_group,
		order = :left
	)

	# Assign stimuli locations
	orderings = [join(p) for p in permutations(["A", "B", "C"])]
	DataFrames.transform!(
		groupby(det_block, :stimulus_group),
		:trial => (
			x -> shuffled_fill(orderings, length(x))
		) => :stimulus_ordering
	)

	@info "Proportion optimal in each location: $([round(mean((x -> x[i] == 'A').(det_block.stimulus_ordering)), digits = 3) for i in 1:3])"

	# Assign stimulus identity to location
	DataFrames.transform!(
		det_block,
		[:stimulus_ordering, :stimulus_A, :stimulus_B, :stimulus_C] =>
		((o, a, b, c) -> [[a b c][i, (Int(o[i][1]) - Int('A') + 1)] for i in 1:length(o)]) => :stimulus_left,
		[:stimulus_ordering, :stimulus_A, :stimulus_B, :stimulus_C] =>
		((o, a, b, c) -> [[a b c][i, (Int(o[i][2]) - Int('A') + 1)] for i in 1:length(o)]) => :stimulus_middle,
		[:stimulus_ordering, :stimulus_A, :stimulus_B, :stimulus_C] =>
		((o, a, b, c) -> [[a b c][i, (Int(o[i][3]) - Int('A') + 1)] for i in 1:length(o)]) => :stimulus_right		
	)

	# Assign feedback to location, by covention stimulus_A is optimal
	det_block.feedback_left = ifelse.(
		(x -> x[1] == 'A').(det_block.stimulus_ordering),
		det_block.feedback_optimal,
		fill(0.01, nrow(det_block))
	)
	
	det_block.feedback_middle = ifelse.(
		(x -> x[2] == 'A').(det_block.stimulus_ordering),
		det_block.feedback_optimal,
		fill(0.01, nrow(det_block))
	)

	det_block.feedback_right = ifelse.(
		(x -> x[3] == 'A').(det_block.stimulus_ordering),
		det_block.feedback_optimal,
		fill(0.01, nrow(det_block))
	)

	# For compatibility with probabilistic block
	det_block.feedback_common .= true

	det_block

end

# ╔═╡ e3bff0b9-306a-4bf9-8cbd-fe0e580bd118
RLWM = let RLWM = RLWM_det_block

	# Session variable
	RLWM.session .= 1

	# Valence variable
	RLWM.valence .= 1

	# Apperance variable
	DataFrames.transform!(
		groupby(RLWM, [:block, :stimulus_group]),
		:trial => (x -> 1:length(x)) => :appearance
	)

	# Cumulative triplet index
	RLWM.stimulus_group_id = RLWM.stimulus_group .+ (maximum(RLWM.stimulus_group) .* (RLWM.block .- 1))

	# Create optimal_side variable
	RLWM.optimal_side = [["left", "middle", "right"][findfirst('A', o)] for o in RLWM.stimulus_ordering]


	# Add variables needed for experiment code
	insertcols!(
		RLWM,
		:n_stimuli => 3,
		:optimal_right => "",
		:present_pavlovian => false,
		:n_groups => maximum(RLWM.stimulus_group),
		:early_stop => false
	)

	

	RLWM
end

# ╔═╡ f9be1490-8e03-445f-b36e-d8ceff894751
# Checks
let task = RLWM
	@assert all(combine(groupby(task, [:session]), 
		:block => issorted => :sorted).sorted) "Task structure not sorted by block"

	@assert all(combine(groupby(task, [:session, :block]), 
		:trial => issorted => :sorted).sorted) "Task structure not sorted by trial number"
	
	@assert all(sign.(task.valence) == sign.(task.feedback_right)) "Valence doesn't match feedback sign"

	@assert all(sign.(task.valence) == sign.(task.feedback_left)) "Valence doesn't match feedback sign"

	@info "Overall proportion of common feedback: $(round(mean(task.feedback_common), digits = 2))"

end

# ╔═╡ eecaac0c-e051-4543-988c-e969de3a8567
let
	save_to_JSON(RLWM, "results/pilot7_WM.json")
	CSV.write("results/pilot7_WM.csv", RLWM)
end

# ╔═╡ 414c9032-6458-4ad4-bd95-e756d221912f
md"""## Post-WM test"""

# ╔═╡ 56fcba3a-da77-42a4-bf91-f5c0962bdbf4
function prepare_for_finding_wm_test_sequence(
	task::DataFrame;
	stimulus_locations::Vector{String} = ["right", "middle", "left"]
)

	# Extract stimuli and their common feedback from task structure
	stimuli = vcat([rename(
		task[task.feedback_common, [:session, :block, :n_groups, :stimulus_group_id,  Symbol("stimulus_$s"), Symbol("feedback_$s")]],
		Symbol("stimulus_$s") => :stimulus,
		Symbol("feedback_$s") => :feedback
	) for s in stimulus_locations]...)

	# Summarize magnitude per stimulus
	stimuli = combine(
		groupby(stimuli, [:session, :block, :n_groups, :stimulus_group_id, :stimulus]),
		:feedback => (x -> mean(unique(x))) => :magnitude
	)

	# Create Dict of unique stimuli by magnitude
	magnitudes = unique(stimuli.magnitude)
	stimuli_magnitude = Dict(m => unique(filter(x -> x.magnitude == m, stimuli).stimulus) for m in magnitudes)

	#  Define existing pairs
	pairer(v) = [[v[i], v[j]] for i in 1:length(v) for j in i+1:length(v)]
	create_pair_list(d) = vcat([pairer(filter(x -> x.stimulus_group_id == p, d).stimulus) 
		for p in unique(stimuli.stimulus_group_id)]...)

	existing_pairs = create_pair_list(stimuli)

	existing_pairs = sort.(existing_pairs)

	return stimuli_magnitude, existing_pairs

end

# ╔═╡ b514eb90-a26c-4c44-82ce-d0d962fc940a
function create_wm_test_pairs(stimuli::Dict{Float64, Vector{String}}, 
                            existing_pairs::Vector{Vector{String}}, 
                            target_n_different::Int64, 
                            target_n_same::Int64)

	@assert all(issorted.(existing_pairs)) "Pairs not sorted within `existing_pairs`"
        
    # Extract unique magnitude values
    magnitudes = collect(keys(stimuli))
    
    # Initialize storage for results
    new_pairs = DataFrame(stimulus_A = String[], stimulus_B = String[],
                          magnitude_A = Float64[], magnitude_B = Float64[])
    
    # Track stimulus usage
    stim_usage = Dict(s => 0 for m in magnitudes for s in stimuli[m])
    
    # Function to sample a pair of stimuli from given magnitude categories
    function sample_pair(mag1, mag2)
        available_stim1 = sort(stimuli[mag1], by=s -> stim_usage[s])
        available_stim2 = sort(stimuli[mag2], by=s -> stim_usage[s])
        
        for stim1 in available_stim1, stim2 in available_stim2
            if stim1 != stim2 && !(sort([stim1, stim2]) in existing_pairs)
                return stim1, stim2
            end
        end
        return rand(available_stim1), rand(available_stim2)
    end
    
    # Generate different-magnitude pairs
    for (i, mag1) in enumerate(magnitudes), mag2 in magnitudes[i+1:end]
        for _ in 1:target_n_different
            stim1, stim2 = sample_pair(mag1, mag2)
            push!(new_pairs, (stim1, stim2, mag1, mag2))
            push!(existing_pairs, sort([stim1, stim2]))
            stim_usage[stim1] += 1
            stim_usage[stim2] += 1
        end
    end
    
    # Generate same-magnitude pairs
    for mag in magnitudes
        if length(stimuli[mag]) > 1  # Ensure at least two stimuli exist
            for _ in 1:target_n_same
                stim1, stim2 = sample_pair(mag, mag)
                push!(new_pairs, (stim1, stim2, mag, mag))
                push!(existing_pairs, sort([stim1, stim2]))
                stim_usage[stim1] += 1
                stim_usage[stim2] += 1
            end
        end
    end
    
    return new_pairs
end


# ╔═╡ 9e4e639f-c078-4000-9f01-63bded0dbd82
md"""## PILT"""

# ╔═╡ 6dc75f3f-7a69-428e-8db8-98e6a40b571b
# PILT parameters
begin
	# PILT Parameters
	PILT_blocks_per_valence = 10
	PILT_trials_per_block = 10
	
	PILT_total_blocks = PILT_blocks_per_valence * 2
	PILT_n_confusing = vcat([0, 0, 1, 1], fill(2, PILT_total_blocks ÷ 2 - 4)) # Per valence
		
	# Post-PILT test parameters
	PILT_test_n_blocks = 5
end

# ╔═╡ 8e137a1d-5261-476d-9921-bc024f9b4382
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

# ╔═╡ 57de85ad-4626-43e0-b7a0-54a70131eb83
# Assign valence and set size per block
PILT_block_attr = let random_seed = 5
	
	# # All combinations of set sizes and valence
	block_attr = DataFrame(
		block = repeat(1:PILT_total_blocks),
		valence = repeat([1, -1], inner = PILT_blocks_per_valence),
		fifty_high = fill(true, PILT_total_blocks)
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

# ╔═╡ 10ee1c0b-46ef-4ec5-8bf7-23ca95cf1e57
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
		model = single_p_QL_recip,
		model_name = "QL_recip",
		unpack_function = unpack_single_p_QL,
		prop_fifty = 0.2,
	) for r in eachrow(n_confusing_wanted)]

	# Unpack results
	common_seqs = [x[2] for x in FI_seqs]
	magn_seqs = [x[3] for x in FI_seqs]

	# # Choose sequences optimizing FI under contraints
	chosen_idx, common_per_pos, EV_per_pos = optimize_FI_distribution(
		n_wanted = n_confusing_wanted.n,
		FIs = [x[1] for x in FI_seqs],
		common_seqs = common_seqs,
		magn_seqs = magn_seqs,
		ω_FI = .15,
		constrain_pairs = false,
		filename = "results/exp_sequences/pilot7_opt.jld2"
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
		
	@assert mean(task.fifty_high) == 1. "Proportion of blocks with 50 pence in high magnitude option expected to be 1."

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

# ╔═╡ 3fb28c3e-4fe1-4476-9362-6bb8d69db60f
# Assign stimulus images
PILT_stimuli = let random_seed = 0

	# Shuffle categories
	shuffle!(Xoshiro(random_seed), categories)

	@info length(categories)

	# Assign stimulus pairs
	stimuli = assign_stimuli_and_optimality(;
		n_phases = 1,
		n_pairs = fill(1, PILT_total_blocks),
		categories = categories,
		random_seed = random_seed
	)

	@info "Proportion of blocks on which the novel category is optimal : $(mean(stimuli.optimal_A))"

	rename!(stimuli, :phase => :session) # For compatibility with multi-phase sessions

	stimuli
end

# ╔═╡ 52c5560f-ac07-44f3-b8ba-42940d10c600
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

	@assert nrow(task) == nrow(PILT_sequences)  "Problem in join operation"

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
		:present_pavlovian => true,
		:early_stop => false
	)

	task
end

# ╔═╡ f5972e81-839b-4c83-b5d6-435a8dcfe83c
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

# ╔═╡ af7a2a50-ec27-481a-b998-8930b3e945d8
let
	save_to_JSON(PILT_task, "results/pilot7_PILT.json")
	CSV.write("results/pilot7_PILT.csv", PILT_task)
end

# ╔═╡ 98ca19c7-a0b3-447a-984d-cae804e36513
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
	


	save("results/pilot7_pilt_trial_plan.png", f, pt_per_unit = 1)

	f

end

# ╔═╡ 85deb936-2204-4fe8-a0dd-a23f527f813d
md"""## Post-PILT test"""

# ╔═╡ 62ca4f41-0b0d-4125-a85b-0a9752714d64
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

# ╔═╡ 6dff1b52-b0f0-4895-89d3-f732791e11c5
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
	chosen = pass_magnitude[argmin(dev_valence)]

	# Return sequence and stats
	return create_test_sequence(task, random_seed = chosen, same_weight = same_weight)
end

# ╔═╡ 2b7204c6-4fc6-41d2-b446-1c6bf75750b7
PILT_test = let task = PILT_task
	
	# Find test sequence for each session
	PILT_test = []
	for s in 1
		test, pb, pv, nm = find_best_test_sequence(
			task,
			n_seeds = 100, # Number of random seeds to try
			same_weight = 25. # Weight reducing the number of same magntiude pairs
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

	# Create feedback_right and feedback_left variables - these determine coins given on this trial
	PILT_test.feedback_left = (x -> abs(x) == 0.01 ? x : sign(x)).(PILT_test.magnitude_left)

	PILT_test.feedback_right = (x -> abs(x) == 0.01 ? x : sign(x)).(PILT_test.magnitude_right)

	PILT_test
end

# ╔═╡ a8fb4243-2bd2-4ca4-b936-842d423c55e6
# Create WM test sequence
RLWM_test = let rng = Xoshiro(0)

	# Process WM sequence to be able to find test pairs
	stimuli_magnitude, existing_pairs = prepare_for_finding_wm_test_sequence(
		RLWM
	)

	# Find test pairs
	RLWM_test = create_wm_test_pairs(stimuli_magnitude, 
						existing_pairs, 
						60, 
						20)

	# Shuffle
	RLWM_test.trial = shuffle(rng, 1:nrow(RLWM_test))

	sort!(RLWM_test, :trial)

	# Add needed variables
	insertcols!(
		RLWM_test,
		1,
		:session => 1,
		:block => 1,
		:original_block_right => 1,
		:original_block_left => 1,
		:same_block => true,
		:valence_left => 1,
		:valence_right => 1,
		:same_valence => true
	)

	# Add magnitude pair variable
	RLWM_test.magnitude_pair = [sort([r.magnitude_A, r.magnitude_B]) for r in eachrow(RLWM_test)]


	# Assign left / right
	DataFrames.transform!(
		groupby(RLWM_test, :magnitude_pair),
		:trial => (x -> shuffled_fill([true, false], length(x), rng = rng)) => :A_on_right
	)

	RLWM_test.stimulus_right = ifelse.(
		RLWM_test.A_on_right,
		RLWM_test.stimulus_A,
		RLWM_test.stimulus_B
	)

	RLWM_test.stimulus_left = ifelse.(
		.!RLWM_test.A_on_right,
		RLWM_test.stimulus_A,
		RLWM_test.stimulus_B
	)

	RLWM_test.magnitude_right = ifelse.(
		RLWM_test.A_on_right,
		RLWM_test.magnitude_A,
		RLWM_test.magnitude_B
	)

	RLWM_test.magnitude_left = ifelse.(
		.!RLWM_test.A_on_right,
		RLWM_test.magnitude_A,
		RLWM_test.magnitude_B
	)

	# Add feedback_right and feedback_left variables - these determine the coins added to the safe for the trial
	RLWM_test.feedback_right = ifelse.(
		RLWM_test.magnitude_right .== 0.75,
		fill(1., nrow(RLWM_test)),
		fill(0.01, nrow(RLWM_test))
	)

	RLWM_test.feedback_left = ifelse.(
		RLWM_test.magnitude_left .== 0.75,
		fill(1., nrow(RLWM_test)),
		fill(0.01, nrow(RLWM_test))
	)

	RLWM_test.block .+= maximum(PILT_test.block)


	RLWM_test

end

# ╔═╡ 16711c7d-4548-4ea3-b7fb-019d2fe80827
# Tests for RLWM_test
let
	# Test even distribution of stimuli appearances
	long_test = vcat(
		select(
			RLWM_test,
			:stimulus_A => :stimulus,
			:magnitude_A => :magnitude
		),
		select(
			RLWM_test,
			:stimulus_B => :stimulus,
			:magnitude_B => :magnitude
		)
	)

	test_n = combine(
		groupby(
			long_test,
			:stimulus
		),
		:stimulus => length => :n,
		:magnitude => unique => :magnitude
	)

	@assert all(combine(
		groupby(
			test_n,
			:magnitude
		),
		:n => (x -> (maximum(x) - minimum(x)) == 1) => :n_diff
	).n_diff) "Number of appearances for each stimulus not balanced"

	# Test left right counterbalancing of magnitude
	@assert mean(RLWM_test.magnitude_left) == mean(RLWM_test.magnitude_right) "Different average magnitudes for stimuli presented on left and right"

	# Make sure all stimuli are in RLWM
	test_stimuli = unique(
		vcat(
			RLWM_test.stimulus_left,
			RLWM_test.stimulus_right
		)
	)

	RLWM_stimuli =  unique(
		vcat(
			RLWM.stimulus_left,
			RLWM.stimulus_right
		)
	)

	@assert all((x -> x in RLWM_stimuli).(test_stimuli)) "Test stimuli not in RLWM sequence"

	@assert all((x -> x in test_stimuli).(RLWM_stimuli)) "Not all RLWM stimuli appear in test"


end

# ╔═╡ dbed9ea8-da67-41e9-8cfa-42dac8712dfc
let
	save_to_JSON(RLWM_test, "results/pilot7_WM_test.json")
	CSV.write("results/pilot7_WM_test.csv", RLWM_test)
end

# ╔═╡ 5a37f9d4-9c27-456e-b9f0-8601dbfee7ca
combine(
	groupby(PILT_test, :magnitude_pair),
	:trial => length => :n
)

# ╔═╡ 0089db22-38ad-4d9c-88a2-12b82361384f
let
	save_to_JSON(PILT_test, "results/pilot7_PILT_test.json")
	CSV.write("results/pilot7_PILT_test.csv", PILT_test)
end

# ╔═╡ cbe8ce1d-b48b-4396-b924-551ce8bbfc14
let
	cat_extract = x -> split(x, "_")[1]
	
	vcat(
		cat_extract.(RLWM.stimulus_A),
		cat_extract.(RLWM.stimulus_B),
		cat_extract.(RLWM.stimulus_C),
		cat_extract.(PILT_task.stimulus_A),
		cat_extract.(PILT_task.stimulus_B)
	) |> unique |> length

end

# ╔═╡ Cell order:
# ╠═2d7211b4-b31e-11ef-3c0b-e979f01c47ae
# ╠═114f2671-1888-4b11-aab1-9ad718ababe6
# ╠═de74293f-a452-4292-b5e5-b4419fb70feb
# ╟─ffe06202-d829-4145-ae26-4a95449d64e6
# ╠═05f25eb8-3a48-4d16-9837-84d1fdf5c806
# ╠═c05d90b6-61a7-4f9e-a03e-3e11791da6d0
# ╠═f5916a9f-ddcc-4c03-9328-7dd76c4c74b2
# ╠═6eadaa9d-5ed0-429b-b446-9cc7fbfb52dc
# ╠═e3bff0b9-306a-4bf9-8cbd-fe0e580bd118
# ╠═f9be1490-8e03-445f-b36e-d8ceff894751
# ╠═eecaac0c-e051-4543-988c-e969de3a8567
# ╠═3fa8c293-ac47-4acd-bdb7-9313286ee464
# ╠═68873d3e-054d-4ab4-9d89-73586bb0370e
# ╠═f89e88c9-ebfc-404f-964d-acff5c7f8985
# ╠═414c9032-6458-4ad4-bd95-e756d221912f
# ╠═a8fb4243-2bd2-4ca4-b936-842d423c55e6
# ╠═16711c7d-4548-4ea3-b7fb-019d2fe80827
# ╠═dbed9ea8-da67-41e9-8cfa-42dac8712dfc
# ╠═56fcba3a-da77-42a4-bf91-f5c0962bdbf4
# ╠═b514eb90-a26c-4c44-82ce-d0d962fc940a
# ╠═9e4e639f-c078-4000-9f01-63bded0dbd82
# ╠═6dc75f3f-7a69-428e-8db8-98e6a40b571b
# ╠═8e137a1d-5261-476d-9921-bc024f9b4382
# ╠═57de85ad-4626-43e0-b7a0-54a70131eb83
# ╠═10ee1c0b-46ef-4ec5-8bf7-23ca95cf1e57
# ╠═3fb28c3e-4fe1-4476-9362-6bb8d69db60f
# ╠═52c5560f-ac07-44f3-b8ba-42940d10c600
# ╠═f5972e81-839b-4c83-b5d6-435a8dcfe83c
# ╠═af7a2a50-ec27-481a-b998-8930b3e945d8
# ╠═98ca19c7-a0b3-447a-984d-cae804e36513
# ╠═85deb936-2204-4fe8-a0dd-a23f527f813d
# ╠═2b7204c6-4fc6-41d2-b446-1c6bf75750b7
# ╠═5a37f9d4-9c27-456e-b9f0-8601dbfee7ca
# ╠═0089db22-38ad-4d9c-88a2-12b82361384f
# ╠═6dff1b52-b0f0-4895-89d3-f732791e11c5
# ╠═62ca4f41-0b0d-4125-a85b-0a9752714d64
# ╠═cbe8ce1d-b48b-4396-b924-551ce8bbfc14
