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
		ForwardDiff, LinearAlgebra, JLD2, FileIO, CSV, Dates, JSON, RCall, Turing, Printf, Combinatorics, JuMP, HiGHS, AlgebraOfGraphics, CategoricalArrays
	using LogExpFunctions: logistic, logit
	using IterTools: product

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

# ╔═╡ 54f6f217-3ae5-49c7-9456-a5abcbbdc62f
# RLX Parameters
begin
	RLX_prop_fifty = 0.2
	RLX_shaping_n = 20
	RLX_test_n_blocks = 2
end

# ╔═╡ 9f300301-b018-4bea-8fc4-4bc889b11afd
triplet_order = let
	triplet_order = DataFrame(CSV.File(
		"generate_experimental_sequences/pilot8_wm_stimulus_sequence.csv"))

	select!(
		triplet_order, 
		:stimset => :stimulus_group,
		:delay
	)
end

# ╔═╡ 184a054c-5a88-44f8-865e-da75a10191ec
md"""## RLWM"""

# ╔═╡ 60c50147-708a-46f8-a813-7667116fc8d2
md"""### Post-WM test"""

# ╔═╡ ffe06202-d829-4145-ae26-4a95449d64e6
md"""# RLLTM"""

# ╔═╡ 3fa8c293-ac47-4acd-bdb7-9313286ee464
function assign_triplet_stimuli_RLLTM(
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

# ╔═╡ b9134153-d9e9-4e35-bfc4-2c5c5a4329ee
RLX_block = let rng = Xoshiro(0)

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
			[1 - RLX_prop_fifty, RLX_prop_fifty],
			length(x),
			rng = rng
			)
		) => :feedback_optimal
	) 

	@info "Proportion fifty pence: $(mean(det_block.feedback_optimal .== 0.5))"

	det_block

end

# ╔═╡ 6417ad94-1852-4cce-867e-a856295ec782
# Create deterministic block
RLWM = let RLWM = copy(RLX_block),
	rng = Xoshiro(0)

	# Assign stimuli --------
	stimuli = [
		popat!(categories, rand(rng, 1:length(categories))) * "_1.jpg" for _ in 1:length(unique(RLWM.stimulus_group))
	]

	stimuli = DataFrame(
		stimulus_group = unique(RLWM.stimulus_group),
		stimulus_left = stimuli
	)
	
		
	leftjoin!(
		RLWM,
		stimuli,
		on = :stimulus_group
	)

	# Replicate - for RLWM there is only one stimulus, but this is requirement of js script
	RLWM.stimulus_middle = RLWM.stimulus_left
	RLWM.stimulus_right = RLWM.stimulus_left
	
	# Assign stimuli locations -----------------------------
	# Count appearances per stimulus_group
	stimulus_ordering = combine(
		groupby(RLWM, [:stimulus_group]),
		:stimulus_group => length => :n
	)

	# Sort by descending n to distribute largest trials first
	shuffle!(rng, stimulus_ordering)
	sort!(stimulus_ordering, :n, rev=true)

	# Track total counts per action
	action_sums = Dict(1 => 0, 2 => 0, 3 => 0)

	# Place holder for optimal action
	stimulus_ordering.optimal_action .= 99
	
	# Assign actions to balance total n
	for row in eachrow(stimulus_ordering)
	    # Pick the action with the smallest current total
	    best_action = argmin(action_sums)
	    row.optimal_action = best_action
	    action_sums[best_action] += row.n
	end


	# Join with data frame
	leftjoin!(
		RLWM,
		select(stimulus_ordering, [:stimulus_group, :optimal_action]),
		on = :stimulus_group
	)

	@info "Proportion optimal in each location: $([round(mean((x -> x == i).(RLWM.optimal_action)), digits = 3) for i in 1:3])"

	# Additional variables --------

	# Assign feedback to action
	for (i, side) in enumerate(["left", "middle", "right"])
		RLWM[!, Symbol("feedback_$side")] = ifelse.(
			(x -> x == i).(RLWM.optimal_action),
			RLWM.feedback_optimal,
			fill(0.01, nrow(RLWM))
		)
	end

	# For compatibility with probabilistic block
	RLWM.feedback_common .= true

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
	RLWM.optimal_side = (x -> ["left", "middle", "right"][x]).(RLWM.optimal_action)
		

	# Add variables needed for experiment code
	insertcols!(
		RLWM,
		:n_stimuli => 1,
		:optimal_right => "",
		:present_pavlovian => false,
		:n_groups => maximum(RLWM.stimulus_group),
		:early_stop => false
	)


	RLWM

end

# ╔═╡ e6f984aa-20dc-4a7d-8a3b-75728995a1f7
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

# ╔═╡ efdfdeb0-2b56-415e-acf7-d6236ee7b199
let
	save_to_JSON(RLWM, "results/pilot8_WM.json")
	CSV.write("results/pilot8_WM.csv", RLWM)
end

# ╔═╡ 1491f0f9-0c40-41ca-b7a9-055259f66eb3
RLWM_test = let rng = Xoshiro(0)

	# Get unique stimuli
	RLWM_stimuli = unique(RLWM.stimulus_left)

	# Get all combinations
	RLWM_pairs = collect(combinations(RLWM_stimuli, 2))

	# Shuffle order within pair
	shuffle!.(rng, RLWM_pairs)

	# Repeat flipped
	RWLM_blocks = [iseven(i) ? reverse.(RLWM_pairs) : RLWM_pairs for i in 1:RLX_test_n_blocks]

	# Assemble into DataFrame
	RLWM_test = vcat([DataFrame(
			block = fill(i, length(stims)),
			stimulus_left = [x[1] for x in stims],
			stimulus_right = [x[2] for x in stims]
		) for (i, stims) in enumerate(RWLM_blocks)]...)

	# Shuffle trial order within block
	DataFrames.transform!(
		groupby(RLWM_test, :block),
		:block => (x -> shuffle(rng, 1:length(x))) => :trial
	)

	sort!(RLWM_test, [:block, :trial])

	# Add variables needed for JS ------------------
	insertcols!(
		RLWM_test,
		:session => 1,
		:feedback_left => 1.,
		:feedback_right => 1.,
		:magnitude_left => 1.,
		:magnitude_right => 1.,
		:same_valence => true,
		:same_block => true,
		:original_block_left => 1,
		:original_block_right => 1
	)
	

end

# ╔═╡ b28f57a2-8aab-45e9-9d16-4c3b9fcf3828
let
	save_to_JSON(RLWM_test, "results/pilot8_WM_test.json")
	CSV.write("results/pilot8_WM_test.csv", RLWM_test)
end

# ╔═╡ 1a6d525f-5317-4b2b-a631-ea646ee20c9f
# Tests for RLWM_test
let

	# Make sure all stimuli are in RLWM
	test_stimuli = unique(
		vcat(
			RLWM_test.stimulus_left,
			RLWM_test.stimulus_right
		)
	)

	RLWM_stimuli =  unique(RLWM.stimulus_left)

	@assert all((x -> x in RLWM_stimuli).(test_stimuli)) "Test stimuli not in RLWM sequence"

	@assert all((x -> x in test_stimuli).(RLWM_stimuli)) "Not all RLWM stimuli appear in test"


end

# ╔═╡ f5916a9f-ddcc-4c03-9328-7dd76c4c74b2
# Create deterministic block
RLLTM_det_block = let det_block = copy(RLX_block),
	rng = Xoshiro(1)
	
	# Assign stimuli categories
	stimuli = assign_triplet_stimuli_RLLTM((categories),
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

	# Assign stimuli locations -----------------------------
	# Count appearances per stimulus_group
	stimulus_ordering = combine(
		groupby(det_block, [:stimulus_group]),
		:stimulus_group => length => :n
	)

	# Sort by descending n to distribute largest trials first
	shuffle!(rng, stimulus_ordering)
	sort!(stimulus_ordering, :n, rev=true)

	# Track total counts per action
	action_sums = Dict(1 => 0, 2 => 0, 3 => 0)

	# Place holder for optimal action
	stimulus_ordering.optimal_action .= 99
	
	# Assign actions to balance total n
	for row in eachrow(stimulus_ordering)
	    # Pick the action with the smallest current total
	    best_action = argmin(action_sums)
	    row.optimal_action = best_action
	    action_sums[best_action] += row.n
	end

	# Function to translate to "ABC" strings
	function stim_ordering(pos::Int64)
		s = join(shuffle(collect("BC")))
		return s[1:pos-1] * "A" * s[pos:end]
	end

	# Translate to "ABC" orderings
	stimulus_ordering.stimulus_ordering = stim_ordering.(stimulus_ordering.optimal_action)

	# Join with data frame
	leftjoin!(
		det_block,
		select(stimulus_ordering, [:stimulus_group, :stimulus_ordering]),
		on = :stimulus_group
	)

	@info "Proportion optimal in each location: $([round(mean((x -> x[i] == 'A').(det_block.stimulus_ordering)), digits = 3) for i in 1:3])"

	# Additional variables --------
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
RLLTM = let RLLTM = RLLTM_det_block

	# Session variable
	RLLTM.session .= 1

	# Valence variable
	RLLTM.valence .= 1

	# Apperance variable
	DataFrames.transform!(
		groupby(RLLTM, [:block, :stimulus_group]),
		:trial => (x -> 1:length(x)) => :appearance
	)

	# Cumulative triplet index
	RLLTM.stimulus_group_id = RLLTM.stimulus_group .+ (maximum(RLLTM.stimulus_group) .* (RLLTM.block .- 1))

	# Create optimal_side variable
	RLLTM.optimal_side = [["left", "middle", "right"][findfirst('A', o)] for o in RLLTM.stimulus_ordering]


	# Add variables needed for experiment code
	insertcols!(
		RLLTM,
		:n_stimuli => 3,
		:optimal_right => "",
		:present_pavlovian => false,
		:n_groups => maximum(RLLTM.stimulus_group),
		:early_stop => false
	)

	

	RLLTM
end

# ╔═╡ f9be1490-8e03-445f-b36e-d8ceff894751
# Checks
let task = RLLTM
	@assert all(combine(groupby(task, [:session]), 
		:block => issorted => :sorted).sorted) "Task structure not sorted by block"

	@assert all(combine(groupby(task, [:session, :block]), 
		:trial => issorted => :sorted).sorted) "Task structure not sorted by trial number"
	
	@assert all(sign.(task.valence) == sign.(task.feedback_right)) "Valence doesn't match feedback sign"

	@assert all(sign.(task.valence) == sign.(task.feedback_left)) "Valence doesn't match feedback sign"

	@info "Overall proportion of common feedback: $(round(mean(task.feedback_common), digits = 2))"

	@assert !all(RLWM.feedback_right .== RLLTM.feedback_right) "Optimal actions should be different on RLWM and RLLTM"

	@assert all(RLWM.stimulus_group .== RLLTM.stimulus_group) "Stimuli sequence not identical in RLWM and RLLTM"

end

# ╔═╡ eecaac0c-e051-4543-988c-e969de3a8567
let
	save_to_JSON(RLLTM, "results/pilot8_LTM.json")
	CSV.write("results/pilot8_LTM.csv", RLLTM)
end

# ╔═╡ bcb73e19-8bf9-4e75-a9f7-8152e3d23201
md"""## Post-LTM test"""

# ╔═╡ df5baa98-27b9-4bbc-9ab0-03ea66e91970
function prepare_for_finding_ltm_test_sequence(
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

# ╔═╡ 391f3f2a-74ae-463b-b109-a47eaaafdf97
function create_ltm_test_pairs(stimuli::Dict{Float64, Vector{String}}, 
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


# ╔═╡ c65d7f1f-224d-4144-aa46-d48a482db95a
# Create LTM test sequence
RLLTM_test = let rng = Xoshiro(0)

	# Process WM sequence to be able to find test pairs
	stimuli_magnitude, existing_pairs = prepare_for_finding_ltm_test_sequence(
		RLLTM
	)
	

	# Find test pairs
	RLLTM_test = create_ltm_test_pairs(stimuli_magnitude, 
						existing_pairs, 
						60, 
						20)

	# Shuffle
	RLLTM_test.trial = shuffle(rng, 1:nrow(RLLTM_test))

	sort!(RLLTM_test, :trial)

	# Add needed variables
	insertcols!(
		RLLTM_test,
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
	RLLTM_test.magnitude_pair = [sort([r.magnitude_A, r.magnitude_B]) for r in eachrow(RLLTM_test)]


	# Assign left / right
	DataFrames.transform!(
		groupby(RLLTM_test, :magnitude_pair),
		:trial => (x -> shuffled_fill([true, false], length(x), rng = rng)) => :A_on_right
	)

	RLLTM_test.stimulus_right = ifelse.(
		RLLTM_test.A_on_right,
		RLLTM_test.stimulus_A,
		RLLTM_test.stimulus_B
	)

	RLLTM_test.stimulus_left = ifelse.(
		.!RLLTM_test.A_on_right,
		RLLTM_test.stimulus_A,
		RLLTM_test.stimulus_B
	)

	RLLTM_test.magnitude_right = ifelse.(
		RLLTM_test.A_on_right,
		RLLTM_test.magnitude_A,
		RLLTM_test.magnitude_B
	)

	RLLTM_test.magnitude_left = ifelse.(
		.!RLLTM_test.A_on_right,
		RLLTM_test.magnitude_A,
		RLLTM_test.magnitude_B
	)

	# Add feedback_right and feedback_left variables - these determine the coins added to the safe for the trial
	RLLTM_test.feedback_right = ifelse.(
		RLLTM_test.magnitude_right .== 0.75,
		fill(1., nrow(RLLTM_test)),
		fill(0.01, nrow(RLLTM_test))
	)

	RLLTM_test.feedback_left = ifelse.(
		RLLTM_test.magnitude_left .== 0.75,
		fill(1., nrow(RLLTM_test)),
		fill(0.01, nrow(RLLTM_test))
	)

	RLLTM_test.block .+= 5 #maximum(PILT_test.block)


	RLLTM_test

end

# ╔═╡ 63b74bde-4a28-4ff8-ae15-a37c891c3166
# Tests for RLWM_test
let
	# Test even distribution of stimuli appearances
	long_test = vcat(
		select(
			RLLTM_test,
			:stimulus_A => :stimulus,
			:magnitude_A => :magnitude
		),
		select(
			RLLTM_test,
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
	@assert mean(RLLTM_test.magnitude_left) == mean(RLLTM_test.magnitude_right) "Different average magnitudes for stimuli presented on left and right"

	# Make sure all stimuli are in RLWM
	test_stimuli = unique(
		vcat(
			RLLTM_test.stimulus_left,
			RLLTM_test.stimulus_right
		)
	)

	RLLTM_stimuli =  unique(
		vcat(
			RLLTM.stimulus_left,
			RLLTM.stimulus_right,
			RLLTM.stimulus_middle
		)
	)

	@assert all((x -> x in RLLTM_stimuli).(test_stimuli)) "Test stimuli not in RLLTM sequence"

	@assert all((x -> x in test_stimuli).(RLLTM_stimuli)) "Not all RLLTM stimuli appear in test"


end

# ╔═╡ 832ccb61-b588-4614-9f5a-efa0f9a6087d
let
	save_to_JSON(RLLTM_test, "results/pilot8_LTM_test.json")
	CSV.write("results/pilot8_LTM_test.csv", RLLTM_test)
end

# ╔═╡ 7a100e62-c028-4a24-a200-658bbca1945a
md"""## PILT"""

# ╔═╡ 18f41358-8554-4142-83f8-1c1c3f38b6fa
# PILT parameters
begin
	# PILT Parameters
	PILT_trials_per_block = 10
			
	# Post-PILT test parameters
	PILT_test_n_blocks = 5
end

# ╔═╡ 926053c8-df86-440f-b397-47c9ed2a6000
# ╠═╡ disabled = true
#=╠═╡
# Draw confsuing trials
PILT_sequence = let rng = Xoshiro(1)

	blocks = groupby(PILT_blocks, :block)
    n_blocks = length(blocks)

    # Pre-allocate best sequence storage
    sequence = DataFrame()
	score = Inf
    while score > 0.0001
        trials_per_block = [
            DataFrame(
                block = first(b.block),
                trial = 1:PILT_trials_per_block,
                feedback_common = shuffle(
					rng, vcat(
						fill(true, PILT_trials_per_block - b.n_confusing[1]),
						fill(false, b.n_confusing[1])
					))
            )
            for b in blocks
        ]

        sequence = vcat(trials_per_block...)

        # Compute trial-wise feedback mean efficiently
        trial_means = combine(
			groupby(sequence, :trial), :feedback_common => mean => :m).m
        score = std(trial_means)

    end

    sequence


end
  ╠═╡ =#

# ╔═╡ 765efd14-e687-4fde-bd0a-4378251a7aa8
#=╠═╡
# Assign outcomes to A and B
PILT = let rng = Xoshiro(0)

	# Merge with block attributes
	PILT = leftjoin(
		PILT_sequence,
		PILT_blocks,
		on = :block,
		order = :left
	)

	# Outcome for stimulus A, B
	PILT.outcome_A = ifelse.(
		PILT.feedback_common,
		PILT.primary_outcome_A,
		PILT.secondary_outcome_A
	)

	PILT.outcome_B = ifelse.(
		PILT.feedback_common,
		PILT.primary_outcome_B,
		PILT.secondary_outcome_B
	)

	PILT

end
  ╠═╡ =#

# ╔═╡ 34229660-e7b2-4e87-b420-334d018a2050
function discretize_to_quantiles(values::Vector{T}, reference_values::Vector{T}, num_quantiles::Int) where T
    quantile_edges = quantile(reference_values, range(0, 1, length=num_quantiles+1))
    
    labels = ["[$(round(quantile_edges[i], sigdigits=3)), $(round(quantile_edges[i+1], sigdigits=3)))" for i in 1:num_quantiles]
    
    function assign_quantile(v)
        for (i, edge) in enumerate(quantile_edges[2:end])
            if v <= edge
                return labels[i]
            end
        end
        return labels[end]
    end
    
    categorical_values = CategoricalArray([assign_quantile(v) for v in values], ordered=true, levels=labels)
    return categorical_values
end


# ╔═╡ 84d2f787-aba2-40be-9816-4924968f5790
PILT_blocks = let rng = Xoshiro(0)

	coins = [-1., -0.5, -0.01, 0.01, 0.5, 1.]

	# Create all magnitdue combinations
	outcome_combinations = collect(product(coins, coins))

	# Select only upper triangular
	outcome_combinations = 
		outcome_combinations[triu(trues(size(outcome_combinations)))]

	# Convert to vector of vectors
	outcome_combinations = collect.(outcome_combinations)

	# Shuffle order within pair, keeping B stim outcome distribution flat
	criteria = false 
	while !criteria

		# Shuffle
		shuffle!.(rng, outcome_combinations)

		# Collect B stimuli values
		B_stim = [x[2] for x in outcome_combinations]

		B_stim_counts = values(countmap(B_stim))

		criteria = maximum(B_stim_counts) - minimum(B_stim_counts) == 1

	end


	# Shuffle order, under constraints
	criteria = false 
	
	while !criteria
		shuffle!(rng, outcome_combinations)

		# Summarize stim B outcome transitions--------
		outcome_B = [x[2] for x in outcome_combinations]

		transitions = [(outcome_B[i], outcome_B[i+1]) for i in 1:(length(outcome_B) - 1)]

		trans_counts = values(countmap(transitions))


		criteria = all(outcome_combinations[1] .> 0) &&
			all((x -> (x[1] * x[2]) < 0).(outcome_combinations[2:3])) &&
			all(outcome_combinations[4] .< 0) &&
			maximum(trans_counts) <= 1

	end

	# Arrange into DataFrame
	PILT_blocks_long = DataFrame(
		block = repeat(1:length(outcome_combinations), outer = 2),
		n_confusing = repeat(vcat(
			fill(0, 4), 
			fill(1, 4), 
			fill(2, length(outcome_combinations) - 8)
		), outer = 2),
		stimulus = repeat(["A", "B"], inner = length(outcome_combinations)),
		primary_outcome = vcat(
			[x[1] for x in outcome_combinations], 
			[x[2] for x in outcome_combinations]
		)
	)

	sort!(PILT_blocks_long, [:block, :stimulus])

	# Function to assign secondary outcomes
	function shuffle_secondary_outcome(n::Int64, primary::Float64)

		secondary = repeat(
			filter(x -> x != primary, coins), 
			n ÷ (length(coins) - 1)
		)

		distances = abs.(coins .- primary)

		furthest = coins[partialsortperm(distances, 1:(rem(n, length(coins) - 1)), rev=true)]

		secondary = vcat(secondary, furthest)

		return shuffle(rng, secondary)

	end

	# Tell apart deterministic from probabilistic blocks
	PILT_blocks_long.deterministic = PILT_blocks_long.n_confusing .== 0

	# Add secondary outcome, shuffle under contraints such that the first two blocks primary outcome matches secondary in sign, and that stimulus B (repeating) has uniform distribution
	criteria2 = false
	B_score = Inf
	while !criteria2
		DataFrames.transform!(
			groupby(PILT_blocks_long, [:deterministic, :primary_outcome]),
			[:deterministic, :primary_outcome] => 
				((d, o) -> ifelse.(
					d,
					fill(missing, length(o)),
					shuffle_secondary_outcome(
						length(o), unique(o)[1]))) => :secondary_outcome
				
		)

		# EV
		PILT_blocks_long.EV = ifelse.(
			ismissing.(PILT_blocks_long.secondary_outcome),
			PILT_blocks_long.primary_outcome,
			PILT_blocks_long.primary_outcome .* 0.8 + PILT_blocks_long.secondary_outcome .* 0.2
		)


		first_mathces = all((r -> sign(r.primary_outcome) == 
			sign(r.secondary_outcome)).(eachrow(filter(x -> x.block in [5,6], PILT_blocks_long))))

		EV_B_bin = discretize_to_quantiles(
			PILT_blocks_long.EV,
			PILT_blocks_long.EV,
			5
		)[PILT_blocks_long.stimulus .== "B"]

		EV_B_bin_counts = values(countmap(EV_B_bin))

		B_uniform = maximum(EV_B_bin_counts) - minimum(EV_B_bin_counts) == 1

		criteria2 = first_mathces && B_uniform

	end
	
	# Long to wide
	PILT_blocks = leftjoin(
		unstack(
			PILT_blocks_long,
			[:block, :n_confusing],
			:stimulus,
			:primary_outcome,
			renamecols = x -> "primary_outcome_$(x)"
		),
		unstack(
			PILT_blocks_long,
			[:block, :n_confusing],
			:stimulus,
			:secondary_outcome,
			renamecols = x -> "secondary_outcome_$(x)"
		),
		on = [:block, :n_confusing]
	)

	# Compute EV
	PILT_blocks.EV_A = ifelse.(
		ismissing.(PILT_blocks.secondary_outcome_A),
		PILT_blocks.primary_outcome_A,
		PILT_blocks.primary_outcome_A .* 0.8 + PILT_blocks.secondary_outcome_A .* 0.2
	)

	PILT_blocks.EV_B = ifelse.(
		ismissing.(PILT_blocks.secondary_outcome_B),
		PILT_blocks.primary_outcome_B,
		PILT_blocks.primary_outcome_B .* 0.8 + PILT_blocks.secondary_outcome_B .* 0.2
	)


	PILT_blocks

end

# ╔═╡ 73399d2c-8256-49a7-b739-cb1f496bcd2f
let
	
	autocor(PILT_blocks.EV_B), autocor(PILT_blocks.EV_A)
	
end

# ╔═╡ cd31c2c9-faa0-4487-b9f3-993b07f297d3
let
	@info quantile(vcat(PILT_blocks.EV_A, PILT_blocks.EV_B), range(0., 1., 6))

	@info minimum(abs.(vcat(PILT_blocks.EV_A, PILT_blocks.EV_B)))
	
	hist(vcat(PILT_blocks.EV_A, PILT_blocks.EV_B), bins = 40)

end

# ╔═╡ 4d115a60-a7ce-48e7-af92-736ac2dbec07
function assign_stimuli_and_optimality(;
	n_phases::Int64,
	n_pairs::Vector{Int64}, # Number of pairs in each block. Assume same for all phases
	categories::Vector{String} = [('A':'Z')[div(i - 1, 26) + 1] * ('a':'z')[rem(i - 1, 26)+1] 
		for i in 1:(sum(n_pairs) * 2 * n_phases + n_phases)],
	random_seed::Int64 = 1,
	ext::String = "jpg"
	)

	total_n_pairs = sum(n_pairs) # Number of pairs needed
	
	# @assert rem(length(n_pairs), 2) == 0 "Code only works for even number of blocks per sesion"

	@assert length(categories) >= sum(total_n_pairs) * n_phases + n_phases "Not enought categories supplied"

	# Compute how many repeating categories we will have
	n_repeating = sum(min.(n_pairs[2:end], n_pairs[1:end-1]))

	# Assign whether repeating is optimal and shuffle
	repeating_optimal = shuffle(
		Xoshiro(random_seed),
		vcat(
			fill(true, div(n_repeating, 2)),
			fill(false, div(n_repeating, 2) + rem(n_repeating, 2))
		)
	)

	# Assign whether categories that cannot repeat are optimal
	rest_optimal = shuffle(
		vcat(
			fill(true, div(total_n_pairs - n_repeating, 2) + 
				rem(total_n_pairs - n_repeating, 2)),
			fill(false, div(total_n_pairs - n_repeating, 2))
		)
	)

	# Initialize vectors for stimuli. A is always novel, B may be repeating
	stimulus_A = []
	stimulus_B = []
	optimal_B = []
	
	for j in 1:n_phases
		for (i, p) in enumerate(n_pairs)
	
			# Choose repeating categories for this block
			n_repeating = ((i > 1) && minimum([p, n_pairs[i - 1]])) * 1
			append!(
				stimulus_B,
				stimulus_A[(end - n_repeating + 1):end]
			)
	
			# Fill up stimulus_repeating with novel categories if not enough to repeat
			for _ in 1:(p - n_repeating)
				push!(
					stimulus_B,
					popfirst!(categories)
				)
			end
			
			# Choose novel categories for this block
			for _ in 1:p
				push!(
					stimulus_A,
					popfirst!(categories)
				)
			end

			# Populate who is optimal vector
			for _ in 1:(n_repeating)
				push!(
					optimal_B,
					popfirst!(repeating_optimal)
				)
			end

			for _ in 1:(p - n_repeating)
				push!(
					optimal_B,
					popfirst!(rest_optimal)
				)
			end
		end
	end

	stimulus_A = (x -> x * "_1.$ext").(stimulus_A)
	stimulus_B = (x -> x * "_2.$ext").(stimulus_B)

	return DataFrame(
		phase = repeat(1:n_phases, inner = total_n_pairs),
		block = repeat(
			vcat([fill(i, p) for (i, p) in enumerate(n_pairs)]...), n_phases),
		pair = repeat(
			vcat([1:p for p in n_pairs]...), n_phases),
		stimulus_A = stimulus_A,
		stimulus_B = stimulus_B,
		optimal_A = .!(optimal_B)
	)

end

# ╔═╡ 76fcde1b-2174-44e3-a993-63a48879403f
# Assign stimulus images
PILT_stimuli = let random_seed = 0

	# Shuffle categories
	shuffle!(Xoshiro(random_seed), categories)

	@info length(categories)

	# Assign stimulus pairs
	stimuli = assign_stimuli_and_optimality(;
		n_phases = 1,
		n_pairs = fill(1, nrow(PILT_blocks)),
		categories = categories,
		random_seed = random_seed
	)

	@info "Proportion of blocks on which the novel category is optimal : $(mean(stimuli.optimal_A))"

	rename!(stimuli, :phase => :session) # For compatibility with multi-phase sessions

	stimuli
end

# ╔═╡ 57aa0daa-0233-416d-802c-d0d93607062b
let

	PILT_blocks_stims = leftjoin(
		PILT_blocks,
		PILT_stimuli,
		on = :block
	)

	PILT_blocks_stims.EV_B_bin = discretize_to_quantiles(
		PILT_blocks_stims.EV_B,
		vcat(PILT_blocks_stims.EV_A, PILT_blocks_stims.EV_B),
		3
	)

	PILT_blocks_stims.previous_EV_B_bin = vcat(
		[missing], (PILT_blocks_stims.EV_B_bin)[1:(end-1)]
	)

	PILT_blocks_stims.repeating_optimal = vcat(
		[missing], (.! PILT_blocks_stims.optimal_A)[2:end]
	)

	PILT_blocks_stims.repeating_previous_optimal = vcat(
		[missing], (.! PILT_blocks_stims.optimal_A)[1:(end-1)]
	)

	combine(
		groupby(
			PILT_blocks_stims, 
			[:EV_B_bin, :previous_EV_B_bin]
		),
		:EV_B_bin => length => :n
	) 

end

# ╔═╡ cbe94f65-56f2-4262-8b6a-76b348ee1f8b
# ╠═╡ disabled = true
#=╠═╡
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
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═2d7211b4-b31e-11ef-3c0b-e979f01c47ae
# ╠═114f2671-1888-4b11-aab1-9ad718ababe6
# ╠═de74293f-a452-4292-b5e5-b4419fb70feb
# ╠═54f6f217-3ae5-49c7-9456-a5abcbbdc62f
# ╠═9f300301-b018-4bea-8fc4-4bc889b11afd
# ╠═b9134153-d9e9-4e35-bfc4-2c5c5a4329ee
# ╟─184a054c-5a88-44f8-865e-da75a10191ec
# ╠═6417ad94-1852-4cce-867e-a856295ec782
# ╠═e6f984aa-20dc-4a7d-8a3b-75728995a1f7
# ╠═efdfdeb0-2b56-415e-acf7-d6236ee7b199
# ╟─60c50147-708a-46f8-a813-7667116fc8d2
# ╠═1491f0f9-0c40-41ca-b7a9-055259f66eb3
# ╠═1a6d525f-5317-4b2b-a631-ea646ee20c9f
# ╠═b28f57a2-8aab-45e9-9d16-4c3b9fcf3828
# ╟─ffe06202-d829-4145-ae26-4a95449d64e6
# ╠═f5916a9f-ddcc-4c03-9328-7dd76c4c74b2
# ╠═e3bff0b9-306a-4bf9-8cbd-fe0e580bd118
# ╠═f9be1490-8e03-445f-b36e-d8ceff894751
# ╠═eecaac0c-e051-4543-988c-e969de3a8567
# ╠═3fa8c293-ac47-4acd-bdb7-9313286ee464
# ╠═68873d3e-054d-4ab4-9d89-73586bb0370e
# ╠═f89e88c9-ebfc-404f-964d-acff5c7f8985
# ╟─bcb73e19-8bf9-4e75-a9f7-8152e3d23201
# ╠═c65d7f1f-224d-4144-aa46-d48a482db95a
# ╠═63b74bde-4a28-4ff8-ae15-a37c891c3166
# ╠═832ccb61-b588-4614-9f5a-efa0f9a6087d
# ╠═df5baa98-27b9-4bbc-9ab0-03ea66e91970
# ╠═391f3f2a-74ae-463b-b109-a47eaaafdf97
# ╠═7a100e62-c028-4a24-a200-658bbca1945a
# ╠═18f41358-8554-4142-83f8-1c1c3f38b6fa
# ╠═84d2f787-aba2-40be-9816-4924968f5790
# ╠═926053c8-df86-440f-b397-47c9ed2a6000
# ╠═765efd14-e687-4fde-bd0a-4378251a7aa8
# ╠═76fcde1b-2174-44e3-a993-63a48879403f
# ╠═57aa0daa-0233-416d-802c-d0d93607062b
# ╠═73399d2c-8256-49a7-b739-cb1f496bcd2f
# ╠═34229660-e7b2-4e87-b420-334d018a2050
# ╠═cd31c2c9-faa0-4487-b9f3-993b07f297d3
# ╠═4d115a60-a7ce-48e7-af92-736ac2dbec07
# ╠═cbe94f65-56f2-4262-8b6a-76b348ee1f8b
