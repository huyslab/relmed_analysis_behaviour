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

# ╔═╡ 3cb53c43-6b38-4c95-b26f-36697095f463
# General parameters
begin

	sessions = [
		"screening",
		"wk0",
		"wk2",
		"wk4",
		"wk24",
		"wk28"
	]

end

# ╔═╡ de74293f-a452-4292-b5e5-b4419fb70feb
categories = let

	categories = DataFrame(CSV.File("generate_experimental_sequences/trial1_stimuli/stimuli.csv")).concept |> unique

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

	# Copy over multiple sessions, skipping screening ------------
	det_block = vcat(
		[insertcols(
			det_block,
			1,
			:session => s
		) for s in sessions[2:end]]...
	)


	det_block

end

# ╔═╡ 6417ad94-1852-4cce-867e-a856295ec782
# Create deterministic block
RLWM = let RLWM = copy(RLX_block),
	rng = Xoshiro(0)


	# Assign stimuli --------

	stimuli = unique(select(RLWM, :session, :stimulus_group))

	stimuli.stimulus_left = [
		popat!(categories, rand(rng, 1:length(categories))) * "_1.jpg" for _ in 1:nrow(stimuli)
	]
		
	leftjoin!(
		RLWM,
		stimuli,
		on = [:session, :stimulus_group]
	)

	# Replicate - for RLWM there is only one stimulus, but this is requirement of js script
	RLWM.stimulus_middle = RLWM.stimulus_left
	RLWM.stimulus_right = RLWM.stimulus_left
	
	# Assign stimuli locations -----------------------------
	# Count appearances per stimulus_group
	stimulus_ordering = combine(
		groupby(RLWM, [:session, :stimulus_group]),
		:stimulus_group => length => :n
	)

	# Sort by descending n to distribute largest trials first
	shuffle!(rng, stimulus_ordering)
	sort!(stimulus_ordering, [:session, :n], rev=true)

	for gdf in groupby(stimulus_ordering, :session)

		# Track total counts per action
		action_sums = Dict(1 => 0, 2 => 0, 3 => 0)
	
		# Placeholder for optimal action
		gdf.optimal_action .= 99
		
		# Assign actions to balance total n
		for row in eachrow(gdf)
		    # Pick the action with the smallest current total
		    best_action = argmin(action_sums)
		    row.optimal_action = best_action
		    action_sums[best_action] += row.n
		end

	end

	# Join with data frame
	leftjoin!(
		RLWM,
		select(stimulus_ordering, [:session, :stimulus_group, :optimal_action]),
		on = [:session, :stimulus_group]
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

	# Valence variable
	RLWM.valence .= 1

	# Apperance variable
	DataFrames.transform!(
		groupby(RLWM, [:session, :block, :stimulus_group]),
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
	save_to_JSON(RLWM, "results/trial1_WM.json")
	CSV.write("results/trial1_WM.csv", RLWM)
end

# ╔═╡ 1491f0f9-0c40-41ca-b7a9-055259f66eb3
RLWM_test = let

	RLWM_test = DataFrame()

	for gdf in groupby(RLWM, :session)

		# Reset random seed so that randomization is the same across sessions
		rng = Xoshiro(0)

		# Get unique stimuli
		RLWM_stimuli = unique(gdf.stimulus_left)
	
		# Get all combinations
		RLWM_pairs = collect(combinations(RLWM_stimuli, 2))
	
		# Shuffle order within pair
		shuffle!.(rng, RLWM_pairs)
	
		# Repeat flipped
		RWLM_blocks = [iseven(i) ? reverse.(RLWM_pairs) : RLWM_pairs for i in 1:RLX_test_n_blocks]
	
		# Assemble into DataFrame
		RLWM_test_session = vcat([DataFrame(
			block = fill(i, length(stims)),
			stimulus_left = [x[1] for x in stims],
			stimulus_right = [x[2] for x in stims]
		) for (i, stims) in enumerate(RWLM_blocks)]...)
	
		# Shuffle trial order within block
		DataFrames.transform!(
			groupby(RLWM_test_session, :block),
			:block => (x -> shuffle(rng, 1:length(x))) => :trial
		)
	
		sort!(RLWM_test_session, [:block, :trial])

		# Add session variable
		insertcols!(
			RLWM_test_session,
			1,
			:session => gdf.session[1],
		)
	
		# Add variables needed for JS ------------------
		insertcols!(
			RLWM_test_session,
			:feedback_left => 1.,
			:feedback_right => 1.,
			:magnitude_left => 1.,
			:magnitude_right => 1.,
			:same_valence => true,
			:same_block => true,
			:original_block_left => 1,
			:original_block_right => 1
		)

		RLWM_test = vcat(RLWM_test, RLWM_test_session)

	end

	RLWM_test
	
end

# ╔═╡ b28f57a2-8aab-45e9-9d16-4c3b9fcf3828
let
	save_to_JSON(RLWM_test, "results/trial1_WM_test.json")
	CSV.write("results/trial1_WM_test.csv", RLWM_test)
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
	stimuli = vcat([insertcols(
		assign_triplet_stimuli_RLLTM(categories,
			maximum(det_block.stimulus_group);
			rng = rng
		),
		1,
		:session => s
	) for s in unique(det_block.session)]...)

	# Merge with trial structure
	det_block = innerjoin(
		det_block,
		stimuli,
		on = [:session, :stimulus_group],
		order = :left
	)

	# Assign stimuli locations -----------------------------
	# Count appearances per stimulus_group
	stimulus_ordering = combine(
		groupby(det_block, [:session, :stimulus_group]),
		:stimulus_group => length => :n
	)

	# Sort by descending n to distribute largest trials first
	shuffle!(rng, stimulus_ordering)
	sort!(stimulus_ordering, [:session, :n], rev=true)

	for gdf in groupby(stimulus_ordering, :session)

		# Track total counts per action
		action_sums = Dict(1 => 0, 2 => 0, 3 => 0)
	
		# Place holder for optimal action
		gdf.optimal_action .= 99
		
		# Assign actions to balance total n
		for row in eachrow(gdf)
		    # Pick the action with the smallest current total
		    best_action = argmin(action_sums)
		    row.optimal_action = best_action
		    action_sums[best_action] += row.n
		end
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
		select(stimulus_ordering, [:session, :stimulus_group, :stimulus_ordering]),
		on = [:session, :stimulus_group]
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

	# Assign feedback to stimulus essence, by covention stimulus_A is optimal
	det_block.feedback_A = det_block.feedback_optimal
	
	det_block.feedback_B = fill(0.01, nrow(det_block))

	det_block.feedback_C = fill(0.01, nrow(det_block))

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
	save_to_JSON(RLLTM, "results/trial1_LTM.json")
	CSV.write("results/trial1_LTM.csv", RLLTM)
end

# ╔═╡ bcb73e19-8bf9-4e75-a9f7-8152e3d23201
md"""## Post-LTM test"""

# ╔═╡ 8d8e4026-2185-4d67-a259-4cdebaab0b94
function prepare_for_finding_ltm_test_sequence(
	task::DataFrame;
)

	# Extract stimuli and their common feedback from task structure
	stimuli = vcat([select(
		filter(x -> x.feedback_common, task),
		:session,
		:block,
		:stimulus_group_id,
		:stimulus_group_id => ByRow(b -> "$(b)_$s") => :stimulus_essence,
		Symbol("feedback_$s") => :feedback
	) for s in ["A", "B", "C"]]...)

	# Summarize magnitude per stimulus
	stimuli = combine(
		groupby(stimuli, [:session, :block, :stimulus_group_id, :stimulus_essence]),
		:feedback => (x -> mean(unique(x))) => :magnitude
	)

	# Create Dict of unique stimuli by magnitude
	magnitudes = unique(stimuli.magnitude)
	stimuli_magnitude = Dict(m => unique(filter(x -> x.magnitude == m, stimuli).stimulus_essence) for m in magnitudes)

	#  Define existing pairs
	pairer(v) = [[v[i], v[j]] for i in 1:length(v) for j in i+1:length(v)]
	create_pair_list(d) = vcat([pairer(filter(x -> x.stimulus_group_id == p, d).stimulus_essence) 
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
    new_pairs = DataFrame(
		stimulus_essence_א = String[], 
		stimulus_essence_ב = String[],
		magnitude_א = Float64[], 
		magnitude_ב = Float64[])
    
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
RLLTM_test_session = let rng = Xoshiro(0)

	# Process LTM sequence to be able to find test pairs
	stimuli_magnitude, existing_pairs = 
		prepare_for_finding_ltm_test_sequence(
		filter(x -> x.session == RLLTM.session[1], RLLTM)
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
		:block => 1,
		:original_block_right => 1,
		:original_block_left => 1,
		:same_block => true,
		:valence_left => 1,
		:valence_right => 1,
		:same_valence => true
	)

	# Add magnitude pair variable
	RLLTM_test.magnitude_pair = [sort([r.magnitude_א, r.magnitude_ב]) for r in eachrow(RLLTM_test)]


	# Assign left / right
	DataFrames.transform!(
		groupby(RLLTM_test, :magnitude_pair),
		:trial => (x -> shuffled_fill([true, false], length(x), rng = rng)) => :א_on_right
	)

	RLLTM_test.magnitude_right = ifelse.(
		RLLTM_test.א_on_right,
		RLLTM_test.magnitude_א,
		RLLTM_test.magnitude_ב
	)

	RLLTM_test.magnitude_left = ifelse.(
		.!RLLTM_test.א_on_right,
		RLLTM_test.magnitude_א,
		RLLTM_test.magnitude_ב
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

	RLLTM_test.block .= 1


	RLLTM_test

end

# ╔═╡ ca34e152-6a53-4e25-a3d2-964c75d70fd5
RLLTM_test = let

	# Get all stimuli
	all_stimuli = unique(vcat([select(
		RLLTM,
		:session,
		:block,
		:stimulus_group_id,
		:stimulus_group_id => ByRow(b -> "$(b)_$stim") => :stimulus_essence,
		Symbol("stimulus_$stim") => :stimulus
	) for stim in ["A", "B", "C"]]...))


	# Join stimulus with stimulus essence
	RLLTM_test = innerjoin(
		RLLTM_test_session,
		select(
			all_stimuli,
			:session,
			:stimulus_essence => :stimulus_essence_א,
			:stimulus => :stimulus_א
		),
		on = :stimulus_essence_א
	)

	RLLTM_test = innerjoin(
		RLLTM_test,
		select(
			all_stimuli,
			:session,
			:stimulus_essence => :stimulus_essence_ב,
			:stimulus => :stimulus_ב
		),
		on = [:session, :stimulus_essence_ב]
	)

	# Sort
	sort!(RLLTM_test, [:session, :trial])

	# Compute stimulus on the right and left
	RLLTM_test.stimulus_right = ifelse.(
		RLLTM_test.א_on_right,
	    RLLTM_test.stimulus_א,
		RLLTM_test.stimulus_ב
	)

	RLLTM_test.stimulus_left = ifelse.(
		.!RLLTM_test.א_on_right,
	    RLLTM_test.stimulus_א,
		RLLTM_test.stimulus_ב
	)

	RLLTM_test

end

# ╔═╡ 30f86e30-e7e9-43c4-9001-1f2f0c6c2bea
# Tests for RLLtM_tests
let
	# Test even distribution of stimuli appearances
	long_test = vcat(
		select(
			RLLTM_test,
			:stimulus_א => :stimulus,
			:magnitude_א => :magnitude
		),
		select(
			RLLTM_test,
			:stimulus_ב => :stimulus,
			:magnitude_ב => :magnitude
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

	@assert mean(RLLTM_test.magnitude_left) ≈ mean(RLLTM_test.magnitude_right) "Different average magnitudes for stimuli presented on left and right"

	# Make sure all stimuli are in RLLTM
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

	# Test assignment of stimuli to stimuli_essence ---------------
	# Extract stimuli and their common feedback from task structure
	stimuli_magntiude = vcat([select(
		filter(x -> x.feedback_common, RLLTM),
		:session,
		:stimulus_group_id,
		Symbol("stimulus_$s") => :stimulus,
		Symbol("feedback_$s") => :feedback
	) for s in ["A", "B", "C"]]...)

	# Summarize magnitude per stimulus
	stimuli_magntiude = combine(
		groupby(stimuli_magntiude, [:session, :stimulus]),
		:feedback => (x -> mean(unique(x))) => :magnitude
	)

	# Extract stimuli and their common feedback from test structure
	test_magntiude = vcat([select(
		RLLTM_test,
		:session,
		Symbol("stimulus_$s") => :stimulus,
		Symbol("magnitude_$s") => :magnitude_test
	) for s in ["א", "ב"]]...)

	magnitdue_compare =
		leftjoin(
			stimuli_magntiude,
			test_magntiude,
			on = [:session, :stimulus]
		)

	@assert all(magnitdue_compare.magnitude .== magnitdue_compare.magnitude_test) "Test structure magnitude does not match learning phase structure magnitude"

end

# ╔═╡ 832ccb61-b588-4614-9f5a-efa0f9a6087d
let
	save_to_JSON(RLLTM_test, "results/trial1_LTM_test.json")
	CSV.write("results/trial1_LTM_test.csv", RLLTM_test)
end

# ╔═╡ Cell order:
# ╠═2d7211b4-b31e-11ef-3c0b-e979f01c47ae
# ╠═114f2671-1888-4b11-aab1-9ad718ababe6
# ╠═3cb53c43-6b38-4c95-b26f-36697095f463
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
# ╠═ca34e152-6a53-4e25-a3d2-964c75d70fd5
# ╠═30f86e30-e7e9-43c4-9001-1f2f0c6c2bea
# ╠═832ccb61-b588-4614-9f5a-efa0f9a6087d
# ╠═8d8e4026-2185-4d67-a259-4cdebaab0b94
# ╠═391f3f2a-74ae-463b-b109-a47eaaafdf97
