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
		"generate_experimental_sequences/pilot7_wm_stimulus_sequence_longer.csv"))

	select!(
		triplet_order, 
		:stimset => :stimulus_group,
		:delay
	)
end

# ╔═╡ 184a054c-5a88-44f8-865e-da75a10191ec
md"""## RLWM"""

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
		categories[rand(rng, 1:length(categories))] * "_1.jpg" for _ in 1:length(unique(RLWM.stimulus_group))
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

# ╔═╡ f5916a9f-ddcc-4c03-9328-7dd76c4c74b2
# Create deterministic block
RLLTM_det_block = let det_block = copy(RLX_block),
	rng = Xoshiro(0)
	
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

	@assert !all(RLWM.feedback_left .== RLLTM.feedback_left) "Optimal actions should be different on RLWM and RLLTM"

	@assert all(RLWM.stimulus_group .== RLLTM.stimulus_group) "Stimuli sequence not identical in RLWM and RLLTM"

end

# ╔═╡ eecaac0c-e051-4543-988c-e969de3a8567
let
	save_to_JSON(RLLTM, "results/pilot8_LTM.json")
	CSV.write("results/pilot8_LTM.csv", RLLTM)
end

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
# ╟─ffe06202-d829-4145-ae26-4a95449d64e6
# ╠═f5916a9f-ddcc-4c03-9328-7dd76c4c74b2
# ╠═e3bff0b9-306a-4bf9-8cbd-fe0e580bd118
# ╠═f9be1490-8e03-445f-b36e-d8ceff894751
# ╠═eecaac0c-e051-4543-988c-e969de3a8567
# ╠═3fa8c293-ac47-4acd-bdb7-9313286ee464
# ╠═68873d3e-054d-4ab4-9d89-73586bb0370e
# ╠═f89e88c9-ebfc-404f-964d-acff5c7f8985
