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
	include("$(pwd())/stats_utils.jl")
	include("$(pwd())/plotting_utils.jl")
	include("$(pwd())/fetch_preprocess_data.jl")
	include("$(pwd())/generate_experimental_sequences/sequence_utils.jl")
	nothing
end

# ╔═╡ de74293f-a452-4292-b5e5-b4419fb70feb
categories = let
	categories = (s -> replace(s, ".jpg" => "")[1:(end-1)]).(readdir("generate_experimental_sequences/pilot7_stims"))

	# Keep only categories where we have two files exactly
	keeps = filter(x -> last(x) == 2, countmap(categories))

	filter(x -> x in keys(keeps), unique(categories))
end

# ╔═╡ c05d90b6-61a7-4f9e-a03e-3e11791da6d0
# RLWM Parameters
begin
	WM_set_sizes = [2, 4, 6]
	WM_blocks_per_cell = [fill(3, 4), [1, 2, 2, 1], fill(1, 4)] # Det. reward - det. punishment - prob. reward - prob. punishment
	WM_trials_per_triplet = 10

	# Total number of blocks
	WM_n_total_blocks = sum(sum.(WM_blocks_per_cell))

	# Total number of triplets
	WM_n_total_tiplets = sum(sum.(WM_set_sizes .* WM_blocks_per_cell))

	# # Full deterministic task
	# WM_n_confusing = fill(0, WM_n_total_blocks) # Per block

	# # For uniform shuffling of triplets in block
	# WM_triplet_mini_block_size = 2

end

# ╔═╡ fdbe5c4e-29cd-4d24-bbe5-40d24d5f98f4
function reorder_with_fixed(v::AbstractVector, fixed::AbstractVector; rng::Xoshiro = Xoshiro(0))
	v = collect(v)
	
    # Ensure fixed vector is of the same length as v, filling with `missing` if needed
    fixed = vcat(fixed, fill(missing, length(v) - length(fixed)))[1:length(v)]
    
    # Identify indices where `fixed` is not `missing`
    fixed_indices = findall(!ismissing, fixed)
    
    # Create a pool of remaining elements by removing one instance of each fixed value
    remaining = v[:]
    for idx in fixed_indices
        value = fixed[idx]
        first_match = findfirst(==(value), remaining)
        if first_match !== nothing
            deleteat!(remaining, first_match)
        else
            error("Fixed value $value not found in v")
        end
    end
    
    # Shuffle the remaining elements
    shuffled_remaining = shuffle(rng, remaining)
    
    # Create the result with the same type as `v` but allow `missing`
    result = Vector{Union{eltype(v), Missing}}(undef, length(v))
    result .= missing
    
    # Place fixed elements in their positions
    for idx in fixed_indices
        result[idx] = fixed[idx]
    end
    
    # Fill the rest with shuffled elements
    remaining_idx = setdiff(1:length(v), fixed_indices)
    result[remaining_idx] .= shuffled_remaining
    
    return result
end

# ╔═╡ 699245d7-1493-4f94-bcfc-83184ca521eb
# Assign valence, probabilistic/determinitistic, and set size per block
block_order = let random_seed = 1

	# Replicate set sizes
	block_order = DataFrame(
		set_size = vcat([fill(s, n) for (i, s) in enumerate(WM_set_sizes) for n in WM_blocks_per_cell[i]]...),
	)

	# Set deterministic/probabilistic, valence
	DataFrames.transform!(
		groupby(block_order, :set_size),
		:set_size => (x -> vcat(
			fill("det", length(x) ÷ 2), 
			fill("prob", length(x) ÷ 2)
		)) => :det_prob,
		:set_size => (x -> [y for pair in zip(
			fill(-1, length(x) ÷ 2), 
			fill(1, length(x) ÷ 2)
		) for y in pair]) => :valence
	)

	# Shuffle with constraints
	gradual_set_size = false
	start_with_reward = false

	rng = Xoshiro(random_seed)
	
	# Shuffle set size order making sure we start with low set sizes
	DataFrames.transform!(
		groupby(block_order, [:det_prob, :valence]),
		:set_size => (x -> reorder_with_fixed(x, [2, 4], rng = rng)) => :set_size
	)

	sort!(block_order, :det_prob)

	# Shuffle valence, set size order, making sure we start with rewards and low set sizes
	block_order.block = vcat(
		invperm(reorder_with_fixed(1:(nrow(block_order) ÷ 2), [2, 4, 1], rng = rng)),
		invperm(reorder_with_fixed(1:(nrow(block_order) ÷ 2), [2, 4, 1], rng = rng)) .+ (nrow(block_order) ÷ 2)
	)

	sort!(block_order, :block)

	# Assign n_confusing
	DataFrames.transform!(
		groupby(block_order, [:det_prob, :valence]),
		:det_prob => (x -> ifelse(
				x[1] == "det",
				fill(0, length(x)),
				vcat([1, 1], fill(2, length(x) - 2))
			)) => :n_confusing
	)

	# Reorder columns
	select!(
		block_order,
		[:block, :det_prob,  :set_size, :n_confusing, :valence]
	)

	# Checks
	@assert nrow(block_order) == WM_n_total_blocks "Number of blocks different from desired"

	@assert sort(combine(
		groupby(block_order, [:set_size, :det_prob, :valence]),
		:block => length => :n
	), [:set_size, :det_prob]).n == [3, 3, 3, 3, 1, 2, 2, 1, 1, 1, 1, 1] "Block numbers don't match desired"

	block_order
	
end

# ╔═╡ f065038f-0ac7-431b-8b51-c7ad28efa04f
reorder_with_fixed(collect(1:10), [3, 5, missing, 6])

# ╔═╡ Cell order:
# ╠═2d7211b4-b31e-11ef-3c0b-e979f01c47ae
# ╠═de74293f-a452-4292-b5e5-b4419fb70feb
# ╠═c05d90b6-61a7-4f9e-a03e-3e11791da6d0
# ╠═699245d7-1493-4f94-bcfc-83184ca521eb
# ╠═fdbe5c4e-29cd-4d24-bbe5-40d24d5f98f4
# ╠═f065038f-0ac7-431b-8b51-c7ad28efa04f
