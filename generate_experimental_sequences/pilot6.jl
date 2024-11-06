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

# ╔═╡ ea917db6-ec27-454f-8b4e-9df65d65064b
md"""## PILT"""

# ╔═╡ 381e61e2-7d51-4070-8ad1-ce9e63015eb6
# PILT parameters
begin
	# PILT Parameters
	PILT_blocks_per_valence = 8
	PILT_trials_per_block = 10
	
	PILT_total_blocks = PILT_blocks_per_valence * 2
	PILT_n_confusing = vcat([0, 1, 1], fill(2, PILT_total_blocks - 3))
		
	# Post-PILT test parameters
	test_n_blocks = 2
end

# ╔═╡ 31128edd-5d2d-49e9-8f65-842bb42639f9
# Assign valence and set size per block
PILT_block_attr = let random_seed = 4
	
	# # All combinations of set sizes and valence
	block_attr = DataFrame(
		block = 1:PILT_total_blocks,
		valence = repeat([1, -1], inner = PILT_blocks_per_valence),
		fifty_high = repeat([true, false], outer = PILT_blocks_per_valence)
	)

	# Shuffle set size and valence, making sure valence is varied in first three blocks, and positive in the first
	rng = Xoshiro(random_seed)
	
	while allequal(block_attr[1:3, :valence]) | 
		(block_attr.valence[1] == -1)

		DataFrames.transform!(
			block_attr,
			:block => (x -> shuffle(rng, x)) => :block
		)
		
		sort!(block_attr, :block)
	end

	# Add n_confusing
	block_attr.n_confusing = PILT_n_confusing

	# Return
	block_attr
end

# ╔═╡ c7c5b78e-ad76-4877-8c80-af9151105544
# Create feedback sequences per pair
# PILT_sequences, common_per_pos, EV_per_pos = 
	let random_seed = 0
	
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

	# # Uncpack FI and sequence arrays
	# FIs = [x[1] for x in FI_seqs]
	# pushfirst!(FIs, zero_seq[1])

	# common_seqs = [x[2] for x in FI_seqs]
	# pushfirst!(common_seqs, zero_seq[2])

	# magn_seqs = [x[3] for x in FI_seqs]
	# pushfirst!(magn_seqs, zero_seq[3])

	# # Choose sequences optimizing FI under contraints
	# chosen_idx, common_per_pos, EV_per_pos = optimize_FI_distribution(
	# 	n_wanted = n_wanted,
	# 	FIs = FIs,
	# 	common_seqs = common_seqs,
	# 	magn_seqs = magn_seqs,
	# 	ω_FI = 0.1,
	# 	filename = "results/exp_sequences/eeg_pilot_FI_opt.jld2"
	# )

	# @assert length(vcat(chosen_idx...)) == sum(valence_set_size.n_pairs) "Number of saved optimize sequences does not match number of sequences needed. Delete file and rerun."

	# # Unpack chosen sequences
	# chosen_common = [[common_seqs[s][idx[1]] for idx in chosen_idx[s]]
	# 	for s in eachindex(common_seqs)]

	# chosen_magn = [[magn_seqs[s][idx[2]] for idx in chosen_idx[s]]
	# 	for s in eachindex(magn_seqs)]

	# # Repack into DataFrame	
	# task = DataFrame(
	# 	appearance = repeat(1:trials_per_pair, n_total_pairs),
	# 	cpair = repeat(1:n_total_pairs, inner = trials_per_pair),
	# 	feedback_common = vcat(vcat(chosen_common...)...),
	# 	variable_magnitude = vcat(vcat(chosen_magn...)...)
	# )

	# # Compute n_confusing and fifty_high per block
	# DataFrames.transform!(
	# 	groupby(task, :cpair),
	# 	:feedback_common => (x -> trials_per_pair - sum(x)) => :n_confusing,
	# 	:variable_magnitude => (x -> 1. in x) => :fifty_high
	# )

	# @assert mean(task.fifty_high) == 0.5 "Proportion of blocks with 50 pence in high magnitude option expected to be 0.5"

	# rng = Xoshiro(random_seed)

	# # Shuffle within n_confusing
	# DataFrames.transform!(
	# 	groupby(task, [:n_confusing, :cpair]),
	# 	:n_confusing => (x -> x .* 10 .+ rand(rng)) => :random_pair
	# )

	# sort!(task, [:n_confusing, :random_pair, :appearance])

	# task.cpair = repeat(1:n_total_pairs, inner = trials_per_pair)

	# select!(task, Not(:random_pair))

	# task, common_per_pos, EV_per_pos
end

# ╔═╡ Cell order:
# ╠═99c994e4-9c36-11ef-2c8f-d5829be639eb
# ╟─ea917db6-ec27-454f-8b4e-9df65d65064b
# ╠═381e61e2-7d51-4070-8ad1-ce9e63015eb6
# ╠═31128edd-5d2d-49e9-8f65-842bb42639f9
# ╠═c7c5b78e-ad76-4877-8c80-af9151105544
