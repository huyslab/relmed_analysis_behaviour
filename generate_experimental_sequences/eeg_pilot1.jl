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
	include("$(pwd())/generate_experimental_sequences/FI_sequences_utils.jl")
	nothing
end

# ╔═╡ 4c34228b-6140-4748-8835-5ee130be4bc3
# General attributes of pilot PILT structure
begin
	set_sizes = 1:3
	block_per_set = 6
	trials_per_pair = 10

	# Total number of blocks
	n_total_blocks = length(set_sizes) * block_per_set

	# Shaping protocol
	n_confusing = vcat([0, 1, 1], fill(2, n_total_blocks - 3)) # Per block

	# Initial Q value for simulation - average outcome
	aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])
end

# ╔═╡ 4c09fe35-8015-43d8-a38f-2434318e74fe
# Assign valence and set size per block
valence_set_size = let random_seed = 10
	
	# # All combinations of set sizes and valence
	@assert iseven(block_per_set) # Requisite for code below
	
	valence_set_size = DataFrame(
		n_pairs = repeat(set_sizes, inner = block_per_set),
		valence = repeat([1, -1], inner = 3, outer = div(n_total_blocks, 6))
	)

	# Shuffle set size and valence, making sure set size rises gradually and valence is varied in first three blocks, and positive in the first
	rng = Xoshiro(random_seed)
	
	while (valence_set_size[1:3, :n_pairs] != [1, 2, 3]) | 
	allequal(valence_set_size[1:3, :valence]) | (valence_set_size.valence[1] == -1)
		valence_set_size.block = shuffle(rng, 1:n_total_blocks)
		sort!(valence_set_size, :block)
	end

	valence_set_size.n_confusing = n_confusing

	# Return
	valence_set_size
end

# ╔═╡ 2018073a-656d-4723-8384-07c9d533245f
# Create feedback sequences per pair
pair_sequences = let random_seed = 321
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
	chosen_idx = optimize_FI_distribution(
		n_wanted = n_wanted,
		FIs = FIs,
		common_seqs = common_seqs,
		magn_seqs = magn_seqs,
		ω_FI = 0.1,
		filename = "results/exp_sequences/eeg_pilot_FI_opt.jld2"
	)

	# Unpack chosen sequences
	chosen_common = [[common_seqs[s][idx[1]] for idx in chosen_idx[s]]
		for s in eachindex(common_seqs)]

	chosen_magn = [[magn_seqs[s][idx[2]] for idx in chosen_idx[s]]
		for s in eachindex(magn_seqs)]

	# Repack into DataFrame
	n_total_pairs = length(vcat(chosen_common...))
	
	task = DataFrame(
		trial = repeat(1:trials_per_pair, n_total_pairs),
		pair = repeat(1:n_total_pairs, inner = trials_per_pair),
		feedback_common = vcat(vcat(chosen_common...)...),
		variable_magnitude = vcat(vcat(chosen_magn...)...)
	)

	# Compute n_confusing and fifty_high per block
	DataFrames.transform!(
		groupby(task, :pair),
		:feedback_common => (x -> trials_per_pair - sum(x)) => :n_confusing,
		:variable_magnitude => (x -> 1. in x) => :fifty_high
	)

	@assert mean(task.fifty_high) == 0.5 "Proportion of blocks with 50 pence in high magnitude option expected to be 0.5"

	rng = Xoshiro(random_seed)

	# Shuffle within n_confusing
	DataFrames.transform!(
		groupby(task, [:n_confusing, :pair]),
		:n_confusing => (x -> x .* 10 .+ rand(rng)) => :random_pair
	)

	sort!(task, [:n_confusing, :random_pair, :trial])

	task.pair = repeat(1:n_total_pairs, inner = trials_per_pair)

	select!(task, Not(:random_pair))

	# Compute feedback_high and feedback_low
	task.feedback_high = ifelse.(
		task.fifty_high,
		task.variable_magnitude,
		fill(1., nrow(task))
	)

	task.feedback_low = ifelse.(
		task.fifty_high,
		fill(0.01, nrow(task)),
		task.variable_magnitude
	)

	# Compute feedback_optimal and feedback_suboptimal
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

	@warn "Must write assertion to make sure statistics are conserved from output of optimizing function"

	task
end

# ╔═╡ 36de90d8-b854-4c87-b8f5-3ab2eaae84f9
	# Load stimulus names
	categories = shuffle(unique([replace(s, ".png" => "")[1:(end-1)] for s in 
		readlines("generate_experimental_sequences/eeg_stim_list.txt")]))


# ╔═╡ Cell order:
# ╠═63e8c382-8560-11ef-1246-0923653e81d2
# ╠═4c34228b-6140-4748-8835-5ee130be4bc3
# ╠═4c09fe35-8015-43d8-a38f-2434318e74fe
# ╠═2018073a-656d-4723-8384-07c9d533245f
# ╠═36de90d8-b854-4c87-b8f5-3ab2eaae84f9
