### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ 62fb2c44-70f7-11ef-2499-9d1ed7c02f46
begin
	cd("/home/jovyan")
	import Pkg
	
	# activate the shared project environment
    Pkg.activate("$(pwd())/relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	using CairoMakie, Random, DataFrames, Distributions, StatsBase,
		ForwardDiff, LinearAlgebra, JLD2, FileIO, CSV, Dates, JSON, RCall, Turing, ParetoSmooth, MCMCDiagnosticTools, Printf
	using LogExpFunctions: logistic, logit

	include("$(pwd())/PILT_models.jl")
	include("$(pwd())/sample_utils.jl")
	include("$(pwd())/stats_utils.jl")
	include("$(pwd())/plotting_utils.jl")
	include("$(pwd())/single_p_QL.jl")
	include("$(pwd())/fetch_preprocess_data.jl")
	nothing
end

# ╔═╡ a4db20c5-a30b-454f-a04b-ffbe25459e07
aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])

# ╔═╡ c1b8f566-a260-4e9d-9712-61b9b53fb74c
# Sample datasets from prior
begin	
	prior_sample = let
		# Load sequence from file
		task = task_vars_for_condition("00")
	
		prior_sample = simulate_single_p_QL(
			2;
			block = task.block,
			outcomes = task.outcomes,
			initV = fill(aao, 1, 2),
			random_seed = 0,
			prior_ρ = truncated(Normal(0., 2.), lower = 0.),
			prior_a = Normal()
		)
	
	
		leftjoin(prior_sample, 
			task.task[!, [:block, :trial, :feedback_optimal, :feedback_suboptimal]],
			on = [:block, :trial]
		)
	
	end

end

# ╔═╡ 2b7e29ee-742c-4afd-95b9-602b3e42a6f2
function map_data_to_single_p_QL(
	data::AbstractDataFrame
)

	tdata = dropmissing(data)

	return (;
		block = collect(tdata.block),
		choice = tdata.choice,
		outcomes = hcat(tdata.feedback_suboptimal, tdata.feedback_optimal)
	)

end

# ╔═╡ a16f068f-a86d-4d91-990b-bbba4a0da511
function FI_for_feedback_sequence(;
	task::AbstractDataFrame,
	ρ::Float64,
	a::Float64,
	initV::Float64,
	n_blocks::Int64 = 500,
	summary_method::Function = tr
)

	# Sample from prior
	prior_sample = simulate_single_p_QL(
			n_blocks;
			block = task.block,
			outcomes = hcat(task.feedback_suboptimal, task.feedback_optimal),
			initV = fill(initV, 1, 2),
			random_seed = 0,
			prior_ρ = Dirac(ρ),
			prior_a = Dirac(a)
		)

	prior_sample = innerjoin(prior_sample, 
			task[!, [:block, :trial, :feedback_optimal, :feedback_suboptimal]],
			on = [:block, :trial]
	)

	prior_sample.block = prior_sample.PID

	prior_sample[!, :PID] .= 1

	# Compute FI
	return FI(
		data = prior_sample,
		model = single_p_QL,
		map_data_to_model = map_data_to_single_p_QL,
		param_names = [:a, :ρ],
		initV = fill(aao, 1, 2),
		summary_method = summary_method
	)
end

	

# ╔═╡ bb3917a6-50a3-4b07-ba23-c64e0efe4097
function sum_FI_for_feedback_sequence(;
	task::AbstractDataFrame,
	ρ_vals::AbstractVector,
	a_vals::AbstractVector,
	initV::Float64,
	n_blocks::Int64 = 200,
	within_summary_method::Function = det,
	across_summary_method::Function = median
)
	
	FIs = Matrix{Float64}(undef, length(ρ_vals), length(a_vals))
	
	for (i, ρ) in enumerate(ρ_vals)
		for (j, a) in enumerate(a_vals)
			FIs[i, j] = FI_for_feedback_sequence(;
				task = task,
				ρ = ρ,
				a = a,
				initV = initV,
				summary_method = within_summary_method,
				n_blocks = n_blocks
			) / n_blocks
		end
	end

	return across_summary_method(FIs)
end

# ╔═╡ e5395ac6-7525-445d-8e7e-d161c9f74f93
function sequence_to_task_df(;
	feedback_common::Vector{Bool}, # Sequence of common (true) / confusing (false) feedback
	feedback_magnitude_high::Vector{Float64}, # Sequence of high magnitude feedback,
	feedback_magnitude_low::Vector{Float64}, # Sequence of low magnitude feedback
)
	# Check inputs
	@assert length(feedback_common) == length(feedback_magnitude_high)
	@assert length(feedback_magnitude_low) == length(feedback_magnitude_high)

	n_trials = length(feedback_common)

	# Build data frame
	task = DataFrame(
		block = fill(1, n_trials),
		feedback_optimal = ifelse.(
			feedback_common, 
			feedback_magnitude_high, 
			feedback_magnitude_low
		), # Swap feedback magnitude on confusing trials
		feedback_suboptimal = ifelse.(
			.!feedback_common, 
			feedback_magnitude_high, 
			feedback_magnitude_low
		),
		trial = 1:n_trials
	)

	return task

end

# ╔═╡ f5b0c228-d73a-4001-b272-b00e6fc2446c
function shuffled_fill(
	values::AbstractVector, # Values to fill vector
	n::Int64; # How many trials overall
	random_seed::Int64 = 0
)	
	# Create vector with as equal number of appearance for each value as possible
	shuffled_values = shuffle(Xoshiro(random_seed), values)
	unshuffled_vector = collect(Iterators.take(Iterators.cycle(shuffled_values), n))

	return shuffle(Xoshiro(random_seed + 1), unshuffled_vector)
end

# ╔═╡ 5dbe7bc7-f348-4efa-91f3-17a1261a4e78
# Look at FI across parameter range for one sequence
FIs = let n_trials = 10,
	FI_res = 10

	task = sequence_to_task_df(;
		feedback_common = shuffled_fill([true, false], n_trials),
		feedback_magnitude_high = fill(1., n_trials),
		feedback_magnitude_low = shuffled_fill([0.01, 0.5], n_trials)
	)

	FIs = sum_FI_for_feedback_sequence(;
		task = task,
		ρ_vals = range(0., 10., length = FI_res),
		a_vals = range(-2., 2., length = FI_res),
		initV = aao,
		across_summary_method = identity
	)

	FIs

end

# ╔═╡ 2624f499-e554-43dc-ac89-e2adb82780e4
let
	f = Figure()

	ax = Axis(
		f[1,1],
		aspect = 1
	)

	heatmap!(
		ax,
		range(0., 10., length = 15),
		range(-2., 2., length = 15),
		FIs
	)

	ax_d = Axis(
		f[1,2]
	)

	hist!(
		ax_d,
		vec(FIs)
	)

	f
end

# ╔═╡ Cell order:
# ╠═62fb2c44-70f7-11ef-2499-9d1ed7c02f46
# ╠═a4db20c5-a30b-454f-a04b-ffbe25459e07
# ╠═c1b8f566-a260-4e9d-9712-61b9b53fb74c
# ╠═2b7e29ee-742c-4afd-95b9-602b3e42a6f2
# ╠═a16f068f-a86d-4d91-990b-bbba4a0da511
# ╠═bb3917a6-50a3-4b07-ba23-c64e0efe4097
# ╠═e5395ac6-7525-445d-8e7e-d161c9f74f93
# ╠═f5b0c228-d73a-4001-b272-b00e6fc2446c
# ╠═5dbe7bc7-f348-4efa-91f3-17a1261a4e78
# ╠═2624f499-e554-43dc-ac89-e2adb82780e4
