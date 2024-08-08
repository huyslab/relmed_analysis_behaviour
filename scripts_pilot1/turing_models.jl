### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ ac7fd352-555d-11ef-0f98-07f8c7c23d25
begin
	cd("/home/jovyan")
	import Pkg
	
	# activate the shared project environment
    Pkg.activate("$(pwd())/relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate
	using CairoMakie, Random, DataFrames, Distributions, Printf, PlutoUI, StatsBase,
		ForwardDiff, LinearAlgebra, Memoization, LRUCache, GLM, JLD2, FileIO, JuMP, CSV, Dates, JSON, RCall, Turing, ParetoSmooth
	using IterTools: product
	using LogExpFunctions: logsumexp, logistic
	using Combinatorics: combinations

	include("$(pwd())/fetch_preprocess_data.jl")
	include("$(pwd())/stan_functions.jl")
	include("$(pwd())/PLT_task_functions.jl")
	include("$(pwd())/fisher_information_functions.jl")
	include("$(pwd())/plotting_functions.jl")

end

# ╔═╡ ae31ab2d-3d59-4c33-b74a-f21dd43ca95e
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

# ╔═╡ 76412db2-9b4c-4aea-a049-3831321347ab
# Load and clean data
begin
	PLT_data = load_PLT_data()

	PLT_data = exclude_PLT_sessions(PLT_data)

	nothing
end

# ╔═╡ 453d0e76-f7da-415a-8ebc-a9fe25032557
md"""
## Testing Turing.jl
"""

# ╔═╡ 0c0a8b5f-efe1-4a21-8d9f-64c2de112847
let
	# Just simple test model
	@model function model(data)
	    μ ~ Normal()
		σ ~ truncated(Normal(), lower = 0.)
	    
		for i in 1:length(data)
	        data[i] ~ Normal(μ, σ)
	    end

		return data
	end

	data = rand(Normal(1, 1), 100)

	chain = sample(model(data), NUTS(), 1000)
	psis_loo(model(data), chain)
end

# ╔═╡ 43d7b28a-97a3-4db7-9e41-a7e73aa18b81
md"""
## Single Q learner
"""

# ╔═╡ 07492d7a-a15a-4e12-97b6-1e85aac23e4f
@model function single_p_QL(;
	N::Int64, # Total number of trials
	n_blocks::Int64, # Number of blocks
	n_trials::Int64, # Number of trials in block
	bl::Vector{Int64}, # Block number
	valence::AbstractVector, # Valence of each block
	choice, # Binary choice, coded true for stimulus A. Not typed so that it can be simulated
	outcomes::Matrix{Float64}, # Outcomes for options, first column optimal
	initV::Matrix{Float64} # Initial Q values
)

	# Priors on parameters
	ρ ~ truncated(Normal(0., 2.), lower = 0.)
	a ~ Normal(0., 1.)

	# Compute learning rate
	α = logistic(π/sqrt(3) * a) # hBayesDM uses Phi_approx from Stan. Here, logistic with the variance of the logistic multiplying a to equate the scales to that of a probit function.

	# Initialize Q values
	Qs = vcat([repeat(initV .* (ρ * valence[i]), n_trials) for i in 1:n_blocks]...)

	# Loop over trials, updating Q values and incrementing log-density
	for i in 1:N
		
		# Define choice distribution
		choice[i] ~ BernoulliLogit(Qs[i, 1] - Qs[i, 2])

		choice_idx::Int64 = choice[i] + 1

		# Prediction error
		PE = outcomes[i, choice_idx] * ρ - Qs[i, choice_idx]

		# Update Q value
		if (i != N) && (bl[i] == bl[i+1])
			Qs[i + 1, choice_idx] = Qs[i, choice_idx] + α * PE
			Qs[i + 1, 3 - choice_idx] = Qs[i, 3 - choice_idx]
		end
	end

	return Qs

end

# ╔═╡ 69f78ddd-2310-4534-997c-6888e2808ea5
function simulate_single_p_QL(
	n::Int64; # How many datasets to simulate
	bl::Vector{Int64}, # Block number
	valence::AbstractVector, # Valence of each block
	outcomes::Matrix{Float64}, # Outcomes for options, first column optimal
	initV::Matrix{Float64}, # Initial Q values
	random_seed::Union{Int64, Nothing} = nothing
)

	# Total trial number
	N = length(bl)

	# Trials per block
	n_trials = div(length(bl), maximum(bl))

	# Prepare model for simulation
	prior_model = single_p_QL(
		N = N,
		n_trials = n_trials,
		n_blocks = maximum(bl),
		bl = bl,
		valence = valence,
		choice = fill(missing, length(bl)),
		outcomes = outcomes,
		initV = initV
	)


	# Draw parameters and simulate choice
	prior_sample = sample(
		isnothing(random_seed) ? Random.default_rng() : Xoshiro(random_seed),
		prior_model,
		Prior(),
		n
	)

	# Arrange choice for return
	sim_data = DataFrame(
		PID = repeat(1:n, inner = N),
		ρ = repeat(prior_sample[:, :ρ, 1], inner = N),
		α = repeat(prior_sample[:, :a, 1], inner = N) .|> a2α,
		block = repeat(bl, n),
		valence = repeat(valence, inner = n_trials, outer = n),
		trial = repeat(1:n_trials, n * maximum(bl)),
		choice = prior_sample[:, [Symbol("choice[$i]") for i in 1:N], 1] |>
			Array |> transpose |> vec
	)

	# Compute Q values
	Qs = generated_quantities(prior_model, prior_sample) |> vec

	sim_data.Q_A = vcat([qs[:, 1] for qs in Qs]...) 
	sim_data.Q_B = vcat([qs[:, 2] for qs in Qs]...) 

	return sim_data
			
end

# ╔═╡ 14b82fda-229b-4a52-bc49-51201d4706be
prior_sample = let
	# Load sequence from file
	task = DataFrame(CSV.File("data/PLT_task_structure_00.csv"))

	# Arrange outcomes such as first column is optimal
	outcomes = hcat(
		ifelse.(task.optimal_A .== 1, task.feedback_A, task.feedback_B),
		ifelse.(task.optimal_A .== 0, task.feedback_A, task.feedback_B)
	)

	# Initial value for Q values
	aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])

	simulate_single_p_QL(
		100;
		bl = task.block .+ (task.session .- 1) * maximum(task.block),
		valence = unique(task[!, [:session, :block, :valence]]).valence,
		outcomes = outcomes,
		initV = fill(aao, 1, 2),
		random_seed = 0
	)

end

# ╔═╡ 78b422d6-c70f-4a29-a433-7173e1b108a0
let
	df = rename(prior_sample, 
		:Q_A => :EV_A,
		:Q_B => :EV_B
	)

	df[!, :group] .= 1
	
	f = Figure(size = (700, 1000))

	g_all = f[1,1] = GridLayout()
	
	plot_sim_q_value_acc!(
		g_all,
		df;
		plw = 1,
		legend = false
	)

	g_reward = f[2,1] = GridLayout()
	
	plot_sim_q_value_acc!(
		g_reward,
		filter(x -> x.valence > 0, df);
		plw = 1,
		legend = false
	)

	g_reward = f[3,1] = GridLayout()
	
	plot_sim_q_value_acc!(
		g_reward,
		filter(x -> x.valence < 0, df);
		plw = 1,
		legend = false
	)

	f
end

# ╔═╡ Cell order:
# ╠═ac7fd352-555d-11ef-0f98-07f8c7c23d25
# ╠═ae31ab2d-3d59-4c33-b74a-f21dd43ca95e
# ╠═76412db2-9b4c-4aea-a049-3831321347ab
# ╟─453d0e76-f7da-415a-8ebc-a9fe25032557
# ╠═0c0a8b5f-efe1-4a21-8d9f-64c2de112847
# ╟─43d7b28a-97a3-4db7-9e41-a7e73aa18b81
# ╠═07492d7a-a15a-4e12-97b6-1e85aac23e4f
# ╠═14b82fda-229b-4a52-bc49-51201d4706be
# ╠═78b422d6-c70f-4a29-a433-7173e1b108a0
# ╠═69f78ddd-2310-4534-997c-6888e2808ea5