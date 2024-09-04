### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ fefc8298-6abf-11ef-0cc9-09ed3cc4c051
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

# ╔═╡ 9b998e37-49e9-427d-963a-c8beb6e2c58b
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

# ╔═╡ 93894285-d47c-49e5-9519-29ecceca4f1d
# Initial value for Q values
aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])

# ╔═╡ b018bc38-bde2-4bff-bbc8-79e4dd81938e
begin
	# Sample single dataset from prior
	function simulate_participant(;
		condition::String,
		ρ::Float64,
		a::Float64,
		repeats::Int64 = 1
		)	
			# Load sequence from file
			task = task_vars_for_condition("110")
			
			prior_sample = simulate_single_p_QL(
				repeats;
				block = task.block,
				valence = task.valence,
				outcomes = task.outcomes,
				initV = fill(aao, 1, 2),
				random_seed = 0,
				prior_ρ = Dirac(ρ),
				prior_a = Dirac(a)
			)
		
		
			prior_sample = leftjoin(prior_sample, 
				task.task[!, [:block, :trial, :feedback_optimal, :feedback_suboptimal]],
				on = [:block, :trial]
			)
	
			# Renumber blocks
			if repeats > 1
					prior_sample.block = prior_sample.block .+ 
						(prior_sample.PID .- 1) .* maximum(prior_sample.block)
			end
	
		return prior_sample
	end
	
	# Sample single dataset from prior, draw task randomly
	function simulate_participant_random_task(;
		ρ::Float64,
		a::Float64,
		n_blocks::Int64 = 48,
		n_trials::Int64 = 13
		)	
			# Create task sequence
			block = repeat(1:n_blocks, inner = n_trials)
			valence = ones(Int64, n_blocks)
			outcomes = hcat(
				rand(Normal(), n_blocks * n_trials),
				rand(Normal(0.5, 1.), n_blocks * n_trials)
			)
			trial = repeat(1:n_trials, n_blocks)
			
			prior_sample = simulate_single_p_QL(
				1;
				block = block,
				valence = valence,
				outcomes = outcomes,
				initV = fill(0., 1, 2),
				random_seed = 0,
				prior_ρ = Dirac(ρ),
				prior_a = Dirac(a)
			)
		
			
			prior_sample.trial = trial
	
			prior_sample.feedback_suboptimal = outcomes[:, 1]
	
			prior_sample.feedback_optimal = outcomes[:, 2]
	
		return prior_sample
	end
end

# ╔═╡ 61dec821-a308-48cb-ab41-7124369810f4
let n_blocks = 48,
	n_trials = 13

	block = repeat(1:n_blocks, inner = n_trials)
	valence = ones(Int64, n_blocks * n_trials)
	length(valence)
end

# ╔═╡ 14e78184-fa18-4489-89d8-41dfd3044203
function plot_turing_ll(
	f::GridPosition;
	data::DataFrame,
	prior_ρ::Distribution,
	prior_a::Distribution,
	grid_ρ::AbstractVector = range(0., 10., length = 200),
	grid_a::AbstractVector = range(-4, 4., length = 200)
)
	
	# Set up model
	post_model = single_p_QL(;
		N = nrow(data),
		n_blocks = maximum(data.block),
		block = data.block,
		valence = data.valence,
		choice = data.choice,
		outcomes = hcat(
			data.feedback_suboptimal, 
			data.feedback_optimal
		),
		initV = fill(aao, 1,2),
		prior_ρ = prior_ρ,
		prior_a = prior_a
	)

	# Set up axis
	ax = Axis(
		f,
		xlabel = "ρ",
		ylabel = "a"
	)

	# Plot loglikelihood
	ρ = repeat(grid_ρ, inner = length(grid_a))
	a = repeat(grid_a, outer = length(grid_ρ))
	ll = [loglikelihood(
			post_model, 
			(ρ = ρ, a = a)) for ρ in grid_ρ for a in grid_a]
	
	contour!(
		ax,
		ρ,
		a, 
		ll,
		levels = 10
	)

	# Plot MLE
	scatter!(
		ax,
		ρ[argmax(ll)],
		a[argmax(ll)],
		marker = :cross,
		markersize = 8,
		color = :blue
	)

	return ax

end

# ╔═╡ 213362ac-4d9e-4dfa-b9da-b9eb23c12a29
function single_p_QL_ll(;
	N::Int64, # Total number of trials
	n_blocks::Int64, # Number of blocks
	block::Vector{Int64}, # Block number
	valence::AbstractVector, # Valence of each block
	choice, # Binary choice, coded true for stimulus A. Not typed so that it can be simulated
	outcomes::Matrix{Float64}, # Outcomes for options, second column optimal
	initV::Matrix{Float64}, # Initial Q values
	ρ::Float64,
	a::Float64
)

	# Compute learning rate
	α = a2α(a) # hBayesDM uses Phi_approx from Stan. Here, logistic with the variance of the logistic multiplying a to equate the scales to that of a probit function.

	# Initialize Q values
	Qs = repeat(initV .* ρ, length(block)) .* valence[block]

	ll = 0.

	# Loop over trials, updating Q values and incrementing log-density
	for i in 1:N
		
		# Define choice distribution
		ll += logpdf(BernoulliLogit(Qs[i, 2] - Qs[i, 1]), choice[i])

		choice_idx::Int64 = choice[i] + 1

		# Prediction error
		PE = outcomes[i, choice_idx] * ρ - Qs[i, choice_idx]

		# Update Q value
		if (i != N) && (block[i] == block[i+1])
			Qs[i + 1, choice_idx] = Qs[i, choice_idx] + α * PE
			Qs[i + 1, 3 - choice_idx] = Qs[i, 3 - choice_idx]
		end
	end

	return ll

end

# ╔═╡ a11a50ae-5aff-49aa-94e7-7f9fd688efb4
function plot_handcrafted_ll(
	f::GridPosition;
	data::DataFrame,
	grid_ρ::AbstractVector = range(0., 10., length = 200),
	grid_a::AbstractVector = range(-4, 4., length = 200)
)
	
	# Set up model
	llf = (ρ, a) -> single_p_QL_ll(;
		N = nrow(data),
		n_blocks = maximum(data.block),
		block = data.block,
		valence = data.valence,
		choice = data.choice,
		outcomes = hcat(
			data.feedback_suboptimal, 
			data.feedback_optimal
		),
		initV = fill(aao, 1,2),
		ρ = ρ,
		a = a
	)

	# Set up axis
	ax = Axis(
		f,
		xlabel = "ρ",
		ylabel = "a"
	)

	# Plot loglikelihood
	ρ = repeat(grid_ρ, inner = length(grid_a))
	a = repeat(grid_a, outer = length(grid_ρ))
	ll = [llf(ρ, a) for ρ in grid_ρ for a in grid_a]
	
	contour!(
		ax,
		ρ,
		a,
		ll,
		levels = 10
	)

	# Plot MLE
	scatter!(
		ax,
		ρ[argmax(ll)],
		a[argmax(ll)],
		marker = :cross,
		markersize = 8,
		color = :blue
	)


	return ax

end

# ╔═╡ 186b86f8-4eba-4b45-9ca4-8d2a1d8e90cb
function simulate_plot_ll!(
	f::GridPosition;
	condition::String,
	ρ::Float64,
	a::Float64,
	title::String = "",
	repeats::Int64 = 1,
)

	if condition == "random"
		prior_sample = simulate_participant_random_task(;
			ρ = ρ,
			a = a,
			n_blocks = repeats
		)
	else
		prior_sample = simulate_participant(;
			condition = condition,
			ρ = ρ,
			a = a,
			repeats = repeats
		)
	end

	ax = plot_turing_ll(
		f;
		data = prior_sample,
		prior_ρ = Dirac(99.),
		prior_a = Dirac(-99.)
	)

	ax.title = title

	return ax
end

# ╔═╡ d0cbfe44-5f88-4833-9313-bad04f51342b
let

	f = Figure(size = (700, 700 / 4))

	for (i, r) in enumerate([1, 5, 10, 20] .* 48)
		ax = simulate_plot_ll!(
			f[1, i];
			condition = "random",
			ρ = 6.,
			a = 0.5,
			repeats = r
		)

		ax.title = "$(r)"

		scatter!(
			ax,
			6.,
			0.5,
			marker = :star5,
			markersize = 15,
			color = :red
		)
	end

	f

end

# ╔═╡ 032842c4-507b-4a58-b860-5a085b93ac47
let

	f = Figure(size = (700, 700 / 4))

	for (i, r) in enumerate([1, 5, 10, 20])
		ax = simulate_plot_ll!(
			f[1, i];
			condition = "00",
			ρ = 6.,
			a = 0.5,
			repeats = r
		)

		ax.title = "$(r*48)"

		scatter!(
			ax,
			6.,
			0.5,
			marker = :star5,
			markersize = 15,
			color = :red
		)
	end

	f

end

# ╔═╡ 2d44b2a8-218b-49dd-ac6d-f4ac45e2c216
let

	f = Figure(size = (700, 700))

	for (i, ρ) in enumerate(range(0.5, 10., length = 4))
		for (j, a) in enumerate(range(-2., 2., length = 4))
			ax = simulate_plot_ll!(
				f[j, i];
				condition = "random",
				ρ = ρ,
				a = a
			)

			scatter!(
				ax,
				ρ,
				a,
				marker = :star5,
				markersize = 15,
				color = :red
			)
		end
	end

	f

end

# ╔═╡ e73d8171-98df-4955-a627-20b40a9a2a5c
let

	f = Figure(size = (700, 700))

	for (i, ρ) in enumerate(range(0.5, 10., length = 4))
		for (j, a) in enumerate(range(-2., 2., length = 4))
			ax = simulate_plot_ll!(
				f[j, i];
				condition = "00",
				ρ = ρ,
				a = a
			)

			scatter!(
				ax,
				ρ,
				a,
				marker = :star5,
				markersize = 15,
				color = :red
			)
		end
	end

	f

end

# ╔═╡ a3c71e4e-03ca-41fd-8a1d-6cedd0da068e
let
	prior_sample = simulate_participant(;
		condition = "00",
		ρ = 8.,
		a = 0.
	)
	
	f = Figure(size = (700, 280))

	ax1 = plot_turing_ll(
		f[1,1];
		data = prior_sample,
		prior_ρ = truncated(Normal(0., 2.), lower = 0.),
		prior_a = Normal()
	)

	ax1.title = "Turing\n with prior 1"

	ax2 = plot_turing_ll(
		f[1,2];
		data = prior_sample,
		prior_ρ = truncated(Normal(99., 2.), lower = 0.),
		prior_a = Normal(-99., 3.)
	)

	ax2.title = "Turing\n with prior 2"

	ax3 = plot_handcrafted_ll(
		f[1,3],
		data = prior_sample
	)

	ax3.title = "My own LL"

	for ax in [ax1, ax2, ax3]

		scatter!(
			ax,
			8.,
			0.,
			marker = :star5,
			markersize = 15,
			color = :red
		)

	end

	f
end

# ╔═╡ Cell order:
# ╠═fefc8298-6abf-11ef-0cc9-09ed3cc4c051
# ╠═9b998e37-49e9-427d-963a-c8beb6e2c58b
# ╠═93894285-d47c-49e5-9519-29ecceca4f1d
# ╠═b018bc38-bde2-4bff-bbc8-79e4dd81938e
# ╠═61dec821-a308-48cb-ab41-7124369810f4
# ╠═14e78184-fa18-4489-89d8-41dfd3044203
# ╠═213362ac-4d9e-4dfa-b9da-b9eb23c12a29
# ╠═a11a50ae-5aff-49aa-94e7-7f9fd688efb4
# ╠═186b86f8-4eba-4b45-9ca4-8d2a1d8e90cb
# ╠═d0cbfe44-5f88-4833-9313-bad04f51342b
# ╠═032842c4-507b-4a58-b860-5a085b93ac47
# ╠═2d44b2a8-218b-49dd-ac6d-f4ac45e2c216
# ╠═e73d8171-98df-4955-a627-20b40a9a2a5c
# ╠═a3c71e4e-03ca-41fd-8a1d-6cedd0da068e
