### A Pluto.jl notebook ###
# v0.19.45

using Markdown
using InteractiveUtils

# ╔═╡ 88f8505c-5fc2-11ef-2ac9-9bb7f4b5b591
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

	include("$(pwd())/sample_utils.jl")
	#include("$(pwd())/fetch_preprocess_data.jl")
	include("$(pwd())/simulate.jl")
	include("$(pwd())/turing_models.jl")
	include("$(pwd())/plotting_utils.jl")
end

# ╔═╡ d49d6f84-5ff6-4100-90db-995ffdeb09ca
md"""
## Notebook setup
"""

# ╔═╡ cf2e2616-d29a-49cb-b9ee-6b6ea89c4ee6
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

# ╔═╡ 533b6f26-adb1-438e-9cbc-5dcc2d19ec1a
md"""
## Comparing Q-learning models with different additional parameters
"""

# ╔═╡ 9bfdd2b3-7885-42b3-8839-c426dae394a4
# Sample datasets from prior
begin
	prior_samples = let
		# Load sequence from file
		task = task_vars_for_condition("00")
		n_ppt = 100
	
		# Initial value for Q values
		aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])
	
		prior_bsl = simulate_from_prior(
			n_ppt;
			model = RL_ss,
			block = task.block,
			valence = task.valence,
			outcomes = task.outcomes,
			initV = fill(aao, 1, 2),
			parameters = [:ρ, :a],
			transformed = Dict(:a => :α),
			priors = Dict(
		        :ρ => truncated(Normal(0., 1.), lower = 0.),
		        :a => Normal(0., 0.5)
		    ),
			random_seed = 123
		)
		insertcols!(prior_bsl, :model => "α|ρ")

		prior_lapse = simulate_from_prior(
			n_ppt;
			model = RL_ss,
			block = task.block,
			valence = task.valence,
			outcomes = task.outcomes,
			initV = fill(aao, 1, 2),
			parameters = [:ρ, :a, :E],
			transformed = Dict(:a => :α, :E => :ε),
			priors = Dict(
		        :ρ => truncated(Normal(0., 1.), lower = 0.),
		        :a => Normal(0., 0.5),
				:E => Normal(0., 0.5)
		    ),
			random_seed = 123
		)
		insertcols!(prior_lapse, :model => "α|ρ|ε")

		prior_forget = simulate_from_prior(
			n_ppt;
			model = RL_ss,
			block = task.block,
			valence = task.valence,
			outcomes = task.outcomes,
			initV = fill(aao, 1, 2),
			parameters = [:ρ, :a, :F],
			transformed = Dict(:a => :α, :F => :φ),
			priors = Dict(
		        :ρ => truncated(Normal(0., 1.), lower = 0.),
		        :a => Normal(0., 0.5),
				:F => Normal(0., 0.5)
		    ),
			random_seed = 123
		)
		insertcols!(prior_forget, :model => "α|ρ|φ")

		prior_full = simulate_from_prior(
			n_ppt;
			model = RL_ss,
			block = task.block,
			valence = task.valence,
			outcomes = task.outcomes,
			initV = fill(aao, 1, 2),
			parameters = [:ρ, :a, :E, :F],
			transformed = Dict(:a => :α, :E => :ε, :F => :φ),
			priors = Dict(
		        :ρ => truncated(Normal(0., 1.), lower = 0.),
		        :a => Normal(0., 0.5),
				:E => Normal(0., 0.5),
				:F => Normal(0., 0.5)
		    ),
			random_seed = 123
		)
		insertcols!(prior_full, :model => "α|ρ|ε|φ")

		prior_samples = vcat(prior_bsl, prior_lapse, prior_forget, prior_full; cols=:union)
	
		leftjoin(prior_samples, 
			task.task[!, [:block, :trial, :feedback_optimal, :feedback_suboptimal]],
			on = [:block, :trial]
		)
	
	end

	describe(prior_samples)
end

# ╔═╡ 3ea2b653-a8f7-4e6b-910d-c5e4d284be53
let
	f = plot_prior_predictive_by_valence(
		prior_samples,
		[:Q_optimal, :Q_suboptimal];
		fig_size = (1000, 1000),
		group = :model,
		legend = true,
		colors = Makie.colorschemes[:seaborn_pastel6]
	)	
	f
end

# ╔═╡ bb8fc03d-f85a-451c-89c4-2303f0cb7fef
md"
## Reinforcement learning + working memory hybrid models
"

# ╔═╡ bfe4c9c1-72f0-4999-bbcb-76bdd4f579da
# ╠═╡ disabled = true
#=╠═╡
# Sample datasets from prior
begin
	prior_rlwm_samples = let
		# Load sequence from file
		task = task_vars_for_condition("00")
		n_ppt = 100
	
		# Initial value for Q values
		aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])
	
		prior_rlwm_sens = simulate_from_prior(
			n_ppt;
			model = RLWM_ss,
			block = task.block,
			valence = task.valence,
			outcomes = task.outcomes,
			initV = fill(aao, 1, 2),
			set_size = fill(2, maximum(task.block)),
			parameters = [:ρ, :a, :F_wm, :W, :C],
			transformed = Dict(:a => :α, :F_wm => :φ_wm, :W => :w0),
			priors = Dict(
		        :ρ => truncated(Normal(0., 1.), lower = 0.),
		        :a => Normal(0., 0.5),
		        :F_wm => Normal(0., 0.5),
		        :W => Normal(0., 0.5),
		        :C => truncated(Normal(3., 2.), lower = 1.)
		    ),
			random_seed = 0
		)
		insertcols!(prior_rlwm_sens, :model => "α|ρ|φ|C|w0")

		# prior_rlwm_temp = simulate_from_prior(
		# 	200;
		# 	model = RLWM_ss,
		# 	block = task.block,
		# 	valence = task.valence,
		# 	outcomes = task.outcomes,
		# 	initV = fill(aao, 1, 2),
		# 	set_size = fill(2, maximum(task.block)),
		# 	random_seed = 0,
		# 	parameters = [:β, :a, :F_wm, :W, :C],
		# 	transformed = Dict(:a => :α, :F_wm => :φ_wm, :W => :w0),
		# 	sigmas = Dict(:β => 2., :a => 0.5, :F_wm => 0.5, :W => 0.5, :C => 2.)
		# )
		# insertcols!(prior_rlwm_temp, :model => "α|β|φ|C|w0")

		prior_rlwm_lapse = simulate_from_prior(
			n_ppt;
			model = RLWM_ss,
			block = task.block,
			valence = task.valence,
			outcomes = task.outcomes,
			initV = fill(aao, 1, 2),
			set_size = fill(2, maximum(task.block)),
			parameters = [:ρ, :a, :E, :F_wm, :W, :C],
			transformed = Dict(:a => :α, :E => :ε, :F_wm => :φ_wm, :W => :w0),
			priors = Dict(
		        :ρ => truncated(Normal(0., 1.), lower = 0.),
		        :a => Normal(0., 0.5),
				:E => Normal(0., 0.5),
		        :F_wm => Normal(0., 0.5),
		        :W => Normal(0., 0.5),
		        :C => truncated(Normal(3., 2.), lower = 1.)
		    ),
			random_seed = 0
		)
		insertcols!(prior_rlwm_lapse, :model => "α|ρ|φ|C|w0|ε")

		prior_rlwm_all = simulate_from_prior(
			n_ppt;
			model = RLWM_ss,
			block = task.block,
			valence = task.valence,
			outcomes = task.outcomes,
			initV = fill(aao, 1, 2),
			set_size = fill(2, maximum(task.block)),
			parameters = [:ρ, :a, :E, :F_rl, :F_wm, :W, :C],
			transformed = Dict(:a => :α, :E => :ε, :F_rl => :φ_rl, :F_wm => :φ_wm, :W => :w0),
			priors = Dict(
		        :ρ => truncated(Normal(0., 1.), lower = 0.),
		        :a => Normal(0., 0.5),
				:E => Normal(0., 0.5),
				:F_rl => Normal(0., 0.5),
		        :F_wm => Normal(0., 0.5),
		        :W => Normal(0., 0.5),
		        :C => truncated(Normal(3., 2.), lower = 1.)
		    ),
			random_seed = 0
		)
		insertcols!(prior_rlwm_all, :model => "α|ρ|φ_rl|C|w0|ε|φ_wm")

		prior_rlwm_samples = vcat(prior_rlwm_sens, prior_rlwm_lapse, prior_rlwm_all; cols=:union)
	
		leftjoin(prior_rlwm_samples, 
			task.task[!, [:block, :trial, :feedback_optimal, :feedback_suboptimal]],
			on = [:block, :trial]
		)
	
	end

	describe(prior_rlwm_samples)
end
  ╠═╡ =#

# ╔═╡ 4b628e68-0625-4cff-bfc4-f146f9673d3c
#=╠═╡
let
	f = plot_prior_predictive_by_valence(
		prior_rlwm_samples,
		[:Q_optimal, :Q_suboptimal];
		W_cols = [:W_optimal, :W_suboptimal],
		ylab = ("Q-value", "W-value"),
		fig_size = (1000, 1000),
		group = :model,
		legend = true,
		legend_rows = 3, 
		colors = Makie.colorschemes[:seaborn_pastel6]
	)	
	f
end
  ╠═╡ =#

# ╔═╡ 91548dc3-0812-4070-9b32-5b5b3bb05eb9
# Sample datasets from prior
begin
	prior_wpmst_samples = let
		# Load sequence from file
		task = task_vars_for_condition("00")
		n_ppt = 100
	
		# Initial value for Q values
		aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])
	
		prior_pmst = simulate_from_prior(
			n_ppt;
			model = RLWM_pmst,
			block = task.block,
			valence = task.valence,
			outcomes = task.outcomes,
			initV = fill(aao, 1, 2),
			set_size = fill(2, maximum(task.block)),
			parameters = [:ρ, :a, :W, :C],
			transformed = Dict(:a => :α, :W => :w0),
			priors = Dict(
		        :ρ => truncated(Normal(0., 1.), lower = 0.),
		        :a => Normal(0., 0.5),
		        :W => Normal(0., 0.5),
		        :C => truncated(Normal(3., 2.), lower = 1.)
		    ),
			random_seed = 0
		)
		insertcols!(prior_pmst, :model => "α|ρ|C|w0")

		prior_pmst_lapse = simulate_from_prior(
			n_ppt;
			model = RLWM_pmst,
			block = task.block,
			valence = task.valence,
			outcomes = task.outcomes,
			initV = fill(aao, 1, 2),
			set_size = fill(2, maximum(task.block)),
			parameters = [:ρ, :a, :E, :W, :C],
			transformed = Dict(:a => :α, :E => :ε, :W => :w0),
			priors = Dict(
		        :ρ => truncated(Normal(0., 1.), lower = 0.),
		        :a => Normal(0., 0.5),
				:E => Normal(0., 0.5),
		        :W => Normal(0., 0.5),
		        :C => truncated(Normal(3., 2.), lower = 1.)
		    ),
			random_seed = 0
		)
		insertcols!(prior_pmst_lapse, :model => "α|ρ|C|w0|ε")

		prior_psmt_samples = vcat(prior_pmst, prior_pmst_lapse; cols=:union)
	
		leftjoin(prior_psmt_samples, 
			task.task[!, [:block, :trial, :feedback_optimal, :feedback_suboptimal]],
			on = [:block, :trial]
		)
	
	end

	describe(prior_wpmst_samples)
end

# ╔═╡ a9a8d13b-4960-4a53-aed5-c104260625b5
let
	f = plot_prior_predictive_by_valence(
		prior_wpmst_samples,
		[:Q_optimal, :Q_suboptimal];
		W_cols = [:W_optimal, :W_suboptimal],
		ylab = ("Q-value", "W-value"),
		fig_size = (1000, 1000),
		group = :model,
		legend = true,
		colors = Makie.colorschemes[:seaborn_pastel6]
	)	
	f
end

# ╔═╡ 76116837-6fa8-4149-b0d4-8e663b53f4ba
function simulate_participant(;
	condition::String,
    model::Function,
    parameters::Vector{Symbol},
    transformed::Dict{Symbol, Symbol},
	priors::Dict,
	aao::Float64 = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])]),
	initV::Matrix{Float64} = fill(aao, 1,2),
	repeats::Int64 = 1
)	
	# Load sequence from file
	task = task_vars_for_condition(condition)
	
	prior_sample = simulate_from_prior(
		repeats;
		model = model,
		block = task.block,
		valence = task.valence,
		outcomes = task.outcomes,
		initV = fill(aao, 1, 2),
		set_size = fill(2, maximum(task.block)),
		parameters = parameters,
		transformed = transformed,
		priors = priors,
		random_seed = 0
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

# ╔═╡ 90035555-50c3-489a-a3ff-140752e1dea6
function plot_turing_ll(
	f::GridPosition;
	data::DataFrame,
	priors::Dict,
	ρ_val::Float64 = 4.,
	a_val::Float64 = 0.,
	grid_W::AbstractVector = range(0.001, 10., length = 200),
	grid_C::AbstractVector = range(1., 13., length = 200),
	aao::Float64 = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])]),
	initV::Matrix{Float64} = fill(aao, 1,2)
)
	# Set up model
	post_model = RLWM_pmst(;
		block = data.block,
		valence = unique(data[!, [:block, :valence]]).valence,
		choice = data.choice,
		outcomes = hcat(
			data.feedback_suboptimal,
			data.feedback_optimal,
		),
		initV = initV,
        set_size = fill(2, maximum(data.block)),
		parameters = [:ρ, :a, :W, :C],
		priors = priors
	)

	# Set up axis
	# ax = Axis(
	# 	f,
	# 	xlabel = "ρ",
	# 	ylabel = "a"
	# )
	
	# ρ = repeat(grid_ρ, inner = length(grid_a))
	# a = repeat(grid_a, outer = length(grid_ρ))
	W = repeat(grid_W, inner = length(grid_W))
	C = repeat(grid_C, outer = length(grid_C))
	
	ll = [loglikelihood(
			post_model, 
			(ρ = ρ_val, a = a_val, W = W, C = C)) for W in grid_W for C in grid_C]
	
	# contour!(
	# 	ax,
	# 	ρ,
	# 	a,
	# 	ll,
	# 	levels = 10
	# )
	

	# # Plot MLE
	# scatter!(
	# 	ax,
	# 	ρ[argmax(ll)],
	# 	a[argmax(ll)],
	# 	marker = :cross,
	# 	markersize = 8,
	# 	color = :blue
	# )

	# Set up axis
	ax = Axis(
		f,
		xlabel = "W",
		ylabel = "C"
	)

	# Plot loglikelihood
	
	contour!(
		ax,
		W,
		C, 
		ll,
		levels = 10
	)

	# Plot MLE
	scatter!(
		ax,
		W[argmax(ll)],
		C[argmax(ll)],
		marker = :cross,
		markersize = 8,
		color = :blue
	)

	return ax

end

# ╔═╡ 2a8dcad9-be29-4db7-9838-aceda47c65c7
begin
	f = Figure(size = (1000, 1000))
	prior_sample = simulate_participant(;
		condition = "00",
	    model = RLWM_pmst,
		parameters = [:ρ, :a, :W, :C],
		transformed = Dict(:a => :α, :W => :w0),
		priors = Dict(
			:ρ => Dirac(4.),
			:a => Dirac(0.),
			:W => Dirac(0.),
			:C => Dirac(5.)
		)
	)
	
	f = Figure(size = (700, 280))

	ax1 = plot_turing_ll(
		f[1,1];
		data = prior_sample,
		priors = Dict(
			:ρ => truncated(Normal(0., 1.), lower = 0.),
			:a => Normal(),
			:W => Normal(),
			:C => truncated(Normal(3., 2.), lower = 1.)
		)
	)

	ax1.title = "Turing\n with prior 1"

	ax2 = plot_turing_ll(
		f[1,2];
		data = prior_sample,
		priors = Dict(
			:ρ => truncated(Normal(99., 2.), lower = 0.),
			:a => Normal(-99., 3.),
			:W => Normal(-99., 3.),
			:C => truncated(Normal(10., 5.), lower = 1., upper = 13.)
		)
	)

	ax2.title = "Turing\n with prior 2"

	f
end

# ╔═╡ 0191bfa9-c613-440a-a8c0-01f210d52763
function plot_turing_ll_wm(
	f::GridPosition;
	data::DataFrame,
	priors::Dict,
	ρ_val::Float64 = 4.,
	a_val::Float64 = 0.,
	f_val::Float64 = 0.,
	grid_W::AbstractVector = range(0.001, 10., length = 200),
	grid_C::AbstractVector = range(1., 13., length = 200),
	aao::Float64 = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])]),
	initV::Matrix{Float64} = fill(aao, 1,2)
)
	# Set up model
	post_model = RLWM_pmst(;
		block = data.block,
		valence = unique(data[!, [:block, :valence]]).valence,
		choice = data.choice,
		outcomes = hcat(
			data.feedback_suboptimal,
			data.feedback_optimal,
		),
		initV = initV,
        set_size = fill(2, maximum(data.block)),
		parameters = [:ρ, :a, :W, :C],
		priors = priors
	)

	# Set up axis
	# ax = Axis(
	# 	f,
	# 	xlabel = "ρ",
	# 	ylabel = "a"
	# )
	
	# ρ = repeat(grid_ρ, inner = length(grid_a))
	# a = repeat(grid_a, outer = length(grid_ρ))
	W = repeat(grid_W, inner = length(grid_W))
	C = repeat(grid_C, outer = length(grid_C))
	
	ll = [loglikelihood(
			post_model, 
			(ρ = ρ_val, a = a_val, F_wm = f_val, W = W, C = C)) for W in grid_W for C in grid_C]
	
	# contour!(
	# 	ax,
	# 	ρ,
	# 	a,
	# 	ll,
	# 	levels = 10
	# )
	

	# # Plot MLE
	# scatter!(
	# 	ax,
	# 	ρ[argmax(ll)],
	# 	a[argmax(ll)],
	# 	marker = :cross,
	# 	markersize = 8,
	# 	color = :blue
	# )

	# Set up axis
	ax = Axis(
		f,
		xlabel = "W",
		ylabel = "C"
	)

	# Plot loglikelihood
	
	contour!(
		ax,
		W,
		C, 
		ll,
		levels = 10
	)

	# Plot MLE
	scatter!(
		ax,
		W[argmax(ll)],
		C[argmax(ll)],
		marker = :cross,
		markersize = 8,
		color = :blue
	)

	return ax

end

# ╔═╡ d6dd0e64-063b-449c-8535-c78e27bae9ba
let
	prior_sample = simulate_participant(;
		condition = "00",
	    model = RLWM_ss,
		parameters = [:ρ, :a, :F_wm, :W, :C],
		transformed = Dict(:a => :α, :F_wm => :φ_wm, :W => :w0),
		priors = Dict(
			:ρ => Dirac(4.),
			:a => Dirac(0.),
			:F_wm => Dirac(0.),
			:W => Dirac(0.),
			:C => Dirac(5.)
		)
	)
	
	f = Figure(size = (700, 280))

	ax1 = plot_turing_ll_wm(
		f[1,1];
		data = prior_sample,
		priors = Dict(
			:ρ => truncated(Normal(0., 1.), lower = 0.),
			:a => Normal(0., 0.5),
			:F_wm => Normal(0., 0.5),
			:W => Normal(0., 0.5),
			:C => truncated(Normal(3., 2.), lower = 1.)
		)
	)

	ax1.title = "Turing\n with prior 1"

	ax2 = plot_turing_ll_wm(
		f[1,2];
		data = prior_sample,
		priors = Dict(
			:ρ => truncated(Normal(99., 2.), lower = 0.),
			:a => Normal(-99., 3.),
			:F_wm => Normal(-99., 3.),
			:W => Normal(-99., 3.),
			:C => truncated(Normal(10., 5.), lower = 1., upper = 13.)
		)
	)

	ax2.title = "Turing\n with prior 2"

	f
end

# ╔═╡ 18b17290-0ab6-43d4-b8ec-792cb694ce64
# ╠═╡ disabled = true
#=╠═╡
begin
    data = simulate_participant(;
		condition = "00",
		model = RLWM_pmst,
		parameters = [:ρ, :a, :W, :C],
		transformed = Dict(:a => :α, :W => :w0),
		priors = Dict(
			:ρ => Dirac(4.),
			:a => Dirac(0.),
			:W => Dirac(0.),
			:C => Dirac(5.)
		)
	)

	aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])

	post_model = RLWM_pmst(;
		block = data.block,
		valence = unique(data[!, [:block, :valence]]).valence,
		choice = data.choice,
		outcomes = hcat(
			data.feedback_suboptimal,
			data.feedback_optimal,
		),
		initV =  fill(aao, 1,2),
		set_size = fill(2, maximum(data.block)),
		parameters = [:ρ, :a, :W, :C],
		priors = Dict(
			:ρ => truncated(Normal(0., 1.), lower = 0.),
			:a => Normal(),
			:W => Normal(),
			:C => truncated(Normal(3., 2.), lower = 1.)
		)
	)

	grid_ρ = range(0.001, 10., length = 200)
	grid_a = range(-4, 4., length = 200)
	grid_W = range(0.001, 10., length = 200)
	grid_C = range(1., 13., length = 200)

	[loglikelihood(
			post_model, 
			(ρ = ρ, a = a, W = W, C = C)) for ρ in grid_ρ for a in grid_a for W in grid_W for C in grid_C]
end
  ╠═╡ =#

# ╔═╡ 24e81e3a-6d47-4e10-b480-1e4afb449612
md"
### Sample from posterior and plot for single participant
"

# ╔═╡ 10516a3d-4deb-46eb-b977-9726dc13d093
# ╠═╡ disabled = true
#=╠═╡
# Sample from posterior and plot for single participant
begin
	rl_fit = let
		# Initial value for Q values
		aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])
		task = task_vars_for_condition("00")
		prior_bsl = filter(x -> x.model == "α|ρ", prior_samples)
		
		rl_fit = posterior_sample_single_p(
			filter(x -> x.PID == 1, prior_bsl);
			model = RL_ss,
			initV = aao,
			random_seed = 0
		)
	
	end

	rlwm_fit = let
		# Initial value for Q values
		aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])
		task = task_vars_for_condition("00")
		prior_rlwm_bsl = filter(x -> x.model == "α|ρ|φ|C|w0", prior_rlwm_samples)
		
		rlwm_fit = posterior_sample_single_p(
			filter(x -> x.PID == 1, prior_rlwm_bsl);
			model = RLWM_ss,
			initV = aao,
			set_size = fill(2, maximum(task.block)),
			random_seed = 0,
			parameters = [:ρ, :a, :F_wm, :W, :C],
			sigmas = Dict(:ρ => 2., :a => 0.5, :F_wm => 0.5, :W => 0.5, :C => 2.)
		)
	
	end
	rlwm_full_fit = let
		# Initial value for Q values
		aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])
		task = task_vars_for_condition("00")
		prior_rlwm_full = filter(x -> x.model == "α|ρ|φ_rl|C|w0|ε|φ_wm", prior_rlwm_samples)
		
		rlwm_full_fit = posterior_sample_single_p(
			filter(x -> x.PID == 1, prior_rlwm_full);
			model = RLWM_ss,
			initV = aao,
			set_size = fill(2, maximum(task.block)),
			random_seed = 0,
			parameters = [:ρ, :a, :E, :F_rl, :F_wm, :W, :C],
			sigmas = Dict(:ρ => 2., :a => 0.5, :E => 0.5, :F_rl => 0.5, :F_wm => 0.5, :W => 0.5, :C => 2.)
		)
	end
end
  ╠═╡ =#

# ╔═╡ b3ec6080-2bd0-42c5-b265-d691efb8cbb2
# ╠═╡ disabled = true
#=╠═╡
begin
	prior_bsl = filter(x -> x.model == "α|ρ", prior_samples)
	rl_one_posterior = plot_posteriors([rl_fit],
		["a", "ρ"];
		true_values = [α2a(prior_bsl[1, :α]), prior_bsl[1, :ρ]]
	)
	
	ax_cor = Axis(
		rl_one_posterior[1,3],
		xlabel = "a",
		ylabel = "ρ",
		aspect = 1,
		xticks = WilkinsonTicks(4)
	)
	
	scatter!(
		ax_cor,
		rl_fit[:, :a, :] |> vec,
		rl_fit[:, :ρ, :] |> vec,
		markersize = 1.5
	)
	
	colsize!(rl_one_posterior.layout, 3, Relative(0.2))
	
	rl_one_posterior
end
  ╠═╡ =#

# ╔═╡ 50420f58-563c-4122-beff-5ca052abdc26
# ╠═╡ disabled = true
#=╠═╡
begin
	prior_rlwm_bsl = filter(x -> x.model == "α|ρ|φ|C|w0", prior_rlwm_samples)
	rlwm_one_posterior = plot_posteriors([rlwm_fit],
		["a", "ρ", "F_wm", "C", "W"];
		true_values = [α2a(prior_rlwm_bsl[1, :α]), prior_rlwm_bsl[1, :ρ], α2a(prior_rlwm_bsl[1, :φ_wm]), prior_rlwm_bsl[1, :C], α2a(prior_rlwm_bsl[1, :w0])]
	)
	rlwm_one_posterior
end
  ╠═╡ =#

# ╔═╡ dbdcb4d8-b02e-49ef-a180-adf97d713f1e
# ╠═╡ disabled = true
#=╠═╡
begin
	prior_rlwm_full = filter(x -> x.model == "α|ρ|φ_rl|C|w0|ε|φ_wm", prior_rlwm_samples)
	rlwm_second_posterior = plot_posteriors([rlwm_full_fit],
		["a", "ρ", "F_rl", "E", "F_wm", "C", "W"];
		true_values = [α2a(prior_rlwm_full[1, :α]), prior_rlwm_full[1, :ρ], α2a(prior_rlwm_full[1, :φ_rl]), α2a(prior_rlwm_full[1, :ε]), α2a(prior_rlwm_full[1, :φ_wm]), prior_rlwm_full[1, :C], α2a(prior_rlwm_full[1, :w0])]
	)
	rlwm_second_posterior
end
  ╠═╡ =#

# ╔═╡ 7609f3bf-b0d1-4353-a074-c96f4a474c82
md"
### Optimise and get MAP estimates
"

# ╔═╡ 89358642-2a4b-4b42-b03d-a8c8f0e817e8
# ╠═╡ disabled = true
#=╠═╡
let
	f_rl = optimization_calibration(
		filter(x -> x.model == "α|ρ", prior_samples),
		optimize_multiple_single_p_QL,
		estimate = "MAP"
	)	

	f_rl
end
  ╠═╡ =#

# ╔═╡ 9c54629b-4aeb-4019-a369-00558270fc96
# ╠═╡ disabled = true
#=╠═╡
let
	f_rfl = optimization_calibration(
		filter(x -> x.model == "α|ρ|ε|φ", prior_samples),
		optimize_multiple_single_p_QL,
		estimate = "MAP",
		parameters = [:ρ, :a, :E, :F],
		initial_params = [mean(truncated(Normal(0., 2.), lower = 0.)), 0.5, 0.5, 0.5, 0.5],
		transformed = Dict(:a => :α, :E => :ε, :F => :φ),
		priors = Dict(
			:ρ => truncated(Normal(0., 1.), lower = 0.),
			:a => Normal(0., 0.5),
			:E => Normal(0., 0.5),
			:F => Normal(0., 0.5)
		)
	)

	f_rfl
end
  ╠═╡ =#

# ╔═╡ 1dfd03d0-a789-4dcc-885e-8546d73a0f0d
# ╠═╡ disabled = true
#=╠═╡
let
	task = task_vars_for_condition("00")
	set_sizes = fill(2, maximum(task.block))
	
	f_rlwm = optimization_calibration(
		filter(x -> x.model == "α|ρ|φ|C|w0", prior_rlwm_samples),
		optimize_multiple_single_p_QL,
		estimate = "MAP",
		model = RLWM_ss,
		set_size = set_sizes,
		initial_params = [mean(truncated(Normal(0., 2.), lower = 0.)), 0.5, 0.5, 0.5, mean(truncated(Normal(2., 2.), lower = 1.))],
		parameters = [:ρ, :a, :F_wm, :W, :C],
		transformed = Dict(:a => :α, :F_wm => :φ_wm, :W => :w0),
		priors = Dict(
			:ρ => truncated(Normal(0., 1.), lower = 0.),
			:a => Normal(0., 0.5),
			:F_wm => Normal(0., 0.5),
			:W => Normal(0., 0.5),
			:C => truncated(Normal(3., 2.), lower = 1.)
		)
	)

	f_rlwm
end
  ╠═╡ =#

# ╔═╡ e0e8b4b4-cc86-4816-b3f6-389df99b437e
# ╠═╡ disabled = true
#=╠═╡
let
	task = task_vars_for_condition("00")
	set_sizes = fill(2, maximum(task.block))
	
	f_pmst = optimization_calibration(
		filter(x -> x.model == "α|ρ|C|w0", prior_wpmst_samples),
		optimize_multiple_single_p_QL,
		estimate = "MAP",
		model = RLWM_pmst,
		set_size = set_sizes,
		initial_params = [mean(truncated(Normal(0., 2.), lower = 0.)), 0.5, 0.5, mean(truncated(Normal(4., 2.), lower = 1.))],
		parameters = [:ρ, :a, :W, :C],
		transformed = Dict(:a => :α, :W => :w0),
		priors = Dict(
			:ρ => truncated(Normal(0., 1.), lower = 0.),
			:a => Normal(0., 0.5),
			:W => Normal(0., 0.5),
			:C => truncated(Normal(3., 2.), lower = 1.)
		)
	)

	f_pmst
end
  ╠═╡ =#

# ╔═╡ 0aaab866-3870-46f3-bda9-e8f2ef37a62a
md"
## Fixing parameters to show effect of RLWM?
"

# ╔═╡ ac08624a-e96f-4d42-8434-0fd6e0cf93c4
# ╠═╡ disabled = true
#=╠═╡
# Sample datasets from prior
begin
	prior_samples_fixed = let
		# Load sequence from file
		task = task_vars_for_condition("00", true)
		n_ppt = 100
	
		# Initial value for Q values
		aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])
	
		prior_rl = simulate_from_prior(
			n_ppt;
			model = RL_ss,
			block = task.block,
			valence = task.valence,
			outcomes = task.outcomes,
			initV = fill(aao, 1, 2),
			parameters = [:ρ, :a],
			transformed = Dict(:a => :α),
			sigmas = Dict(:ρ => 2., :a => 0.5),
			fixed_params = Dict(:ρ => 1.5, :α => 0.3)
		)
		insertcols!(prior_rl, :model => "α|ρ")

		prior_rlwm = simulate_from_prior(
			n_ppt;
			model = RLWM_pmst,
			block = task.block,
			valence = task.valence,
			outcomes = task.outcomes,
			initV = fill(aao, 1, 2),
			set_size = fill(2, maximum(task.block)),
			parameters = [:ρ, :a, :W, :C],
			transformed = Dict(:a => :α, :W => :w0),
			sigmas = Dict(:ρ => 2., :a => 0.5, :W => 0.5, :C => 4.),
			fixed_params = Dict(:ρ => 1.5, :α => 0.3)
		)
		insertcols!(prior_rlwm, :model => "α|ρ|C|w0")

		prior_samples = vcat(prior_rl, prior_rlwm; cols=:union)
	
		leftjoin(prior_samples, 
			task.task[!, [:block, :trial, :feedback_optimal, :feedback_suboptimal, :early_confusion, :confusion_time]],
			on = [:block, :trial]
		)
	
	end

	describe(prior_samples_fixed)
end
  ╠═╡ =#

# ╔═╡ d38d6106-b878-465a-8d15-7704c0702c51
#=╠═╡
let
	f = Figure(size = (1400, 1400))
    df = prior_samples_fixed

	plot_prior_expectations!(
		GridLayout(f[1,1]),
		df;
		colA = :Q_optimal,
		colB = :Q_suboptimal,
		norm = :ρ,
		group = :model,
		plw = 1,
		legend = false,
		colors = Makie.colorschemes[:seaborn_pastel6],
	)
	
	plot_prior_expectations!(
		GridLayout(f[1,2]),
		df;
		colA = :W_optimal,
		colB = :W_suboptimal,
		ylab = "W-value",
		norm = :ρ,
		#ylims = (-1., 5.),
		group = :model,
		plw = 1,
		legend = false,
		colors = Makie.colorschemes[:seaborn_pastel6],
	)

	#df[!, :optimal_choice] .= df.choice .== 1

	plot_prior_accuracy!(
		GridLayout(f[1,3]),
		df;
		group = :model,
		pid_col = :PID,
		acc_col = :choice,
		legend = true,
		legend_rows = 2,
		colors = Makie.colorschemes[:seaborn_pastel6],
		error_band = "PI"
	)

	Label(f[0,:], "All blocks", fontsize = 30, font = :bold)
	
	plot_prior_expectations!(
		GridLayout(f[3,1]),
		filter(x -> x.valence > 0, df);
		colA = :Q_optimal,
		colB = :Q_suboptimal,
		norm = :ρ,
		group = :model,
		plw = 1,
		legend = false,
		colors = Makie.colorschemes[:seaborn_deep6]
	)

	plot_prior_expectations!(
		GridLayout(f[3,2]),
		filter(x -> x.valence > 0, df);
		colA = :W_optimal,
		colB = :W_suboptimal,
		ylab = "W-value",
		norm = :ρ,
		#ylims = (-1., 5.),
		group = :model,
		plw = 1,
		legend = false,
		colors = Makie.colorschemes[:seaborn_deep6],
	)

	plot_prior_accuracy!(
		GridLayout(f[3,3]),
		filter(x -> x.valence > 0, df);
		group = :model,
		pid_col = :PID,
		acc_col = :choice,
		legend = true,
		legend_rows = 2,
		colors = Makie.colorschemes[:seaborn_deep6],
		error_band = "PI"
	)

	Label(f[2,:], "Reward blocks", fontsize = 30, font = :bold)

	plot_prior_expectations!(
		GridLayout(f[5,1]),
		filter(x -> x.valence < 0, df);
		colA = :Q_optimal,
		colB = :Q_suboptimal,
		norm = :ρ,
		group = :model,
		plw = 1,
		legend = false,
		colors = Makie.colorschemes[:seaborn_colorblind6]
	)

	plot_prior_expectations!(
		GridLayout(f[5,2]),
		filter(x -> x.valence < 0, df);
		colA = :W_optimal,
		colB = :W_suboptimal,
		ylab = "W-value",
		norm = :ρ,
		#ylims = (-1., 5.),
		group = :model,
		plw = 1,
		legend = false,
		colors = Makie.colorschemes[:seaborn_colorblind6],
	)

	plot_prior_accuracy!(
		GridLayout(f[5,3]),
		filter(x -> x.valence < 0, df);
		group = :model,
		pid_col = :PID,
		acc_col = :choice,
		legend = true,
		legend_rows = 2,
		colors = Makie.colorschemes[:seaborn_colorblind6],
		error_band = "PI"
	)

	Label(f[4,:], "Punishment blocks", fontsize = 30, font = :bold)

	#save("results/single_p_QL_prior_predictive.png", f, pt_per_unit = 1)

	f
end
  ╠═╡ =#

# ╔═╡ e3556019-ed4f-4398-9178-a3dd372db650
md"
### Early confusing feedback (trial 5 or earlier)
"

# ╔═╡ 69bc167a-ee86-465c-b86e-7754d2cf8892
#=╠═╡
let
	f = Figure(size = (1400, 1400))
	df_nm = filter(x -> !ismissing(x.early_confusion), prior_samples_fixed)
    df_early = filter(x -> x.early_confusion == true, df_nm)

	plot_prior_expectations!(
		GridLayout(f[1,1]),
		df_early;
		colA = :Q_optimal,
		colB = :Q_suboptimal,
		norm = :ρ,
		group = :model,
		plw = 1,
		legend = false,
		colors = Makie.colorschemes[:seaborn_pastel6],
	)
	
	plot_prior_expectations!(
		GridLayout(f[1,2]),
		df_early;
		colA = :W_optimal,
		colB = :W_suboptimal,
		ylab = "W-value",
		norm = :ρ,
		#ylims = (-1., 5.),
		group = :model,
		plw = 1,
		legend = false,
		colors = Makie.colorschemes[:seaborn_pastel6],
	)

	#df[!, :optimal_choice] .= df.choice .== 1

	plot_prior_accuracy!(
		GridLayout(f[1,3]),
		df_early;
		group = :model,
		pid_col = :PID,
		acc_col = :choice,
		legend = true,
		legend_rows = 2,
		colors = Makie.colorschemes[:seaborn_pastel6],
		error_band = "PI"
	)

	Label(f[0,:], "All blocks", fontsize = 30, font = :bold)
	
	plot_prior_expectations!(
		GridLayout(f[3,1]),
		filter(x -> x.valence > 0, df_early);
		colA = :Q_optimal,
		colB = :Q_suboptimal,
		norm = :ρ,
		group = :model,
		plw = 1,
		legend = false,
		colors = Makie.colorschemes[:seaborn_deep6]
	)

	plot_prior_expectations!(
		GridLayout(f[3,2]),
		filter(x -> x.valence > 0, df_early);
		colA = :W_optimal,
		colB = :W_suboptimal,
		ylab = "W-value",
		norm = :ρ,
		#ylims = (-1., 5.),
		group = :model,
		plw = 1,
		legend = false,
		colors = Makie.colorschemes[:seaborn_deep6],
	)

	plot_prior_accuracy!(
		GridLayout(f[3,3]),
		filter(x -> x.valence > 0, df_early);
		group = :model,
		pid_col = :PID,
		acc_col = :choice,
		legend = true,
		legend_rows = 2,
		colors = Makie.colorschemes[:seaborn_deep6],
		error_band = "PI"
	)

	Label(f[2,:], "Reward blocks", fontsize = 30, font = :bold)

	plot_prior_expectations!(
		GridLayout(f[5,1]),
		filter(x -> x.valence < 0, df_early);
		colA = :Q_optimal,
		colB = :Q_suboptimal,
		norm = :ρ,
		group = :model,
		plw = 1,
		legend = false,
		colors = Makie.colorschemes[:seaborn_colorblind6]
	)

	plot_prior_expectations!(
		GridLayout(f[5,2]),
		filter(x -> x.valence < 0, df_early);
		colA = :W_optimal,
		colB = :W_suboptimal,
		ylab = "W-value",
		norm = :ρ,
		#ylims = (-1., 5.),
		group = :model,
		plw = 1,
		legend = false,
		colors = Makie.colorschemes[:seaborn_colorblind6],
	)

	plot_prior_accuracy!(
		GridLayout(f[5,3]),
		filter(x -> x.valence < 0, df_early);
		group = :model,
		pid_col = :PID,
		acc_col = :choice,
		legend = true,
		legend_rows = 2,
		colors = Makie.colorschemes[:seaborn_colorblind6],
		error_band = "PI"
	)

	Label(f[4,:], "Punishment blocks", fontsize = 30, font = :bold)

	#save("results/single_p_QL_prior_predictive.png", f, pt_per_unit = 1)

	f
end
  ╠═╡ =#

# ╔═╡ 5d03c678-2e9b-4844-85f6-6be0a3927fc7
md"
### Later confusing feedback (trial 6 or later)
"

# ╔═╡ 207f4f66-26c0-4833-a911-498c5b344542
#=╠═╡
let
	f = Figure(size = (1400, 1400))
	df_nm = filter(x -> !ismissing(x.early_confusion), prior_samples_fixed)
    df_late = filter(x -> x.early_confusion == false, df_nm)

	plot_prior_expectations!(
		GridLayout(f[1,1]),
		df_late;
		colA = :Q_optimal,
		colB = :Q_suboptimal,
		norm = :ρ,
		group = :model,
		plw = 1,
		legend = false,
		colors = Makie.colorschemes[:seaborn_pastel6],
	)
	
	plot_prior_expectations!(
		GridLayout(f[1,2]),
		df_late;
		colA = :W_optimal,
		colB = :W_suboptimal,
		ylab = "W-value",
		norm = :ρ,
		#ylims = (-1., 5.),
		group = :model,
		plw = 1,
		legend = false,
		colors = Makie.colorschemes[:seaborn_pastel6],
	)

	#df[!, :optimal_choice] .= df.choice .== 1

	plot_prior_accuracy!(
		GridLayout(f[1,3]),
		df_late;
		group = :model,
		pid_col = :PID,
		acc_col = :choice,
		legend = true,
		legend_rows = 2,
		colors = Makie.colorschemes[:seaborn_pastel6],
		error_band = "PI"
	)

	Label(f[0,:], "All blocks", fontsize = 30, font = :bold)
	
	plot_prior_expectations!(
		GridLayout(f[3,1]),
		filter(x -> x.valence > 0, df_late);
		colA = :Q_optimal,
		colB = :Q_suboptimal,
		norm = :ρ,
		group = :model,
		plw = 1,
		legend = false,
		colors = Makie.colorschemes[:seaborn_deep6]
	)

	plot_prior_expectations!(
		GridLayout(f[3,2]),
		filter(x -> x.valence > 0, df_late);
		colA = :W_optimal,
		colB = :W_suboptimal,
		ylab = "W-value",
		norm = :ρ,
		#ylims = (-1., 5.),
		group = :model,
		plw = 1,
		legend = false,
		colors = Makie.colorschemes[:seaborn_deep6],
	)

	plot_prior_accuracy!(
		GridLayout(f[3,3]),
		filter(x -> x.valence > 0, df_late);
		group = :model,
		pid_col = :PID,
		acc_col = :choice,
		legend = true,
		legend_rows = 2,
		colors = Makie.colorschemes[:seaborn_deep6],
		error_band = "PI"
	)

	Label(f[2,:], "Reward blocks", fontsize = 30, font = :bold)

	plot_prior_expectations!(
		GridLayout(f[5,1]),
		filter(x -> x.valence < 0, df_late);
		colA = :Q_optimal,
		colB = :Q_suboptimal,
		norm = :ρ,
		group = :model,
		plw = 1,
		legend = false,
		colors = Makie.colorschemes[:seaborn_colorblind6]
	)

	plot_prior_expectations!(
		GridLayout(f[5,2]),
		filter(x -> x.valence < 0, df_late);
		colA = :W_optimal,
		colB = :W_suboptimal,
		ylab = "W-value",
		norm = :ρ,
		#ylims = (-1., 5.),
		group = :model,
		plw = 1,
		legend = false,
		colors = Makie.colorschemes[:seaborn_colorblind6],
	)

	plot_prior_accuracy!(
		GridLayout(f[5,3]),
		filter(x -> x.valence < 0, df_late);
		group = :model,
		pid_col = :PID,
		acc_col = :choice,
		legend = true,
		legend_rows = 2,
		colors = Makie.colorschemes[:seaborn_colorblind6],
		error_band = "PI"
	)

	Label(f[4,:], "Punishment blocks", fontsize = 30, font = :bold)

	#save("results/single_p_QL_prior_predictive.png", f, pt_per_unit = 1)

	f
end
  ╠═╡ =#

# ╔═╡ Cell order:
# ╟─d49d6f84-5ff6-4100-90db-995ffdeb09ca
# ╠═88f8505c-5fc2-11ef-2ac9-9bb7f4b5b591
# ╠═cf2e2616-d29a-49cb-b9ee-6b6ea89c4ee6
# ╟─533b6f26-adb1-438e-9cbc-5dcc2d19ec1a
# ╠═9bfdd2b3-7885-42b3-8839-c426dae394a4
# ╠═3ea2b653-a8f7-4e6b-910d-c5e4d284be53
# ╟─bb8fc03d-f85a-451c-89c4-2303f0cb7fef
# ╠═bfe4c9c1-72f0-4999-bbcb-76bdd4f579da
# ╠═4b628e68-0625-4cff-bfc4-f146f9673d3c
# ╠═91548dc3-0812-4070-9b32-5b5b3bb05eb9
# ╠═a9a8d13b-4960-4a53-aed5-c104260625b5
# ╠═76116837-6fa8-4149-b0d4-8e663b53f4ba
# ╠═90035555-50c3-489a-a3ff-140752e1dea6
# ╠═2a8dcad9-be29-4db7-9838-aceda47c65c7
# ╠═0191bfa9-c613-440a-a8c0-01f210d52763
# ╠═d6dd0e64-063b-449c-8535-c78e27bae9ba
# ╠═18b17290-0ab6-43d4-b8ec-792cb694ce64
# ╟─24e81e3a-6d47-4e10-b480-1e4afb449612
# ╠═10516a3d-4deb-46eb-b977-9726dc13d093
# ╠═b3ec6080-2bd0-42c5-b265-d691efb8cbb2
# ╠═50420f58-563c-4122-beff-5ca052abdc26
# ╠═dbdcb4d8-b02e-49ef-a180-adf97d713f1e
# ╟─7609f3bf-b0d1-4353-a074-c96f4a474c82
# ╠═89358642-2a4b-4b42-b03d-a8c8f0e817e8
# ╠═9c54629b-4aeb-4019-a369-00558270fc96
# ╠═1dfd03d0-a789-4dcc-885e-8546d73a0f0d
# ╠═e0e8b4b4-cc86-4816-b3f6-389df99b437e
# ╟─0aaab866-3870-46f3-bda9-e8f2ef37a62a
# ╠═ac08624a-e96f-4d42-8434-0fd6e0cf93c4
# ╠═d38d6106-b878-465a-8d15-7704c0702c51
# ╟─e3556019-ed4f-4398-9178-a3dd372db650
# ╠═69bc167a-ee86-465c-b86e-7754d2cf8892
# ╟─5d03c678-2e9b-4844-85f6-6be0a3927fc7
# ╠═207f4f66-26c0-4833-a911-498c5b344542
