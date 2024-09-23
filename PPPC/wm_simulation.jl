### A Pluto.jl notebook ###
# v0.19.45

using Markdown
using InteractiveUtils

# ╔═╡ 772362f4-75c0-11ef-25de-55ffd5418ed2
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
	include("$(pwd())/simulate.jl")
	include("$(pwd())/turing_models.jl")
	include("$(pwd())/plotting_utils.jl")
end

# ╔═╡ 2c3e4f77-3ad4-42cd-be28-9a078fd1b449
include("$(pwd())/turing_models.jl")

# ╔═╡ 4eb03947-90d6-4281-9d63-372f76bb4900
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

# ╔═╡ 24289c6f-4974-41ad-822b-6f3df77f7cb4
begin
	aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])
	prior_sample = simulate_from_prior(
		100;
		model = RL_ss,
		priors = Dict(
			:ρ => truncated(Normal(0., 1.), lower = 0.),
			:a => Normal(0., 0.5)
		),
		initial = aao,
		transformed = Dict(:a => :α),
		structure = (
            n_blocks = 48, n_trials = 13, n_confusing = 3, set_sizes = [2, 4, 6]
		),
		gq = true,
		random_seed = 123
	)
end

# ╔═╡ f99c4eb3-f018-4d3f-995c-88765ad6633d
let
	f = plot_prior_predictive_by_valence(
		prior_sample,
		[:Q_optimal, :Q_suboptimal];
		group = :set_size,
		legend = true,
		legend_title = "Set size",
		fig_size = (1000, 1000),
		colors = Makie.colorschemes[:seaborn_pastel6]
	)	
	f
end

# ╔═╡ f743bbd4-44aa-437e-910a-6c4830f08ef4
begin
	prior_sample_wm = simulate_from_prior(
		100;
		model = RLWM_ss,
		priors = Dict(
			:ρ => truncated(Normal(0., 1.), lower = 0.),
			:a => Normal(0., 0.5),
			:F_wm => Normal(0., 0.5),
			:W => Normal(0., 0.5),
			:C => truncated(Normal(3., 2.), lower = 1.)
		),
		initial = aao,
		transformed = Dict(:a => :α, :F_wm => :φ_wm, :W => :w0),
		structure = (
            n_blocks = 48, n_trials = 13, n_confusing = 3, set_sizes = [2, 4, 6]
		),
		gq = true,
		random_seed = 123
	)
end

# ╔═╡ e9a9cecb-c5c6-422d-b089-b1bb0c11e79f
let
	f = plot_prior_predictive_by_valence(
		prior_sample_wm,
		[:Q_optimal, :Q_suboptimal];
		W_cols = [:W_optimal, :W_suboptimal],
		group = :set_size,
		ylab = ("Q-value", "W-value"),
		legend = true,
		legend_title = "Set size",
		fig_size = (1000, 1000),
		colors = Makie.colorschemes[:seaborn_pastel6]
	)	
	f
end

# ╔═╡ f1918ce1-9435-4757-b970-68e795a49557
begin
	prior_sample_pmst = simulate_from_prior(
		100;
		model = RLWM_pmst,
		priors = Dict(
			:ρ => truncated(Normal(0., 1.), lower = 0.),
			:a => Normal(0., 0.5),
			:W => Normal(0., 0.5),
			:C => truncated(Normal(3., 2.), lower = 1.)
		),
		initial = aao,
		transformed = Dict(:a => :α, :W => :w0),
		structure = (
			n_blocks = 48, n_trials = 13, n_confusing = 3, set_sizes = [2, 4, 6]
		),
		gq = true,
		random_seed = 123
	)
end

# ╔═╡ acf1ba59-c202-4924-a58b-6ba3d6f7f338
let
	f = plot_prior_predictive_by_valence(
		prior_sample_pmst,
		[:Q_optimal, :Q_suboptimal];
		W_cols = [:W_optimal, :W_suboptimal],
		group = :set_size,
		ylab = ("Q-value", "W-value"),
		legend = true,
		legend_title = "Set size",
		fig_size = (1000, 1000),
		colors = Makie.colorschemes[:seaborn_pastel6]
	)	
	f
end

# ╔═╡ Cell order:
# ╠═772362f4-75c0-11ef-25de-55ffd5418ed2
# ╠═4eb03947-90d6-4281-9d63-372f76bb4900
# ╠═24289c6f-4974-41ad-822b-6f3df77f7cb4
# ╠═f99c4eb3-f018-4d3f-995c-88765ad6633d
# ╠═f743bbd4-44aa-437e-910a-6c4830f08ef4
# ╠═e9a9cecb-c5c6-422d-b089-b1bb0c11e79f
# ╠═2c3e4f77-3ad4-42cd-be28-9a078fd1b449
# ╠═f1918ce1-9435-4757-b970-68e795a49557
# ╠═acf1ba59-c202-4924-a58b-6ba3d6f7f338
