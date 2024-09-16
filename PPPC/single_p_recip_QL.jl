### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ 0234eb34-5efa-11ef-14b3-fde3a2bb1104
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
	include("$(pwd())/fetch_preprocess_data.jl")
	nothing
end

# ╔═╡ 735f2c78-b6de-4d6b-b97c-1427edb5df21
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

# ╔═╡ 1fb9bfe6-0e35-43c3-aa89-1870f07112f4
# Initial value for Q values
aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])

# ╔═╡ 38c03cc9-deb4-4853-9f00-06d40c675555
# Sample datasets from prior
begin	
	prior_sample = let
		# Load sequence from file
		task = task_vars_for_condition("00")
	
		prior_sample = simulate_single_p_PILT(
			200;
			model = single_p_recip_QL,
			block = task.block,
			valence = task.valence,
			outcomes = task.outcomes,
			initV = fill(aao, 1, 2),
			random_seed = 0
		)
	
	
		leftjoin(prior_sample, 
			task.task[!, [:block, :trial, :feedback_optimal, :feedback_suboptimal]],
			on = [:block, :trial]
		)
	
	end

	describe(prior_sample)
end

# ╔═╡ 43f4356f-6ef3-47b6-a1cd-67d022447a7d
plot_prior_predictive_by_valence(
	prior_sample,
	[:Q_optimal, :Q_suboptimal]
)

# ╔═╡ Cell order:
# ╠═0234eb34-5efa-11ef-14b3-fde3a2bb1104
# ╠═735f2c78-b6de-4d6b-b97c-1427edb5df21
# ╠═1fb9bfe6-0e35-43c3-aa89-1870f07112f4
# ╠═38c03cc9-deb4-4853-9f00-06d40c675555
# ╠═43f4356f-6ef3-47b6-a1cd-67d022447a7d
