### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ 6f90ca0b-4c72-42ac-9e67-02c2a05331b6
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
	
	Turing.setprogress!(false)
	
	include("$(pwd())/working_memory/simulate.jl")
	include("$(pwd())/working_memory/collins_RLWM.jl")	
	include("$(pwd())/working_memory/plotting_utils.jl")
	include("$(pwd())/working_memory/model_utils.jl")

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

# ╔═╡ 49bf0b54-25b9-42b4-b25a-3e75488ee8cb
include("$(pwd())/working_memory/collins_RLWM.jl")

# ╔═╡ 5b92e714-a7eb-11ef-2626-91c405db34d1
md"
### WM + habit learning 
"

# ╔═╡ acc3d38b-fa39-4f5b-bb10-6112a0c60b45
# ╠═╡ disabled = true
#=╠═╡
begin
	## load working memory task sequence for Pilot 6
	pilot6_wm = load_wm_structure_csv("pilot6_WM")
	filter!(x -> x.block <= 10, pilot6_wm)
	nothing
end
  ╠═╡ =#

# ╔═╡ 72b105fc-564b-4a9c-ac4f-6cf20f5f0913
begin
	prior_sample_rlwm = simulate_from_prior(
	    100;
		model = RLWM_collins24,
		priors = Dict(
	        :β => 25., # fixed inverse temperature
	        :a => Normal(-2., 1.), # RL reward learning rate
	        :bRL => Normal(1., 1.), # LR difference between RL reward / punishment
	        :bWM => Normal(1., 1.), # punishment learning rate for working memory
	        :E => Normal(-2., 1.), # undirected noise
	        :F_wm => Normal(-2., 1.), # working memory forgetting rate
	        :w0 => Normal(2., 1.), # initial working memory weight
        	:C => Uniform(2, 5) # working memory capacity
		),
		parameters = [:a_pos, :bRL, :bWM, :E, :F_wm, :w0, :C],
		structure = (
            n_blocks = 10, n_trials = 10, n_confusing = 0,
		    set_sizes = [2, 4, 6], n_options = 3,
		    coins = [0.01, 1.0], punish = false
		),
		gq = true,
		random_seed = 1
	)
	nothing
end

# ╔═╡ a38b7482-922a-4678-8380-a931bb8143f3
let
	f = Figure(size=(1000, 400))
	pos = f[1, 1] = GridLayout()
	axqr, ax_acc = plot_sim_q_value_acc!(
		pos,
		prior_sample_rlwm;
		colA = [:Q_optimal, :W_optimal],
		colB = [[:Q_suboptimal1, :Q_suboptimal2], [:W_suboptimal1, :W_suboptimal2]],
		choice_val = 3.0,
		ylab = ("Q-value", "W-value"),
		group = :set_size,
		norm = nothing, # as we are using inverse temp
		legend = true,
		colors = Makie.colorschemes[:seaborn_pastel6]
	)
	linkyaxes!(axqr...)
	f
end

# ╔═╡ 90bb80e0-d473-4eb7-8171-a02c191a13b8
# ╠═╡ disabled = true
#=╠═╡
# if we include punishment blocks, this plots them separately from reward
let
	f = plot_prior_predictive_by_valence(
		prior_sample_rlwm,
		[:Q_optimal, :Q_suboptimal1, :Q_suboptimal2];
		W_cols = [:W_optimal, :W_suboptimal1, :W_suboptimal2],
		choice_val = 3.0,
		ylab = ("Q-value", "W-value"),
		fig_size = (1000, 1000),
		group = :set_size,
		norm = nothing,
		legend = true,
		colors = Makie.colorschemes[:seaborn_pastel6]
	)
	f
end
  ╠═╡ =#

# ╔═╡ 470de2e8-84ef-417b-bbec-35383a1feedc
begin
	f_rlwm = optimization_calibration(
		prior_sample_rlwm,
		optimize_multiple,
		estimate = "MAP",
		model = RLWM_collins24,
		priors = Dict(
	        :β => 25., # fixed inverse temperature
	        :a => Normal(-2., 3.), # RL reward learning rate
	        :bRL => Normal(1., 3.), # LR difference between RL reward / punishment
	        :bWM => Normal(1., 3.), # punishment learning rate for working memory
	        :E => Normal(-2., 3.), # undirected noise
	        :F_wm => Normal(-2., 3.), # working memory forgetting rate
	        :w0 => Normal(2., 3.), # initial working memory weight
        	:C => Uniform(2, 5) # working memory capacity
		),
		parameters = [:a_pos, :bRL, :bWM, :E, :F_wm, :w0, :C],
		n_starts = 5
	)

	f_rlwm
end

# ╔═╡ Cell order:
# ╟─5b92e714-a7eb-11ef-2626-91c405db34d1
# ╠═6f90ca0b-4c72-42ac-9e67-02c2a05331b6
# ╠═acc3d38b-fa39-4f5b-bb10-6112a0c60b45
# ╠═49bf0b54-25b9-42b4-b25a-3e75488ee8cb
# ╠═72b105fc-564b-4a9c-ac4f-6cf20f5f0913
# ╠═a38b7482-922a-4678-8380-a931bb8143f3
# ╠═90bb80e0-d473-4eb7-8171-a02c191a13b8
# ╠═470de2e8-84ef-417b-bbec-35383a1feedc
