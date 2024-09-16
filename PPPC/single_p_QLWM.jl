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
	include("$(pwd())/plotting.jl")
end

# ╔═╡ c300ec0c-cdd2-437c-bfac-94eaa531fc4a
begin
	include("$(pwd())/simulate.jl")
	include("$(pwd())/turing_models.jl")
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

# ╔═╡ 7a3b57c9-51ca-4ec1-b81f-bb45ed456b9d
let
    f = Figure(size = (1000, 1000))
	
	plot_prior_expectations!(
		GridLayout(f[1,1]),
		prior_samples;
		colA = :Q_optimal,
		colB = :Q_suboptimal,
		group = :model,
		plw = 1,
		legend = false,
		colors = Makie.colorschemes[:seaborn_pastel6],
	)

	prior_samples[!, :optimal_choice] .= prior_samples.choice .== 1

	plot_prior_accuracy!(
		GridLayout(f[1,2]),
		prior_samples;
		group = :model,
		pid_col = :PID,
		acc_col = :choice,
		legend = true,
		colors = Makie.colorschemes[:seaborn_pastel6],
		error_band = "PI"
	)

	Label(f[0,:], "All blocks", fontsize = 18, font = :bold)

	plot_prior_expectations!(
		GridLayout(f[3,1]),
		filter(x -> x.valence > 0, prior_samples);
		colA = :Q_optimal,
		colB = :Q_suboptimal,
		group = :model,
		plw = 1,
		legend = false,
		colors = Makie.colorschemes[:seaborn_deep6]
	)

	plot_prior_accuracy!(
		GridLayout(f[3,2]),
		filter(x -> x.valence > 0, prior_samples);
		group = :model,
		pid_col = :PID,
		acc_col = :optimal_choice,
		legend = true,
		colors = Makie.colorschemes[:seaborn_deep6],
		error_band = "PI"
	)

	Label(f[2,:], "Reward blocks", fontsize = 18, font = :bold)

	plot_prior_expectations!(
		GridLayout(f[5,1]),
		filter(x -> x.valence < 0, prior_samples);
		colA = :Q_optimal,
		colB = :Q_suboptimal,
		group = :model,
		plw = 1,
		legend = false,
		colors = Makie.colorschemes[:seaborn_colorblind6]
	)

	plot_prior_accuracy!(
		GridLayout(f[5,2]),
		filter(x -> x.valence < 0, prior_samples);
		group = :model,
		pid_col = :PID,
		acc_col = :optimal_choice,
		legend = true,
		colors = Makie.colorschemes[:seaborn_colorblind6],
		error_band = "PI"
	)

	Label(f[4,:], "Punishment blocks", fontsize = 18, font = :bold)

	# save("results/single_p_QL_prior_predictive.png", f, pt_per_unit = 1)

	f
end

# ╔═╡ bb8fc03d-f85a-451c-89c4-2303f0cb7fef
md"
## Reinforcement learning + working memory hybrid models
"

# ╔═╡ bfe4c9c1-72f0-4999-bbcb-76bdd4f579da
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

# ╔═╡ 7d6b2cfc-500f-469b-a294-ae25b0ca4571
let
	f = Figure(size = (1400, 1400))

    df = prior_rlwm_samples

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
		ylims = (-1., 5.),
		group = :model,
		plw = 1,
		legend = false,
		colors = Makie.colorschemes[:seaborn_pastel6],
	)

	df[!, :optimal_choice] .= df.choice .== 1

	plot_prior_accuracy!(
		GridLayout(f[1,3]),
		df;
		group = :model,
		pid_col = :PID,
		acc_col = :optimal_choice,
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
		ylims = (-1., 5.),
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
		acc_col = :optimal_choice,
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
		ylims = (-1., 5.),
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
		acc_col = :optimal_choice,
		legend = true,
		legend_rows = 2,
		colors = Makie.colorschemes[:seaborn_colorblind6],
		error_band = "PI"
	)

	Label(f[4,:], "Punishment blocks", fontsize = 30, font = :bold)

	# save("results/single_p_QL_prior_predictive.png", f, pt_per_unit = 1)

	f
end

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

# ╔═╡ 43ae709e-588c-4371-bfbd-65f333972867
let
	f = Figure(size = (1400, 1400))

    df = prior_wpmst_samples

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
		#ylims = (-.5, .5),
		group = :model,
		plw = 1,
		legend = false,
		colors = Makie.colorschemes[:seaborn_pastel6],
	)

	df[!, :optimal_choice] .= df.choice .== 1

	plot_prior_accuracy!(
		GridLayout(f[1,3]),
		df;
		group = :model,
		pid_col = :PID,
		acc_col = :optimal_choice,
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
		#ylims = (0., 1.),
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
		acc_col = :optimal_choice,
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
		#ylims = (-1., 0.),
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
		acc_col = :optimal_choice,
		legend = true,
		legend_rows = 2,
		colors = Makie.colorschemes[:seaborn_colorblind6],
		error_band = "PI"
	)

	Label(f[4,:], "Punishment blocks", fontsize = 30, font = :bold)

	# save("results/single_p_QL_prior_predictive.png", f, pt_per_unit = 1)

	f
end

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

# ╔═╡ 1dfd03d0-a789-4dcc-885e-8546d73a0f0d
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

# ╔═╡ e0e8b4b4-cc86-4816-b3f6-389df99b437e
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
# ╠═c300ec0c-cdd2-437c-bfac-94eaa531fc4a
# ╠═9bfdd2b3-7885-42b3-8839-c426dae394a4
# ╠═7a3b57c9-51ca-4ec1-b81f-bb45ed456b9d
# ╟─bb8fc03d-f85a-451c-89c4-2303f0cb7fef
# ╠═bfe4c9c1-72f0-4999-bbcb-76bdd4f579da
# ╠═7d6b2cfc-500f-469b-a294-ae25b0ca4571
# ╠═91548dc3-0812-4070-9b32-5b5b3bb05eb9
# ╠═43ae709e-588c-4371-bfbd-65f333972867
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
