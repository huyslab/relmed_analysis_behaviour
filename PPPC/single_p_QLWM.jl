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
	include("$(pwd())/fetch_preprocess_data.jl")
	include("$(pwd())/simulate.jl")
	include("$(pwd())/turing_models.jl")
	include("$(pwd())/plotting.jl")
end

# ╔═╡ 3e5083bf-89e9-405d-8496-a53e66e115ef
include("$(pwd())/turing_models.jl")

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
	
		# Initial value for Q values
		aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])
	
		prior_bsl = simulate_single_p(
			200;
			block = task.block,
			valence = task.valence,
			outcomes = task.outcomes,
			initV = fill(aao, 1, 2),
			random_seed = 0,
			parameters = [:ρ, :a],
			transformed = Dict(:a => :α),
			sigmas = Dict(:ρ => 2., :a => 0.5)
		)
		insertcols!(prior_bsl, :model => "α|ρ")

		prior_lapse = simulate_single_p(
			200;
			block = task.block,
			valence = task.valence,
			outcomes = task.outcomes,
			initV = fill(aao, 1, 2),
			random_seed = 0,
			parameters = [:ρ, :a, :E],
			transformed = Dict(:a => :α, :E => :ε),
			sigmas = Dict(:ρ => 2., :a => 0.5, :E => 0.5)
		)
		insertcols!(prior_lapse, :model => "α|ρ|ε")

		prior_forget = simulate_single_p(
			200;
			block = task.block,
			valence = task.valence,
			outcomes = task.outcomes,
			initV = fill(aao, 1, 2),
			random_seed = 0,
			parameters = [:ρ, :a, :F],
			transformed = Dict(:a => :α, :F => :φ),
			sigmas = Dict(:ρ => 2., :a => 0.5, :F => 0.5)
		)
		insertcols!(prior_forget, :model => "α|ρ|φ")

		prior_full = simulate_single_p(
			200;
			block = task.block,
			valence = task.valence,
			outcomes = task.outcomes,
			initV = fill(aao, 1, 2),
			random_seed = 0,
			parameters = [:ρ, :a, :E, :F],
			transformed = Dict(:a => :α, :E => :ε, :F => :φ),
			sigmas = Dict(:ρ => 2., :a => 0.5, :E => 0.5, :F => 0.5)
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
		acc_col = :optimal_choice,
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
	
		# Initial value for Q values
		aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])
	
		prior_rlwm_sens = simulate_single_p(
			200;
			wm = true,
			block = task.block,
			valence = task.valence,
			outcomes = task.outcomes,
			initV = fill(aao, 1, 2),
			set_size = fill(2, maximum(task.block)),
			random_seed = 0,
			parameters = [:ρ, :a, :F_wm, :W, :C],
			transformed = Dict(:a => :α, :F_wm => :φ_wm, :W => :w0),
			sigmas = Dict(:ρ => 2., :a => 0.5, :F_wm => 0.5, :W => 0.5, :C => 2.)
		)
		insertcols!(prior_rlwm_sens, :model => "α|ρ|φ|C|w0")

		prior_rlwm_temp = simulate_single_p(
			200;
			wm = true,
			block = task.block,
			valence = task.valence,
			outcomes = task.outcomes,
			initV = fill(aao, 1, 2),
			set_size = fill(2, maximum(task.block)),
			random_seed = 0,
			parameters = [:β, :a, :F_wm, :W, :C],
			transformed = Dict(:a => :α, :F_wm => :φ_wm, :W => :w0),
			sigmas = Dict(:β => 2., :a => 0.5, :F_wm => 0.5, :W => 0.5, :C => 2.)
		)
		insertcols!(prior_rlwm_temp, :model => "α|β|φ|C|w0")

		prior_rlwm_lapse = simulate_single_p(
			200;
			wm = true,
			block = task.block,
			valence = task.valence,
			outcomes = task.outcomes,
			initV = fill(aao, 1, 2),
			set_size = fill(2, maximum(task.block)),
			random_seed = 0,
			parameters = [:ρ, :a, :E, :F_wm, :W, :C],
			transformed = Dict(:a => :α, :E => :ε, :F_wm => :φ_wm, :W => :w0),
			sigmas = Dict(:ρ => 2., :a => 0.5, :E => 0.5, :F_wm => 0.5, :W => 0.5, :C => 2.)
		)
		insertcols!(prior_rlwm_lapse, :model => "α|ρ|φ|C|w0|ε")

		prior_rlwm_all = simulate_single_p(
			200;
			wm = true,
			block = task.block,
			valence = task.valence,
			outcomes = task.outcomes,
			initV = fill(aao, 1, 2),
			set_size = fill(2, maximum(task.block)),
			random_seed = 0,
			parameters = [:ρ, :a, :E, :F_rl, :F_wm, :W, :C],
			transformed = Dict(:a => :α, :E => :ε, :F_rl => :φ_rl, :F_wm => :φ_wm, :W => :w0),
			sigmas = Dict(:ρ => 2., :a => 0.5, :E => 0.5, :F_rl => 0.5, :F_wm => 0.5, :W => 0.5, :C => 2.)
		)
		insertcols!(prior_rlwm_all, :model => "α|ρ|φ_rl|C|w0|ε|φ_wm")

		prior_rlwm_samples = vcat(prior_rlwm_sens, prior_rlwm_temp, prior_rlwm_lapse, prior_rlwm_all; cols=:union)
	
		leftjoin(prior_rlwm_samples, 
			task.task[!, [:block, :trial, :feedback_optimal, :feedback_suboptimal]],
			on = [:block, :trial]
		)
	
	end

	describe(prior_rlwm_samples)
end

# ╔═╡ 40f10954-93c2-48bc-8c81-53ad6138fe97
let
	f = Figure(size = (1400, 1400))
	df1 = filter(x -> !ismissing(x.β), prior_rlwm_samples)
	df1[!, :ρ] .= 1
	df2 = filter(x -> ismissing(x.β), prior_rlwm_samples)

    df = vcat(df1, df2)

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

	save("results/single_p_QL_prior_predictive.png", f, pt_per_unit = 1)

	f
end

# ╔═╡ 10516a3d-4deb-46eb-b977-9726dc13d093
# Sample from posterior and plot for single participant
begin
	rl_fit = let
		# Initial value for Q values
		aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])
		task = task_vars_for_condition("00")
		prior_bsl = filter(x -> x.model == "α|ρ", prior_samples)
		
		rl_fit = posterior_sample_single_p(
			filter(x -> x.PID == 1, prior_bsl);
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
			wm = true,
			initV = aao,
			set_size = fill(2, maximum(task.block)),
			random_seed = 0,
			parameters = [:ρ, :a, :F_wm, :W, :C],
			sigmas = Dict(:ρ => 2., :a => 0.5, :F_wm => 0.5, :W => 0.5, :C => 2.)
		)
	
	end
end

# ╔═╡ b3ec6080-2bd0-42c5-b265-d691efb8cbb2
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

# ╔═╡ 50420f58-563c-4122-beff-5ca052abdc26
begin
	prior_rlwm_bsl = filter(x -> x.model == "α|ρ|φ|C|w0", prior_rlwm_samples)
	rlwm_one_posterior = plot_posteriors([rlwm_fit],
		["a", "ρ", "F_wm", "C", "W"];
		true_values = [α2a(prior_rlwm_bsl[1, :α]), prior_rlwm_bsl[1, :ρ], α2a(prior_rlwm_bsl[1, :φ_wm]), prior_rlwm_bsl[1, :C], α2a(prior_rlwm_bsl[1, :w0])]
	)
	rlwm_one_posterior
end

# ╔═╡ Cell order:
# ╟─d49d6f84-5ff6-4100-90db-995ffdeb09ca
# ╠═88f8505c-5fc2-11ef-2ac9-9bb7f4b5b591
# ╠═cf2e2616-d29a-49cb-b9ee-6b6ea89c4ee6
# ╟─533b6f26-adb1-438e-9cbc-5dcc2d19ec1a
# ╠═9bfdd2b3-7885-42b3-8839-c426dae394a4
# ╠═7a3b57c9-51ca-4ec1-b81f-bb45ed456b9d
# ╟─bb8fc03d-f85a-451c-89c4-2303f0cb7fef
# ╠═3e5083bf-89e9-405d-8496-a53e66e115ef
# ╠═bfe4c9c1-72f0-4999-bbcb-76bdd4f579da
# ╠═40f10954-93c2-48bc-8c81-53ad6138fe97
# ╠═10516a3d-4deb-46eb-b977-9726dc13d093
# ╠═b3ec6080-2bd0-42c5-b265-d691efb8cbb2
# ╠═50420f58-563c-4122-beff-5ca052abdc26
