### A Pluto.jl notebook ###
# v0.19.45

using Markdown
using InteractiveUtils

# ╔═╡ 6dab1aae-66e9-11ef-33d9-b9cd9f2ad67b
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

# ╔═╡ cb618858-d046-4b19-a532-e79decd5ff41
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

# ╔═╡ 178803a6-d5ed-4643-934e-b5e096500d61
# Simulate datasets for some number of participants from prior
prior_samples = let n_participants = 5
		
	# Load sequence from file
	task = task_vars_for_condition("00")

	# Arrange values for simulation
	block = repeat([task.block], n_participants)
	valence = repeat([task.valence], n_participants)
	outcomes = repeat([task.outcomes], n_participants)

	# Initial value for Q values
	aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])
	
	prior_rl = simulate_from_hierarchical_prior(
		n_participants;
		model = RL,
		block = block,
		valence = valence,
		outcomes = outcomes,
		initV = fill(aao, 1, 2),
		random_seed = 0,
		parameters = [:ρ, :a],
		transformed = Dict(:a => :α),
		sigmas = Dict(:ρ => 2., :a => 0.5)
	)
	insertcols!(prior_rl, :model => "RL")
	
	set_sizes = repeat([fill(2, maximum(task.block))], n_participants)

	prior_rlwm = simulate_from_hierarchical_prior(
		n_participants;
		model = RLWM,
		block = block,
		valence = valence,
		outcomes = outcomes,
		initV = fill(aao, 1, 2),
		set_size = set_sizes,
		random_seed = 0,
		parameters = [:ρ, :a, :F_wm, :W, :C],
		transformed = Dict(:a => :α, :F_wm => :φ_wm, :W => :w0),
		sigmas = Dict(:ρ => 2., :a => 0.5, :F_wm => 0.5, :W => 0.5, :C => 2.)
	)
	insertcols!(prior_rlwm, :model => "RLWM")

	all_samples = vcat(prior_rl, prior_rlwm; cols=:union)

	leftjoin(all_samples, 
		task.task[!, [:block, :trial, :feedback_optimal, :feedback_suboptimal]],
		on = [:block, :trial]
	)
end

# ╔═╡ f56610f0-5e72-4023-b4a8-d425b717dac2
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

# ╔═╡ 6bcb616c-3862-4201-9f26-4648c93ae7e3
let
	f_rl = optimization_calibration(
		filter(x -> x.model == "RL", prior_samples),
		optimize_multiple_single_p_QL,
		estimate = "MAP"
	)

	f_rl
end

# ╔═╡ 29ef2300-78e6-4e81-a98d-2f9e1e69540a
let
	task = task_vars_for_condition("00")
	set_sizes = fill(2, maximum(task.block))
	
	f_rlwm = optimization_calibration(
		filter(x -> x.model == "RLWM", prior_samples),
		optimize_multiple_single_p_QL,
		estimate = "MAP",
		model = RLWM_ss,
		set_size = set_sizes,
		initial_params = [mean(truncated(Normal(0., 2.), lower = 0.)), 0.5, 0.5, 0.5, mean(truncated(Normal(0., 2.), lower = 0.))],
		parameters = [:ρ, :a, :F_wm, :W, :C],
		transformed = Dict(:a => :α, :F_wm => :φ_wm, :W => :w0),
		sigmas = Dict(:ρ => 1., :a => 0.5, :F_wm => 0.5, :W => 0.5, :C => 1.)
	)

	f_rlwm
end

# ╔═╡ 60ffdcb8-2977-4cb8-88c6-ab368e8e1ce4
# Load and clean data
begin
	PLT_data = load_PLT_data()

	PLT_data = exclude_PLT_sessions(PLT_data)

	DataFrames.transform!(groupby(PLT_data, [:prolific_pid, :session, :block]),
		:isOptimal => count_consecutive_ones => :consecutiveOptimal
	)

	PLT_data = exclude_PLT_trials(PLT_data)

	nothing
end

# ╔═╡ f1b16881-a385-49c6-a516-82b3e02c6baf
let
	pilot_data = prepare_for_fit(PLT_data)
	aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])
	f_rl = optimize_multiple_single_p_QL(
		pilot_data[1],
		estimate = "MAP",
		initV = aao,
		include_true = false
	)

	f_rl
end

# ╔═╡ 7081c467-0ce5-43b1-955e-2c73427c6da2
let
	pilot_data = prepare_for_fit(PLT_data)
	aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])
	
	f_rlwm = optimize_multiple_single_p_QL(
		pilot_data[1],
		estimate = "MAP",
		initV = aao,
		include_true = false,
		model = RLWM_ss,
		set_size = fill(2, maximum(pilot_data[1].block)),
		initial_params = [mean(truncated(Normal(0., 2.), lower = 0.)), 0.5, 0.5, 0.5, mean(truncated(Normal(0., 2.), lower = 0.))],
		parameters = [:ρ, :a, :F_wm, :W, :C],
		transformed = Dict(:a => :α, :F_wm => :φ_wm, :W => :w0),
		sigmas = Dict(:ρ => 1., :a => 0.5, :F_wm => 0.5, :W => 0.5, :C => 1.)
	)

	f_rlwm
end

# ╔═╡ Cell order:
# ╠═6dab1aae-66e9-11ef-33d9-b9cd9f2ad67b
# ╠═cb618858-d046-4b19-a532-e79decd5ff41
# ╠═178803a6-d5ed-4643-934e-b5e096500d61
# ╠═f56610f0-5e72-4023-b4a8-d425b717dac2
# ╠═6bcb616c-3862-4201-9f26-4648c93ae7e3
# ╠═29ef2300-78e6-4e81-a98d-2f9e1e69540a
# ╠═60ffdcb8-2977-4cb8-88c6-ab368e8e1ce4
# ╠═f1b16881-a385-49c6-a516-82b3e02c6baf
# ╠═7081c467-0ce5-43b1-955e-2c73427c6da2
