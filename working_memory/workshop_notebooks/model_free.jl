### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ 54a03c1e-a2b8-11ef-2c40-695bce51ab75
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
	
	include("$(pwd())/fetch_preprocess_data.jl")
	include("$(pwd())/working_memory/simulate.jl")
	include("$(pwd())/working_memory/RL+WM_multiaction_models.jl")	
	include("$(pwd())/working_memory/plotting_utils.jl")

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

# ╔═╡ 11ebe9db-0809-48e7-ab4b-d5bee1fae9b1
md"
## Model-free plots for working memory in workshop
"

# ╔═╡ 2a0c2f77-4af2-44f0-adbe-3d9e52d24a59
begin
	# Load data
	_, _, _, _, _, WM_data, _, _ = load_pilot6_data()
	wm_df_clean = let
		# Clean data
		wm_df_clean = exclude_PLT_sessions(WM_data, required_n_blocks = 10)
		wm_df_clean = filter(x -> x.response != "noresp", wm_df_clean)
		wm_df_clean = prepare_for_fit(wm_df_clean; pilot = 6)[1]
		filter!(x -> x.PID != 7, wm_df_clean) # double taker still in data?
	end
	nothing
end

# ╔═╡ 419ae40b-f067-43ab-aff7-af7817ab93db
let
	f = Figure(size = (700, 1000))
	plot_prior_accuracy!(
		f[1,:],
		wm_df_clean;
		group = :set_size,
		pid_col = :PID,
		legend = true,
		legend_rows = 1,
		legend_pos = :bottom,
		title = "Overall",
		colors = Makie.colorschemes[:seaborn_pastel6],
		error_band = "se"
	)
	plot_prior_accuracy!(
		f[2,1],
		filter(x -> x.valence > 0, wm_df_clean);
		group = :set_size,
		pid_col = :PID,
		legend = true,
		legend_rows = 1,
		legend_pos = :bottom,
		title = "Reward blocks",
		colors = Makie.colorschemes[:seaborn_pastel6],
		error_band = "se"
	)
	plot_prior_accuracy!(
		f[3, 1],
		filter(x -> x.valence < 0, wm_df_clean);
		group = :set_size,
		pid_col = :PID,
		legend = true,
		legend_rows = 1,
		legend_pos = :bottom,
		title = "Punishment blocks",
		colors = Makie.colorschemes[:seaborn_pastel6],
		error_band = "se"
	)
	f
end

# ╔═╡ Cell order:
# ╟─11ebe9db-0809-48e7-ab4b-d5bee1fae9b1
# ╠═54a03c1e-a2b8-11ef-2c40-695bce51ab75
# ╠═2a0c2f77-4af2-44f0-adbe-3d9e52d24a59
# ╠═419ae40b-f067-43ab-aff7-af7817ab93db
