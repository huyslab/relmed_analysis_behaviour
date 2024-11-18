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
	wm_df_clean = prepare_WM_data(WM_data)
	# extra double taker which missed by exclude_double_takers!
	filter!(x -> x.PID != 7, wm_df_clean)
	nothing
end

# ╔═╡ 419ae40b-f067-43ab-aff7-af7817ab93db
let
	f = Figure(size = (700, 1000))
	plot_prior_accuracy!(
		f[1,:],
		wm_df_clean;
		choice_val = 3.,
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
		choice_val = 3.,
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
		choice_val = 3.,
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

# ╔═╡ 64b9070c-106c-4274-a8b3-013c2b53bb30
wm_df_clean.session

# ╔═╡ 946360aa-3a1a-4ea0-b4ef-c32453e65b19
md"
### Model fits
"

# ╔═╡ 3108a5a5-37b9-4b76-a529-dab5e3eecff2
begin
	# single update model
	qs_ests, qs_choices, qs_covs = optimize_multiple(
		wm_df_clean;
		model = RL_multi_2set,
		estimate = "MAP",
		include_true = false,
		priors = Dict(
			:ρ => truncated(Normal(0., 3.), lower = 0.),
			:a1 => Normal(0., 0.5),
        	:a2 => Normal(-0.2, 0.5),
		),
		parameters = [:ρ, :a1, :a2],
		transformed = Dict(:a1 => :α1, :a2 => :α2)
	)
	# reciprocal update model
	qr_ests, qr_choices, qr_covs = optimize_multiple(
		wm_df_clean;
		model = RL_multi_2set_recip,
		estimate = "MAP",
		include_true = false,
		priors = Dict(
			:ρ => truncated(Normal(0., 3.), lower = 0.),
			:a1 => Normal(0., 0.5),
        	:a2 => Normal(-0.2, 0.5),
		),
		parameters = [:ρ, :a1, :a2],
		transformed = Dict(:a1 => :α1, :a2 => :α2)
	)
	nothing
end

# ╔═╡ 19d06113-a41f-4b5f-a353-225cec794db0
let
	f = Figure(size = (800, 300))
	
	# Define parameters
	ρ1, α1s, α2s = qs_ests.ρ, qs_ests.α1, qs_ests.α2
	ρ2, α1r, α2r = qr_ests.ρ, qr_ests.α1, qr_ests.α2

	# Set labels
	labs1 = (xlabel = "ρ", ylabel = "α1", zlabel = "α2", title = "Single")
	labs2 = (xlabel = "ρ", ylabel = "α1", zlabel = "α2", title = "Reciprocal")

	# Plot
	ax1, ax2 = Axis3(f[1,1]; labs1...), Axis3(f[1,2]; labs2...)
	scatter!(ax1, ρ1, α1s, α2s)
	scatter!(ax2, ρ2, α1r, α2r)
	f
end

# ╔═╡ Cell order:
# ╟─11ebe9db-0809-48e7-ab4b-d5bee1fae9b1
# ╠═54a03c1e-a2b8-11ef-2c40-695bce51ab75
# ╠═2a0c2f77-4af2-44f0-adbe-3d9e52d24a59
# ╠═419ae40b-f067-43ab-aff7-af7817ab93db
# ╠═64b9070c-106c-4274-a8b3-013c2b53bb30
# ╟─946360aa-3a1a-4ea0-b4ef-c32453e65b19
# ╠═3108a5a5-37b9-4b76-a529-dab5e3eecff2
# ╠═19d06113-a41f-4b5f-a353-225cec794db0
