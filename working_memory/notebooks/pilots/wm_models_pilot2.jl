### A Pluto.jl notebook ###
# v0.19.45

using Markdown
using InteractiveUtils

# ╔═╡ 2511ec84-857d-11ef-2e80-43218bddef4e
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
	include("$(pwd())/working_memory/RL+RLWM_models.jl")
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

# ╔═╡ 8a215c05-805a-4960-aca8-3c84b9452c3c
md"
*N.B. need to download data from osf and place in data/ folder!*
"

# ╔═╡ 5c9edffe-1197-44a2-b6d5-2f4d80a18518
begin
	PLT_data, test_data, vigour_data, jspsych_data = load_pilot2_data()
	df = prepare_for_fit(PLT_data, pilot2=true)[1]
	df = filter(gdf -> any(gdf.set_size .== 6), groupby(df, :prolific_pid); ungroup=true)
	df = DataFrames.transform(groupby(df, [:prolific_pid, :block, :pair]), eachindex => :trial; ungroup=true)
	df = filter(x -> x.PID != 27, df)
end

# ╔═╡ b5744ee4-7f10-4c41-94d6-185edcfff6dc
md"
### Collins & Frank RLWM model
"

# ╔═╡ 7243e277-a9c8-4c33-ac10-8573cb88fb69
begin
	rlwm_ests, rlwm_choices, rlwm_cov = optimize_multiple(
		df;
		model = RLWM_ss,
		estimate = "MAP",
		include_true=false,
		transformed = Dict(:a => :α, :F_wm => :φ_wm, :W => :w),
		priors = Dict(
			:ρ => truncated(Normal(0., 1.), lower = 0.),
			:a => Normal(0., 0.5),
			:F_wm => Normal(0., 0.5),
			:W => Normal(0., 0.5),
			:C => truncated(Normal(3., 2.), lower = 1.)
		)
		#bootstraps=100
	)

	rlwm_ests
end

# ╔═╡ 3c25a381-4fe1-4b94-8acf-7da4c14b9a3c
let
	f = Figure(size = (1000, 300))
	
	labs1 = (xlabel = "α", ylabel = "ρ", zlabel = "log-likelihood")
	labs2 = (xlabel = "C", ylabel = "w", zlabel = "log-likelihood")
	labs3 = (xlabel = "C", ylabel = "φ_wm", zlabel = "log-likelihood")
	
	ax1, ax2, ax3 = Axis3(f[1,1]; labs1...), Axis3(f[1,2]; labs2...), Axis3(f[1,3]; labs3...)
	scatter!(ax1, rlwm_ests.α, rlwm_ests.ρ, rlwm_ests.loglike)
	scatter!(ax2, rlwm_ests.C, rlwm_ests.w, rlwm_ests.loglike)
	scatter!(ax3, rlwm_ests.C, rlwm_ests.φ_wm, rlwm_ests.loglike)
	f
end

# ╔═╡ efdb30b8-65c8-4885-8066-d0be1ea655a6
let
	choice_df = rlwm_choices
	choice_df.true_choice = Int.(choice_df.true_choice)
	choice_df.predicted_choice = Int.(choice_df.predicted_choice)
	choice_df = stack(choice_df, [:true_choice, :predicted_choice])

	f = Figure(size = (700, 400))
	plot_prior_accuracy!(
		f[1,1],
		choice_df;
		group = :variable,
		pid_col = :PID,
		acc_col = :value,
		legend = true,
		legend_rows = 1,
		colors = Makie.colorschemes[:seaborn_pastel6],
		error_band = "PI"
	)
	f
end

# ╔═╡ 3c7f50e8-d766-45a7-ba3d-f241eedbb9e7
md"
### Palimpsest RLWM model (strict capacity)
"

# ╔═╡ a0e0399e-a982-4bc4-856f-2e712d1e182b
begin
	pmst_ests, pmst_choices, pmst_covs = optimize_multiple(
		df;
		model = RLWM_pmst,
		estimate = "MAP",
		include_true=false,
		priors = Dict(
			:ρ => truncated(Normal(0., 1.), lower = 0.),
			:a => Normal(0., 0.5),
			:W => Normal(0., 0.5),
			:C => truncated(Normal(3., 2.), lower = 1.)
		),
		parameters = [:ρ, :a, :W, :C],
		transformed = Dict(:a => :α, :W => :w)
		#bootstraps=100
	)
	pmst_ests
end

# ╔═╡ 5ab6c587-0a1f-40ee-b8f6-6cb9a2075c1e
let
	# Set the labels for the axes
	f = Figure(size = (1000, 300))
	
	labs1 = (xlabel = "α", ylabel = "ρ", zlabel = "log-likelihood")
	labs2 = (xlabel = "C", ylabel = "w", zlabel = "log-likelihood")
	
	ax1, ax2 = Axis3(f[1,1]; labs1...), Axis3(f[1,2]; labs2...)
	scatter!(ax1, pmst_ests.α, pmst_ests.ρ, pmst_ests.loglike)
	scatter!(ax2, pmst_ests.C, pmst_ests.w, pmst_ests.loglike)
	f
end

# ╔═╡ fb2b80b8-9206-499e-8b55-d84f47b96a5a
let
	choice_df = pmst_choices
	choice_df.true_choice = Int.(choice_df.true_choice)
	choice_df.predicted_choice = Int.(choice_df.predicted_choice)
	choice_df = stack(choice_df, [:true_choice, :predicted_choice])

	f = Figure(size = (700, 400))
	plot_prior_accuracy!(
		f[1,1],
		choice_df;
		group = :variable,
		pid_col = :PID,
		acc_col = :value,
		legend = true,
		legend_rows = 1,
		colors = Makie.colorschemes[:seaborn_pastel6],
		error_band = "PI"
	)
	f
end

# ╔═╡ 2de21839-789d-4e48-be95-efa6a45cc906
md"
### Palimpsest RLWM model (with sigmoid weight)
"

# ╔═╡ 7af0c1fb-e32d-4fa2-96a2-094c58c57888
begin
	pmst_sg_ests, pmst_sg_choices, pmst_sg_covs = optimize_multiple(
		df;
		model = RLWM_pmst_sgd,
		estimate = "MAP",
		include_true = false,
		priors = Dict(
			:ρ => truncated(Normal(0., 1.), lower = 0.),
			:a => Normal(0., 0.5),
			:W => Normal(0., 0.5),
			:C => truncated(Normal(3., 2.), lower = 1.)
		),
		parameters = [:ρ, :a, :W, :C],
		transformed = Dict(:a => :α, :W => :w)
		#bootstraps=100
	)
	pmst_sg_ests
end

# ╔═╡ 72ac4969-16b7-4eb5-bfb4-c60be499228c
let
	# Set the labels for the axes
	f = Figure(size = (1000, 300))
	
	labs1 = (xlabel = "α", ylabel = "ρ", zlabel = "log-likelihood")
	labs2 = (xlabel = "C", ylabel = "w", zlabel = "log-likelihood")
	
	ax1, ax2 = Axis3(f[1,1]; labs1...), Axis3(f[1,2]; labs2...)
	scatter!(ax1, pmst_sg_ests.α, pmst_sg_ests.ρ, pmst_sg_ests.loglike)
	scatter!(ax2, pmst_sg_ests.C, pmst_sg_ests.w, pmst_sg_ests.loglike)
	f
end

# ╔═╡ 8304bcf4-f27d-434d-8952-a5047a5f1d78
let
	choice_df = pmst_sg_choices
	choice_df.true_choice = Int.(choice_df.true_choice)
	choice_df.predicted_choice = Int.(choice_df.predicted_choice)
	choice_df = stack(choice_df, [:true_choice, :predicted_choice])

	f = Figure(size = (700, 400))
	plot_prior_accuracy!(
		f[1,1],
		choice_df;
		group = :variable,
		pid_col = :PID,
		acc_col = :value,
		legend = true,
		legend_rows = 1,
		colors = Makie.colorschemes[:seaborn_pastel6],
		error_band = "PI"
	)
	f
end

# ╔═╡ 54e7dc54-f19d-4f3a-a0c4-4348d9c655c9
md"
### Palimpsest RLWM model (outcome-nonspecific C)
"

# ╔═╡ 384270b1-0c13-4d7f-8cb1-fbdee5e8fc10
begin
	pmst_ovlC_ests, pmst_ovlC_choices, pmst_ovlC_covs = optimize_multiple(
		df;
		model = RLWM_all_outc_pmst_sgd,
		estimate = "MAP",
		include_true = false,
		priors = Dict(
			:ρ => truncated(Normal(0., 1.), lower = 0.),
			:a => Normal(0., 0.5),
			:W => Normal(0, 0.5),
			:C => truncated(Normal(6., 4.), lower = 1.)
		),
		parameters = [:ρ, :a, :W, :C],
		transformed = Dict(:a => :α, :W => :w)
		#bootstraps=100
	)
	pmst_ovlC_ests
end

# ╔═╡ 1e3b5bda-903b-42b0-a774-7e78d9524614
let
	# Set the labels for the axes
	f = Figure(size = (1000, 300))
	
	labs1 = (xlabel = "α", ylabel = "ρ", zlabel = "log-likelihood")
	labs2 = (xlabel = "C", ylabel = "w", zlabel = "log-likelihood")
	
	ax1, ax2 = Axis3(f[1,1]; labs1...), Axis3(f[1,2]; labs2...)
	scatter!(ax1, pmst_ovlC_ests.α, pmst_ovlC_ests.ρ, pmst_ovlC_ests.loglike)
	scatter!(ax2, pmst_ovlC_ests.C, pmst_ovlC_ests.w, pmst_ovlC_ests.loglike)
	f
end

# ╔═╡ 48c6907b-a64e-4690-bd09-4c6f5a5f4c47
let
	choice_df = pmst_ovlC_choices
	choice_df.true_choice = Int.(choice_df.true_choice)
	choice_df.predicted_choice = Int.(choice_df.predicted_choice)
	choice_df = stack(choice_df, [:true_choice, :predicted_choice])

	f = Figure(size = (700, 400))
	plot_prior_accuracy!(
		f[1,1],
		choice_df;
		group = :variable,
		pid_col = :PID,
		acc_col = :value,
		legend = true,
		legend_rows = 1,
		colors = Makie.colorschemes[:seaborn_pastel6],
		error_band = "PI"
	)
	f
end

# ╔═╡ 4f353003-24df-4671-8198-04d4ba7cb735
md"
### Working memory alone (palimpsest)
"

# ╔═╡ 08c886a0-b4a9-47ad-ac71-aaa42c021925
begin
	pmst_wm_ests, pmst_wm_choices, pmst_wm_covs = optimize_multiple(
		df;
		model = WM_pmst_sgd,
		estimate = "MAP",
		include_true = false,
		priors = Dict(
			:ρ => truncated(Normal(0., 2.), lower = 0.),
			:C => truncated(Normal(3., 2), lower = 1.)
		),
		parameters = [:ρ, :C]
		#bootstraps=100
	)
	pmst_wm_ests
end

# ╔═╡ 47c03556-301f-4607-953a-ed16092b165d
let
	# Set the labels for the axes
	f = Figure(size = (500, 300))
	
	labs1 = (xlabel = "ρ", ylabel = "C", zlabel = "log-likelihood")
	
	ax1 = Axis3(f[1,1]; labs1...)
	scatter!(ax1, pmst_wm_ests.ρ, pmst_wm_ests.C, pmst_wm_ests.loglike)
	f
end

# ╔═╡ a84787d9-e51c-4ed5-8ea0-b23d887cac1d
let
	choice_df = pmst_wm_choices
	choice_df.true_choice = Int.(choice_df.true_choice)
	choice_df.predicted_choice = Int.(choice_df.predicted_choice)
	choice_df = stack(choice_df, [:true_choice, :predicted_choice])

	f = Figure(size = (700, 400))
	plot_prior_accuracy!(
		f[1,1],
		choice_df;
		group = :variable,
		pid_col = :PID,
		acc_col = :value,
		legend = true,
		legend_rows = 1,
		colors = Makie.colorschemes[:seaborn_pastel6],
		error_band = "PI"
	)
	f
end

# ╔═╡ Cell order:
# ╠═2511ec84-857d-11ef-2e80-43218bddef4e
# ╟─8a215c05-805a-4960-aca8-3c84b9452c3c
# ╠═5c9edffe-1197-44a2-b6d5-2f4d80a18518
# ╟─b5744ee4-7f10-4c41-94d6-185edcfff6dc
# ╠═7243e277-a9c8-4c33-ac10-8573cb88fb69
# ╠═3c25a381-4fe1-4b94-8acf-7da4c14b9a3c
# ╠═efdb30b8-65c8-4885-8066-d0be1ea655a6
# ╟─3c7f50e8-d766-45a7-ba3d-f241eedbb9e7
# ╠═a0e0399e-a982-4bc4-856f-2e712d1e182b
# ╠═5ab6c587-0a1f-40ee-b8f6-6cb9a2075c1e
# ╠═fb2b80b8-9206-499e-8b55-d84f47b96a5a
# ╟─2de21839-789d-4e48-be95-efa6a45cc906
# ╠═7af0c1fb-e32d-4fa2-96a2-094c58c57888
# ╠═72ac4969-16b7-4eb5-bfb4-c60be499228c
# ╠═8304bcf4-f27d-434d-8952-a5047a5f1d78
# ╟─54e7dc54-f19d-4f3a-a0c4-4348d9c655c9
# ╠═384270b1-0c13-4d7f-8cb1-fbdee5e8fc10
# ╠═1e3b5bda-903b-42b0-a774-7e78d9524614
# ╠═48c6907b-a64e-4690-bd09-4c6f5a5f4c47
# ╟─4f353003-24df-4671-8198-04d4ba7cb735
# ╠═08c886a0-b4a9-47ad-ac71-aaa42c021925
# ╠═47c03556-301f-4607-953a-ed16092b165d
# ╠═a84787d9-e51c-4ed5-8ea0-b23d887cac1d
