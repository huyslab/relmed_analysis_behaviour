### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ c621989a-978e-11ef-170f-198420390baf
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

# ╔═╡ 45bcccda-b7d3-46c8-8dca-0c5dbc913356
md"
## Variants of working-memory alone models
"

# ╔═╡ 49da8a4e-3f5d-4ceb-b095-5a5f2305de0c
begin
	random_task = create_random_task(;
	    n_blocks = 18, n_trials = 10, n_confusing = 0, set_sizes = [2, 6, 14]
		#n_blocks = 18, n_trials = 10, n_confusing = 2, set_sizes = [2, 4, 6]
	)
	chce = fill(missing, nrow(random_task))
	
	nothing
end

# ╔═╡ 577a6f57-dc98-4d3b-802f-389204b12e36
md"
### Working memory models
"

# ╔═╡ bc1b401e-2f54-48e1-8299-ab93c0eb43f0
md"
##### Single update
"

# ╔═╡ 536d4e3e-1036-4c6f-bfde-94522701cfc7
md"
A) Palimpsest with **overall** capacity (sigmoid k=3)
"

# ╔═╡ 6ff1e36d-af92-4922-9744-2afb5e924e7b
begin
	prior_sample_wma = simulate_from_prior(
	    100;
		model = WM_all_outc_pmst_sgd,
		priors = Dict(
			:ρ => truncated(Normal(0., 2.), lower = 0.),
			:C => truncated(Normal(5., 3.), lower = 1.)
		),
		parameters = [:ρ, :C],
		fixed_struct = random_task, 
		gq = true,
		random_seed = 123
	)
	
	f_wma = optimization_calibration(
		prior_sample_wma,
		optimize_multiple,
		estimate = "MAP",
		model = WM_all_outc_pmst_sgd,
		priors = Dict(
			:ρ => truncated(Normal(0., 2.), lower = 0.),
			:C => truncated(Normal(5., 3.), lower = 1.)
		),
		parameters = [:ρ, :C]
	)

	f_wma
end

# ╔═╡ b6fd701e-b353-4fd3-bb95-573076de2706
let
	f = plot_prior_predictive_by_valence(
		prior_sample_wma,
		[:W_optimal, :W_suboptimal];
		ylab = "W-value",
		fig_size = (1000, 1000),
		group = :set_size,
		legend = true,
		colors = Makie.colorschemes[:seaborn_pastel6]
	)	
	f
end

# ╔═╡ a09f6496-58e4-4229-af54-c3e25672698e
md"
B) Palimpsest with **stimulus-specific** capacity (sigmoid k=3)
"

# ╔═╡ 2cf5da29-7b11-45fa-ac16-6d39dcbd5126
begin
	prior_sample_wmb = simulate_from_prior(
	    100;
		model = WM_pmst_sgd,
		priors = Dict(
			:ρ => truncated(Normal(0., 2.), lower = 0.),
			:C => truncated(Normal(3., 2.), lower = 1.)
		),
		parameters = [:ρ, :C],
		fixed_struct = random_task, 
		gq = true,
		random_seed = 123
	)
	
	f_wms = optimization_calibration(
		prior_sample_wmb,
		optimize_multiple,
		estimate = "MAP",
		model = WM_pmst_sgd,
		priors = Dict(
			:ρ => truncated(Normal(0., 2.), lower = 0.),
			:C => truncated(Normal(3., 2.), lower = 1.)
		),
		parameters = [:ρ, :C]
	)

	f_wms
end

# ╔═╡ ca112cc5-5312-47ea-88b6-55069d212e9c
let
	f = plot_prior_predictive_by_valence(
		prior_sample_wmb,
		[:W_optimal, :W_suboptimal];
		ylab = "W-value",
		fig_size = (1000, 1000),
		group = :set_size,
		legend = true,
		colors = Makie.colorschemes[:seaborn_pastel6]
	)	
	f
end

# ╔═╡ aee8e115-68d9-4626-af8d-8b3e237a7f77
md"
C) Palimpsest with **overall** capacity (sigmoid k=3) + **no averaging**
"

# ╔═╡ 0059b111-235c-4fd5-a140-ef4cd985a76d
begin
	prior_sample_wmc = simulate_from_prior(
	    100;
		model = WM_all_outc_pmst_sgd_sum,
		priors = Dict(
			:ρ => truncated(Normal(0., 2.), lower = 0.),
			:C => truncated(Normal(5., 3.), lower = 1.)
		),
		parameters = [:ρ, :C],
		fixed_struct = random_task, 
		gq = true,
		random_seed = 123
	)
	
	f_wmc = optimization_calibration(
		prior_sample_wmc,
		optimize_multiple,
		estimate = "MAP",
		model = WM_all_outc_pmst_sgd_sum,
		priors = Dict(
			:ρ => truncated(Normal(0., 2.), lower = 0.),
			:C => truncated(Normal(5., 3.), lower = 1.)
		),
		parameters = [:ρ, :C]
	)

	f_wmc
end

# ╔═╡ 294ecb17-d4d8-45b8-a644-eb6665a73028
let
	f = plot_prior_predictive_by_valence(
		prior_sample_wmc,
		[:W_optimal, :W_suboptimal];
		ylab = "W-value",
		fig_size = (1000, 1000),
		group = :set_size,
		legend = true,
		colors = Makie.colorschemes[:seaborn_pastel6]
	)	
	f
end

# ╔═╡ 47c25912-113c-4418-adfe-5799b5d74834
md"
D) Palimpsest with **stimulus-specific** capacity (sigmoid k=3) + **no averaging**
"

# ╔═╡ daed8ac8-1a4c-4440-9743-15147dc0e05e
begin
	prior_sample_wmd = simulate_from_prior(
	    100;
		model = WM_pmst_sgd_sum,
		priors = Dict(
			:ρ => truncated(Normal(0., 2.), lower = 0.),
			:C => truncated(Normal(3., 2), lower = 1.)
		),
		parameters = [:ρ, :C],
		fixed_struct = random_task, 
		gq = true,
		random_seed = 123
	)
	
	f_wmd = optimization_calibration(
		prior_sample_wmd,
		optimize_multiple,
		estimate = "MAP",
		model = WM_pmst_sgd_sum,
		priors = Dict(
			:ρ => truncated(Normal(0., 2.), lower = 0.),
			:C => truncated(Normal(3., 2.), lower = 1.)
		),
		parameters = [:ρ, :C]
	)

	f_wmd
end

# ╔═╡ 882f4764-d6a6-4a4b-94c7-f86a902d72bb
let
	f = plot_prior_predictive_by_valence(
		prior_sample_wmd,
		[:W_optimal, :W_suboptimal];
		ylab = "W-value",
		fig_size = (1000, 1000),
		group = :set_size,
		legend = true,
		colors = Makie.colorschemes[:seaborn_pastel6]
	)	
	f
end

# ╔═╡ ed5d0c07-28d6-4232-8d1f-7b3350ba0db9
md"
##### Reciprocal update
"

# ╔═╡ 034eae62-f2bc-4be3-b8b5-e48c9ec54621
md"
These models work by assuming the agent has some weak knowledge about the environment, namely what (other) possible outcomes there are (i.e. if they win/lose 50p, the other possibilities are 1p or £1).

If they get one of the larger outcomes, then we assume they update the other W-value based on winning/losing 1p. If they get 1p, we (currently) assume an update of *randomly* either 50p or £1 for the alternative option in that pair.

In all cases we use averaging rather than summing.
"

# ╔═╡ b50cb8ab-7f25-43c7-b795-f70cc0c9e15d
md"
E) **Overall** capacity, with weights on alternative choice also updated
"

# ╔═╡ 13ef37d0-ec7e-4750-8a40-7e11f5bb076b
begin
	prior_sample_wme = simulate_from_prior(
	    100;
		model = WM_all_outc_pmst_sgd_recip,
		priors = Dict(
			:ρ => truncated(Normal(0., 2.), lower = 0.),
			:C => truncated(Normal(5., 3.), lower = 1.)
		),
		parameters = [:ρ, :C],
		fixed_struct = random_task, 
		gq = true,
		random_seed = 123
	)
	
	f_wme = optimization_calibration(
		prior_sample_wme,
		optimize_multiple,
		estimate = "MAP",
		model = WM_all_outc_pmst_sgd_recip,
		priors = Dict(
			:ρ => truncated(Normal(0., 2.), lower = 0.),
			:C => truncated(Normal(5., 3.), lower = 1.)
		),
		parameters = [:ρ, :C]
	)

	f_wme
end

# ╔═╡ 14d0aad7-4834-49b7-b1e4-8df8d8211606
let
	f = plot_prior_predictive_by_valence(
		prior_sample_wme,
		[:W_optimal, :W_suboptimal];
		ylab = "W-value",
		fig_size = (1000, 1000),
		group = :set_size,
		legend = true,
		colors = Makie.colorschemes[:seaborn_pastel6]
	)
	f
end

# ╔═╡ d25b5f86-7c30-44fa-8e73-9e9a68e2a2ed
md"
F) **Stimulus-specific** capacity, with weights on alternative choice also updated
"

# ╔═╡ 8da91c18-eb34-4085-93be-4218c880e2aa
begin
	prior_sample_wmf = simulate_from_prior(
	    100;
		model = WM_pmst_sgd_recip,
		priors = Dict(
			:ρ => truncated(Normal(0., 2.), lower = 0.),
			:C => truncated(Normal(3., 2.), lower = 1.)
		),
		parameters = [:ρ, :C],
		fixed_struct = random_task, 
		gq = true,
		random_seed = 123
	)
	
	f_wmf = optimization_calibration(
		prior_sample_wmf,
		optimize_multiple,
		estimate = "MAP",
		model = WM_pmst_sgd_recip,
		priors = Dict(
			:ρ => truncated(Normal(0., 2.), lower = 0.),
			:C => truncated(Normal(3., 2.), lower = 1.)
		),
		parameters = [:ρ, :C]
	)

	f_wmf
end

# ╔═╡ 0aaa2631-c5a7-4ea1-ba49-ff3bb17f6049
let
	f = plot_prior_predictive_by_valence(
		prior_sample_wmf,
		[:W_optimal, :W_suboptimal];
		ylab = "W-value",
		fig_size = (1000, 1000),
		group = :set_size,
		legend = true,
		colors = Makie.colorschemes[:seaborn_pastel6]
	)
	f
end

# ╔═╡ 37753b04-73d8-4136-82ce-aab01bc4db34
md"
G) **Overall** capacity, reciprocal, **no averaging**
"

# ╔═╡ a9f10ec1-e712-471f-ab58-8c02f1b8ca48
begin
	prior_sample_wmg = simulate_from_prior(
	    100;
		model = WM_all_outc_pmst_sgd_sum_recip,
		priors = Dict(
			:ρ => truncated(Normal(0., 2.), lower = 0.),
			:C => truncated(Normal(5., 3.), lower = 1.)
		),
		parameters = [:ρ, :C],
		fixed_struct = random_task, 
		gq = true,
		random_seed = 123
	)
	
	f_wmg = optimization_calibration(
		prior_sample_wmg,
		optimize_multiple,
		estimate = "MAP",
		model = WM_all_outc_pmst_sgd_sum_recip,
		priors = Dict(
			:ρ => truncated(Normal(0., 2.), lower = 0.),
			:C => truncated(Normal(5., 3.), lower = 1.)
		),
		parameters = [:ρ, :C]
	)

	f_wmg
end

# ╔═╡ 3b279ca2-b398-49d5-b309-3b3d0dbd25d5
let
	f = plot_prior_predictive_by_valence(
		prior_sample_wmg,
		[:W_optimal, :W_suboptimal];
		ylab = "W-value",
		fig_size = (1000, 1000),
		group = :set_size,
		legend = true,
		colors = Makie.colorschemes[:seaborn_pastel6]
	)
	f
end

# ╔═╡ 70f3cf79-455a-42a0-9843-e1c357d6ce37
md"
H) **Stimulus-specific** capacity, reciprocal, **no averaging**
"

# ╔═╡ 09b941b0-ccdf-4cdc-bf2a-df7091365936
begin
	prior_sample_wmh = simulate_from_prior(
	    100;
		model = WM_pmst_sgd_sum_recip,
		priors = Dict(
			:ρ => truncated(Normal(0., 2.), lower = 0.),
			:C => truncated(Normal(3., 2.), lower = 1.)
		),
		parameters = [:ρ, :C],
		fixed_struct = random_task, 
		gq = true,
		random_seed = 123
	)
	
	f_wmh = optimization_calibration(
		prior_sample_wmh,
		optimize_multiple,
		estimate = "MAP",
		model = WM_pmst_sgd_sum_recip,
		priors = Dict(
			:ρ => truncated(Normal(0., 2.), lower = 0.),
			:C => truncated(Normal(3., 2.), lower = 1.)
		),
		parameters = [:ρ, :C]
	)

	f_wmh
end

# ╔═╡ 8d2dc096-827c-48ce-936d-171057950970
let
	f = plot_prior_predictive_by_valence(
		prior_sample_wmh,
		[:W_optimal, :W_suboptimal];
		ylab = "W-value",
		fig_size = (1000, 1000),
		group = :set_size,
		legend = true,
		colors = Makie.colorschemes[:seaborn_pastel6]
	)
	f
end

# ╔═╡ fae2deee-fbba-4170-be1c-2256d1958555
md"
### Can we implement this in a RLWM model?
"

# ╔═╡ c0d8fb59-ec34-4278-a16a-f2e91085e30f
md"
A) **With averaging**
"

# ╔═╡ 481d4882-48a4-479b-9b03-fb46f479748c
begin
	prior_sample_rlwm_recip = simulate_from_prior(
	    100;
		model = RLWM_pmst_sgd_recip,
		priors = Dict(
			:ρ => truncated(Normal(0., 2.), lower = 0.),
			:a => Normal(0., 0.5),
			:W => Normal(0., 0.5),
			:C => truncated(Normal(3., 2.), lower = 1.)
		),
		parameters = [:ρ, :a, :W, :C],
		transformed = Dict(:a => :α, :W => :w),
		fixed_struct = random_task,
		gq = true,
		random_seed = 123
	)

	f_rlwm = optimization_calibration(
		prior_sample_rlwm_recip,
		optimize_multiple,
		estimate = "MAP",
		model = RLWM_pmst_sgd_recip,
		priors = Dict(
			:ρ => truncated(Normal(0., 2.), lower = 0.),
			:a => Normal(0., 0.5),
			:W => Normal(0., 0.5),
			:C => truncated(Normal(3., 2.), lower = 1.)
		),
		parameters = [:ρ, :a, :W, :C],
		transformed = Dict(:a => :α, :W => :w)
	)

	f_rlwm
end

# ╔═╡ 8522c9e0-e988-4f84-8d27-ce5bf40ed683
let
	f = plot_prior_predictive_by_valence(
		prior_sample_rlwm_recip,
		[:Q_optimal, :Q_suboptimal];
		W_cols = [:W_optimal, :W_suboptimal],
		ylab = ("Q-value", "W-value"),
		fig_size = (1000, 1000),
		group = :set_size,
		legend = true,
		colors = Makie.colorschemes[:seaborn_pastel6]
	)	
	f
end

# ╔═╡ 4313b8da-ee54-4f31-b0a8-6bbdbe7ced84
md"
B) **Without averaging**
"

# ╔═╡ 122ff088-2015-4a5e-a921-d41847a83d8d
begin
	prior_sample_rlwm_sum_recip = simulate_from_prior(
	    100;
		model = RLWM_pmst_sgd_sum_recip,
		priors = Dict(
			:ρ => truncated(Normal(0., 2.), lower = 0.),
			:a => Normal(0., 0.5),
			:W => Normal(0., 0.5),
			:C => truncated(Normal(3., 2.), lower = 1.)
		),
		parameters = [:ρ, :a, :W, :C],
		transformed = Dict(:a => :α, :W => :w),
		fixed_struct = random_task,
		gq = true,
		random_seed = 123
	)

	f_rlwm_sum = optimization_calibration(
		prior_sample_rlwm_sum_recip,
		optimize_multiple,
		estimate = "MAP",
		model = RLWM_pmst_sgd_sum_recip,
		priors = Dict(
			:ρ => truncated(Normal(0., 2.), lower = 0.),
			:a => Normal(0., 0.5),
			:W => Normal(0., 0.5),
			:C => truncated(Normal(3., 2.), lower = 1.)
		),
		parameters = [:ρ, :a, :W, :C],
		transformed = Dict(:a => :α, :W => :w)
	)

	f_rlwm_sum
end

# ╔═╡ b25b5545-ba41-4e9e-a382-5ec4197f2973
let
	f = plot_prior_predictive_by_valence(
		prior_sample_rlwm_sum_recip,
		[:Q_optimal, :Q_suboptimal];
		W_cols = [:W_optimal, :W_suboptimal],
		ylab = ("Q-value", "W-value"),
		fig_size = (1000, 1000),
		group = :set_size,
		legend = true,
		colors = Makie.colorschemes[:seaborn_pastel6]
	)	
	f
end

# ╔═╡ Cell order:
# ╟─45bcccda-b7d3-46c8-8dca-0c5dbc913356
# ╠═c621989a-978e-11ef-170f-198420390baf
# ╠═49da8a4e-3f5d-4ceb-b095-5a5f2305de0c
# ╟─577a6f57-dc98-4d3b-802f-389204b12e36
# ╟─bc1b401e-2f54-48e1-8299-ab93c0eb43f0
# ╟─536d4e3e-1036-4c6f-bfde-94522701cfc7
# ╠═6ff1e36d-af92-4922-9744-2afb5e924e7b
# ╠═b6fd701e-b353-4fd3-bb95-573076de2706
# ╟─a09f6496-58e4-4229-af54-c3e25672698e
# ╠═2cf5da29-7b11-45fa-ac16-6d39dcbd5126
# ╠═ca112cc5-5312-47ea-88b6-55069d212e9c
# ╟─aee8e115-68d9-4626-af8d-8b3e237a7f77
# ╠═0059b111-235c-4fd5-a140-ef4cd985a76d
# ╠═294ecb17-d4d8-45b8-a644-eb6665a73028
# ╟─47c25912-113c-4418-adfe-5799b5d74834
# ╠═daed8ac8-1a4c-4440-9743-15147dc0e05e
# ╠═882f4764-d6a6-4a4b-94c7-f86a902d72bb
# ╟─ed5d0c07-28d6-4232-8d1f-7b3350ba0db9
# ╟─034eae62-f2bc-4be3-b8b5-e48c9ec54621
# ╟─b50cb8ab-7f25-43c7-b795-f70cc0c9e15d
# ╠═13ef37d0-ec7e-4750-8a40-7e11f5bb076b
# ╠═14d0aad7-4834-49b7-b1e4-8df8d8211606
# ╟─d25b5f86-7c30-44fa-8e73-9e9a68e2a2ed
# ╠═8da91c18-eb34-4085-93be-4218c880e2aa
# ╠═0aaa2631-c5a7-4ea1-ba49-ff3bb17f6049
# ╟─37753b04-73d8-4136-82ce-aab01bc4db34
# ╠═a9f10ec1-e712-471f-ab58-8c02f1b8ca48
# ╠═3b279ca2-b398-49d5-b309-3b3d0dbd25d5
# ╟─70f3cf79-455a-42a0-9843-e1c357d6ce37
# ╠═09b941b0-ccdf-4cdc-bf2a-df7091365936
# ╠═8d2dc096-827c-48ce-936d-171057950970
# ╟─fae2deee-fbba-4170-be1c-2256d1958555
# ╟─c0d8fb59-ec34-4278-a16a-f2e91085e30f
# ╠═481d4882-48a4-479b-9b03-fb46f479748c
# ╠═8522c9e0-e988-4f84-8d27-ce5bf40ed683
# ╟─4313b8da-ee54-4f31-b0a8-6bbdbe7ced84
# ╠═122ff088-2015-4a5e-a921-d41847a83d8d
# ╠═b25b5545-ba41-4e9e-a382-5ec4197f2973
