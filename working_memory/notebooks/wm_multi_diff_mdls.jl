### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ 8b660dfb-a035-4726-ae47-0bd2b0d09ed6
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

# ╔═╡ 67f6ff9d-024d-48af-8eda-3d90d7fdf92d
md"
### Recovery for RL- and WM-alone models for multi-action task version
"

# ╔═╡ 50b996f0-e737-4c2b-9822-201f9d14fc71
begin
	## load working memory task sequence for Pilot 6
	pilot6_wm = load_wm_structure_csv("pilot6_WM")
		
	# random_task = create_random_task(;
	#     n_blocks = 18, n_trials = 10, n_confusing = 0, 
	#     set_sizes = [3, 3, 3, 21], n_options = 3
	# )
	nothing
end

# ╔═╡ bb7a4106-29e0-4711-8086-a0df85d7225a
md"
### Multi-action RL models
"

# ╔═╡ bdf5c2d9-90b8-4895-b0a5-708ac0094b79
md"
##### Two separate learning rates
"

# ╔═╡ ae020d75-e72b-4706-a778-9af63d77fb94
md"
A) **Single-update**
"

# ╔═╡ c953d1c0-24cb-4593-94b4-5b7f2cb6f2cd
begin
	prior_sample_qla = simulate_from_prior(
	    100;
		model = RL_multi_2set,
		priors = Dict(
			:ρ => truncated(Normal(0., 3.), lower = 0.),
			:a1 => Normal(0., 0.5),
        	:a2 => Normal(-0.2, 0.5),
		),
		parameters = [:ρ, :a1, :a2],
		transformed = Dict(:a1 => :α1, :a2 => :α2),
		fixed_struct = pilot6_wm, 
		gq = true,
		random_seed = 123
	)
	
	f_qla = optimization_calibration(
		prior_sample_qla,
		optimize_multiple,
		estimate = "MAP",
		model = RL_multi_2set,
		priors = Dict(
			:ρ => truncated(Normal(0., 3.), lower = 0.),
			:a1 => Normal(0., 0.5),
        	:a2 => Normal(-0.2, 0.5),
		),
		parameters = [:ρ, :a1, :a2],
		transformed = Dict(:a1 => :α1, :a2 => :α2),
	)

	f_qla
end

# ╔═╡ 1d8be900-42c0-4e31-a44d-df91f4af29e7
let
	f = plot_prior_predictive_by_valence(
		prior_sample_qla,
		[:Q_optimal, :Q_suboptimal1, :Q_suboptimal2];
		choice_val = 3.0,
		ylab = "Q-value",
		fig_size = (1000, 1000),
		group = :set_size,
		legend = true,
		colors = Makie.colorschemes[:seaborn_pastel6]
	)	
	f
end

# ╔═╡ 953d2e6f-ab5f-4479-a6e1-2c6970d15120
md"
B) **Reciprocal update**
"

# ╔═╡ e4121c93-97ef-47b8-a916-331d06a8ec40
begin
	prior_sample_qlb = simulate_from_prior(
	    100;
		model = RL_multi_2set_recip,
		priors = Dict(
			:ρ => truncated(Normal(0., 3.), lower = 0.),
			:a1 => Normal(0., 0.5),
        	:a2 => Normal(-0.2, 0.5),
		),
		parameters = [:ρ, :a1, :a2],
		transformed = Dict(:a1 => :α1, :a2 => :α2),
		fixed_struct = pilot6_wm, 
		gq = true,
		random_seed = 123
	)
	
	f_qlb = optimization_calibration(
		prior_sample_qlb,
		optimize_multiple,
		estimate = "MAP",
		model = RL_multi_2set_recip,
		priors = Dict(
			:ρ => truncated(Normal(0., 3.), lower = 0.),
			:a1 => Normal(0., 0.5),
        	:a2 => Normal(-0.2, 0.5),
		),
		parameters = [:ρ, :a1, :a2],
		transformed = Dict(:a1 => :α1, :a2 => :α2),
	)

	f_qlb
end

# ╔═╡ 9f759ab5-2930-498f-9222-d8e675ef62b8
let
	f = plot_prior_predictive_by_valence(
		prior_sample_qlb,
		[:Q_optimal, :Q_suboptimal1, :Q_suboptimal2];
		choice_val = 3.0,
		ylab = "Q-value",
		fig_size = (1000, 1000),
		group = :set_size,
		legend = true,
		colors = Makie.colorschemes[:seaborn_pastel6]
	)	
	f
end

# ╔═╡ 57c4724b-c3b3-49bc-ba63-e4ce36ad3168
md"
##### Difference in learning rate
"

# ╔═╡ a38dd46b-4d22-48cf-94b8-a319a0845704
md"
C) **Single-update**
"

# ╔═╡ 0dd99525-73f4-458e-9b62-bc7f8e885ac7
begin
	prior_sample_qlc = simulate_from_prior(
	    100;
		model = RL_multi_2set_diff,
		priors = Dict(
			:ρ => truncated(Normal(0., 3.), lower = 0.),
			:a => Normal(0., 0.5),
        	:Δa => Normal(-0.2, 0.5)
		),
		parameters = [:ρ, :a, :Δa],
		transformed = Dict(:a => :α),
		fixed_struct = pilot6_wm, 
		gq = true,
		random_seed = 123
	)
	
	f_qlc = optimization_calibration(
		prior_sample_qlc,
		optimize_multiple,
		estimate = "MAP",
		model = RL_multi_2set_diff,
		priors = Dict(
			:ρ => truncated(Normal(0., 3.), lower = 0.),
			:a => Normal(0., 0.5),
        	:Δa => Normal(-0.2, 0.5)
		),
		parameters = [:ρ, :a, :Δa],
		transformed = Dict(:a => :α)		
	)

	f_qlc
end

# ╔═╡ 9657c2af-1981-4282-99f2-7c1b55886ce9
let
	f = plot_prior_predictive_by_valence(
		prior_sample_qlc,
		[:Q_optimal, :Q_suboptimal1, :Q_suboptimal2];
		choice_val = 3.0,
		ylab = "Q-value",
		fig_size = (1000, 1000),
		group = :set_size,
		legend = true,
		colors = Makie.colorschemes[:seaborn_pastel6]
	)	
	f
end

# ╔═╡ aa9ae337-d07d-41f6-926a-5845ddf4048e
md"
D) **Reciprocal update**
"

# ╔═╡ 9a8c9fe0-7749-4650-9a66-443faeeef75e
begin
	prior_sample_qld = simulate_from_prior(
	    100;
		model = RL_multi_2set_diff_recip,
		priors = Dict(
			:ρ => truncated(Normal(0., 3.), lower = 0.),
			:a => Normal(0., 0.5),
        	:Δa => Normal(-0.2, 0.5)
		),
		parameters = [:ρ, :a, :Δa],
		transformed = Dict(:a => :α),
		fixed_struct = pilot6_wm, 
		gq = true,
		random_seed = 123
	)
	
	f_qld = optimization_calibration(
		prior_sample_qld,
		optimize_multiple,
		estimate = "MAP",
		model = RL_multi_2set_diff_recip,
		priors = Dict(
			:ρ => truncated(Normal(0., 3.), lower = 0.),
			:a => Normal(0., 0.5),
        	:Δa => Normal(-0.2, 0.5)
		),
		parameters = [:ρ, :a, :Δa],
		transformed = Dict(:a => :α)		
	)

	f_qld
end

# ╔═╡ cee13f49-f2db-420a-a3fa-ce2e1f208649
let
	f = plot_prior_predictive_by_valence(
		prior_sample_qld,
		[:Q_optimal, :Q_suboptimal1, :Q_suboptimal2];
		choice_val = 3.0,
		ylab = "Q-value",
		fig_size = (1000, 1000),
		group = :set_size,
		legend = true,
		colors = Makie.colorschemes[:seaborn_pastel6]
	)	
	f
end

# ╔═╡ c5b1e95b-6478-4c3f-8610-f174511e818c
md"
### Working memory models
"

# ╔═╡ bc4f34ce-3fd2-4ae4-b308-21cfe823c828
md"
##### Single update
"

# ╔═╡ a90de11d-74b5-4919-8aef-6c72b9a78d16
md"
A) Palimpsest with **overall** capacity (sigmoid k=3)
"

# ╔═╡ 8fca6913-127b-4e71-b27b-fe70d4651d41
begin
	prior_sample_wma = simulate_from_prior(
	    100;
		model = WM_multi_all_outc_pmst_sgd,
		priors = Dict(
			:ρ => truncated(Normal(0., 3.), lower = 0.),
			:C => truncated(Normal(5., 3.), lower = 1.)
		),
		parameters = [:ρ, :C],
		fixed_struct = pilot6_wm, 
		gq = true,
		random_seed = 123
	)
	
	f_wma = optimization_calibration(
		prior_sample_wma,
		optimize_multiple,
		estimate = "MAP",
		model = WM_multi_all_outc_pmst_sgd,
		priors = Dict(
			:ρ => truncated(Normal(0., 3.), lower = 0.),
			:C => truncated(Normal(5., 3.), lower = 1.)
		),
		parameters = [:ρ, :C]
	)

	f_wma
end

# ╔═╡ 49f1711f-a219-45a0-8e8d-8e66956ad68c
let
	f = plot_prior_predictive_by_valence(
		prior_sample_wma,
		[:W_optimal, :W_suboptimal1, :W_suboptimal2];
		choice_val = 3.0,
		ylab = "W-value",
		fig_size = (1000, 1000),
		group = :set_size,
		legend = true,
		colors = Makie.colorschemes[:seaborn_pastel6]
	)	
	f
end

# ╔═╡ 9f0d7748-27f6-478e-8bc2-fd414d1b8e0a
md"
B) Palimpsest with **stimulus-specific** capacity (sigmoid k=3)
"

# ╔═╡ a713462d-6117-4949-8f14-cccdb0a2db4f
begin
	prior_sample_wmb = simulate_from_prior(
	    100;
		model = WM_multi_pmst_sgd,
		priors = Dict(
			:ρ => truncated(Normal(0., 3.), lower = 0.),
			:C => truncated(Normal(3., 2.), lower = 1.)
		),
		parameters = [:ρ, :C],
		fixed_struct = pilot6_wm, 
		gq = true,
		random_seed = 123
	)
	
	f_wms = optimization_calibration(
		prior_sample_wmb,
		optimize_multiple,
		estimate = "MAP",
		model = WM_multi_pmst_sgd,
		priors = Dict(
			:ρ => truncated(Normal(0., 3.), lower = 0.),
			:C => truncated(Normal(3., 2.), lower = 1.)
		),
		parameters = [:ρ, :C]
	)

	f_wms
end

# ╔═╡ f498156c-ca7e-42a7-84a4-0ca549ad89b5
let
	f = plot_prior_predictive_by_valence(
		prior_sample_wmb,
		[:W_optimal, :W_suboptimal1, :W_suboptimal2];
		choice_val = 3.0,
		ylab = "W-value",
		fig_size = (1000, 1000),
		group = :set_size,
		legend = true,
		colors = Makie.colorschemes[:seaborn_pastel6]
	)	
	f
end

# ╔═╡ d03367aa-63af-40f7-bb86-8446c156bd07
md"
C) Palimpsest with **overall** capacity (sigmoid k=3) + **no averaging**
"

# ╔═╡ c42179be-ef5b-427a-9ce2-8da19fd6414f
begin
	prior_sample_wmc = simulate_from_prior(
	    100;
		model = WM_multi_all_outc_pmst_sgd_sum,
		priors = Dict(
			:ρ => truncated(Normal(0., 3.), lower = 0.),
			:C => truncated(Normal(5., 3.), lower = 1.)
		),
		parameters = [:ρ, :C],
		fixed_struct = pilot6_wm, 
		gq = true,
		random_seed = 123
	)
	
	f_wmc = optimization_calibration(
		prior_sample_wmc,
		optimize_multiple,
		estimate = "MAP",
		model = WM_multi_all_outc_pmst_sgd_sum,
		priors = Dict(
			:ρ => truncated(Normal(0., 3.), lower = 0.),
			:C => truncated(Normal(5., 3.), lower = 1.)
		),
		parameters = [:ρ, :C]
	)

	f_wmc
end

# ╔═╡ 056844f3-eb07-4e9c-885e-7567b91cc60e
let
	f = plot_prior_predictive_by_valence(
		prior_sample_wmc,
		[:W_optimal, :W_suboptimal1, :W_suboptimal2];
		choice_val = 3.0,
		ylab = "W-value",
		fig_size = (1000, 1000),
		group = :set_size,
		legend = true,
		colors = Makie.colorschemes[:seaborn_pastel6]
	)	
	f
end

# ╔═╡ d9d18869-0b38-40cd-ae9b-6ab9c2415c77
md"
D) Palimpsest with **stimulus-specific** capacity (sigmoid k=3) + **no averaging**
"

# ╔═╡ 3ad40b7a-d801-4e3a-9156-fbaf4ce60b44
begin
	prior_sample_wmd = simulate_from_prior(
	    100;
		model = WM_multi_pmst_sgd_sum,
		priors = Dict(
			:ρ => truncated(Normal(0., 3.), lower = 0.),
			:C => truncated(Normal(3., 2), lower = 1.)
		),
		parameters = [:ρ, :C],
		fixed_struct = pilot6_wm, 
		gq = true,
		random_seed = 123
	)
	
	f_wmd = optimization_calibration(
		prior_sample_wmd,
		optimize_multiple,
		estimate = "MAP",
		model = WM_multi_pmst_sgd_sum,
		priors = Dict(
			:ρ => truncated(Normal(0., 3.), lower = 0.),
			:C => truncated(Normal(3., 2.), lower = 1.)
		),
		parameters = [:ρ, :C]
	)

	f_wmd
end

# ╔═╡ b0d809fa-11e1-4ba5-aa08-bf8123135a51
let
	f = plot_prior_predictive_by_valence(
		prior_sample_wmd,
		[:W_optimal, :W_suboptimal1, :W_suboptimal2];
		choice_val = 3.0,
		ylab = "W-value",
		fig_size = (1000, 1000),
		group = :set_size,
		legend = true,
		colors = Makie.colorschemes[:seaborn_pastel6]
	)	
	f
end

# ╔═╡ 73e5923c-016f-47d4-aa22-0bdc58e02d95
md"
##### Reciprocal update
"

# ╔═╡ ae3afa92-768e-4630-9abd-173dc7de0f90
md"
These models work by assuming the agent has some weak knowledge about the environment, namely what (other) possible outcomes there are (i.e. if they win/lose 50p, the other possibilities are 1p or £1).

If they get one of the larger outcomes, then we assume they update the other W-value based on winning/losing 1p. If they get 1p, we (currently) assume an update of *randomly* either 50p or £1 for the alternative option in that pair.
"

# ╔═╡ 06367719-7329-4572-a8eb-e68073b14747
md"
E) **Overall** capacity, with weights on alternative choice also updated
"

# ╔═╡ ca78df2a-b420-4f8b-acd8-0f5f2dc086b4
begin
	prior_sample_wme = simulate_from_prior(
	    100;
		model = WM_multi_all_outc_pmst_sgd_recip,
		priors = Dict(
			:ρ => truncated(Normal(0., 3.), lower = 0.),
			:C => truncated(Normal(5., 3.), lower = 1.)
		),
		parameters = [:ρ, :C],
		fixed_struct = pilot6_wm, 
		gq = true,
		random_seed = 123
	)
	
	f_wme = optimization_calibration(
		prior_sample_wme,
		optimize_multiple,
		estimate = "MAP",
		model = WM_multi_all_outc_pmst_sgd_recip,
		priors = Dict(
			:ρ => truncated(Normal(0., 3.), lower = 0.),
			:C => truncated(Normal(5., 3.), lower = 1.)
		),
		parameters = [:ρ, :C]
	)

	f_wme
end

# ╔═╡ aa36ca5d-fb2d-4d03-a257-5c6f89c70042
let
	f = plot_prior_predictive_by_valence(
		prior_sample_wme,
		[:W_optimal, :W_suboptimal1, :W_suboptimal2];
		choice_val = 3.0,
		ylab = "W-value",
		fig_size = (1000, 1000),
		group = :set_size,
		legend = true,
		colors = Makie.colorschemes[:seaborn_pastel6]
	)
	f
end

# ╔═╡ a6325c60-c248-485c-b51e-583e33358a94
md"
F) **Stimulus-specific** capacity, with weights on alternative choice also updated
"

# ╔═╡ 0e50ba82-0f3c-4328-b2eb-042769543345
begin
	prior_sample_wmf = simulate_from_prior(
	    100;
		model = WM_multi_pmst_sgd_recip,
		priors = Dict(
			:ρ => truncated(Normal(0., 3.), lower = 0.),
			:C => truncated(Normal(3., 2.), lower = 1.)
		),
		parameters = [:ρ, :C],
		fixed_struct = pilot6_wm, 
		gq = true,
		random_seed = 123
	)
	
	f_wmf = optimization_calibration(
		prior_sample_wmf,
		optimize_multiple,
		estimate = "MAP",
		model = WM_multi_pmst_sgd_recip,
		priors = Dict(
			:ρ => truncated(Normal(0., 3.), lower = 0.),
			:C => truncated(Normal(3., 2.), lower = 1.)
		),
		parameters = [:ρ, :C]
	)

	f_wmf
end

# ╔═╡ 878bb7f6-6dc1-49a2-b68b-aa56914e41e3
let
	f = plot_prior_predictive_by_valence(
		prior_sample_wmf,
		[:W_optimal, :W_suboptimal1, :W_suboptimal2];
		choice_val = 3.0,
		ylab = "W-value",
		fig_size = (1000, 1000),
		group = :set_size,
		legend = true,
		colors = Makie.colorschemes[:seaborn_pastel6]
	)
	f
end

# ╔═╡ fb2d42a6-d35f-4fc2-8a2d-32545f7abdb6
md"
G) **Overall** capacity, reciprocal, **no averaging**
"

# ╔═╡ 2ff08a9f-4158-46db-a459-f7d4b58ea727
begin
	prior_sample_wmg = simulate_from_prior(
	    100;
		model = WM_multi_all_outc_pmst_sgd_sum_recip,
		priors = Dict(
			:ρ => truncated(Normal(0., 3.), lower = 0.),
			:C => truncated(Normal(5., 3.), lower = 1.)
		),
		parameters = [:ρ, :C],
		fixed_struct = pilot6_wm, 
		gq = true,
		random_seed = 123
	)
	
	f_wmg = optimization_calibration(
		prior_sample_wmg,
		optimize_multiple,
		estimate = "MAP",
		model = WM_multi_all_outc_pmst_sgd_sum_recip,
		priors = Dict(
			:ρ => truncated(Normal(0., 3.), lower = 0.),
			:C => truncated(Normal(5., 3.), lower = 1.)
		),
		parameters = [:ρ, :C]
	)

	f_wmg
end

# ╔═╡ 1f563765-4152-4ebd-b05b-bd841c496652
let
	f = plot_prior_predictive_by_valence(
		prior_sample_wmg,
		[:W_optimal, :W_suboptimal1, :W_suboptimal2];
		choice_val = 3.0,
		ylab = "W-value",
		fig_size = (1000, 1000),
		group = :set_size,
		legend = true,
		colors = Makie.colorschemes[:seaborn_pastel6]
	)
	f
end

# ╔═╡ 6a33a65d-fc04-4f0c-b70c-0a73ce8089b5
md"
H) **Stimulus-specific** capacity, reciprocal, **no averaging**
"

# ╔═╡ 243707f3-4c16-407a-b39e-95b120b9db62
begin
	prior_sample_wmh = simulate_from_prior(
	    100;
		model = WM_multi_pmst_sgd_sum_recip,
		priors = Dict(
			:ρ => truncated(Normal(0., 3.), lower = 0.),
			:C => truncated(Normal(3., 2.), lower = 1.)
		),
		parameters = [:ρ, :C],
		fixed_struct = pilot6_wm, 
		gq = true,
		random_seed = 123
	)
	
	f_wmh = optimization_calibration(
		prior_sample_wmh,
		optimize_multiple,
		estimate = "MAP",
		model = WM_multi_pmst_sgd_sum_recip,
		priors = Dict(
			:ρ => truncated(Normal(0., 3.), lower = 0.),
			:C => truncated(Normal(3., 2.), lower = 1.)
		),
		parameters = [:ρ, :C]
	)

	f_wmh
end

# ╔═╡ 02b2c48b-34c0-4c24-bced-2f39dc795d03
let
	f = plot_prior_predictive_by_valence(
		prior_sample_wmh,
		[:W_optimal, :W_suboptimal1, :W_suboptimal2];
		choice_val = 3.0,
		ylab = "W-value",
		fig_size = (1000, 1000),
		group = :set_size,
		legend = true,
		colors = Makie.colorschemes[:seaborn_pastel6]
	)
	f
end

# ╔═╡ 904de860-9ef7-4952-bde3-03d786e6fab1
# ╠═╡ disabled = true
#=╠═╡
md"
### Can we implement this in a RLWM model?
"
  ╠═╡ =#

# ╔═╡ 08d92d11-15ba-49cd-95d5-5464f46625c3
# ╠═╡ disabled = true
#=╠═╡
md"
A) **With averaging**
"
  ╠═╡ =#

# ╔═╡ 4b0449b8-fc3f-4971-a019-732012b83cd2
# ╠═╡ disabled = true
#=╠═╡
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
  ╠═╡ =#

# ╔═╡ f195d5fb-de42-482a-8ffd-a960b5ad2055
#=╠═╡
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
  ╠═╡ =#

# ╔═╡ bde84efd-4b47-4d0d-b2d6-60ede47de1e2
md"
B) **Without averaging**
"

# ╔═╡ b8311355-6ff0-4ab5-b4ab-7ee855b3dc38
# ╠═╡ disabled = true
#=╠═╡
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
  ╠═╡ =#

# ╔═╡ 4fab7905-02a6-4a19-b485-88d181e86bf4
#=╠═╡
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
  ╠═╡ =#

# ╔═╡ Cell order:
# ╟─67f6ff9d-024d-48af-8eda-3d90d7fdf92d
# ╠═8b660dfb-a035-4726-ae47-0bd2b0d09ed6
# ╠═50b996f0-e737-4c2b-9822-201f9d14fc71
# ╟─bb7a4106-29e0-4711-8086-a0df85d7225a
# ╟─bdf5c2d9-90b8-4895-b0a5-708ac0094b79
# ╟─ae020d75-e72b-4706-a778-9af63d77fb94
# ╠═c953d1c0-24cb-4593-94b4-5b7f2cb6f2cd
# ╠═1d8be900-42c0-4e31-a44d-df91f4af29e7
# ╟─953d2e6f-ab5f-4479-a6e1-2c6970d15120
# ╠═e4121c93-97ef-47b8-a916-331d06a8ec40
# ╠═9f759ab5-2930-498f-9222-d8e675ef62b8
# ╟─57c4724b-c3b3-49bc-ba63-e4ce36ad3168
# ╟─a38dd46b-4d22-48cf-94b8-a319a0845704
# ╠═0dd99525-73f4-458e-9b62-bc7f8e885ac7
# ╠═9657c2af-1981-4282-99f2-7c1b55886ce9
# ╟─aa9ae337-d07d-41f6-926a-5845ddf4048e
# ╠═9a8c9fe0-7749-4650-9a66-443faeeef75e
# ╠═cee13f49-f2db-420a-a3fa-ce2e1f208649
# ╟─c5b1e95b-6478-4c3f-8610-f174511e818c
# ╟─bc4f34ce-3fd2-4ae4-b308-21cfe823c828
# ╟─a90de11d-74b5-4919-8aef-6c72b9a78d16
# ╠═8fca6913-127b-4e71-b27b-fe70d4651d41
# ╠═49f1711f-a219-45a0-8e8d-8e66956ad68c
# ╟─9f0d7748-27f6-478e-8bc2-fd414d1b8e0a
# ╠═a713462d-6117-4949-8f14-cccdb0a2db4f
# ╠═f498156c-ca7e-42a7-84a4-0ca549ad89b5
# ╟─d03367aa-63af-40f7-bb86-8446c156bd07
# ╠═c42179be-ef5b-427a-9ce2-8da19fd6414f
# ╠═056844f3-eb07-4e9c-885e-7567b91cc60e
# ╟─d9d18869-0b38-40cd-ae9b-6ab9c2415c77
# ╠═3ad40b7a-d801-4e3a-9156-fbaf4ce60b44
# ╠═b0d809fa-11e1-4ba5-aa08-bf8123135a51
# ╟─73e5923c-016f-47d4-aa22-0bdc58e02d95
# ╟─ae3afa92-768e-4630-9abd-173dc7de0f90
# ╟─06367719-7329-4572-a8eb-e68073b14747
# ╠═ca78df2a-b420-4f8b-acd8-0f5f2dc086b4
# ╠═aa36ca5d-fb2d-4d03-a257-5c6f89c70042
# ╟─a6325c60-c248-485c-b51e-583e33358a94
# ╠═0e50ba82-0f3c-4328-b2eb-042769543345
# ╠═878bb7f6-6dc1-49a2-b68b-aa56914e41e3
# ╟─fb2d42a6-d35f-4fc2-8a2d-32545f7abdb6
# ╠═2ff08a9f-4158-46db-a459-f7d4b58ea727
# ╠═1f563765-4152-4ebd-b05b-bd841c496652
# ╟─6a33a65d-fc04-4f0c-b70c-0a73ce8089b5
# ╠═243707f3-4c16-407a-b39e-95b120b9db62
# ╠═02b2c48b-34c0-4c24-bced-2f39dc795d03
# ╟─904de860-9ef7-4952-bde3-03d786e6fab1
# ╟─08d92d11-15ba-49cd-95d5-5464f46625c3
# ╠═4b0449b8-fc3f-4971-a019-732012b83cd2
# ╠═f195d5fb-de42-482a-8ffd-a960b5ad2055
# ╟─bde84efd-4b47-4d0d-b2d6-60ede47de1e2
# ╠═b8311355-6ff0-4ab5-b4ab-7ee855b3dc38
# ╠═4fab7905-02a6-4a19-b485-88d181e86bf4
