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
		ForwardDiff, LinearAlgebra, Turing, CSV
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

# ╔═╡ 5b92e714-a7eb-11ef-2626-91c405db34d1
md"
### WM + habit learning 
"

# ╔═╡ 0f8e1ec4-4fa2-46d5-b364-8e1b3acb5920
begin
	# structure
	pilot6_wm = load_wm_structure_csv("pilot6_WM") # task structure for ppcs
	sess1_str = filter(x -> x.block <= 10, pilot6_wm)
	nothing
end

# ╔═╡ 901987c4-1d4e-43b6-a044-a4ffb7d8f928
md"
## Standard model
"

# ╔═╡ 72b105fc-564b-4a9c-ac4f-6cf20f5f0913
# ╠═╡ disabled = true
#=╠═╡
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
	        :F_wm => Normal(-2., 4.), # working memory forgetting rate
	        :w0 => Normal(2., 1.), # initial working memory weight
        	:C => truncated(Normal(4., 2.), lower=1, upper=7) # WM capacity
		),
		parameters = [:a_pos, :bRL, :bWM, :E, :F_wm, :w0, :C],
		fixed_struct = sess1_str,
		# structure = (
        #    n_blocks = 20, n_trials = 10, n_confusing = 0,
		#     set_sizes = [2, 4, 6], n_options = 3,
		#     coins = [-1.0, -0.01, 0.01, 1.0], punish = true
		# ),
		gq = true,
		random_seed = 1
	)
	nothing
end
  ╠═╡ =#

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
# ╠═╡ disabled = true
#=╠═╡
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
	        :F_wm => Normal(-2., 6.), # working memory forgetting rate
	        :w0 => Normal(2., 3.), # initial working memory weight
        	:C => truncated(Normal(4., 2.), lower=1, upper=7) # WM capacity
		),
		parameters = [:a_pos, :bRL, :bWM, :E, :F_wm, :w0, :C],
		n_starts = 5
	)

	f_rlwm
end
  ╠═╡ =#

# ╔═╡ 26d9b023-fcb8-471e-9dfd-dfe5f4ec3092
md"
## Habit model
"

# ╔═╡ 991c57ff-d36f-4259-9cd4-1f25732025d8
begin
	prior_sample_hlwm = simulate_from_prior(
	    100;
		model = HLWM_collins_mk2,
		priors = Dict(
			:β => 25., # fixed inverse temperature
	        :a => Normal(-2., 1.), # RL reward learning rate
	        :bWM => Normal(1., 1.), # punishment learning rate for working memory
			:E => Normal(-3., 1.), # punishment learning rate for RL
			:F_wm => Normal(-1., 1.), # working memory forgetting rate
	        :w0 => Beta(1., 2.) # prop. of WM to RL weight (i.e., 0.5 ===)
		),
		parameters = [:a_pos, :bWM, :E, :F_wm, :w0],
		# fixed_struct = sess1_str,
		structure = (
           n_blocks = 20, n_trials = 10, n_confusing = 0,
		    set_sizes = [2, 4, 6], n_options = 3,
		    coins = [-1.0, -0.01, 0.01, 1.0], punish = true
		),
		gq = true,
		random_seed = 1
	)
	nothing
end

# ╔═╡ 9bfa255f-50f5-4651-8835-9ad7e07909bd
# if we include punishment blocks, this plots them separately from reward
let
	f = plot_prior_predictive_by_valence(
		prior_sample_hlwm,
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

# ╔═╡ f05ef93f-a3b7-4fea-aa2a-70837fc9f8a0
begin
	f_hlwm = optimization_calibration(
		prior_sample_hlwm,
		optimize_multiple,
		estimate = "MAP",
		model = HLWM_collins_mk2,
		priors = Dict(
			:β => 25., # fixed inverse temperature
	        :a => Normal(0., 4.), # RL reward learning rate
	        :bWM => Normal(0., 4.), # punishment learning rate for working memory
			:E => Normal(0., 4.), # undirected noise
			:F_wm => Normal(0., 4.), # working memory forgetting rate
	        :w0 => Beta(1.1, 1.1), # prop. of WM to RL weight (i.e., 0.5 ===)
		),
		parameters = [:a_pos, :bWM, :E, :F_wm, :w0],
		n_starts = 5
	)

	f_hlwm
end

# ╔═╡ 10b5970b-a7af-479d-af7c-acd44f2a3792
md"
#### Reparameterised weights
"

# ╔═╡ 6ca64c1b-ada4-45cb-9d08-cc7b749d1c84
md"
##### Linear weight difference
"

# ╔═╡ b1398f9e-8097-41cf-99c8-50970b8356b6
# ╠═╡ disabled = true
#=╠═╡
begin
	prior_sample_hlwm_lnw = simulate_from_prior(
	    100;
		model = HLWM_linewt,
		priors = Dict(
	        :β => 25., # fixed inverse temperature
	        :a => Normal(-2., 1.), # RL reward learning rate
	        :bWM => Normal(1., 1.), # punishment learning rate for working memory
	        :E => Normal(-2., 1.), # undirected noise
	        :F_wm => Normal(-2., 4.), # working memory forgetting rate
	        :w0 => Normal(3., 1.), # initial working memory weight
			:Δw => Normal(1., 1.) # change in WM weight
		),
		parameters = [:a_pos, :bWM, :E, :F_wm, :w0, :Δw],
		fixed_struct = sess1_str,
		# structure = (
        #    n_blocks = 20, n_trials = 10, n_confusing = 0,
		#     set_sizes = [2, 4, 6], n_options = 3,
		#     coins = [-1.0, -0.01, 0.01, 1.0], punish = true
		# ),
		gq = true,
		random_seed = 123
	)
	nothing
end
  ╠═╡ =#

# ╔═╡ be69157a-fac5-4eba-af7f-2883ac8b6084
# ╠═╡ disabled = true
#=╠═╡
# if we include punishment blocks, this plots them separately from reward
let
	f = plot_prior_predictive_by_valence(
		prior_sample_hlwm_lnw,
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

# ╔═╡ b2c7318d-431e-4149-bdad-e1ac869f8569
# ╠═╡ disabled = true
#=╠═╡
begin
	f_hlwm_lnw = optimization_calibration(
		prior_sample_hlwm_lnw,
		optimize_multiple,
		estimate = "MAP",
		model = HLWM_linewt,
		priors = Dict(
	        :β => 25., # fixed inverse temperature
	        :a => Normal(-2., 3.), # RL reward learning rate
	        # :bWM => Normal(1., 3.), # punishment learning rate for working memory
	        :E => Normal(-2., 3.), # undirected noise
	        :F_wm => Normal(-2., 12.), # working memory forgetting rate
	        :w0 => Normal(3., 3.), # initial working memory weight
			:Δw => Normal(1., 3.) # change in WM weight
		),
		parameters = [:a_pos, :E, :F_wm, :w0, :Δw],
		n_starts = 5
	)

	f_hlwm_lnw
end
  ╠═╡ =#

# ╔═╡ 309d0ce6-8658-4e27-82bd-8bc8b75ac535
md"
##### Exponential weight difference
"

# ╔═╡ 411fd8ab-c8cf-425e-a519-842fe52b6d08
# ╠═╡ disabled = true
#=╠═╡
begin
	prior_sample_hlwm_expwt = simulate_from_prior(
	    100;
		model = HLWM_expwt,
		priors = Dict(
	        :β => 25., # fixed inverse temperature
	        :a => Normal(-2., 1.), # RL reward learning rate
	        :bWM => Normal(1., 1.), # punishment learning rate for working memory
	        :E => Normal(-2., 1.), # undirected noise
	        :F_wm => Normal(-2., 1.), # working memory forgetting rate
	        :w0 => Normal(3., 1.), # initial working memory weight
        	:C => Gamma(7., 1.) # WM capacity
		),
		parameters = [:a_pos, :bWM, :E, :F_wm, :w0, :C],
		fixed_struct = sess1_str,
		# structure = (
        #    n_blocks = 20, n_trials = 10, n_confusing = 0,
		#     set_sizes = [2, 4, 6], n_options = 3,
		#     coins = [-1.0, -0.01, 0.01, 1.0], punish = true
		# ),
		gq = true,
		random_seed = 123
	)
	nothing
end
  ╠═╡ =#

# ╔═╡ 521b22ea-b47a-40ce-b87f-0f699d5eb052
# ╠═╡ disabled = true
#=╠═╡
# if we include punishment blocks, this plots them separately from reward
let
	f = plot_prior_predictive_by_valence(
		prior_sample_hlwm_expwt,
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

# ╔═╡ 5b7fec05-759b-4716-9828-07fb15a151a6
# ╠═╡ disabled = true
#=╠═╡
begin
	f_hlwm_expwt = optimization_calibration(
		prior_sample_hlwm_expwt,
		optimize_multiple,
		estimate = "MAP",
		model = HLWM_expwt,
		priors = Dict(
	        :β => 25., # fixed inverse temperature
	        :a => Normal(-2., 3.), # RL reward learning rate
	        :bWM => Normal(1., 3.), # punishment learning rate for working memory
	        :E => Normal(-2., 3.), # undirected noise
	        :F_wm => Normal(-2., 3.), # working memory forgetting rate
	        :w0 => Normal(3., 3.), # initial working memory weight
        	:C => Gamma(7., 2.) # WM capacity
		),
		parameters = [:a_pos, :bWM, :E, :F_wm, :w0, :C],
		n_starts = 5
	)

	f_hlwm_expwt
end
  ╠═╡ =#

# ╔═╡ 0d8668ad-8cd4-4e36-88be-4c40e66dd5cf
md"
#### Noisy W-values?
"

# ╔═╡ ff5abd26-c373-4bc1-8bb4-7b62f1cd8bde
# ╠═╡ disabled = true
#=╠═╡
begin
	prior_sample_hlwm_noisy = simulate_from_prior(
	    100;
		model = HLWM_noise,
		priors = Dict(
	        :β => 25., # fixed inverse temperature
	        :a => Normal(-2., 1.), # RL reward learning rate
	        :bWM => Normal(1., 1.), # punishment learning rate for working memory
	        :E => Normal(-2., 1.), # undirected noise
	        :F_wm => Normal(-2., 1.), # working memory forgetting rate
	        :w0 => Normal(3., 1.), # initial working memory weight
        	:σ_wm => Exponential(1.) # WM noise which scales w/ set-size
		),
		parameters = [:a_pos, :bWM, :E, :F_wm, :w0, :σ_wm],
		fixed_struct = sess1_str,
		# structure = (
        #    n_blocks = 20, n_trials = 10, n_confusing = 0,
		#     set_sizes = [2, 4, 6], n_options = 3,
		#     coins = [-1.0, -0.01, 0.01, 1.0], punish = true
		# ),
		gq = true,
		random_seed = 123
	)
	nothing
end
  ╠═╡ =#

# ╔═╡ 11602f69-849b-4831-92eb-90b1d6de1e37
# ╠═╡ disabled = true
#=╠═╡
# if we include punishment blocks, this plots them separately from reward
let
	f = plot_prior_predictive_by_valence(
		prior_sample_hlwm_noisy,
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

# ╔═╡ 081579fa-df87-420c-a088-63a6e606f5e6
# ╠═╡ disabled = true
#=╠═╡
begin
	f_hlwm_noise = optimization_calibration(
		prior_sample_hlwm_noisy,
		optimize_multiple,
		estimate = "MAP",
		model = HLWM_noise,
		priors = Dict(
	        :β => 25., # fixed inverse temperature
	        :a => Normal(-2., 1.), # RL reward learning rate
	        :bWM => Normal(1., 1.), # punishment learning rate for working memory
	        :E => Normal(-2., 1.), # undirected noise
	        :F_wm => Normal(-2., 1.), # working memory forgetting rate
	        :w0 => Normal(3., 1.), # initial working memory weight
        	:σ_wm => Exponential(1.) # WM noise which scales w/ set-size
		),
		parameters = [:a_pos, :bWM, :E, :F_wm, :w0, :σ_wm],
		n_starts = 5
	)

	f_hlwm_noise
end
  ╠═╡ =#

# ╔═╡ 89e1051b-7ad9-453e-95c0-b757673130f0
md"
#### Forgetful models
"

# ╔═╡ 9215221b-8ac5-46e0-831d-8fed04fde387
begin
	prior_sample_hlwm_forget = simulate_from_prior(
	    100;
		model = HLWM_forget,
		priors = Dict(
			:β => 25., # fixed inverse temperature
	        :a => Normal(-2., 1.), # RL reward learning rate
	        :bWM => Normal(1., 1.), # punishment learning rate for working memory
			:F_wm => Normal(-1., 1.), # working memory forgetting rate
	        :w0 => Beta(1., 2.) # prop. of WM to RL weight (i.e., 0.5 ===)
		),
		parameters = [:a_pos, :bWM, :F_wm, :w0],
		# fixed_struct = sess1_str,
		structure = (
           n_blocks = 20, n_trials = 10, n_confusing = 0,
		    set_sizes = [2, 4, 6], n_options = 3,
		    coins = [-1.0, -0.01, 0.01, 1.0], punish = true
		),
		gq = true,
		random_seed = 888
	)
	nothing
end

# ╔═╡ d59db927-3654-41e1-861b-5497feb9e1b9
# if we include punishment blocks, this plots them separately from reward
let
	f = plot_prior_predictive_by_valence(
		prior_sample_hlwm_forget,
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

# ╔═╡ 2ea504d6-3d0f-41ff-a026-5aae6f93cf5d
begin
	f_hlwm_elig = optimization_calibration(
		prior_sample_hlwm_forget,
		optimize_multiple,
		estimate = "MAP",
		model = HLWM_forget,
		priors = Dict(
			:β => 25., # fixed inverse temperature
	        :a => Normal(0., 4.), # RL reward learning rate
	        :bWM => Normal(0., 4.), # punishment learning rate for working memory
			:F_wm => Normal(0., 4.), # working memory forgetting rate
	        :w0 => Beta(1.1, 1.1), # prop. of WM to RL weight (i.e., 0.5 ===)
		),
		parameters = [:a_pos, :bWM, :F_wm, :w0],
		n_starts = 5
	)

	f_hlwm_elig
end

# ╔═╡ d84a8aac-359f-465e-a05e-92765aab2aa3
md"
#### Using RL instead of habit?
"

# ╔═╡ 8f195665-9fee-441a-aa50-36453f706f90
begin
	prior_sample_rlwm_forget = simulate_from_prior(
	    100;
		model = RLWM_forget,
		priors = Dict(
			:β => 25., # fixed inverse temperature
	        :a => Normal(-2., 1.), # RL reward learning rate
			:bRL => Normal(-1., 1.), # punishment learning rate for RL
	        :bWM => Normal(1., 1.), # punishment learning rate for working memory
			:F_wm => Normal(-1., 1.), # working memory forgetting rate
	        :w0 => Beta(1., 2.) # prop. of WM to RL weight (i.e., 0.5 ===)
		),
		parameters = [:a_pos, :bRL, :bWM, :F_wm, :w0],
		#fixed_struct = sess1_str,
		structure = (
  		    n_blocks = 20, n_trials = 10, n_confusing = 0,
		    set_sizes = [2, 4, 6], n_options = 3,
		    coins = [-1.0, -0.01, 0.01, 1.0], punish = true
		),
		gq = true,
		random_seed = 123
	)
	nothing
end

# ╔═╡ 93fd8120-f8f7-4adf-962d-244315c04e1f
# if we include punishment blocks, this plots them separately from reward
let
	f = plot_prior_predictive_by_valence(
		prior_sample_rlwm_forget,
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

# ╔═╡ 4663cff3-8305-4a94-8cf0-722c896fb135
begin
	f_rlwm_elig = optimization_calibration(
		prior_sample_rlwm_forget,
		optimize_multiple,
		estimate = "MAP",
		model = RLWM_forget,
		priors = Dict(
			:β => 25., # fixed inverse temperature
	        :a => Normal(0., 4.), # RL reward learning rate
			:bRL => Normal(0., 4.), # punishment learning rate for RL
	        :bWM => Normal(0., 4.), # punishment learning rate for working memory
			:F_wm => Normal(0., 4.), # working memory forgetting rate
	        :w0 => Beta(1.1, 1.1), # prop. of WM to RL weight (i.e., 0.5 ===)
		),
		parameters = [:a_pos, :bRL, :bWM, :F_wm, :w0],
		n_starts = 5
	)

	f_rlwm_elig
end

# ╔═╡ Cell order:
# ╟─5b92e714-a7eb-11ef-2626-91c405db34d1
# ╠═6f90ca0b-4c72-42ac-9e67-02c2a05331b6
# ╠═0f8e1ec4-4fa2-46d5-b364-8e1b3acb5920
# ╟─901987c4-1d4e-43b6-a044-a4ffb7d8f928
# ╠═72b105fc-564b-4a9c-ac4f-6cf20f5f0913
# ╠═90bb80e0-d473-4eb7-8171-a02c191a13b8
# ╠═470de2e8-84ef-417b-bbec-35383a1feedc
# ╟─26d9b023-fcb8-471e-9dfd-dfe5f4ec3092
# ╠═991c57ff-d36f-4259-9cd4-1f25732025d8
# ╠═9bfa255f-50f5-4651-8835-9ad7e07909bd
# ╠═f05ef93f-a3b7-4fea-aa2a-70837fc9f8a0
# ╟─10b5970b-a7af-479d-af7c-acd44f2a3792
# ╟─6ca64c1b-ada4-45cb-9d08-cc7b749d1c84
# ╠═b1398f9e-8097-41cf-99c8-50970b8356b6
# ╠═be69157a-fac5-4eba-af7f-2883ac8b6084
# ╠═b2c7318d-431e-4149-bdad-e1ac869f8569
# ╟─309d0ce6-8658-4e27-82bd-8bc8b75ac535
# ╠═411fd8ab-c8cf-425e-a519-842fe52b6d08
# ╠═521b22ea-b47a-40ce-b87f-0f699d5eb052
# ╠═5b7fec05-759b-4716-9828-07fb15a151a6
# ╟─0d8668ad-8cd4-4e36-88be-4c40e66dd5cf
# ╠═ff5abd26-c373-4bc1-8bb4-7b62f1cd8bde
# ╠═11602f69-849b-4831-92eb-90b1d6de1e37
# ╠═081579fa-df87-420c-a088-63a6e606f5e6
# ╟─89e1051b-7ad9-453e-95c0-b757673130f0
# ╠═9215221b-8ac5-46e0-831d-8fed04fde387
# ╠═d59db927-3654-41e1-861b-5497feb9e1b9
# ╠═2ea504d6-3d0f-41ff-a026-5aae6f93cf5d
# ╟─d84a8aac-359f-465e-a05e-92765aab2aa3
# ╠═8f195665-9fee-441a-aa50-36453f706f90
# ╠═93fd8120-f8f7-4adf-962d-244315c04e1f
# ╠═4663cff3-8305-4a94-8cf0-722c896fb135
