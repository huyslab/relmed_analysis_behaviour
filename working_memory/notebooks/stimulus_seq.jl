### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ ef1a09ce-d1b5-11ef-2994-91cdad7ede58
begin
	cd("/home/jovyan")
	import Pkg
	
	# activate the shared project environment
    Pkg.activate("$(pwd())/relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
		using CairoMakie, Random, DataFrames, Distributions, StatsBase,
		ForwardDiff, LinearAlgebra, JSON, CSV, JLD2, AlgebraOfGraphics, Dates, Turing, HypothesisTests
		using LogExpFunctions: logistic, logit
	
	Turing.setprogress!(false)
	
	include("$(pwd())/working_memory/simulate.jl")
	include("$(pwd())/working_memory/collins_RLWM.jl")
	include("$(pwd())/working_memory/plotting_utils.jl")
	include("$(pwd())/working_memory/model_utils.jl")
	#include("$(pwd())/fetch_preprocess_data.jl")

	# Load data
	#_, _, _, _, _, WM_data, _, _ = load_pilot6_data()

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

# ╔═╡ 82326a87-0cba-4027-97f7-1e0fa60e1549
begin
	function uniform_chi_squared(df::DataFrame, mdel::Int64)
	    # Simple chi-squared helper
	    chisq(o, e) = sum((o .- e).^2 .// e)
		pds = 1:mdel
		cto = countmap(df.delay)
		observed = [get(cto, d, 0) for d in pds]
		expected = fill(nrow(df)÷length(pds), length(pds))
	    return chisq(observed, expected)
	end
	
	function best_seed_for_delays(;
		ns::Int, mc::Int, d::Float64, ntries::Int, ntol::Int, md::Int
	)
	    best_seed  = -1
	    best_score = Inf
		best_tol = -1
	
	    for s in 1:ntries #rand(1:1000000, ntries)
			for tol in 1:ntol
			    df = generate_delay_sequence(;
					no_sets=ns,
				    max_count=mc,
				    difficulty=d,
				    seed=s,
				    tolerance=tol,
				    max_delay=md,
					return_struct=false
				)
		        # Example "score": mean absolute deviation from target "delay_g"
		        score = uniform_chi_squared(df, md)
		        if score < best_score
		            best_score = score
		            best_seed  = s
					best_tol = tol
		        end
			end
	    end
	
	    return best_seed, best_score, best_tol
	end
end

# ╔═╡ 2cfbbb2f-457a-40d3-b676-0eb9f99b805c
md"
### Optimise sequence
#####
Here we try to optimise for an approximately uniform distribution of delays, by looping over seeds and tolerances and picking that with the lowest chi-squared test score compared to a perfectly uniform distribution of delay counts.
"

# ╔═╡ 3461ed6b-fa89-4bd9-b94a-6ef55f3bdf2c
begin
    se, nc, dd, md = 7, 24, .3, 11
	bse, bsc, bst = best_seed_for_delays(
		ns=se, mc=nc, d=dd, ntries=10000, ntol=5, md=md
	)
end

# ╔═╡ 878db40c-4b7a-45bb-bddd-a24d08202359
begin
	ts = Dict()
	ts["rew_detm"] = generate_delay_sequence(
		no_sets=se,
		max_count=nc,
		difficulty=dd,
		seed=bse,
		tolerance=bst,
		max_delay=md,
		return_struct=true,
		n_confusing=0,
		n_options = 3, # number of options to choose from
	    coins = [0.01, 1.0],
	    punish = false
	)
	ts["rew_int_detm"] = generate_delay_sequence(
		no_sets=se,
		max_count=nc,
		difficulty=dd,
		seed=bse,
		tolerance=bst,
		max_delay=md,
		return_struct=true,
		n_confusing=0,
		n_options = 3, # number of options to choose from
	    coins = [0.01, 0.5, 1.0],
	    punish = false
	)
	# ts["rew_pun_detm"] = generate_delay_sequence(
	# 	no_sets=se,
	# 	max_count=nc,
	# 	difficulty=dd,
	# 	seed=bse,
	# 	tolerance=bst,
	# 	return_struct=true,
	# 	n_confusing=0,
	# 	n_options = 3, # number of options to choose from
	#     coins = [0.01, 1.0],
	#     punish = true
	# )
	# ts["rew_pun_int_detm"] = generate_delay_sequence(
	# 	no_sets=se,
	# 	max_count=nc,
	# 	difficulty=dd,
	# 	seed=bse,
	# 	tolerance=bst,
	# 	return_struct=true,
	# 	n_confusing=0,
	# 	n_options = 3, # number of options to choose from
	#     coins = [0.01, 0.5, 1.0],
	#     punish = true
	# )
	# ts["rew_conf"] = generate_delay_sequence(
	# 	no_sets=se,
	# 	max_count=nc,
	# 	difficulty=dd,
	# 	seed=bse,
	# 	tolerance=bst,
	# 	return_struct=true,
	# 	n_confusing = floor(Int, 0.2*nc),
	# 	n_options = 3, # number of options to choose from
	#     coins = [0.01, 1.0],
	#     punish = false
	# )
	# ts["rew_int_conf"] = generate_delay_sequence(
	# 	no_sets=se,
	# 	max_count=nc,
	# 	difficulty=dd,
	# 	seed=bse,
	# 	tolerance=bst,
	# 	return_struct=true,
	# 	n_confusing = floor(Int, 0.2*nc),
	# 	n_options = 3, # number of options to choose from
	#     coins = [0.01, 0.5, 1.0],
	#     punish = false
	# )
	# ts["rew_pun_conf"] = generate_delay_sequence(
	# 	no_sets=se,
	# 	max_count=nc,
	# 	difficulty=dd,
	# 	seed=bse,
	# 	tolerance=bst,
	# 	return_struct=true,
	# 	n_confusing = floor(Int, 0.2*nc),
	# 	n_options = 3, # number of options to choose from
	#     coins = [0.01, 1.0],
	#     punish = true
	# )
	# ts["rew_pun_int_conf"] = generate_delay_sequence(
	# 	no_sets=se,
	# 	max_count=nc,
	# 	difficulty=dd,
	# 	seed=bse,
	# 	tolerance=bst,
	# 	return_struct=true,
	# 	n_confusing = floor(Int, 0.2*nc),
	# 	n_options = 3, # number of options to choose from
	#     coins = [0.01, 0.5, 1.0],
	#     punish = true
	# )
	for (task, df) in zip(keys(ts), values(ts))
       df[!, :task] .= task
    end;
	tasks = vcat(values(ts)...)
end

# ╔═╡ 2f9398ce-6520-454d-b131-2d32082a9850
let
	set_aog_theme!()
	update_theme!(fontsize=20)
	delay_seq = deepcopy(filter(x->x.task == "rew_detm", tasks))

	fig = Figure(size=(800, 300))
	delay_seq.nstim = string.(delay_seq.set_size)
	delay_pt = data(delay_seq) * mapping(:trial_ovl, :delay, color=:nstim)
	delay_freq = data(delay_seq) * frequency() * mapping(:delay)
		
	legend!(fig[1, 2], draw!(fig[1,1], delay_pt))
	draw!(fig[1, 3], delay_freq)
	fig
end

# ╔═╡ 0e288774-79bd-41bb-94be-911307356834
md"
### Comparison of parameter recovery
"

# ╔═╡ 0ec1b51c-64de-4dda-8257-06b3661c60ec
md"
#### Rewards alone
"

# ╔═╡ 5ccea4c6-b470-49e7-92b4-4588a3283025
md"
**Deterministic; rewards ∈ {0, 1}**
"

# ╔═╡ 24b87114-c3d2-4e69-8cd4-a08a82ea0c5f
begin
	set_theme!(th)
	prior_sample_t1 = simulate_from_prior(
	    100;
		model = HLWM_collins_continuous,
		priors = Dict(
			:β => 50., # fixed inverse temperature
	        :a => Normal(0., 2.), # RL reward learning rate
			:F_wm => Normal(0., 2.), # working memory forgetting rate
	        :w0 => Beta(2, 2), # prop. of WM to RL weight (i.e., 0.5 ===)
		),
		parameters = [:a_pos, :F_wm, :w0],
		fixed_struct = filter(x->x.task == "rew_detm", tasks),
		gq = true,
		random_seed = 1
	)
	f_hlwm_t1 = optimization_calibration(
		prior_sample_t1,
		optimize_multiple,
		estimate = "MAP",
		model = HLWM_collins_continuous,
		priors = Dict(
			:β => 50., # fixed inverse temperature
	        :a => Normal(0., 4.), # RL reward learning rate
			:F_wm => Normal(0., 4.), # working memory forgetting rate
	        :w0 => Beta(1.1, 1.1), # prop. of WM to RL weight (i.e., 0.5 ===)
		),
		parameters = [:a_pos, :F_wm, :w0],
		n_starts = 5
	)

	f_hlwm_t1
end

# ╔═╡ f153aa3b-985d-494f-bb8b-e1446d0633e9
let
	tstrct = filter(x->x.task == "rew_detm", tasks)
	df = leftjoin(
		prior_sample_t1, tstrct;
		on=[:block, :trial, :valence, :stimset, :set_size, :feedback_optimal, :feedback_suboptimal1, :feedback_suboptimal2],
		order=:left
	)

	f = Figure(size=(1000, 300))
	plot_sim_q_value_acc!(
		f,
		df;
		colA = [:Q_optimal, :W_optimal],
		colB = [[:Q_suboptimal1, :Q_suboptimal2], [:W_suboptimal1, :W_suboptimal2]],
		choice_val = 3.0,
		ylab = ("Q-value", "W-value"),
		group = nothing,
		norm = nothing,
		legend = false,
		colors = Makie.colorschemes[:seaborn_pastel]
	)
	f
end

# ╔═╡ 4a211644-d646-4465-ab7f-a30f49df897f
md"
**Deterministic; rewards ∈ {0, 0.5, 1}**
"

# ╔═╡ 10799432-c9dc-4023-a3e6-ca5b4c65ddc6
begin
	set_theme!(th)
	prior_sample_t2 = simulate_from_prior(
	    100;
		model = HLWM_collins_continuous,
		priors = Dict(
			:β => 50., # fixed inverse temperature
	        :a => Normal(0., 2.), # RL reward learning rate
			:F_wm => Normal(0., 2.), # working memory forgetting rate
	        :w0 => Beta(2, 2), # prop. of WM to RL weight (i.e., 0.5 ===)
		),
		parameters = [:a_pos, :F_wm, :w0],
		fixed_struct = filter(x->x.task == "rew_int_detm", tasks),
		gq = true,
		random_seed = 1
	)
	f_hlwm_t2 = optimization_calibration(
		prior_sample_t2,
		optimize_multiple,
		estimate = "MAP",
		model = HLWM_collins_continuous,
		priors = Dict(
			:β => 50., # fixed inverse temperature
	        :a => Normal(0., 4.), # RL reward learning rate
			:F_wm => Normal(0., 4.), # working memory forgetting rate
	        :w0 => Beta(1.1, 1.1), # prop. of WM to RL weight (i.e., 0.5 ===)
		),
		parameters = [:a_pos, :F_wm, :w0],
		n_starts = 5
	)

	f_hlwm_t2
end

# ╔═╡ e8afb04e-cee8-4a36-b74b-c9a667b1e760
let
	tstrct = filter(x->x.task == "rew_int_detm", tasks)
	df = leftjoin(
		prior_sample_t2, tstrct;
		on=[:block, :trial, :trial_ovl, :valence, :stimset, :set_size, :feedback_optimal, :feedback_suboptimal1, :feedback_suboptimal2],
		order=:left
	)

	f = Figure(size=(1000, 300))
	plot_sim_q_value_acc!(
		f,
		df;
		colA = [:Q_optimal, :W_optimal],
		colB = [[:Q_suboptimal1, :Q_suboptimal2], [:W_suboptimal1, :W_suboptimal2]],
		choice_val = 3.0,
		ylab = ("Q-value", "W-value"),
		group = nothing,
		norm = nothing,
		legend = false,
		colors = Makie.colorschemes[:seaborn_pastel]
	)
	f
end

# ╔═╡ 3527b3e3-bf97-45fa-b962-0c66410f1acb
# ╠═╡ disabled = true
#=╠═╡
begin
	fseq = filter(x->x.task == "rew_int_detm", tasks)
	fseq_tr = fseq[:, 1:end-2]
	CSV.write("wm_stimulus_sequence_longer.csv", fseq_tr)
end
  ╠═╡ =#

# ╔═╡ f00b9a42-70d6-44d7-8438-5efeb7b67585
md"
**Probabilistic (0.8/0.2); rewards ∈ {0, 1}**
"

# ╔═╡ 4c5c7c1f-df1d-49f9-9202-388e4eb60d9d
# ╠═╡ disabled = true
#=╠═╡
begin
	set_theme!(th)
	prior_sample_t3 = simulate_from_prior(
	    100;
		model = HLWM_collins_continuous,
		priors = Dict(
			:β => 50., # fixed inverse temperature
	        :a => Normal(0., 2.), # RL reward learning rate
			:F_wm => Normal(0., 2.), # working memory forgetting rate
	        :w0 => Beta(2, 2), # prop. of WM to RL weight (i.e., 0.5 ===)
		),
		parameters = [:a_pos, :F_wm, :w0],
		fixed_struct = filter(x->x.task == "rew_conf", tasks),
		gq = true,
		random_seed = 1
	)
	f_hlwm_t3 = optimization_calibration(
		prior_sample_t3,
		optimize_multiple,
		estimate = "MAP",
		model = HLWM_collins_continuous,
		priors = Dict(
			:β => 50., # fixed inverse temperature
	        :a => Normal(0., 4.), # RL reward learning rate
			:F_wm => Normal(0., 4.), # working memory forgetting rate
	        :w0 => Beta(1.1, 1.1), # prop. of WM to RL weight (i.e., 0.5 ===)
		),
		parameters = [:a_pos, :F_wm, :w0],
		n_starts = 5
	)

	f_hlwm_t3
end
  ╠═╡ =#

# ╔═╡ 3436bfb4-2850-4fd0-9529-06c8e37c244d
#=╠═╡
let
	tstrct = filter(x->x.task == "rew_conf", tasks)
	df = leftjoin(
		prior_sample_t3, tstrct;
		on=[:block, :trial, :valence, :stimset, :set_size, :feedback_optimal, :feedback_suboptimal1, :feedback_suboptimal2],
		order=:left
	)

	f = Figure(size=(1000, 300))
	plot_sim_q_value_acc!(
		f,
		df;
		colA = [:Q_optimal, :W_optimal],
		colB = [[:Q_suboptimal1, :Q_suboptimal2], [:W_suboptimal1, :W_suboptimal2]],
		choice_val = 3.0,
		ylab = ("Q-value", "W-value"),
		group = nothing,
		norm = nothing,
		legend = false,
		colors = Makie.colorschemes[:seaborn_pastel]
	)
	f
end
  ╠═╡ =#

# ╔═╡ c6dc958b-6dc3-41b9-9d77-69fcad2b2960
md"
**Probabilistic (0.8/0.2); rewards ∈ {0, 0.5, 1}**
"

# ╔═╡ 4b57fb04-4cb8-42d3-8b5b-531e57b87124
# ╠═╡ disabled = true
#=╠═╡
begin
	set_theme!(th)
	prior_sample_t4 = simulate_from_prior(
	    100;
		model = HLWM_collins_continuous,
		priors = Dict(
			:β => 50., # fixed inverse temperature
	        :a => Normal(0., 2.), # RL reward learning rate
			:F_wm => Normal(0., 2.), # working memory forgetting rate
	        :w0 => Beta(2, 2), # prop. of WM to RL weight (i.e., 0.5 ===)
		),
		parameters = [:a_pos, :F_wm, :w0],
		fixed_struct = filter(x->x.task == "rew_int_conf", tasks),
		gq = true,
		random_seed = 1
	)
	f_hlwm_t4 = optimization_calibration(
		prior_sample_t4,
		optimize_multiple,
		estimate = "MAP",
		model = HLWM_collins_continuous,
		priors = Dict(
			:β => 50., # fixed inverse temperature
	        :a => Normal(0., 4.), # RL reward learning rate
			:F_wm => Normal(0., 4.), # working memory forgetting rate
	        :w0 => Beta(1.1, 1.1), # prop. of WM to RL weight (i.e., 0.5 ===)
		),
		parameters = [:a_pos, :F_wm, :w0],
		n_starts = 5
	)

	f_hlwm_t4
end
  ╠═╡ =#

# ╔═╡ 9d239aa3-9b09-4115-8a05-c9eb46be97d0
#=╠═╡
let
	tstrct = filter(x->x.task == "rew_int_conf", tasks)
	df = leftjoin(
		prior_sample_t4, tstrct;
		on=[:block, :trial, :valence, :stimset, :set_size, :feedback_optimal, :feedback_suboptimal1, :feedback_suboptimal2],
		order=:left
	)

	f = Figure(size=(1000, 300))
	plot_sim_q_value_acc!(
		f,
		df;
		colA = [:Q_optimal, :W_optimal],
		colB = [[:Q_suboptimal1, :Q_suboptimal2], [:W_suboptimal1, :W_suboptimal2]],
		choice_val = 3.0,
		ylab = ("Q-value", "W-value"),
		group = nothing,
		norm = nothing,
		legend = false,
		colors = Makie.colorschemes[:seaborn_pastel]
	)
	f
end
  ╠═╡ =#

# ╔═╡ 7964d3d9-def8-4be1-b847-02e308c00761
md"
#### Rewards and punishments
"

# ╔═╡ 45aeb5cb-f5da-408f-816b-68af3841831c
md"
**Deterministic; rewards ∈ {0, 1}**
"

# ╔═╡ 727f5c63-541b-4c40-ba3c-aa9c4138f2aa
# ╠═╡ disabled = true
#=╠═╡
begin
	set_theme!(th)
	prior_sample_t5 = simulate_from_prior(
	    100;
		model = HLWM_collins_continuous,
		priors = Dict(
			:β => 50., # fixed inverse temperature
	        :a => Normal(0., 2.), # RL reward learning rate
			:F_wm => Normal(0., 2.), # working memory forgetting rate
	        :w0 => Beta(2, 2), # prop. of WM to RL weight (i.e., 0.5 ===)
		),
		parameters = [:a_pos, :F_wm, :w0],
		fixed_struct = filter(x->x.task == "rew_pun_detm", tasks),
		gq = true,
		random_seed = 1
	)
	f_hlwm_t5 = optimization_calibration(
		prior_sample_t5,
		optimize_multiple,
		estimate = "MAP",
		model = HLWM_collins_continuous,
		priors = Dict(
			:β => 50., # fixed inverse temperature
	        :a => Normal(0., 4.), # RL reward learning rate
			:F_wm => Normal(0., 4.), # working memory forgetting rate
	        :w0 => Beta(1.1, 1.1), # prop. of WM to RL weight (i.e., 0.5 ===)
		),
		parameters = [:a_pos, :F_wm, :w0],
		n_starts = 5
	)

	f_hlwm_t5
end
  ╠═╡ =#

# ╔═╡ bbe49b34-7bf7-40e8-8b16-273d10f8444b
#=╠═╡
let
	tstrct = filter(x->x.task == "rew_pun_detm", tasks)
	df = leftjoin(
		prior_sample_t5, tstrct;
		on=[:block, :trial, :valence, :stimset, :set_size, :feedback_optimal, :feedback_suboptimal1, :feedback_suboptimal2],
		order=:left
	)

	f = plot_prior_predictive_by_valence(
		df,
		[:Q_optimal, :Q_suboptimal1, :Q_suboptimal2];
		W_cols = [:W_optimal, :W_suboptimal1, :W_suboptimal2],
		choice_val = 3.0,
		ylab = ("Q-value", "W-value"),
		fig_size = (1000, 1000),
		group = nothing,
		norm = nothing,
		legend = false,
		colors = Makie.colorschemes[:seaborn_pastel6]
	)
	f
end
  ╠═╡ =#

# ╔═╡ 72e4bf3a-165b-42b5-8fd3-f6af3934e032
md"
**Deterministic; rewards ∈ {0, 0.5, 1}**
"

# ╔═╡ 2233a230-f1a9-424b-8159-d2db32ea16f1
# ╠═╡ disabled = true
#=╠═╡
begin
	set_theme!(th)
	prior_sample_t6 = simulate_from_prior(
	    100;
		model = HLWM_collins_continuous,
		priors = Dict(
			:β => 50., # fixed inverse temperature
	        :a => Normal(0., 2.), # RL reward learning rate
			:F_wm => Normal(0., 2.), # working memory forgetting rate
	        :w0 => Beta(2, 2), # prop. of WM to RL weight (i.e., 0.5 ===)
		),
		parameters = [:a_pos, :F_wm, :w0],
		fixed_struct = filter(x->x.task == "rew_pun_int_detm", tasks),
		gq = true,
		random_seed = 1
	)
	f_hlwm_t6 = optimization_calibration(
		prior_sample_t6,
		optimize_multiple,
		estimate = "MAP",
		model = HLWM_collins_continuous,
		priors = Dict(
			:β => 50., # fixed inverse temperature
	        :a => Normal(0., 4.), # RL reward learning rate
			:F_wm => Normal(0., 4.), # working memory forgetting rate
	        :w0 => Beta(1.1, 1.1), # prop. of WM to RL weight (i.e., 0.5 ===)
		),
		parameters = [:a_pos, :F_wm, :w0],
		n_starts = 5
	)

	f_hlwm_t6
end
  ╠═╡ =#

# ╔═╡ f29af358-5c23-4574-90ab-16b6e313b562
#=╠═╡
let
	tstrct = filter(x->x.task == "rew_pun_int_detm", tasks)
	df = leftjoin(
		prior_sample_t6, tstrct;
		on=[:block, :trial, :valence, :stimset, :set_size, :feedback_optimal, :feedback_suboptimal1, :feedback_suboptimal2],
		order=:left
	)

	f = plot_prior_predictive_by_valence(
		df,
		[:Q_optimal, :Q_suboptimal1, :Q_suboptimal2];
		W_cols = [:W_optimal, :W_suboptimal1, :W_suboptimal2],
		choice_val = 3.0,
		ylab = ("Q-value", "W-value"),
		fig_size = (1000, 1000),
		group = nothing,
		norm = nothing,
		legend = false,
		colors = Makie.colorschemes[:seaborn_pastel]
	)
	f
end
  ╠═╡ =#

# ╔═╡ 5d04212e-6347-40f3-a428-78ce83852156
md"
**Probabilistic (0.8/0.2); rewards ∈ {0, 1}**
"

# ╔═╡ 3ba4aca4-1c88-4fcb-a4b9-25f6ba4b1d18
# ╠═╡ disabled = true
#=╠═╡
begin
	set_theme!(th)
	prior_sample_t7 = simulate_from_prior(
	    100;
		model = HLWM_collins_continuous,
		priors = Dict(
			:β => 50., # fixed inverse temperature
	        :a => Normal(0., 2.), # RL reward learning rate
			:F_wm => Normal(0., 2.), # working memory forgetting rate
	        :w0 => Beta(2, 2), # prop. of WM to RL weight (i.e., 0.5 ===)
		),
		parameters = [:a_pos, :F_wm, :w0],
		fixed_struct = filter(x->x.task == "rew_pun_conf", tasks),
		gq = true,
		random_seed = 1
	)
	f_hlwm_t7 = optimization_calibration(
		prior_sample_t7,
		optimize_multiple,
		estimate = "MAP",
		model = HLWM_collins_continuous,
		priors = Dict(
			:β => 50., # fixed inverse temperature
	        :a => Normal(0., 4.), # RL reward learning rate
			:F_wm => Normal(0., 4.), # working memory forgetting rate
	        :w0 => Beta(1.1, 1.1), # prop. of WM to RL weight (i.e., 0.5 ===)
		),
		parameters = [:a_pos, :F_wm, :w0],
		n_starts = 5
	)

	f_hlwm_t7
end
  ╠═╡ =#

# ╔═╡ 6da19384-68a1-4af1-9ecb-3a627ce87a24
#=╠═╡
let
	tstrct = filter(x->x.task == "rew_pun_conf", tasks)
	df = leftjoin(
		prior_sample_t7, tstrct;
		on=[:block, :trial, :valence, :stimset, :set_size, :feedback_optimal, :feedback_suboptimal1, :feedback_suboptimal2],
		order=:left
	)

	f = plot_prior_predictive_by_valence(
		df,
		[:Q_optimal, :Q_suboptimal1, :Q_suboptimal2];
		W_cols = [:W_optimal, :W_suboptimal1, :W_suboptimal2],
		choice_val = 3.0,
		ylab = ("Q-value", "W-value"),
		fig_size = (1000, 1000),
		group = nothing,
		norm = nothing,
		legend = false,
		colors = Makie.colorschemes[:seaborn_pastel]
	)
	f
end
  ╠═╡ =#

# ╔═╡ b85ee18a-2926-4728-abce-7eb1f932f0fb
md"
**Probabilistic (0.8/0.2); rewards ∈ {0, 0.5, 1}**
"

# ╔═╡ ea2d9308-e2cc-4f48-8ce4-278964e990bb
# ╠═╡ disabled = true
#=╠═╡
begin
	set_theme!(th)
	prior_sample_t8 = simulate_from_prior(
	    100;
		model = HLWM_collins_continuous,
		priors = Dict(
			:β => 50., # fixed inverse temperature
	        :a => Normal(0., 2.), # RL reward learning rate
			:F_wm => Normal(0., 2.), # working memory forgetting rate
	        :w0 => Beta(2, 2), # prop. of WM to RL weight (i.e., 0.5 ===)
		),
		parameters = [:a_pos, :F_wm, :w0],
		fixed_struct = filter(x->x.task == "rew_pun_int_conf", tasks),
		gq = true,
		random_seed = 1
	)
	f_hlwm_t8 = optimization_calibration(
		prior_sample_t8,
		optimize_multiple,
		estimate = "MAP",
		model = HLWM_collins_continuous,
		priors = Dict(
			:β => 50., # fixed inverse temperature
	        :a => Normal(0., 4.), # RL reward learning rate
			:F_wm => Normal(0., 4.), # working memory forgetting rate
	        :w0 => Beta(1.1, 1.1), # prop. of WM to RL weight (i.e., 0.5 ===)
		),
		parameters = [:a_pos, :F_wm, :w0],
		n_starts = 5
	)

	f_hlwm_t8
end
  ╠═╡ =#

# ╔═╡ 8f9c50e4-0045-4dc9-83df-0d3008ba8dd3
#=╠═╡
let
	tstrct = filter(x->x.task == "rew_pun_int_conf", tasks)
	df = leftjoin(
		prior_sample_t8, tstrct;
		on=[:block, :trial, :valence, :stimset, :set_size, :feedback_optimal, :feedback_suboptimal1, :feedback_suboptimal2],
		order=:left
	)

	f = plot_prior_predictive_by_valence(
		df,
		[:Q_optimal, :Q_suboptimal1, :Q_suboptimal2];
		W_cols = [:W_optimal, :W_suboptimal1, :W_suboptimal2],
		choice_val = 3.0,
		ylab = ("Q-value", "W-value"),
		fig_size = (1000, 1000),
		group = nothing,
		norm = nothing,
		legend = false,
		colors = Makie.colorschemes[:seaborn_pastel]
	)
	f
end
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═ef1a09ce-d1b5-11ef-2994-91cdad7ede58
# ╠═82326a87-0cba-4027-97f7-1e0fa60e1549
# ╟─2cfbbb2f-457a-40d3-b676-0eb9f99b805c
# ╠═3461ed6b-fa89-4bd9-b94a-6ef55f3bdf2c
# ╠═878db40c-4b7a-45bb-bddd-a24d08202359
# ╠═2f9398ce-6520-454d-b131-2d32082a9850
# ╟─0e288774-79bd-41bb-94be-911307356834
# ╟─0ec1b51c-64de-4dda-8257-06b3661c60ec
# ╟─5ccea4c6-b470-49e7-92b4-4588a3283025
# ╠═24b87114-c3d2-4e69-8cd4-a08a82ea0c5f
# ╠═f153aa3b-985d-494f-bb8b-e1446d0633e9
# ╟─4a211644-d646-4465-ab7f-a30f49df897f
# ╠═10799432-c9dc-4023-a3e6-ca5b4c65ddc6
# ╠═e8afb04e-cee8-4a36-b74b-c9a667b1e760
# ╠═3527b3e3-bf97-45fa-b962-0c66410f1acb
# ╟─f00b9a42-70d6-44d7-8438-5efeb7b67585
# ╠═4c5c7c1f-df1d-49f9-9202-388e4eb60d9d
# ╠═3436bfb4-2850-4fd0-9529-06c8e37c244d
# ╟─c6dc958b-6dc3-41b9-9d77-69fcad2b2960
# ╠═4b57fb04-4cb8-42d3-8b5b-531e57b87124
# ╠═9d239aa3-9b09-4115-8a05-c9eb46be97d0
# ╟─7964d3d9-def8-4be1-b847-02e308c00761
# ╟─45aeb5cb-f5da-408f-816b-68af3841831c
# ╠═727f5c63-541b-4c40-ba3c-aa9c4138f2aa
# ╠═bbe49b34-7bf7-40e8-8b16-273d10f8444b
# ╟─72e4bf3a-165b-42b5-8fd3-f6af3934e032
# ╠═2233a230-f1a9-424b-8159-d2db32ea16f1
# ╠═f29af358-5c23-4574-90ab-16b6e313b562
# ╟─5d04212e-6347-40f3-a428-78ce83852156
# ╠═3ba4aca4-1c88-4fcb-a4b9-25f6ba4b1d18
# ╠═6da19384-68a1-4af1-9ecb-3a627ce87a24
# ╟─b85ee18a-2926-4728-abce-7eb1f932f0fb
# ╠═ea2d9308-e2cc-4f48-8ce4-278964e990bb
# ╠═8f9c50e4-0045-4dc9-83df-0d3008ba8dd3
