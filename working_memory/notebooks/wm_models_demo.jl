### A Pluto.jl notebook ###
# v0.19.45

using Markdown
using InteractiveUtils

# ╔═╡ 52da319e-824e-11ef-18dc-31b1ab4d2e75
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
	
	include("$(pwd())/simulate.jl")
	include("$(pwd())/turing_models.jl")
	include("$(pwd())/plotting_utils.jl")

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

# ╔═╡ 2fd2db62-49a5-4cae-8169-919c748bfc95
md"
## Simulate a task structure
"

# ╔═╡ 3fb61265-25f8-456d-8e4c-5e6d2064467a
begin
	random_task = create_random_task(;
	    n_blocks = 48, n_trials = 13, n_confusing = 3, set_sizes = [2, 4, 6]
	)
	chce = fill(missing, nrow(random_task))
end

# ╔═╡ 4dae9630-b32a-4f51-8ef3-d707bf260c5a
md"
## Basic Q-learning model
"

# ╔═╡ ac5e7807-db3c-43da-90a0-f49a4dbd195c
let
	m = RL_ss(unpack_data(random_task), chce)
	c = sample(m, Prior(), 500)
	gq = generated_quantities(m, c)
	ll_df = DataFrame(
		α = a2α.(c[:, :a, 1]),
		ρ = c[:, :ρ, 1],
		loglike = [pt.loglike for pt in gq] |> vec
	)
	f = Figure(size = (700, 400))
	labs = (xlabel = "α", ylabel = "ρ", zlabel = "log-likelihood")
	ax = Axis3(f[1,1]; labs...)
	scatter!(ax, eachcol(ll_df)...)
	f
end

# ╔═╡ ae42c592-1f69-414e-a263-bcf67934f65a
md"
### Simulated behaviour and Q-values
"

# ╔═╡ ba70f606-a366-4f9d-962a-a1245e5b0f06
begin
	prior_sample_ql = simulate_from_prior(
		100;
		model = RL_ss,
		priors = Dict(
			:ρ => truncated(Normal(0., 1.), lower = 0.),
			:a => Normal(0., 0.5)
		),
		transformed = Dict(:a => :α),
		fixed_struct = random_task,
		# structure = (
        #     n_blocks = 48, n_trials = 13, n_confusing = 3, set_sizes = [2, 4, 6]
		# ),
		gq = true,
		random_seed = 123
	)
	describe(prior_sample_ql)
end

# ╔═╡ 02716563-b8ab-4f0c-907d-a773b3618ee1
let
	f = plot_prior_predictive_by_valence(
		prior_sample_ql,
		[:Q_optimal, :Q_suboptimal];
		group = :set_size,
		legend = true,
		legend_title = "Set size",
		fig_size = (1000, 1000),
		colors = Makie.colorschemes[:seaborn_pastel6]
	)	
	f
end

# ╔═╡ 01740ca2-dda9-4c48-a52a-8c361d036b57
# ╠═╡ disabled = true
#=╠═╡
let
	f_rl = optimization_calibration(
		prior_sample_ql,
		optimize_multiple,
		estimate = "MAP"
	)	
	f_rl
end
  ╠═╡ =#

# ╔═╡ 5fb23da2-258c-4125-a349-4165cdf6afe2
md"
## Reciprocal Q-learning model
"

# ╔═╡ d2a118fa-700f-44b7-9cf4-dbe0f785784b
let
	m = RL_recip_ss(unpack_data(random_task), chce)
	c = sample(m, Prior(), 500)
	gq = generated_quantities(m, c)
	ll_df = DataFrame(
		α = a2α.(c[:, :a, 1]),
		ρ = c[:, :ρ, 1],
		loglike = [pt.loglike for pt in gq] |> vec
	)
	f = Figure(size = (700, 400))
	labs = (xlabel = "α", ylabel = "ρ", zlabel = "log-likelihood")
	ax = Axis3(f[1,1]; labs...)
	scatter!(ax, eachcol(ll_df)...)
	f
end

# ╔═╡ 94c92aa8-04ce-4208-9ef8-fd032b2b4177
begin
	prior_sample_ql_rec = simulate_from_prior(
		100;
		model = RL_recip_ss,
		priors = Dict(
			:ρ => truncated(Normal(0., 1.), lower = 0.),
			:a => Normal(0., 0.5)
		),
		transformed = Dict(:a => :α),
		fixed_struct = random_task,
		# structure = (
        #     n_blocks = 48, n_trials = 13, n_confusing = 3, set_sizes = [2, 4, 6]
		# ),
		gq = true,
		random_seed = 123
	)
	describe(prior_sample_ql_rec)
end

# ╔═╡ c874c7cb-422f-4330-8ed3-a35a47dcf5a3
let
	f = plot_prior_predictive_by_valence(
		prior_sample_ql_rec,
		[:Q_optimal, :Q_suboptimal];
		group = :set_size,
		legend = true,
		legend_title = "Set size",
		fig_size = (1000, 1000),
		colors = Makie.colorschemes[:seaborn_pastel6]
	)	
	f
end

# ╔═╡ fac0f8df-c3c5-4f7e-bd69-adbbaa8ab757
# ╠═╡ disabled = true
#=╠═╡
let
	f_rl = optimization_calibration(
		prior_sample_ql_rec,
		optimize_multiple,
		estimate = "MAP"
	)	
	f_rl
end
  ╠═╡ =#

# ╔═╡ 4de24130-d8dd-476e-91af-998d2c57f760
md"
## Collins & Frank RLWM model
"

# ╔═╡ 8a0b4210-0525-4485-ad5e-1478948f7d7a
let
	m = RLWM_ss(unpack_data(random_task), chce)
	c = sample(m, Prior(), 500)
	gq = generated_quantities(m, c)
	ll_df = DataFrame(
		α = a2α.(c[:, :a, 1]),
		ρ = c[:, :ρ, 1],
		C = c[:, :C, 1],
		w0 = a2α.(c[:, :W, 1]),
		φ_wm = a2α.(c[:, :F_wm, 1]),
		loglike = [pt.loglike for pt in gq] |> vec
	)
	# Set the labels for the axes
	f = Figure(size = (1000, 300))
	
	labs1 = (xlabel = "α", ylabel = "ρ", zlabel = "log-likelihood")
	labs2 = (xlabel = "C", ylabel = "w0", zlabel = "log-likelihood")
	labs3 = (xlabel = "C", ylabel = "φ_wm", zlabel = "log-likelihood")
	
	ax1, ax2, ax3 = Axis3(f[1,1]; labs1...), Axis3(f[1,2]; labs2...), Axis3(f[1,3]; labs3...)
	scatter!(ax1, ll_df.α, ll_df.ρ, ll_df.loglike)
	scatter!(ax2, ll_df.C, ll_df.w0, ll_df.loglike)
	scatter!(ax3, ll_df.C, ll_df.φ_wm, ll_df.loglike)
	f
end

# ╔═╡ 7afd9cab-a329-4839-9b56-32fa439df50e
begin
	prior_sample_rlwm = simulate_from_prior(
		100;
		model = RLWM_ss,
		priors = Dict(
			:ρ => truncated(Normal(0., 1.), lower = 0.),
			:a => Normal(0., 0.5),
			:F_wm => Normal(0., 0.5),
			:W => Normal(0., 0.5),
			:C => truncated(Normal(3., 2.), lower = 1.)
		),
		transformed = Dict(:a => :α, :F_wm => :φ_wm, :W => :w0),
		fixed_struct = random_task,
		# structure = (
        #     n_blocks = 48, n_trials = 13, n_confusing = 3, set_sizes = [2, 4, 6]
		# ),
		gq = true,
		random_seed = 123
	)
	describe(prior_sample_rlwm)
end

# ╔═╡ 1dbe2f06-7075-486d-a70e-52752b28e869
let
	f = plot_prior_predictive_by_valence(
		prior_sample_rlwm,
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

# ╔═╡ 2635c4bf-1941-46d8-b20c-de982c610a6b
# ╠═╡ disabled = true
#=╠═╡
let
	f_rlwm = optimization_calibration(
		prior_sample_rlwm,
		optimize_multiple,
		estimate = "MAP",
		model = RLWM_ss,
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
  ╠═╡ =#

# ╔═╡ 85024085-eb50-4698-a7c4-0b51097cac81
md"
## Palimpsest RLWM model
"

# ╔═╡ 2ec1567e-86cc-4fe5-894e-81651cec5cc1
let
	m = RLWM_pmst(unpack_data(random_task), chce)
	c = sample(m, Prior(), 500)
	gq = generated_quantities(m, c)
	ll_df = DataFrame(
		α = a2α.(c[:, :a, 1]),
		ρ = c[:, :ρ, 1],
		C = c[:, :C, 1],
		w0 = a2α.(c[:, :W, 1]),
		loglike = [pt.loglike for pt in gq] |> vec
	)
	# Set the labels for the axes
	f = Figure(size = (1000, 300))
	
	labs1 = (xlabel = "α", ylabel = "ρ", zlabel = "log-likelihood")
	labs2 = (xlabel = "C", ylabel = "w0", zlabel = "log-likelihood")
	
	ax1, ax2 = Axis3(f[1,1]; labs1...), Axis3(f[1,2]; labs2...)
	scatter!(ax1, ll_df.α, ll_df.ρ, ll_df.loglike)
	scatter!(ax2, ll_df.C, ll_df.w0, ll_df.loglike)
	f
end

# ╔═╡ 4ca11ee5-0135-4618-9125-ffa4303f15f1
begin
	prior_sample_pmstwm = simulate_from_prior(
		100;
		model = RLWM_pmst,
		priors = Dict(
			:ρ => truncated(Normal(0., 1.), lower = 0.),
			:a => Normal(0., 0.5),
			:W => Normal(0., 0.5),
			:C => truncated(Normal(3., 2.), lower = 1.)
		),
		transformed = Dict(:a => :α, :W => :w0),
		fixed_struct = random_task,
		# structure = (
        #     n_blocks = 48, n_trials = 13, n_confusing = 3, set_sizes = [2, 4, 6]
		# ),
		gq = true,
		random_seed = 123
	)
	describe(prior_sample_pmstwm)
end

# ╔═╡ d2fb5d7a-305b-42ef-a83f-7345f7c424e4
let
	f = plot_prior_predictive_by_valence(
		prior_sample_pmstwm,
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

# ╔═╡ 66136216-e21f-4229-b0a5-9186238e7da8
# ╠═╡ disabled = true
#=╠═╡
let
	f_pmst = optimization_calibration(
		prior_sample_pmstwm,
		optimize_multiple,
		estimate = "MAP",
		model = RLWM_pmst,
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
  ╠═╡ =#

# ╔═╡ 810da379-c8f1-4173-9d85-10d1631cfec0
md"
## Palimpsest RLWM model with sigmoid weighting
"

# ╔═╡ b6ee37b6-06d2-48b2-9d4b-cd04aa0938d7
let
	m = RLWM_pmst_sgd(unpack_data(random_task), chce)
	c = sample(m, Prior(), 500)
	gq = generated_quantities(m, c)
	ll_df = DataFrame(
		α = a2α.(c[:, :a, 1]),
		ρ = c[:, :ρ, 1],
		C = c[:, :C, 1],
		w0 = a2α.(c[:, :W, 1]),
		loglike = [pt.loglike for pt in gq] |> vec
	)	
	# Set the labels for the axes
	f = Figure(size = (1000, 300))
	
	labs1 = (xlabel = "α", ylabel = "ρ", zlabel = "log-likelihood")
	labs2 = (xlabel = "C", ylabel = "w0", zlabel = "log-likelihood")
	
	ax1, ax2 = Axis3(f[1,1]; labs1...), Axis3(f[1,2]; labs2...)
	scatter!(ax1, ll_df.α, ll_df.ρ, ll_df.loglike)
	scatter!(ax2, ll_df.C, ll_df.w0, ll_df.loglike)
	f
end

# ╔═╡ 003dbd73-315c-4a65-8c16-25b00d67052f
begin
	prior_sample_pmstsg = simulate_from_prior(
		100;
		model = RLWM_pmst_sgd,
		priors = Dict(
			:ρ => truncated(Normal(0., 1.), lower = 0.),
			:a => Normal(0., 0.5),
			:W => Normal(0., 0.5),
			:C => truncated(Normal(3., 2.), lower = 1.)
		),
		transformed = Dict(:a => :α, :W => :w0),
		fixed_struct = random_task,
		# structure = (
        #     n_blocks = 48, n_trials = 13, n_confusing = 3, set_sizes = [2, 4, 6]
		# ),
		gq = true,
		random_seed = 123
	)
	describe(prior_sample_pmstsg)
end

# ╔═╡ 6d10b413-9bf3-42e0-97a8-113d6f0bb228
let
	f = plot_prior_predictive_by_valence(
		prior_sample_pmstsg,
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

# ╔═╡ 01555619-dae0-4a7e-95f7-e17c82df8e1a
# ╠═╡ disabled = true
#=╠═╡
let
	f_pmst = optimization_calibration(
		prior_sample_pmstsg,
		optimize_multiple,
		estimate = "MAP",
		model = RLWM_pmst_sgd,
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
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═52da319e-824e-11ef-18dc-31b1ab4d2e75
# ╟─2fd2db62-49a5-4cae-8169-919c748bfc95
# ╠═3fb61265-25f8-456d-8e4c-5e6d2064467a
# ╟─4dae9630-b32a-4f51-8ef3-d707bf260c5a
# ╠═ac5e7807-db3c-43da-90a0-f49a4dbd195c
# ╟─ae42c592-1f69-414e-a263-bcf67934f65a
# ╠═ba70f606-a366-4f9d-962a-a1245e5b0f06
# ╠═02716563-b8ab-4f0c-907d-a773b3618ee1
# ╠═01740ca2-dda9-4c48-a52a-8c361d036b57
# ╟─5fb23da2-258c-4125-a349-4165cdf6afe2
# ╠═d2a118fa-700f-44b7-9cf4-dbe0f785784b
# ╠═94c92aa8-04ce-4208-9ef8-fd032b2b4177
# ╠═c874c7cb-422f-4330-8ed3-a35a47dcf5a3
# ╠═fac0f8df-c3c5-4f7e-bd69-adbbaa8ab757
# ╟─4de24130-d8dd-476e-91af-998d2c57f760
# ╠═8a0b4210-0525-4485-ad5e-1478948f7d7a
# ╠═7afd9cab-a329-4839-9b56-32fa439df50e
# ╠═1dbe2f06-7075-486d-a70e-52752b28e869
# ╠═2635c4bf-1941-46d8-b20c-de982c610a6b
# ╟─85024085-eb50-4698-a7c4-0b51097cac81
# ╠═2ec1567e-86cc-4fe5-894e-81651cec5cc1
# ╠═4ca11ee5-0135-4618-9125-ffa4303f15f1
# ╠═d2fb5d7a-305b-42ef-a83f-7345f7c424e4
# ╠═66136216-e21f-4229-b0a5-9186238e7da8
# ╟─810da379-c8f1-4173-9d85-10d1631cfec0
# ╠═b6ee37b6-06d2-48b2-9d4b-cd04aa0938d7
# ╠═003dbd73-315c-4a65-8c16-25b00d67052f
# ╠═6d10b413-9bf3-42e0-97a8-113d6f0bb228
# ╠═01555619-dae0-4a7e-95f7-e17c82df8e1a