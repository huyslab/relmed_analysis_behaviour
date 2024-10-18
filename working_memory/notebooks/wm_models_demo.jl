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

# ╔═╡ 961a4264-9cd1-48d6-b169-80588c07e98c
include("$(pwd())/working_memory/RL+RLWM_models.jl")

# ╔═╡ 2fd2db62-49a5-4cae-8169-919c748bfc95
md"
## Simulate a task structure
"

# ╔═╡ 3fb61265-25f8-456d-8e4c-5e6d2064467a
begin
	random_task = create_random_task(;
	    n_blocks = 18, n_trials = 10, n_confusing = 2, set_sizes = [2, 4, 6]
	)
	chce = fill(missing, nrow(random_task))
	
	nothing
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
let
	f_ql = optimization_calibration(
		prior_sample_ql,
		optimize_multiple,
		estimate = "MAP"
	)	
	f_ql
end

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
			:ρ => truncated(Normal(0., 2), lower = 0.),
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
let
	f_ql_rec = optimization_calibration(
		prior_sample_ql_rec,
		optimize_multiple,
		estimate = "MAP"
	)	
	f_ql_rec
end

# ╔═╡ 4de24130-d8dd-476e-91af-998d2c57f760
md"
## Collins & Frank RLWM model
"

# ╔═╡ 8a0b4210-0525-4485-ad5e-1478948f7d7a
let
	m = RLWM_ss(
		unpack_data(random_task),
		chce;
		priors = Dict(
	        :ρ => truncated(Normal(0., 1.), lower = 0.),
	        :a => Normal(0., 0.5),
	        :F_wm => Normal(0., 0.5),
	        :W => Normal(0., 0.5),
	        :C => truncated(Normal(3., 2.), lower = 1.)
	    )
	)
	c = sample(m, Prior(), 500)
	gq = generated_quantities(m, c)
	ll_df = DataFrame(
		α = a2α.(c[:, :a, 1]),
		ρ = c[:, :ρ, 1],
		C = c[:, :C, 1],
		w = a2α.(c[:, :W, 1]),
		φ_wm = a2α.(c[:, :F_wm, 1]),
		loglike = [pt.loglike for pt in gq] |> vec
	)
	# Set the labels for the axes
	f = Figure(size = (1000, 300))
	
	labs1 = (xlabel = "α", ylabel = "ρ", zlabel = "log-likelihood")
	labs2 = (xlabel = "C", ylabel = "w", zlabel = "log-likelihood")
	labs3 = (xlabel = "C", ylabel = "φ_wm", zlabel = "log-likelihood")
	
	ax1, ax2, ax3 = Axis3(f[1,1]; labs1...), Axis3(f[1,2]; labs2...), Axis3(f[1,3]; labs3...)
	scatter!(ax1, ll_df.α, ll_df.ρ, ll_df.loglike)
	scatter!(ax2, ll_df.C, ll_df.w, ll_df.loglike)
	scatter!(ax3, ll_df.C, ll_df.φ_wm, ll_df.loglike)
	f
end

# ╔═╡ 7afd9cab-a329-4839-9b56-32fa439df50e
begin
	prior_sample_rlwm = simulate_from_prior(
		100;
		model = RLWM_ss,
		priors = Dict(
			:ρ => truncated(Normal(0., 2.), lower = 0.),
			:a => Normal(0., 0.5),
			:F_wm => Normal(0., 0.5),
			:W => Normal(0., 0.5),
			:C => truncated(Normal(3., 2.), lower = 1.)
		),
		transformed = Dict(:a => :α, :F_wm => :φ_wm, :W => :w),
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
let
	f_rlwm = optimization_calibration(
		prior_sample_rlwm,
		optimize_multiple,
		estimate = "MAP",
		model = RLWM_ss,
		transformed = Dict(:a => :α, :F_wm => :φ_wm, :W => :w),
		priors = Dict(
			:ρ => truncated(Normal(0., 2.), lower = 0.),
			:a => Normal(0., 0.5),
			:F_wm => Normal(0., 0.5),
			:W => Normal(0., 0.5),
			:C => truncated(Normal(3., 2.), lower = 1.)
		)
	)

	f_rlwm
end

# ╔═╡ 85024085-eb50-4698-a7c4-0b51097cac81
md"
## Palimpsest RLWM model
"

# ╔═╡ 2ec1567e-86cc-4fe5-894e-81651cec5cc1
let
	m = RLWM_pmst(
		unpack_data(random_task), chce;
		priors = Dict(
			:ρ => truncated(Normal(0., 2.), lower = 0.),
			:a => Normal(0., 0.5),
			Symbol("W[2]") => Normal(0., 0.5),
			Symbol("W[4]") => Normal(0., 0.5),
			Symbol("W[6]") => Normal(0., 0.5),
			:C => truncated(Normal(8., 4.), lower = 1.)
		)
	)
	c = sample(m, Prior(), 500)
	gq = generated_quantities(m, c)
	ll_df = DataFrame(
		α = a2α.(c[:, :a, 1]),
		ρ = c[:, :ρ, 1],
		C = c[:, :C, 1],
		w2 = a2α.(c[:, Symbol("W[2]"), 1]),
		w4 = a2α.(c[:, Symbol("W[4]"), 1]),
		w6 = a2α.(c[:, Symbol("W[6]"), 1]),
		loglike = [pt.loglike for pt in gq] |> vec
	)
	# Set the labels for the axes
	f = Figure(size = (1000, 300))
	
	labs1 = (xlabel = "α", ylabel = "ρ", zlabel = "log-likelihood")
	labs2 = (xlabel = "C", ylabel = "w2", zlabel = "log-likelihood")
	labs3 = (xlabel = "w4", ylabel = "w6", zlabel = "log-likelihood")
	
	ax1, ax2, ax3 = Axis3(f[1,1]; labs1...), Axis3(f[1,2]; labs2...), Axis3(f[1,3]; labs3...)
	scatter!(ax1, ll_df.α, ll_df.ρ, ll_df.loglike)
	scatter!(ax2, ll_df.C, ll_df.w2, ll_df.loglike)
	scatter!(ax3, ll_df.w4, ll_df.w6, ll_df.loglike)
	f
end

# ╔═╡ 4ca11ee5-0135-4618-9125-ffa4303f15f1
begin
	prior_sample_pmstwm = simulate_from_prior(
		100;
		model = RLWM_pmst,
		priors = Dict(
			:ρ => truncated(Normal(0., 2.), lower = 0.),
			:a => Normal(0., 0.5),
			Symbol("W[2]") => Normal(0., 0.5),
			Symbol("W[4]") => Normal(0., 0.5),
			Symbol("W[6]") => Normal(0., 0.5),
			:C => truncated(Normal(4., 2.), lower = 1.)
		),
		parameters = [:ρ, :a, Symbol("W[2]"), Symbol("W[4]"), Symbol("W[6]"), :C],
		transformed = Dict(
			:a => :α, Symbol("W[2]") => :w2, Symbol("W[4]") => :w4, Symbol("W[6]") => :w6
		),
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
let
	f_pmst = optimization_calibration(
		prior_sample_pmstwm,
		optimize_multiple,
		estimate = "MAP",
		model = RLWM_pmst,
		priors = Dict(
			:ρ => truncated(Normal(0., 2.), lower = 0.),
			:a => Normal(0., 0.5),
			Symbol("W[2]") => Normal(0., 0.5),
			Symbol("W[4]") => Normal(0., 0.5),
			Symbol("W[6]") => Normal(0., 0.5),
			:C => truncated(Normal(4., 2.), lower = 1.)
		),
		parameters = [:ρ, :a, Symbol("W[2]"), Symbol("W[4]"), Symbol("W[6]"), :C],
		transformed = Dict(
			:a => :α, Symbol("W[2]") => :w2, Symbol("W[4]") => :w4, Symbol("W[6]") => :w6
		)
	)
	f_pmst
end

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
		w = a2α.(c[:, :W, 1]),
		loglike = [pt.loglike for pt in gq] |> vec
	)	
	# Set the labels for the axes
	f = Figure(size = (1000, 300))
	
	labs1 = (xlabel = "α", ylabel = "ρ", zlabel = "log-likelihood")
	labs2 = (xlabel = "C", ylabel = "w", zlabel = "log-likelihood")
	
	ax1, ax2 = Axis3(f[1,1]; labs1...), Axis3(f[1,2]; labs2...)
	scatter!(ax1, ll_df.α, ll_df.ρ, ll_df.loglike)
	scatter!(ax2, ll_df.C, ll_df.w, ll_df.loglike)
	f
end

# ╔═╡ 003dbd73-315c-4a65-8c16-25b00d67052f
begin
	prior_sample_pmstsg = simulate_from_prior(
	    100;
		model = RLWM_pmst_sgd,
		priors = Dict(
			:ρ => truncated(Normal(0., 2.), lower = 0.),
			:a => Normal(0., 0.5),
			:W => Normal(0., 0.5),
			:C => truncated(Normal(4., 2.), lower = 1.)
		),
		parameters = [:ρ, :a, :W, :C],
		transformed = Dict(:a => :α, :W => :w),
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
let
	f_pmst = optimization_calibration(
		prior_sample_pmstsg,
		optimize_multiple,
		estimate = "MAP",
		model = RLWM_pmst_sgd,
		priors = Dict(
			:ρ => truncated(Normal(0., 2.), lower = 0.),
			:a => Normal(0., 0.5),
			:W => Normal(0., 0.5),
			:C => truncated(Normal(4., 2.), lower = 1.)
		),
		parameters = [:ρ, :a, :W, :C],
		transformed = Dict(:a => :α, :W => :w)
	)

	f_pmst
end

# ╔═╡ 464b052a-82be-4358-89d8-f1885773edf5
md"
### Palimpsest model with non-specific capacity
"

# ╔═╡ e036e2b3-a01b-480e-b62d-f8c2be1de4f5
let
	m = RLWM_all_outc_pmst_sgd(unpack_data(random_task), chce)
	c = sample(m, Prior(), 500)
	gq = generated_quantities(m, c)
	ll_df = DataFrame(
		α = a2α.(c[:, :a, 1]),
		ρ = c[:, :ρ, 1],
		C = c[:, :C, 1],
		w = a2α.(c[:, :W, 1]),
		loglike = [pt.loglike for pt in gq] |> vec
	)	
	# Set the labels for the axes
	f = Figure(size = (1000, 300))
	
	labs1 = (xlabel = "α", ylabel = "ρ", zlabel = "log-likelihood")
	labs2 = (xlabel = "C", ylabel = "w", zlabel = "log-likelihood")
	
	ax1, ax2 = Axis3(f[1,1]; labs1...), Axis3(f[1,2]; labs2...)
	scatter!(ax1, ll_df.α, ll_df.ρ, ll_df.loglike)
	scatter!(ax2, ll_df.C, ll_df.w, ll_df.loglike)
	f
end

# ╔═╡ e20dcac6-c949-4d20-973b-3333516cd12e
begin
	prior_sample_allO = simulate_from_prior(
	    100;
		model = RLWM_all_outc_pmst_sgd,
		priors = Dict(
			:ρ => truncated(Normal(0., 2.), lower = 0.),
			:a => Normal(0., 0.5),
			:W => Normal(0., 0.5),
			:C => truncated(Normal(6., 4.), lower = 1.)
		),
		parameters = [:ρ, :a, :W, :C],
		transformed = Dict(:a => :α, :W => :w),
		fixed_struct = random_task,
		# structure = (
        #     n_blocks = 48, n_trials = 13, n_confusing = 3, set_sizes = [2, 4, 6]
		# ),
		gq = true,
		random_seed = 123
	)
	describe(prior_sample_allO)
end

# ╔═╡ 8235f9e0-7110-49ac-ae43-1a765583402d
let
	f = plot_prior_predictive_by_valence(
		prior_sample_allO,
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

# ╔═╡ dd86b930-f737-4431-a1d8-755415d8e1e0
let
	f_pmst_allO = optimization_calibration(
		prior_sample_allO,
		optimize_multiple,
		estimate = "MAP",
		model = RLWM_all_outc_pmst_sgd,
		priors = Dict(
			:ρ => truncated(Normal(0., 2.), lower = 0.),
			:a => Normal(0., 0.5),
			:W => Normal(0., 0.5),
			:C => truncated(Normal(6., 4.), lower = 1.)
		),
		parameters = [:ρ, :a, :W, :C],
		transformed = Dict(:a => :α, :W => :w)
	)

	f_pmst_allO
end

# ╔═╡ b109be02-5849-4d76-a06a-78c99a70ca64
md"
### Working memory alone
"

# ╔═╡ 79fddeaf-b66b-4098-9a15-8a73ecbf8b2b
let
	m = WM_all_outc_pmst_sgd(unpack_data(random_task), chce)
	c = sample(m, Prior(), 500)
	gq = generated_quantities(m, c)
	ll_df = DataFrame(
		ρ = c[:, :ρ, 1],
		C = c[:, :C, 1],
		loglike = [pt.loglike for pt in gq] |> vec
	)
	# Set the labels for the axes
	f = Figure(size = (1000, 300))
	
	labs1 = (xlabel = "C", ylabel = "ρ", zlabel = "log-likelihood")
	
	ax1 = Axis3(f[1,1]; labs1...)
	scatter!(ax1, ll_df.C, ll_df.ρ, ll_df.loglike)
	f
end

# ╔═╡ 8d19cf83-6ec8-42fc-8d79-689edaf2bdc9
begin
	prior_sample_wma = simulate_from_prior(
	    100;
		model = WM_all_outc_pmst_sgd,
		priors = Dict(
			:ρ => truncated(Normal(0., 2.), lower = 0.),
			:C => truncated(Normal(3., 2), lower = 1.)
		),
		parameters = [:ρ, :C],
		fixed_struct = random_task,
		# structure = (
        #     n_blocks = 48, n_trials = 13, n_confusing = 3, set_sizes = [2, 4, 6]
		# ),
		gq = true,
		random_seed = 123
	)
	describe(prior_sample_wma)
end

# ╔═╡ d30fb160-ecbb-490f-b0a8-3901ef61ca46
let
	using CategoricalArrays
 	sort!(prior_sample_wma, :C)
	prior_sample_wma.capacity = categorical(round.(prior_sample_wma.C))
	f = plot_prior_predictive_by_valence(
		prior_sample_wma,
		[:W_optimal, :W_suboptimal];
		ylab = "W-value",
		fig_size = (1000, 1000),
		group = :capacity,
		legend = true,
		colors = Makie.colorschemes[:YlGnBu]
	)	
	f
end

# ╔═╡ e1c6a1ef-2c19-4309-ac7d-c23f3a2cb51b
let
	f_wma = optimization_calibration(
		prior_sample_wma,
		optimize_multiple,
		estimate = "MAP",
		model = WM_all_outc_pmst_sgd,
		priors = Dict(
			:ρ => truncated(Normal(0., 2.), lower = 0.),
			:C => truncated(Normal(3., 2), lower = 1.)
		),
		parameters = [:ρ, :C]
	)

	f_wma
end

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
# ╠═961a4264-9cd1-48d6-b169-80588c07e98c
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
# ╟─464b052a-82be-4358-89d8-f1885773edf5
# ╠═e036e2b3-a01b-480e-b62d-f8c2be1de4f5
# ╠═e20dcac6-c949-4d20-973b-3333516cd12e
# ╠═8235f9e0-7110-49ac-ae43-1a765583402d
# ╠═dd86b930-f737-4431-a1d8-755415d8e1e0
# ╟─b109be02-5849-4d76-a06a-78c99a70ca64
# ╠═79fddeaf-b66b-4098-9a15-8a73ecbf8b2b
# ╠═8d19cf83-6ec8-42fc-8d79-689edaf2bdc9
# ╠═d30fb160-ecbb-490f-b0a8-3901ef61ca46
# ╠═e1c6a1ef-2c19-4309-ac7d-c23f3a2cb51b
