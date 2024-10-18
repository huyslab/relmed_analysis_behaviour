### A Pluto.jl notebook ###
# v0.19.45

using Markdown
using InteractiveUtils

# ╔═╡ 4d396fea-8d2d-11ef-2100-0156aa928c6e
begin
	cd("/home/jovyan")
	import Pkg
	
	# activate the shared project environment
    Pkg.activate("$(pwd())/relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
		using CairoMakie, Random, DataFrames, Distributions, StatsBase,
		ForwardDiff, LinearAlgebra, JLD2, FileIO, CSV, Dates, JSON, RCall, Turing, ParetoSmooth, MCMCDiagnosticTools, Printf, CategoricalArrays
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

# ╔═╡ 271aa371-4d01-4857-a66c-368cd17e1981
begin
	PLT_data, test_data, vigour_data, jspsych_data = load_pilot2_data()
	PLT_data
	df = prepare_for_fit(PLT_data, pilot2=true)[1]
	# df = filter(gdf -> any(gdf.set_size .== 6), groupby(df, :prolific_pid); ungroup=true)
	df = DataFrames.transform(groupby(df, [:prolific_pid, :block, :pair]), eachindex => :trial; ungroup=true)
	df = filter(x -> x.PID != 27, df)
end

# ╔═╡ 03828753-65d8-42a1-8fbc-7c677a9a1323
md"
### Choices by set-size: pilot 2
"

# ╔═╡ 15be818f-de9d-44cb-b58a-daaef8e81ec7
let
	f = Figure(size = (700, 400))
	df.set_size = string.(df.set_size)
	plot_prior_accuracy!(
		f[1,1],
		df;
		group = :set_size,
		pid_col = :PID,
		acc_col = :choice,
		legend = true,
		# legend_rows = 1,
		colors = Makie.colorschemes[:seaborn_pastel6],
		error_band = "se"
	)
	f
end

# ╔═╡ 2e050589-6665-4236-9971-b7502b4a107c
md"
#### Simulate a task structure
"

# ╔═╡ 8522dedf-ca8d-4d71-a0b9-198d6d6a1e9e
begin
	random_task = create_random_task(;
	    n_blocks = 48, n_trials = 10, n_confusing = 2, set_sizes = [2, 4, 6]
	)
	chce = fill(missing, nrow(random_task))
	
	nothing
end

# ╔═╡ f22b0faf-f000-40b3-ad37-81cb71372aee
begin
	ps = []
	for r in [1, 3, 4, 5, 6, 9]
		psr = []
		for (i, c) in enumerate(2:8)
			p = simulate_from_prior(
			    1;
				model = WM_all_outc_pmst_sgd,
				priors = Dict(
					:ρ => Dirac(r),
					##:a => Dirac(0.2),
					##:W => Dirac(0.5),
					:C => Dirac(c)
				),
				parameters = [:ρ, :C],
				#parameters = [:ρ, :a, :W, :C],
				#transformed = Dict(:a => :α, :W => :w),
				fixed_struct = random_task,
				gq = true,
				random_seed = 123
			)
			push!(psr, p)
	    end
		psr = vcat(psr...)
		sort!(psr, :C)
		psr.capacity = categorical(psr.C)
		push!(ps, psr)
	end
end

# ╔═╡ 6a3d59f6-aad3-4643-b474-fef6cd26a2ab
md"
#### ρ=1
"

# ╔═╡ 0c5b1c3d-a1c6-4f21-916c-ec0b45481da6
let
	f = plot_prior_predictive_by_valence(
		ps[1],
		[:W_optimal, :W_suboptimal];
		ylab = "W-value",
		fig_size = (1000, 1000),
		group = :capacity,
		legend = true,
		colors = Makie.colorschemes[:YlGnBu]
	)	
	f
end

# ╔═╡ a230c54b-7357-4fa7-8295-0b884ea30ed4
let
	f = plot_prior_predictive_by_valence(
		ps[2],
		[:W_optimal, :W_suboptimal];
		ylab = "W-value",
		fig_size = (1000, 1000),
		group = :capacity,
		legend = true,
		colors = Makie.colorschemes[:YlGnBu]
	)	
	f
end

# ╔═╡ 45ae7ccd-50ef-4ac5-b55c-df01d393cd3c
md"
#### ρ=3
"

# ╔═╡ 49119241-d2d9-4185-9443-6da62cac0e76
let
	f = plot_prior_predictive_by_valence(
		ps[2],
		[:W_optimal, :W_suboptimal];
		ylab = "W-value",
		fig_size = (1000, 1000),
		group = :capacity,
		legend = true,
		colors = Makie.colorschemes[:YlGnBu]
	)	
	f
end

# ╔═╡ 0017deaa-53f4-4235-a3ed-793d1d269b28
md"
#### ρ=4
"

# ╔═╡ 38dc308e-58ca-4329-b590-9af4232c0321
let
	f = plot_prior_predictive_by_valence(
		ps[3],
		[:W_optimal, :W_suboptimal];
		ylab = "W-value",
		fig_size = (1000, 1000),
		group = :capacity,
		legend = true,
		colors = Makie.colorschemes[:YlGnBu]
	)	
	f
end

# ╔═╡ dc85af67-68e7-4c7b-9bcb-4d148e95b231
md"
#### ρ=5
"

# ╔═╡ b12ab057-c2fa-47c3-8bc0-64f0cbcedb6f
let
	f = plot_prior_predictive_by_valence(
		ps[4],
		[:W_optimal, :W_suboptimal];
		ylab = "W-value",
		fig_size = (1000, 1000),
		group = :capacity,
		legend = true,
		colors = Makie.colorschemes[:YlGnBu]
	)	
	f
end

# ╔═╡ 76013f04-1b81-4e5d-a323-870be8c956ac
md"
#### ρ=6
"

# ╔═╡ 50fde835-4853-4210-82b0-0733e73360fe
let
	f = plot_prior_predictive_by_valence(
		ps[5],
		[:W_optimal, :W_suboptimal];
		ylab = "W-value",
		fig_size = (1000, 1000),
		group = :capacity,
		legend = true,
		colors = Makie.colorschemes[:YlGnBu]
	)	
	f
end

# ╔═╡ 11938b5c-e400-4cc5-ac6b-f8f5b9f4f31c
md"
#### ρ=9
"

# ╔═╡ 14346941-ee69-4268-824f-0e881f9d8e7f
let
	f = plot_prior_predictive_by_valence(
		ps[6],
		[:W_optimal, :W_suboptimal];
		ylab = "W-value",
		fig_size = (1000, 1000),
		group = :capacity,
		legend = true,
		colors = Makie.colorschemes[:YlGnBu]
	)	
	f
end

# ╔═╡ c83ebf69-01a7-4fec-a4ca-0a95a508d657
md"
### Add RL
"

# ╔═╡ 172b5334-95df-437c-bc56-3940e3fb4b4c
begin
	psrl = []
	for r in [1, 3, 4, 5, 6, 9]
		psr = []
		for (i, c) in enumerate(2:8)
			p = simulate_from_prior(
			    1;
				model = RLWM_all_outc_pmst_sgd,
				priors = Dict(
					:ρ => Dirac(r),
					:a => Dirac(0.2),
					Symbol("W[2]") => Dirac(0.7),
					Symbol("W[4]") => Dirac(0.5),
					Symbol("W[6]") => Dirac(0.2),
					:C => Dirac(c)
				),
				parameters = [:ρ, :a, Symbol("W[2]"), Symbol("W[4]"), Symbol("W[6]"), :C],
				transformed = Dict(
					:a => :α, Symbol("W[2]") => :w2, Symbol("W[4]") => :w4, Symbol("W[6]") => :w6
				),
				fixed_struct = random_task,
				gq = true,
				random_seed = 123
			)
			push!(psr, p)
	    end
		psr = vcat(psr...)
		sort!(psr, :C)
		psr.capacity = categorical(psr.C)
		push!(psrl, psr)
	end
end

# ╔═╡ d89f693b-e266-488b-96ef-c8240a3a001c
md"
#### ρ=1
"

# ╔═╡ a114a277-b135-4f34-af5b-0ff857264a38
let
	f = plot_prior_predictive_by_valence(
		psrl[1],
		[:Q_optimal, :Q_suboptimal];
		W_cols = [:W_optimal, :W_suboptimal],
		ylab = ("Q-value", "W-value"),
		fig_size = (1000, 1000),
		group = :capacity,
		legend = true,
		colors = Makie.colorschemes[:YlGnBu]
	)	
	f
end

# ╔═╡ 444faed4-4891-4698-89df-5406b2d1a4b9
let
	f = plot_prior_predictive_by_valence(
		psrl[2],
		[:Q_optimal, :Q_suboptimal];
		W_cols = [:W_optimal, :W_suboptimal],
		ylab = ("Q-value", "W-value"),
		fig_size = (1000, 1000),
		group = :capacity,
		legend = true,
		colors = Makie.colorschemes[:YlGnBu]
	)	
	f
end

# ╔═╡ 05d363f7-e21a-4f56-9303-85a468d4a0e4
md"
#### ρ=3
"

# ╔═╡ 9142bbc2-8730-4104-a35d-7480fc3c4026
let
	f = plot_prior_predictive_by_valence(
		psrl[2],
		[:Q_optimal, :Q_suboptimal];
		W_cols = [:W_optimal, :W_suboptimal],
		ylab = ("Q-value", "W-value"),
		fig_size = (1000, 1000),
		group = :capacity,
		legend = true,
		colors = Makie.colorschemes[:YlGnBu]
	)	
	f
end

# ╔═╡ 9a4aa535-2196-4a66-a10c-2580a2ecafaa
md"
#### ρ=4
"

# ╔═╡ f831a4b2-0faa-438e-8cca-88b5e21ddf89
let
	f = plot_prior_predictive_by_valence(
		psrl[3],
		[:W_optimal, :W_suboptimal];
		ylab = "W-value",
		fig_size = (1000, 1000),
		group = :capacity,
		legend = true,
		colors = Makie.colorschemes[:YlGnBu]
	)	
	f
end

# ╔═╡ dcc088c8-7cb8-4733-bb49-e74379c4a6b8
md"
#### ρ=5
"

# ╔═╡ ded45294-d299-4438-b687-ecc860162fb5
let
	f = plot_prior_predictive_by_valence(
		psrl[4],
		[:Q_optimal, :Q_suboptimal];
		W_cols = [:W_optimal, :W_suboptimal],
		ylab = ("Q-value", "W-value"),
		fig_size = (1000, 1000),
		group = :capacity,
		legend = true,
		colors = Makie.colorschemes[:YlGnBu]
	)	
	f
end

# ╔═╡ cf5e18a9-ce36-450c-aeb6-218c52568cb6
md"
#### ρ=6
"

# ╔═╡ 94973231-4670-42fa-b5fe-8b2e670693e8
let
	f = plot_prior_predictive_by_valence(
		psrl[5],
		[:Q_optimal, :Q_suboptimal];
		W_cols = [:W_optimal, :W_suboptimal],
		ylab = ("Q-value", "W-value"),
		fig_size = (1000, 1000),
		group = :capacity,
		legend = true,
		colors = Makie.colorschemes[:YlGnBu]
	)	
	f
end

# ╔═╡ 09018b1d-e0d7-4f4a-9efd-972f6dc6e5bf
md"
#### ρ=9
"

# ╔═╡ 7a4f4024-a504-417b-b471-1b67c0848bc0
let
	f = plot_prior_predictive_by_valence(
		psrl[6],
		[:Q_optimal, :Q_suboptimal];
		W_cols = [:W_optimal, :W_suboptimal],
		ylab = ("Q-value", "W-value"),
		fig_size = (1000, 1000),
		group = :capacity,
		legend = true,
		colors = Makie.colorschemes[:YlGnBu]
	)	
	f
end

# ╔═╡ Cell order:
# ╠═4d396fea-8d2d-11ef-2100-0156aa928c6e
# ╠═271aa371-4d01-4857-a66c-368cd17e1981
# ╟─03828753-65d8-42a1-8fbc-7c677a9a1323
# ╠═15be818f-de9d-44cb-b58a-daaef8e81ec7
# ╟─2e050589-6665-4236-9971-b7502b4a107c
# ╠═8522dedf-ca8d-4d71-a0b9-198d6d6a1e9e
# ╠═f22b0faf-f000-40b3-ad37-81cb71372aee
# ╟─6a3d59f6-aad3-4643-b474-fef6cd26a2ab
# ╠═0c5b1c3d-a1c6-4f21-916c-ec0b45481da6
# ╠═a230c54b-7357-4fa7-8295-0b884ea30ed4
# ╟─45ae7ccd-50ef-4ac5-b55c-df01d393cd3c
# ╠═49119241-d2d9-4185-9443-6da62cac0e76
# ╟─0017deaa-53f4-4235-a3ed-793d1d269b28
# ╠═38dc308e-58ca-4329-b590-9af4232c0321
# ╟─dc85af67-68e7-4c7b-9bcb-4d148e95b231
# ╠═b12ab057-c2fa-47c3-8bc0-64f0cbcedb6f
# ╟─76013f04-1b81-4e5d-a323-870be8c956ac
# ╠═50fde835-4853-4210-82b0-0733e73360fe
# ╟─11938b5c-e400-4cc5-ac6b-f8f5b9f4f31c
# ╠═14346941-ee69-4268-824f-0e881f9d8e7f
# ╟─c83ebf69-01a7-4fec-a4ca-0a95a508d657
# ╠═172b5334-95df-437c-bc56-3940e3fb4b4c
# ╟─d89f693b-e266-488b-96ef-c8240a3a001c
# ╠═a114a277-b135-4f34-af5b-0ff857264a38
# ╠═444faed4-4891-4698-89df-5406b2d1a4b9
# ╟─05d363f7-e21a-4f56-9303-85a468d4a0e4
# ╠═9142bbc2-8730-4104-a35d-7480fc3c4026
# ╟─9a4aa535-2196-4a66-a10c-2580a2ecafaa
# ╠═f831a4b2-0faa-438e-8cca-88b5e21ddf89
# ╟─dcc088c8-7cb8-4733-bb49-e74379c4a6b8
# ╠═ded45294-d299-4438-b687-ecc860162fb5
# ╟─cf5e18a9-ce36-450c-aeb6-218c52568cb6
# ╠═94973231-4670-42fa-b5fe-8b2e670693e8
# ╟─09018b1d-e0d7-4f4a-9efd-972f6dc6e5bf
# ╠═7a4f4024-a504-417b-b471-1b67c0848bc0
