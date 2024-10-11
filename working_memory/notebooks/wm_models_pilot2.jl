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
end

# ╔═╡ a0e0399e-a982-4bc4-856f-2e712d1e182b
begin
	include("$(pwd())/working_memory/RL+RLWM_models.jl")
	pmst_ests, pmst_choices = optimize_multiple(
		df;
		model = RLWM_pmst,
		estimate = "MAP",
		include_true=false,
		initial_params = [mean(truncated(Normal(0., 2.), lower = 0.)), 0.5, 0.5, mean(truncated(Normal(2., 2.), lower = 1.))],
		transformed = Dict(:a => :α, :W => :w0),
		priors = Dict(
			:ρ => truncated(Normal(0., 1.), lower = 0.),
			:a => Normal(0., 0.5),
			:W => Normal(0., 0.5),
			:C => truncated(Normal(3., 2.), lower = 1.)
		)
	)
	pmst_ests
end

# ╔═╡ 7af0c1fb-e32d-4fa2-96a2-094c58c57888
begin
	include("$(pwd())/working_memory/RL+RLWM_models.jl")
	pmst_sg_ests, pmst_sg_choices = optimize_multiple(
		df;
		model = RLWM_pmst_sgd,
		estimate = "MAP",
		include_true=false,
		initial_params = [mean(truncated(Normal(0., 2.), lower = 0.)), 0.5, 0.5, mean(truncated(Normal(2., 2.), lower = 1.))],
		transformed = Dict(:a => :α, :W => :w0),
		priors = Dict(
			:ρ => truncated(Normal(0., 1.), lower = 0.),
			:a => Normal(0., 0.5),
			:W => Normal(0., 0.5),
			:C => truncated(Normal(3., 2.), lower = 1.)
		)
	)
	pmst_sg_ests
end

# ╔═╡ b5744ee4-7f10-4c41-94d6-185edcfff6dc
md"
### Collins & Frank RLWM model
"

# ╔═╡ 7243e277-a9c8-4c33-ac10-8573cb88fb69
begin
	rlwm_ests, rlwm_choices = optimize_multiple(
		df;
		model = RLWM_ss,
		estimate = "MAP",
		include_true=false,
		initial_params = [mean(truncated(Normal(0., 2.), lower = 0.)), 0.5, 0.5, 0.5, mean(truncated(Normal(2., 2.), lower = 1.))],
		transformed = Dict(:a => :α, :F_wm => :φ_wm, :W => :w0),
		priors = Dict(
			:ρ => truncated(Normal(0., 1.), lower = 0.),
			:a => Normal(0., 0.5),
			:F_wm => Normal(0., 0.5),
			:W => Normal(0., 0.5),
			:C => truncated(Normal(3., 2.), lower = 1.)
		)
	)

	rlwm_ests
end

# ╔═╡ 3c25a381-4fe1-4b94-8acf-7da4c14b9a3c
let
	f = Figure(size = (1000, 300))
	
	labs1 = (xlabel = "α", ylabel = "ρ", zlabel = "log-likelihood")
	labs2 = (xlabel = "C", ylabel = "w0", zlabel = "log-likelihood")
	labs3 = (xlabel = "C", ylabel = "φ_wm", zlabel = "log-likelihood")
	
	ax1, ax2, ax3 = Axis3(f[1,1]; labs1...), Axis3(f[1,2]; labs2...), Axis3(f[1,3]; labs3...)
	scatter!(ax1, a2α.(rlwm_ests.a), rlwm_ests.ρ, rlwm_ests.loglike)
	scatter!(ax2, rlwm_ests.C, a2α.(rlwm_ests.W), rlwm_ests.loglike)
	scatter!(ax3, rlwm_ests.C, a2α.(rlwm_ests.F_wm), rlwm_ests.loglike)
	f
end

# ╔═╡ 3c7f50e8-d766-45a7-ba3d-f241eedbb9e7
md"
### Palimpsest RLWM model (strict capacity)
"

# ╔═╡ 5ab6c587-0a1f-40ee-b8f6-6cb9a2075c1e
let
	# Set the labels for the axes
	f = Figure(size = (1000, 300))
	
	labs1 = (xlabel = "α", ylabel = "ρ", zlabel = "log-likelihood")
	labs2 = (xlabel = "C", ylabel = "w0", zlabel = "log-likelihood")
	
	ax1, ax2 = Axis3(f[1,1]; labs1...), Axis3(f[1,2]; labs2...)
	scatter!(ax1, a2α.(pmst_ests.a), pmst_ests.ρ, pmst_ests.loglike)
	scatter!(ax2, pmst_ests.C, a2α.(pmst_ests.W), pmst_ests.loglike)
	f
end

# ╔═╡ 2de21839-789d-4e48-be95-efa6a45cc906
md"
### Palimpsest RLWM model (with sigmoid weight)
"

# ╔═╡ 72ac4969-16b7-4eb5-bfb4-c60be499228c
let
	# Set the labels for the axes
	f = Figure(size = (1000, 300))
	
	labs1 = (xlabel = "α", ylabel = "ρ", zlabel = "log-likelihood")
	labs2 = (xlabel = "C", ylabel = "w0", zlabel = "log-likelihood")
	
	ax1, ax2 = Axis3(f[1,1]; labs1...), Axis3(f[1,2]; labs2...)
	scatter!(ax1, a2α.(pmst_sg_ests.a), pmst_sg_ests.ρ, pmst_sg_ests.loglike)
	scatter!(ax2, pmst_sg_ests.C, a2α.(pmst_sg_ests.W), pmst_sg_ests.loglike)
	f
end

# ╔═╡ Cell order:
# ╠═2511ec84-857d-11ef-2e80-43218bddef4e
# ╟─8a215c05-805a-4960-aca8-3c84b9452c3c
# ╠═5c9edffe-1197-44a2-b6d5-2f4d80a18518
# ╟─b5744ee4-7f10-4c41-94d6-185edcfff6dc
# ╠═7243e277-a9c8-4c33-ac10-8573cb88fb69
# ╠═3c25a381-4fe1-4b94-8acf-7da4c14b9a3c
# ╟─3c7f50e8-d766-45a7-ba3d-f241eedbb9e7
# ╠═a0e0399e-a982-4bc4-856f-2e712d1e182b
# ╠═5ab6c587-0a1f-40ee-b8f6-6cb9a2075c1e
# ╟─2de21839-789d-4e48-be95-efa6a45cc906
# ╠═7af0c1fb-e32d-4fa2-96a2-094c58c57888
# ╠═72ac4969-16b7-4eb5-bfb4-c60be499228c
