### A Pluto.jl notebook ###
# v0.19.45

using Markdown
using InteractiveUtils

# ╔═╡ 3af44b1c-8705-11ef-1f5b-4d06044e3a52
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

# ╔═╡ ae3243b7-b75c-4921-8713-41eb7a4fcb16
include("$(pwd())/working_memory/RL+RLWM_models.jl")

# ╔═╡ cbfb0051-fdd9-4514-b8d7-8453789247c7
begin
	PLT_data, test_data, vigour_data, jspsych_data = load_pilot2_data()
	df = prepare_for_fit(PLT_data, pilot2=true)[1]
	nothing
end

# ╔═╡ 1f8a2014-896b-47e6-aa9b-e839c05a8d86
md"
## Hierarchical RL models: fit to data
"

# ╔═╡ 77289ca7-13f4-44ed-a2cc-0c8f5349ad05
begin
	m = RL(df)
	fit = maximum_a_posteriori(m)
end

# ╔═╡ Cell order:
# ╠═3af44b1c-8705-11ef-1f5b-4d06044e3a52
# ╠═cbfb0051-fdd9-4514-b8d7-8453789247c7
# ╟─1f8a2014-896b-47e6-aa9b-e839c05a8d86
# ╠═ae3243b7-b75c-4921-8713-41eb7a4fcb16
# ╠═77289ca7-13f4-44ed-a2cc-0c8f5349ad05
