### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ 32109870-a1ae-11ef-3dca-57321e58b0e8
begin
	cd("/home/jovyan/")
    import Pkg
    # activate the shared project environment
    Pkg.activate("relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	using Random, DataFrames, JSON, CSV, StatsBase, JLD2, HTTP, CairoMakie, Printf, Distributions, CategoricalArrays, AlgebraOfGraphics, Dates, Turing, SHA, HypothesisTests
	using LogExpFunctions: logistic, logit
	import OpenScienceFramework as OSF
	include("fetch_preprocess_data.jl")
	include("sample_utils.jl")
	include("plotting_utils.jl")
	include("stats_utils.jl")
	include("osf_utils.jl")
	include("model_utils.jl")
	include("PILT_models.jl")
	Turing.setprogress!(false)
end

# ╔═╡ d79c72d4-adda-4cde-bc46-d4be516261ea
begin
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

# ╔═╡ ffc74f42-8ca4-45e0-acee-40086ff8eba4
# Set up saving to OSF
begin
	osf_folder = "Workshop figures/reversal"
	proj = setup_osf("Task development")
end

# ╔═╡ 377a69d3-a5ab-4a1f-ae3c-1e685bc00982
begin
	# Load data
	_, _, _, _, _, _, reversal_data, _ = load_pilot6_data()
	nothing
end

# ╔═╡ Cell order:
# ╠═32109870-a1ae-11ef-3dca-57321e58b0e8
# ╠═d79c72d4-adda-4cde-bc46-d4be516261ea
# ╠═ffc74f42-8ca4-45e0-acee-40086ff8eba4
# ╠═377a69d3-a5ab-4a1f-ae3c-1e685bc00982
