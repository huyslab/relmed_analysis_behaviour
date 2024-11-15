### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ 926b1c86-a34a-11ef-1787-03cf4275cddb
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

# ╔═╡ 56cafcfb-90c3-4310-9b19-aac5ec231512
begin
	# Set theme
	inter_bold = assetpath(pwd() * "/fonts/Inter/Inter-Bold.ttf")
	
	th = merge(theme_minimal(), Theme(
		font = "Helvetica",
		fontsize = 16,
		Axis = (
			xticklabelsize = 14,
			yticklabelsize = 14,
			spinewidth = 1.5,
			xtickwidth = 1.5,
			ytickwidth = 1.5
		)))
	set_theme!(th)
end

# ╔═╡ ea6eb668-de64-4aa5-b3ea-8a5bc0475250
# Set up saving to OSF
begin
	osf_folder = "Workshop figures/Generalization"
	proj = setup_osf("Task development")
end

# ╔═╡ 1a1eb012-16e2-4318-be51-89b2e6a3b55b
begin
	# Load data
	PILT_data, test_data, _, _, _, _,
		_, _ = load_pilot6_data()
	nothing
end

# ╔═╡ 120babf5-f4c4-4c43-aab4-b3537111d15d
# Prepare data
let

	# Select post-PILT test
	test_data_clean = filter(x -> isa(x.block, Int64), test_data)

	# Remove missing values
	filter!(x -> !isnothing(x.response), test_data_clean)

end

# ╔═╡ Cell order:
# ╠═926b1c86-a34a-11ef-1787-03cf4275cddb
# ╠═56cafcfb-90c3-4310-9b19-aac5ec231512
# ╠═ea6eb668-de64-4aa5-b3ea-8a5bc0475250
# ╠═1a1eb012-16e2-4318-be51-89b2e6a3b55b
# ╠═120babf5-f4c4-4c43-aab4-b3537111d15d
