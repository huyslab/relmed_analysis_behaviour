### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ f4428174-30cc-11f0-29db-4d087f316fe5
begin
	cd("/home/jovyan/")
    import Pkg
    # activate the shared project environment
    Pkg.activate("relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	using Random, DataFrames, JSON, CSV, StatsBase, JLD2, HTTP, CairoMakie, Printf, Distributions, CategoricalArrays, AlgebraOfGraphics, Dates, SHA
	using Tidier, GLM, MixedModels, PlutoUI, LaTeXStrings, ColorSchemes
	using LogExpFunctions: logistic, logit
	import OpenScienceFramework as OSF
	include("fetch_preprocess_data.jl")
	include("sample_utils.jl")
	include("plotting_utils.jl")
	include("osf_utils.jl")
	include("vigour_utils.jl")
	nothing
end

# ╔═╡ 5e6cb37e-7b4d-4188-8d2f-58362a3cf981
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

	spearman_brown(
	r;
	n = 2 # Number of splits
	) = (n * r) / (1 + (n - 1) * r)
end

# ╔═╡ 7f61cd3a-0e9a-47ba-8708-ca4e5816262d


# ╔═╡ Cell order:
# ╠═f4428174-30cc-11f0-29db-4d087f316fe5
# ╠═5e6cb37e-7b4d-4188-8d2f-58362a3cf981
# ╠═7f61cd3a-0e9a-47ba-8708-ca4e5816262d
