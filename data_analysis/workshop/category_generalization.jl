### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ c4f778a8-a207-11ef-1db0-f57fc0a2a769
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

# ╔═╡ d6f8130a-3527-4c89-aff2-0c0e64d494d9
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

# ╔═╡ 52ca98ce-1349-4d98-8e8b-8e8faa3aeba4
# Set up saving to OSF
begin
	osf_folder = "Workshop figures/Generalization"
	proj = setup_osf("Task development")
end

# ╔═╡ 0f1cf0ad-3a49-4c8e-8b51-607b7237e02f
begin
	# Load data
	PILT_data, _, _, _, _, _,
		_, _ = load_pilot6_data()
	nothing
end

# ╔═╡ 3b019f83-64f3-428a-96d6-42d9cc1969fd
PILT_data_clean = let
	# Clean data
	PILT_data_clean = exclude_PLT_sessions(PILT_data, required_n_blocks = 20)
	PILT_data_clean = filter(x -> x.response != "noresp", PILT_data_clean)
end

# ╔═╡ Cell order:
# ╠═c4f778a8-a207-11ef-1db0-f57fc0a2a769
# ╠═d6f8130a-3527-4c89-aff2-0c0e64d494d9
# ╠═52ca98ce-1349-4d98-8e8b-8e8faa3aeba4
# ╠═0f1cf0ad-3a49-4c8e-8b51-607b7237e02f
# ╠═3b019f83-64f3-428a-96d6-42d9cc1969fd
