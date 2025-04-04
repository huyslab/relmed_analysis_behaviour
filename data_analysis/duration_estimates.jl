### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ cdfb1b44-1140-11f0-2d13-eb38e7da3866
begin
	cd("/home/jovyan/")
    import Pkg
    # activate the shared project environment
    Pkg.activate("relmed_environment")
    Pkg.instantiate()
	using Random, DataFrames, JSON, CSV, StatsBase, JLD2, HTTP, CairoMakie, Printf, Distributions, CategoricalArrays, AlgebraOfGraphics, Dates
	using LogExpFunctions: logistic, logit
	using Tidier
	include("fetch_preprocess_data.jl")
	include("sample_utils.jl")
	include("plotting_utils.jl")
	nothing
end

# ╔═╡ d02d8396-98ac-439e-9f59-1d41d752281c


# ╔═╡ Cell order:
# ╠═cdfb1b44-1140-11f0-2d13-eb38e7da3866
# ╠═d02d8396-98ac-439e-9f59-1d41d752281c
