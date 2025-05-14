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
begin
	PILT_data, test_data, _, _, jspsych_data = load_pilot9_data(;
		force_download=true)
	nothing
end

# ╔═╡ 23f77f57-7880-4415-9f97-59178d0d4b4d
PILT_data_clean = let
	# Clean data
	PILT_data_clean = exclude_PLT_sessions(PILT_data, required_n_blocks = 21)
	filter!(x -> x.response != "noresp", PILT_data_clean)

	# Remove empty columns
	select!(
		PILT_data_clean,
		Not([:EV_right, :EV_left])
	)
end

# ╔═╡ 46eb5012-f5b5-4174-bf5a-32fc305bd4e9
let
	# Sumarrize by participant and trial
	acc_curve = combine(
		groupby(PILT_data_clean, [:prolific_pid, :trial]),
		:response_optimal => mean => :acc
	)

	sort!(acc_curve, [:prolific_pid, :trial])

	# Summarize by trial
	acc_curve_sum = combine(
		groupby(acc_curve, :trial),
		:acc => mean => :acc,
		:acc => sem => :se
	)

	# Plot
	mp = (data(acc_curve) * mapping(
		:trial => "Trial #",
		:acc => "Prop. optimal choice",
		group = :prolific_pid,
		color = :prolific_pid
	) * visual(Lines, linewidth = 1, alpha = 0.7)) +
	(data(acc_curve_sum) * 
	mapping(
		:trial => "Trial #",
		:acc => "Prop. optimal choice"
	) * visual(Lines, linewidth = 4))
	
	
	draw(mp; legend = (; show = false))
end

# ╔═╡ 93ef440c-4c75-4239-84d2-ba1fe3c9a2ea
let

	# Extract trial plan from data
	unique_feedback = unique(select(PILT_data_clean, [:session, :block, :trial, :stimulus_right, :stimulus_left, :feedback_right, :feedback_left]))

	# Make sure it is the same for all participants
	@assert nrow(unique_feedback) == maximum(PILT_data_clean.block) * 10 "Trial plan not the same for all participants"

	# Feedback per stimulus, regardless of presentations side
	unique_feedback = vcat(
		[select(
			unique_feedback,
			:session,
			:block,
			:trial,
			Symbol("stimulus_$side") => :stimulus,
			Symbol("feedback_$side") => :feedback
		) for side in ["right", "left"]]...
	)

	# Average for EV
	EVs = combine(
		groupby(unique_feedback, [:session, :block, :stimulus]),
		:feedback => mean => :EV
	)

	# Merge back into data
	for side in ["right", "left"]
		leftjoin!(
			PILT_data_clean,
			select(
				EVs,
				:session,
				:block,
				:stimulus => Symbol("stimulus_$side"),
				:EV => Symbol("EV_$side")
			),
			on = [:session, :block, Symbol("stimulus_$side")]
		)
	end

	# EV difference variable
	PILT_data_clean.EV_diff = PILT_data_clean.EV_right .- PILT_data_clean.EV_left
end

# ╔═╡ 1d5ad1bc-2e93-4211-a08e-f5d5be23ccc1
"""
    quantile_bin_centers(x::AbstractVector, nbins::Int)

Bins the vector `x` into `nbins` equal-probability (quantile) bins and labels each value by its bin center.

# Arguments
- `x::AbstractVector`: Input data to bin.
- `nbins::Int`: Number of quantile bins.

# Returns
- `centers::Vector{Float64}`: Vector of bin center values, same length as `x`.

# Example
```julia
x = randn(100)
centers = quantile_bin_centers(x, 4)
```
"""
function quantile_bin_centers(x::AbstractVector, nbins::Int)
    # Compute quantile edges
    edges = quantile(x, range(0, 1; length=nbins+1))
    # Assign each value to a bin
    bin_idx = map(v -> searchsortedlast(edges, v), x)
    # Fix rightmost edge
    bin_idx = map(i -> min(i, nbins), bin_idx)
    # Compute bin centers
    centers = [round((edges[i] + edges[i+1]) / 2, digits = 2) for i in 1:nbins]
    # Label each value by its bin center
    return [centers[i] for i in bin_idx]
end

# ╔═╡ 2d8e283c-9215-4eaa-8447-2772b7e26645
let df = copy(PILT_data_clean)

	# Absolute value of EV difference
	df.abs_EV_diff = abs.(df.EV_diff)

	# Bin variable
	df.EV_bin = quantile_bin_centers(df.abs_EV_diff, 4)

	
	# Sumarrize by participant and trial
	acc_curve = combine(
		groupby(df, [:prolific_pid, :EV_bin, :trial]),
		:response_optimal => mean => :acc
	)

	# Summarize by trial
	acc_curve_sum = combine(
		groupby(acc_curve, [:EV_bin, :trial]),
		:acc => mean => :acc,
		:acc => sem => :se
	)

	# Error bands
	acc_curve_sum.lb = acc_curve_sum.acc - acc_curve_sum.se
	acc_curve_sum.ub = acc_curve_sum.acc + acc_curve_sum.se

	# Sort
	sort!(acc_curve_sum, [:EV_bin, :trial])

	# Plot
	mp = data(acc_curve_sum) * 
	(
		mapping(
			:trial => "Trial #",
			:acc => "Prop. optimal choice",
			color = :EV_bin => nonnumeric => "Abs. EV difference"
		) * visual(Lines, linewidth = 4) +
		mapping(
			:trial => "Trial #",
			:lb => "Prop. optimal choice",
			:ub => "Prop. optimal choice",
			color = :EV_bin => nonnumeric => "Abs. EV difference"
		) * visual(Band, alpha = 0.5)
	) + mapping([5]) * visual(VLines, color = :grey, linestyle = :dash) +
	mapping([0.5]) * visual(HLines, color = :grey, linestyle = :dash)
	
	f = Figure()
	plt = draw!(f[1,1], mp)
	legend!(
		f[0,1], 
		plt,
		tellwidth = false,
		framevisible = false,
		orientation = :horizontal,
		titleposition = :left
	)

	f

end

# ╔═╡ fe7479df-fde4-41de-936a-af4f88b8cf8a
describe(PILT_data_clean)

# ╔═╡ Cell order:
# ╠═f4428174-30cc-11f0-29db-4d087f316fe5
# ╠═5e6cb37e-7b4d-4188-8d2f-58362a3cf981
# ╠═7f61cd3a-0e9a-47ba-8708-ca4e5816262d
# ╠═23f77f57-7880-4415-9f97-59178d0d4b4d
# ╠═46eb5012-f5b5-4174-bf5a-32fc305bd4e9
# ╠═93ef440c-4c75-4239-84d2-ba1fe3c9a2ea
# ╠═2d8e283c-9215-4eaa-8447-2772b7e26645
# ╠═1d5ad1bc-2e93-4211-a08e-f5d5be23ccc1
# ╠═fe7479df-fde4-41de-936a-af4f88b8cf8a
