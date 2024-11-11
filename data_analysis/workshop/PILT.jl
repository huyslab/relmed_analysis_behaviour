### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ 8cf30b5e-a020-11ef-23b2-2da6e9116b54
begin
	cd("/home/jovyan/")
    import Pkg
    # activate the shared project environment
    Pkg.activate("relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	using Random, DataFrames, JSON, CSV, StatsBase, JLD2, HTTP, CairoMakie, Printf, Distributions, CategoricalArrays, AlgebraOfGraphics, Dates, Turing
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

# ╔═╡ 82ef300e-536f-40ce-9cde-72056e6f4b5e
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

# ╔═╡ 14a292db-43d4-45d8-97a5-37ffc03bdc5c
begin
	# Load data
	PILT_data, _, _, _, _, _,
		_, _ = load_pilot6_data()
	nothing
end

# ╔═╡ 6ed82686-35ab-4afd-a1b2-6fa19ae67168
PILT_data_clean = let
	# Clean data
	PILT_data_clean = exclude_PLT_sessions(PILT_data, required_n_blocks = 20)
	PILT_data_clean = filter(x -> x.response != "noresp", PILT_data_clean)
end

# ╔═╡ b5b75f4e-7b91-4287-a409-6f0ebdf20f4e
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

	# Compute bounds
	insertcols!(
		acc_curve_sum,
		:lb => acc_curve_sum.acc .- acc_curve_sum.se,
		:ub => acc_curve_sum.acc .+ acc_curve_sum.se
	)

	# Plot
	mp = data(acc_curve_sum) * 
	(
	# Error band
		mapping(
		:trial => "Trial #",
		:lb,
		:ub
	) * visual(Band) +
	# Average line	
		mapping(
		:trial => "Trial #",
		:acc => "Prop. optimal choice"
	) * visual(Lines, linewidth = 4)
	)
	
	
	draw(mp; legend = (; show = false))
end

# ╔═╡ Cell order:
# ╠═8cf30b5e-a020-11ef-23b2-2da6e9116b54
# ╠═82ef300e-536f-40ce-9cde-72056e6f4b5e
# ╠═14a292db-43d4-45d8-97a5-37ffc03bdc5c
# ╠═6ed82686-35ab-4afd-a1b2-6fa19ae67168
# ╠═b5b75f4e-7b91-4287-a409-6f0ebdf20f4e
