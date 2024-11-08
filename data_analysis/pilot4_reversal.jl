### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ 170953f6-923f-11ef-1436-0d463bcd07db
begin
	cd("/home/jovyan/")
    import Pkg
    # activate the shared project environment
    Pkg.activate("relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	using Random, DataFrames, JSON, CSV, StatsBase, JLD2, HTTP, CairoMakie, Printf, Distributions, CategoricalArrays, AlgebraOfGraphics, Dates, Turing
	using LogExpFunctions: logistic, logit
	include("fetch_preprocess_data.jl")
	include("sample_utils.jl")
	include("plotting_utils.jl")
	include("stats_utils.jl")
	nothing
end

# ╔═╡ 1233cc00-fffe-4338-b86e-b95711b08009
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

# ╔═╡ e667a15d-933a-4f93-9098-55a1785ba497
begin
	_, _, _, _, _, reversal_data = load_pilot4_data()
	
	reversal_data = exclude_reversal_sessions(reversal_data)

	filter!(x -> !isnothing(x.response_optimal), reversal_data)
end

# ╔═╡ b30a0f8f-5832-45e2-b27d-7e560d8b0284
begin
	# Number trials leading to reversal
	DataFrames.transform!(
		groupby(reversal_data, [:prolific_pid, :block]),
		:trial => (x -> x .- maximum(x) .- 1) => :trial_pre_reversal
	)

	# Count number of block
	DataFrames.transform!(
		groupby(reversal_data, :prolific_pid),
		:block => (x -> maximum(x)) => :n_blocks
	)
end

# ╔═╡ ef806082-d952-40f5-92bb-b0de914b77ac
let
	# Summarize accuracy pre reversal
	sum_pre = combine(
		groupby(
			filter(x -> (x.trial_pre_reversal > -4) && 
				(x.block < x.n_blocks) && (x.trial < 48), reversal_data), 
			[:prolific_pid, :trial_pre_reversal]
		),
		:response_optimal => mean => :acc
	)

	sum_pre = combine(
		groupby(sum_pre, :trial_pre_reversal),
		:acc => mean => :acc,
		:acc => sem => :se
	)

	rename!(sum_pre, :trial_pre_reversal => :trial)

	# Summarize post reversal
	sum_post = combine(
		groupby(
			filter(x -> x.trial < 10, reversal_data),
			[:prolific_pid, :trial]
		),
		:response_optimal => mean => :acc
	)

	sum_post = combine(
		groupby(sum_post, :trial),
		:acc => mean => :acc,
		:acc => sem => :se
	)

	# Concatenate pre and post
	sum_pre_post = vcat(sum_pre, sum_post)

	# Create group variable to break line plot
	sum_pre_post.group = sign.(sum_pre_post.trial)

	# Plot
	mp = data(sum_pre_post) *
		(
			mapping(
				:trial => "Trial relative to reversal",
				:acc  => "Prop. optimal choice",
				:se
			) * visual(Errorbars) +
			mapping(
				:trial => "Trial relative to reversal",
				:acc => "Prop. optimal choice"
			) * 
			visual(Scatter) +
			mapping(
				:trial => "Trial relative to reversal",
				:acc => "Prop. optimal choice",
				group = :group => nonnumeric 
			) * 
			visual(Lines)
		) +
		mapping([0]) * visual(VLines, color = :grey, linestyle = :dash)

	f = draw(mp; axis = (; xticks = -3:9))
	
	save("results/pilot4_reversal.png", f, pt_per_unit = 1)

	f
end

# ╔═╡ Cell order:
# ╠═170953f6-923f-11ef-1436-0d463bcd07db
# ╠═1233cc00-fffe-4338-b86e-b95711b08009
# ╠═e667a15d-933a-4f93-9098-55a1785ba497
# ╠═b30a0f8f-5832-45e2-b27d-7e560d8b0284
# ╠═ef806082-d952-40f5-92bb-b0de914b77ac
