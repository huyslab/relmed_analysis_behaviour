### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ 23172eca-d33f-11ef-2eec-d5ccbae4cea9
begin
	cd("/home/jovyan/")
    import Pkg
    # activate the shared project environment
    Pkg.activate("relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	using Random, DataFrames, JSON, CSV, StatsBase, JLD2, HTTP, CairoMakie, Printf, Distributions, CategoricalArrays, AlgebraOfGraphics, Dates, Turing, SHA, HypothesisTests, GLM, MixedModels
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

# ╔═╡ d67226b1-b5bf-4f25-914b-8d2959618904
context = "poster" # or "abstract"

# ╔═╡ 77bfbda2-3aba-456c-bcb3-3e16ef351ce2
begin
	# Set theme
	inter_bold = assetpath(pwd() * "/fonts/Inter/Inter-Bold.ttf")

	inch = 96
	pt = 4/3
	mm = inch / 25.4

	sizes = Dict(
		"abstract" => (46.5mm, 31.3mm),
		"poster" => (22.86, 11.42) .* inch ./ 2.54
	)


	
	th =  Dict()
	
	th["abstract"] = merge(theme_minimal(), Theme(
		font = "Helvetica",
		fontsize = 7pt,
		Axis = (
			xticklabelsize = 5pt,
			yticklabelsize = 5pt,
			spinewidth = 0.5pt,
			xlabelpadding = 0,
			ylabelpadding = 0
		)
	))

	th["poster"] = merge(
			theme_minimal(),
			Theme(
			font = "Helvetica",
			fontsize = 28pt,
			Axis = (
				xticklabelsize = 24pt,
				yticklabelsize = 24pt,
				spinewidth = 3pt,
				xtickwidth = 3pt,
				ytickwidth = 3pt
			)
		)
	)
	
	set_theme!(th[context])

end

# ╔═╡ 8834884d-e403-4e1d-943c-d1e6a1238331
begin
	# Load data
	_, _, _, _, _, _, reversal_data, _ = load_pilot6_data()
	nothing
end

# ╔═╡ 3217e1c8-a4c2-4a07-acb9-22a29a698d3f
# Prepare data
reversal_data_clean = let
	# Exclude sessions
	reversal_data_clean = exclude_reversal_sessions(reversal_data; required_n_trials = 120)

	# Sort
	sort!(reversal_data_clean, [:prolific_pid, :session, :block, :trial])

	# Cumulative trial number
	DataFrames.transform!(
		groupby(reversal_data_clean, [:prolific_pid, :session]),
		:trial => (x -> 1:length(x)) => :ctrial
	)

	# Exclude trials
	filter!(x -> !isnothing(x.response_optimal), reversal_data_clean)

	# Auxillary variables --------------------------
	# Make session into Int64
	reversal_data_clean.session = parse.(Int, reversal_data_clean.session)
		
	# Number trials leading to reversal
	DataFrames.transform!(
		groupby(reversal_data_clean, [:prolific_pid, :session, :block]),
		:trial => (x -> x .- maximum(x) .- 1) => :trial_pre_reversal
	)

	# Count number of block
	DataFrames.transform!(
		groupby(reversal_data_clean, [:prolific_pid, :session]),
		:block => (x -> maximum(x)) => :n_blocks
	)

	# Split half
	reversal_data_clean.half = ifelse.(
		reversal_data_clean.ctrial .< median(unique(reversal_data_clean.ctrial)),
		fill(1, nrow(reversal_data_clean)),
		fill(2, nrow(reversal_data_clean))
	)

	# Create feedback_optimal and feedback_suboptimal
	reversal_data_clean.feedback_optimal = ifelse.(
		reversal_data_clean.optimal_right .== 1,
		reversal_data_clean.feedback_right,
		reversal_data_clean.feedback_left
	)

	reversal_data_clean.feedback_suboptimal = ifelse.(
		reversal_data_clean.optimal_right .== 0,
		reversal_data_clean.feedback_right,
		reversal_data_clean.feedback_left
	)

	@assert all(combine(
		groupby(reversal_data_clean, [:prolific_pid, :session]),
		:ctrial => issorted => :trial_sorted
	).trial_sorted)

	reversal_data_clean
	
end

# ╔═╡ e065d13e-eff6-46d2-9a1c-3ad9c9f2d7da
# Plot reversal accuracy curve
let
	# Summarize accuracy pre reversal
	sum_pre = combine(
		groupby(
			filter(x -> (x.trial_pre_reversal > -4) && 
				(x.block < x.n_blocks) && (x.trial < 48), reversal_data_clean), 
			[:prolific_pid, :trial_pre_reversal]
		),
		:response_optimal => mean => :acc
	)

	rename!(sum_pre, :trial_pre_reversal => :trial)

	sum_sum_pre = combine(
		groupby(sum_pre, :trial),
		:acc => mean => :acc,
		:acc => sem => :se
	)


	# Summarize post reversal
	sum_post = combine(
		groupby(
			filter(x -> x.trial < 6, reversal_data_clean),
			[:prolific_pid, :trial]
		),
		:response_optimal => mean => :acc
	)

	sum_sum_post = combine(
		groupby(sum_post, :trial),
		:acc => mean => :acc,
		:acc => sem => :se
	)

	# Concatenate pre and post
	sum_sum_pre_post = vcat(sum_sum_pre, sum_sum_post)
	sum_pre_post = vcat(sum_pre, sum_post)

	# Create group variable to break line plot
	sum_sum_pre_post.group = sign.(sum_sum_pre_post.trial)
	sum_pre_post.group = sign.(sum_pre_post.trial) .* 
		map(val -> findfirst(==(val), unique(sum_pre_post.prolific_pid)), 
			sum_pre_post.prolific_pid)

	# Color by accuracy on trial - 3
	DataFrames.transform!(
		groupby(sum_pre_post, :prolific_pid),
		[:trial, :acc] => ((t, a) -> mean(a[t .== -3])) => :color
	)

	# Sort for plotting
	sort!(sum_pre_post, [:prolific_pid, :trial])

	# Plot
	lws = Dict(
		"abstract" => 1pt,
		"poster" => 3pt
	)

	mp = data(sum_pre_post) *
		mapping(
			:trial => "Trial relative to reversal",
			:acc => "Prop. optimal choice",
			group = :group => nonnumeric,
			color = :color
		) * visual(Lines, linewidth = 1pt, alpha = 0.1) +
		
	data(sum_sum_pre_post) *
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
			visual(Scatter, markersize = context == "abstract" ? 6pt : 16pt) +
			mapping(
				:trial => "Trial relative to reversal",
				:acc => "Prop. optimal choice",
				group = :group => nonnumeric 
			) * 
			visual(Lines, linewidth = lws[context])
		) +
		mapping([0]) * visual(VLines, color = :grey, linestyle = :dash)

	f1 = Figure(size = sizes[context],
		figure_padding = 8)
	draw!(f1[1,1], mp, scales(Color = (; colormap = :roma)); 
		axis = (; xticks = -3:5, yticks = 0:0.25:1.))

	# Save
	filepath = "results/rldm/$(context)_reversal_acc_curve.pdf"

	save(filepath, f1)

	f1

end

# ╔═╡ cd7a2ef8-fc44-4286-b5ce-91ac10486128
# Tell fitting functions the column names
reversal_columns = Dict(
	"block" => :session,
	"trial" => :ctrial,
	"feedback_optimal" => :feedback_optimal,
	"feedback_suboptimal" => :feedback_suboptimal,
	"choice" => :response_optimal
)

# ╔═╡ be35ea4b-57c1-424d-bca0-5e13c1b1a11a
# Fit by session
QL_retest = let	
	fits = optimize_multiple_by_factor(
		reversal_data_clean;
		model = single_p_QL_recip,
		factor = :session,
		priors = Dict(
			:ρ => truncated(Normal(0., 5.), lower = 0.),
			:a => Normal(0., 2.)
		),
		unpack_function = unpack_single_p_QL,
		remap_columns = reversal_columns
	)
end

# ╔═╡ 0a1a9d26-3256-4135-8003-9cc6151c2178
function RLDM_reliability_scatter!(
		f::GridPosition;
		df::AbstractDataFrame,
		xlabel::AbstractString,
		ylabel::AbstractString,
		xcol::Symbol = :x,
		ycol::Symbol = :y,
		subtitle::AbstractString = "",
		tickformat::Union{Function, Makie.Automatic} = Makie.automatic,
		correct_r::Bool = true, # Whether to apply Spearman Brown
		markersize::Union{Int64,Float64} = 9,
		label_valign = 0.025
	)	

		# Compute correlation
		r = cor(df[!, xcol], df[!, ycol])
		
		# Spearman-Brown correction
		if correct_r
			r = spearman_brown(r)
		end

		# Text
		r_text = "n = $(nrow(df)),$(correct_r ? " SB" : "") r = $(round(r; digits = 2))"

		# Plot
		mp = data(df) *
				mapping(xcol, ycol) *
				(visual(Scatter; markersize = markersize, alpha = 0.75) + linear()) +
			mapping([0], [1]) *
				visual(ABLines, linestyle = :dash, color = :gray70)
		
		draw!(f, mp; axis=(;
			xlabel = xlabel, 
			ylabel = ylabel,
			xtickformat = tickformat,
			ytickformat = tickformat,
			subtitle = subtitle
		))

		if r > 0
			Label(
				f,
				r_text,
				fontsize = 24pt,
				font = :bold,
				halign = 0.975,
				valign = label_valign,
				tellheight = false,
				tellwidth = false
			)
		end

	end

# ╔═╡ 77d885d3-196a-4d61-a819-8fed889a10ab
# Test retest of mean, trials +2 - +4
let

	# Post reversal ------

	# Summarize post reversal
	sum_post = combine(
		groupby(
			filter(x -> (x.trial in 2:4), reversal_data_clean),
			[:prolific_pid, :session]
		),
		:response_optimal => mean => :acc
	)

	# Long to wide
	sum_post_wide = unstack(
		sum_post,
		:prolific_pid,
		:session,
		:acc,
		renamecols = x -> "sess_$x"
	)

	# Plot
	f = Figure(size = sizes[context],
		figure_padding = (0, 11.5, 10, 5))

	ms = Dict(
		"abstract" => 2,
		"poster" => 9
	)
	
	RLDM_reliability_scatter!(
		f[1, 1];
		df = dropmissing!(sum_post_wide),
		xcol = :sess_1,
		ycol = :sess_2,
		xlabel = "Session 1",
		ylabel = "Session 2",
		subtitle = "Post-reversal acc.",
		correct_r = false,
		markersize = ms[context]
	)


	# Reward sensitivty --------------
	# Long to wide
	this_retest = unstack(
		QL_retest,
		:prolific_pid,
		:session,
		:ρ,
		renamecols = (x -> "ρ_$x")
	)

	# Plot
	RLDM_reliability_scatter!(
		f[1, 2];
		df = dropmissing!(this_retest),
		xcol = :ρ_1,
		ycol = :ρ_2,
		xlabel = "Session 1",
		ylabel = "",
		subtitle = "Reward sensitivity",
		correct_r = false,
		markersize = ms[context] 
	)

	save("results/rldm/$(context)_reversal_test_retest.pdf", f)


	f
end

# ╔═╡ Cell order:
# ╠═23172eca-d33f-11ef-2eec-d5ccbae4cea9
# ╠═d67226b1-b5bf-4f25-914b-8d2959618904
# ╠═77bfbda2-3aba-456c-bcb3-3e16ef351ce2
# ╠═8834884d-e403-4e1d-943c-d1e6a1238331
# ╠═3217e1c8-a4c2-4a07-acb9-22a29a698d3f
# ╠═e065d13e-eff6-46d2-9a1c-3ad9c9f2d7da
# ╠═cd7a2ef8-fc44-4286-b5ce-91ac10486128
# ╠═be35ea4b-57c1-424d-bca0-5e13c1b1a11a
# ╠═77d885d3-196a-4d61-a819-8fed889a10ab
# ╠═0a1a9d26-3256-4135-8003-9cc6151c2178
