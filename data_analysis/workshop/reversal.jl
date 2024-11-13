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
	osf_folder = "Workshop figures/Reversal"
	proj = setup_osf("Task development")
end

# ╔═╡ 377a69d3-a5ab-4a1f-ae3c-1e685bc00982
begin
	# Load data
	_, _, _, _, _, _, reversal_data, _ = load_pilot6_data()
	nothing
end

# ╔═╡ cdef82d7-0c48-48ba-9518-5d31ee63f936
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

	reversal_data_clean
	
end

# ╔═╡ 4ba8c548-ff95-4796-91b6-0f5c1ac4847a
describe(reversal_data_clean)

# ╔═╡ 4292e779-332c-4e5b-adc8-63559c0f5cbb
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
	mp = data(sum_pre_post) *
		mapping(
			:trial => "Trial relative to reversal",
			:acc => "Prop. optimal choice",
			group = :group => nonnumeric,
			color = :color
		) * visual(Lines, linewidth = 2, alpha = 0.1) +
		
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
			visual(Scatter) +
			mapping(
				:trial => "Trial relative to reversal",
				:acc => "Prop. optimal choice",
				group = :group => nonnumeric 
			) * 
			visual(Lines)
		) +
		mapping([0]) * visual(VLines, color = :grey, linestyle = :dash)

	f = Figure()
	draw!(f[1,1], mp, scales(Color = (; colormap = :roma)); 
		axis = (; xticks = -3:5, yticks = 0:0.25:1.))

	# Save
	filepath = "results/workshop/reversal_acc_curve.png"

	save(filepath, f)

	# upload_to_osf(
	# 	filepath,
	# 	proj,
	# 	osf_folder
	# )

	f
end

# ╔═╡ 518b2148-8d82-4135-96a8-5ce332c446a3
# Split half of mean, trials +2 - +4
let

	fs = []

	for s in unique(reversal_data_clean.session)
		# Summarize post reversal
		sum_post = combine(
			groupby(
				filter(x -> (x.trial in 2:4) && (x.session == s), reversal_data_clean),
				[:prolific_pid, :half]
			),
			:response_optimal => mean => :acc
		)
	
		# Long to wide
		sum_post = unstack(
			sum_post,
			:prolific_pid,
			:half,
			:acc,
			renamecols = x -> "half_$x"
		)
	
		# Plot
		f = Figure()
		
		workshop_reliability_scatter!(
			f[1, 1];
			df = sum_post,
			xcol = :half_1,
			ycol = :half_2,
			xlabel = "First half",
			ylabel = "Second half",
			subtitle = "Session $s post reversal accuracy"
		)

		# Save
		filepath = "results/workshop/reversal_sess$(s)_trials_2-4_splithalf.png"
	
		save(filepath, f)
	
		upload_to_osf(
			filepath,
			proj,
			osf_folder
		)

		# Push for notebook plotting
		push!(fs, f)

	end

	fs

end

# ╔═╡ b14b4021-7a41-4066-b38b-be70776eebb4
# ╠═╡ disabled = true
#=╠═╡
# Test retest of mean, trials +2 - +4
let

	# Summarize post reversal
	sum_post = combine(
		groupby(
			filter(x -> (x.trial in 2:4), reversal_data_clean),
			[:prolific_pid, :session]
		),
		:response_optimal => mean => :acc
	)

	# Long to wide
	sum_post = unstack(
		sum_post,
		:prolific_pid,
		:session,
		:acc,
		renamecols = x -> "sess_$x"
	)

	# Plot
	f = Figure()
	
	workshop_reliability_scatter!(
		f[1, 1];
		df = sum_post,
		xcol = :sess_1,
		ycol = :sess_2,
		xlabel = "Session 1",
		ylabel = "Session 2",
		subtitle = "Post reversal accuracy"
	)

	# Save
	filepath = "results/workshop/reversal_trials_2-4_test_retest.png"

	save(filepath, f)

	# upload_to_osf(
	# 	filepath,
	# 	proj,
	# 	osf_folder
	# )

	# Push for notebook plotting
	push!(fs, f)

	fs

end
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═32109870-a1ae-11ef-3dca-57321e58b0e8
# ╠═d79c72d4-adda-4cde-bc46-d4be516261ea
# ╠═ffc74f42-8ca4-45e0-acee-40086ff8eba4
# ╠═377a69d3-a5ab-4a1f-ae3c-1e685bc00982
# ╠═cdef82d7-0c48-48ba-9518-5d31ee63f936
# ╠═4ba8c548-ff95-4796-91b6-0f5c1ac4847a
# ╠═4292e779-332c-4e5b-adc8-63559c0f5cbb
# ╠═518b2148-8d82-4135-96a8-5ce332c446a3
# ╠═b14b4021-7a41-4066-b38b-be70776eebb4
