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

	f1 = Figure(size = (45, 28) .* 72 ./ 2.54 ./ 2)
	draw!(f1[1,1], mp, scales(Color = (; colormap = :roma)); 
		axis = (; xticks = -3:5, yticks = 0:0.25:1.))

	# Save
	filepath = "results/workshop/reversal_acc_curve.png"

	save(filepath, f1)

	# upload_to_osf(
	# 	filepath,
	# 	proj,
	# 	osf_folder
	# )

	f1

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
		f = Figure(size = (19.5, 19.5) .* 72 ./ 2.54 ./ 2)
		
		workshop_reliability_scatter!(
			f[1, 1];
			df = dropmissing!(sum_post),
			xcol = :half_1,
			ycol = :half_2,
			xlabel = "First half",
			ylabel = "Second half",
			subtitle = "Session $s",
			markersize = 5
		)

		# Save
		filepath = "results/workshop/reversal_sess$(s)_trials_2-4_splithalf.png"
	
		save(filepath, f)
	
		# upload_to_osf(
		# 	filepath,
		# 	proj,
		# 	osf_folder
		# )

		# Push for notebook plotting
		push!(fs, f)

	end

	fs

end

# ╔═╡ b14b4021-7a41-4066-b38b-be70776eebb4
# Test retest of mean, trials +2 - +4
f_sum_post, sum_post = let

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
	f = Figure(size = (19.5, 19.5) .* 72 ./ 2.54 ./ 2)
	
	workshop_reliability_scatter!(
		f[1, 1];
		df = dropmissing!(sum_post_wide),
		xcol = :sess_1,
		ycol = :sess_2,
		xlabel = "Session 1",
		ylabel = "Session 2",
		subtitle = " ",
		correct_r = false,
		markersize = 5
	)

	# Save
	filepath = "results/workshop/reversal_trials_2-4_test_retest.png"

	save(filepath, f)

	# upload_to_osf(
	# 	filepath,
	# 	proj,
	# 	osf_folder
	# )

	f, sum_post
end

# ╔═╡ d5996434-7bdc-4aa1-95f9-5e9223a6b08d
# Logistic regression to recovery from reversal - split half
let

	fs = []

	for s in unique(reversal_data_clean.session)

		# Summarize post reversal
		sum_post = combine(
			groupby(
				filter(x -> (x.trial in 1:4), reversal_data_clean),
				[:prolific_pid, :trial, :half]
			),
			:response_optimal => mean => :acc,
			:response_optimal => length => :n
		)
	
		# Zscore trial
		sum_post.trial_s = (sum_post.trial .- mean(1:4)) ./ std(1:4)
	
		# GLM fit function
		glm_coef(dat) = coef(glm(@formula(acc ~ trial_s), dat, Binomial(), LogitLink(), wts = dat.n))[2]
	
		# Fit per participant and half
		post_coef = combine(
			groupby(sum_post, [:prolific_pid, :half]),
			AsTable([:acc, :trial_s, :n]) => glm_coef => :β
		)
	
		# Long to wide
		post_coef = unstack(
			post_coef,
			:prolific_pid,
			:half,
			:β,
			renamecols = x -> "half_$x"
		)
	
		# Plot
		f = Figure(size = (19.5, 19.5) .* 72 ./ 2.54 ./ 2)
		
		workshop_reliability_scatter!(
			f[1, 1];
			df = post_coef,
			xcol = :half_1,
			ycol = :half_2,
			xlabel = "First half",
			ylabel = "Second half",
			subtitle = "Session $s",
			markersize = 5
		)

			# Save
		filepath = "results/workshop/reversal_sess$(s)_logistic_recovery_splithalf.png"
	
		save(filepath, f)
	
		# upload_to_osf(
		# 	filepath,
		# 	proj,
		# 	osf_folder
		# )

		# Push for notebook plotting
		push!(fs, f)

	end

	fs


end

# ╔═╡ 6521764f-80cb-4c0a-90bb-09a40fe15136
# Logistic regression to recovery from reversal - test retest
f_post_coef, post_coef = let

	# Summarize post reversal
	sum_post = combine(
		groupby(
			filter(x -> (x.trial in 1:4), reversal_data_clean),
			[:prolific_pid, :trial, :session]
		),
		:response_optimal => mean => :acc,
		:response_optimal => length => :n
	)

	# Zscore trial
	sum_post.trial_s = (sum_post.trial .- mean(1:4)) ./ std(1:4)

	# GLM fit function
	glm_coef(dat) = coef(glm(@formula(acc ~ trial_s), dat, Binomial(), LogitLink(), wts = dat.n))[2]

	# Fit per participant and half
	post_coef = combine(
		groupby(sum_post, [:prolific_pid, :session]),
		AsTable([:acc, :trial_s, :n]) => glm_coef => :β
	)

	# Long to wide
	post_coef_wide = unstack(
		post_coef,
		:prolific_pid,
		:session,
		:β,
		renamecols = x -> "sess_$x"
	)

	# Plot
	f = Figure(size = (19.5, 19.5) .* 72 ./ 2.54 ./ 2)
	
	workshop_reliability_scatter!(
		f[1, 1];
		df = dropmissing!(post_coef_wide),
		xcol = :sess_1,
		ycol = :sess_2,
		xlabel = "Session 1",
		ylabel = "Session 2",
		subtitle = " ",
		correct_r = false,
		markersize = 5
	)

		# Save
	filepath = "results/workshop/reversal_logistic_recovery_test_retest.png"

	save(filepath, f)

	# upload_to_osf(
	# 	filepath,
	# 	proj,
	# 	osf_folder
	# )

	f, post_coef

end

# ╔═╡ 0cc9cab2-53e5-4049-8523-4f4bb4dfb5bb
# Tell fitting functions the column names
reversal_columns = Dict(
	"block" => :session,
	"trial" => :ctrial,
	"feedback_optimal" => :feedback_optimal,
	"feedback_suboptimal" => :feedback_suboptimal,
	"choice" => :response_optimal
)

# ╔═╡ f65ada7d-2f82-4cdd-9cb0-04a2105b1187
QL_splithalf = let
	fits =  Dict(s => optimize_multiple_by_factor(
		filter(x -> x.session == s, reversal_data_clean);
		model = single_p_QL_recip,
		factor = :half,
		priors = Dict(
			:ρ => truncated(Normal(0., 5.), lower = 0.),
			:a => Normal(0., 2.)
		),
		unpack_function = unpack_single_p_QL,
		remap_columns = reversal_columns
		) for s in unique(reversal_data_clean.session))
end

# ╔═╡ b239a169-b284-4d7d-98a4-8f15961ebad7
# Split half reliability of Q learning model
let
	fs = []

	# Run over sessions and parameters
	for (s, fits) in QL_splithalf

		for (p, st, tf) in zip(
			[:a, :ρ], 
			["learning rate", "reward sensitivity"],
			[x -> string.(round.(a2α.(x), digits = 2)), Makie.automatic]
		)

			f = Figure(size = (19.5, 19.5) .* 72 ./ 2.54 ./ 2)

			# Long to wide
			splithalf = unstack(
				QL_splithalf[s],
				:prolific_pid,
				:half,
				p,
				renamecols = (x -> "$(p)_$x")
			)

			# Plot
			workshop_reliability_scatter!(
				f[1, 1];
				df = splithalf,
				xcol = Symbol("$(p)_1"),
				ycol = Symbol("$(p)_2"),
				xlabel = "First half",
				ylabel = "Second half",
				subtitle = "Session $s",
				tickformat = tf,
				markersize = 5
			)

			# Save
			filepath = "results/workshop/reversal_sess$(s)_QL_$(string(p))_split_half.png"
			save(filepath, f)

			# Push for plotting in notebook
			push!(fs, f)
		end
	end

	fs
end

# ╔═╡ 34dcb1f1-3171-4a2d-a4f4-e9d880052d4f
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

# ╔═╡ 91fa774d-38d5-44ec-bc08-8b71dae24b9b
# Test retest of QL parameters
let
	fs = []

	# Run over parameters
	for (p, st, tf) in zip(
		[:a, :ρ], 
		["Learning rate", "Reward Sensitivity"],
		[x -> string.(round.(a2α.(x), digits = 2)), Makie.automatic]
	)

		f = Figure(size = (19.5, 19.5) .* 72 ./ 2.54 ./ 2)

		# Long to wide
		this_retest = unstack(
			QL_retest,
			:prolific_pid,
			:session,
			p,
			renamecols = (x -> "$(p)_$x")
		)

		# Plot
		workshop_reliability_scatter!(
			f[1, 1];
			df = dropmissing!(this_retest),
			xcol = Symbol("$(p)_1"),
			ycol = Symbol("$(p)_2"),
			xlabel = "First session",
			ylabel = "Second session",
			# subtitle = st,
			tickformat = tf,
			correct_r = false,
			markersize = 5
		)

		# Save
		filepath = "results/workshop/reversal_QL_$(string(p))_test_retest.png"
		save(filepath, f)

		# Push for plotting in notebook
		push!(fs, f)
	end
	fs
end

# ╔═╡ 9e24120b-a105-4c0b-a37b-c85e4faadf7c
# Export params
let
	# Combine all parameter dataframes exported previously
	params = outerjoin(
		select(
			QL_retest,
			:prolific_pid,
			:session,
			:a => :reversal_QL_recip_learning_rate_unconstrained,
			:ρ => :reversal_QL_recip_reward_sensitivity,
			:a => ByRow(a2α) => :reversal_QL_recip_learning_rate,
		),
		select(
			sum_post,
			:prolific_pid,
			:session,
			:acc => Symbol("reversal_accuracy_2-4_post_reversal")
		),
		on = [:prolific_pid, :session]
	)

	params = outerjoin(
		params,
		select(
			post_coef,
			:prolific_pid,
			:session,
			:β => :reversal_logistic_slope_acc_post_reversal
		),
		on = [:prolific_pid, :session]
	)

	CSV.write("results/workshop/reversal_params.csv", params)
	

end

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
# ╠═d5996434-7bdc-4aa1-95f9-5e9223a6b08d
# ╠═6521764f-80cb-4c0a-90bb-09a40fe15136
# ╠═0cc9cab2-53e5-4049-8523-4f4bb4dfb5bb
# ╠═f65ada7d-2f82-4cdd-9cb0-04a2105b1187
# ╠═b239a169-b284-4d7d-98a4-8f15961ebad7
# ╠═34dcb1f1-3171-4a2d-a4f4-e9d880052d4f
# ╠═91fa774d-38d5-44ec-bc08-8b71dae24b9b
# ╠═9e24120b-a105-4c0b-a37b-c85e4faadf7c
