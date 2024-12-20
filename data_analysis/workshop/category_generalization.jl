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

# ╔═╡ 52ca98ce-1349-4d98-8e8b-8e8faa3aeba4
# Set up saving to OSF
begin
	osf_folder = "Workshop figures/Generalization"
	proj = setup_osf("Task development")
end

# ╔═╡ 0f1cf0ad-3a49-4c8e-8b51-607b7237e02f
begin
	# Load data
	PILT_data, _, _, _, _, WM_data,
		_, _ = load_pilot6_data()
	nothing
end

# ╔═╡ 31f77ce8-9a2a-4ea4-b1d7-f4f843870286
WM_data_clean = let
	
	# Clean data
	WM_data_clean = exclude_PLT_sessions(WM_data, required_n_blocks = 10)
	WM_data_clean = filter(x -> x.response != "noresp", WM_data_clean)

	# Auxillary variables ----------------
	# Stimulus category
	WM_data_clean.category_right = 
		(s -> replace(s, "imgs/PILT_stims/" => "", ".jpg" => "")[1:(end-1)]).(WM_data_clean.stimulus_right)

	WM_data_clean.category_left = 
		(s -> replace(s, "imgs/PILT_stims/" => "", ".jpg" => "")[1:(end-1)]).(WM_data_clean.stimulus_left)

	WM_data_clean.category_middle = 
		(s -> replace(s, "imgs/PILT_stims/" => "", ".jpg" => "")[1:(end-1)]).(WM_data_clean.stimulus_middle)

	# Optimal category
	WM_data_clean.optimal_category = (r -> r[Symbol("category_$(r.optimal_side)")]).(eachrow(WM_data_clean))

	# Repeating category
	categories = unique(WM_data_clean[!, [:session, :block, :stimulus_group, :category_right, :category_left, :category_middle, :valence, :optimal_category]])

	# Combine positions
	categories = stack(
		categories,
		[:category_right, :category_left, :category_middle],
		[:session, :block, :stimulus_group, :valence, :optimal_category],
		value_name = :category
	)

	# Keep unique
	select!(categories, Not(:variable))

	unique!(categories)

	sort!(categories, [:session, :block, :stimulus_group])

	# Compute repeating, previous valence, repeating_previous_optimal
	repeating = []
	prev_valence = []
	repeating_previous_optimal = []
	for (i, r) in enumerate(eachrow(categories))

		if r.block == 1
			# Missing / default values for block 1
			push!(repeating, false)
			push!(prev_valence, missing)
		else
			# Push whether reapting
			push!(
				repeating,
				r.category in (filter(x -> (x.block .== (r.block - 1)) && (x.session == r.session), categories).category)
			)
	
			# Push previous valence
			push!(
				prev_valence,
				filter(x -> (x.block .== (r.block - 1)) && (x.session == r.session), categories).valence[1]
			)
		end

		# Compute and push repeating_previously_optimal
		if !repeating[i]
			push!(repeating_previous_optimal, missing)
		else
			push!(
				repeating_previous_optimal,
				filter(x -> (x.block .== (r.block - 1)) && (x.session == r.session), categories).optimal_category[1] == r.category
			)
		end
	end

	categories.repeating = repeating
	categories.previous_valence = prev_valence
	categories.repeating_previous_optimal = repeating_previous_optimal

end

# ╔═╡ 3b019f83-64f3-428a-96d6-42d9cc1969fd
PILT_data_clean = let
	# Clean data
	PILT_data_clean = exclude_PLT_sessions(PILT_data, required_n_blocks = 20)
	PILT_data_clean = filter(x -> x.response != "noresp", PILT_data_clean)

	# Auxillary variables ----------------
	# Stimulus category
	PILT_data_clean.category_right = 
		(s -> replace(s, "imgs/PILT_stims/" => "", ".jpg" => "")[1:(end-1)]).(PILT_data_clean.stimulus_right)

	PILT_data_clean.category_left = 
		(s -> replace(s, "imgs/PILT_stims/" => "", ".jpg" => "")[1:(end-1)]).(PILT_data_clean.stimulus_left)

	# Optimal category
	PILT_data_clean.optimal_category = ifelse.(
		PILT_data_clean.optimal_right .== 1,
		PILT_data_clean.category_right,
		PILT_data_clean.category_left
	)

	# Repeating category
	categories = unique(PILT_data_clean[!, [:session, :block, :category_right, :category_left, :valence, :optimal_category]])

	# Combine left and right
	categories = stack(
		categories,
		[:category_right, :category_left],
		[:session, :block, :valence, :optimal_category],
		value_name = :category
	)

	# Keep unique
	select!(categories, Not(:variable))

	unique!(categories)

	# Compute repeating, previous valence, repeating_previous_optimal
	repeating = []
	prev_valence = []
	repeating_previous_optimal = []
	for (i, r) in enumerate(eachrow(categories))

		if r.block == 1
			# Missing / default values for block 1
			push!(repeating, false)
			push!(prev_valence, missing)
		else
			# Push whether reapting
			push!(
				repeating,
				r.category in (filter(x -> (x.block .== (r.block - 1)) && (x.session == r.session), categories).category)
			)
	
			# Push previous valence
			push!(
				prev_valence,
				filter(x -> (x.block .== (r.block - 1)) && (x.session == r.session), categories).valence[1]
			)
		end

		# Compute and push repeating_previously_optimal
		if !repeating[i]
			push!(repeating_previous_optimal, missing)
		else
			push!(
				repeating_previous_optimal,
				filter(x -> (x.block .== (r.block - 1)) && (x.session == r.session), categories).optimal_category[1] == r.category
			)
		end
	end

	categories.repeating = repeating
	categories.previous_valence = prev_valence
	categories.repeating_previous_optimal = repeating_previous_optimal
	
	# Join into data
	n_rows_pre = nrow(PILT_data_clean)
	PILT_data_clean = leftjoin(
		PILT_data_clean,
		rename(categories, 
			:category => :category_right,
			:repeating => :category_right_repeating,
			:repeating_previous_optimal => :repeating_previous_optimal_right
		),
		on = [:session, :block, :category_right, :valence, :optimal_category],
		order = :left
	)

	PILT_data_clean = leftjoin(
		PILT_data_clean,
		select(rename(categories, 
			:category => :category_left,
			:repeating => :category_left_repeating,
			:repeating_previous_optimal => :repeating_previous_optimal_left
		), Not(:previous_valence)),
		on = [:session, :block, :category_left, :valence, :optimal_category],
		order = :left
	)

	@assert nrow(PILT_data_clean) == n_rows_pre

	# Unify into one repeating_previous_optimal variable
	PILT_data_clean.repeating_previous_optimal = 
		coalesce.(PILT_data_clean.repeating_previous_optimal_right,
		PILT_data_clean.repeating_previous_optimal_left)

	select!(PILT_data_clean, 
		Not([:repeating_previous_optimal_left, :repeating_previous_optimal_right]))

	# Repeating optimal
	PILT_data_clean.repeating_optimal = ifelse.(
		PILT_data_clean.category_right_repeating .|| PILT_data_clean.category_left_repeating,
		ifelse.(
			PILT_data_clean.category_right_repeating,
			PILT_data_clean.optimal_right .== 1,
			PILT_data_clean.optimal_right .== 0
		),
		fill(missing, nrow(PILT_data_clean))
	)

	# Repeating chosen
	PILT_data_clean.repeating_chosen = ifelse.(
		PILT_data_clean.category_right_repeating .|| PILT_data_clean.category_left_repeating,
		ifelse.(
			PILT_data_clean.category_right_repeating,
			PILT_data_clean.response .== "right",
			PILT_data_clean.response .== "left"
		),
		fill(missing, nrow(PILT_data_clean))
	)

		
	PILT_data_clean
end

# ╔═╡ 3cb8fb6f-7457-499d-9aad-47e0f2e8ec5c
# Plot first trial in PILT
let

	# Summarize repeating_chosen on 1st trial by participant, previous valence, repeating_previous_optimal
	repeat_sum = combine(
		groupby(
			filter(x -> !ismissing(x.repeating_chosen) && (x.trial == 1), PILT_data_clean), 
			[:prolific_pid, :trial, :previous_valence, :repeating_previous_optimal]),
		:repeating_chosen => mean => :repeating_chosen,
		:repeating_chosen => length => :n
	)

	# Summarize repeating_chosen on 1st trial by previous valence, repeating_previous_optimal
	repeat_sum = combine(
		groupby(
			repeat_sum, 
			[:trial, :previous_valence, :repeating_previous_optimal]),
		:repeating_chosen => mean => :repeating_chosen,
		:repeating_chosen => sem => :se
	)
	

	# Create mapping for plot
	mp = data(repeat_sum) *
	mapping(
		:repeating_previous_optimal => "Repeating category on previous block",
		:repeating_chosen => "Prop. repeating category chosen",
		:se,
		color = :previous_valence => nonnumeric => "Previous block"
	) * (visual(Errorbars) +visual(Lines) + visual(Scatter)) +
	mapping([0.5]) * visual(HLines, linestyle = :dash, color = :grey)

	# Plot
	f = Figure(size = (30, 28) .* 36 ./ 2.54)

	plt = draw!(f[1,1], mp, scales(
		    X = (; categories = [false => "Suboptimal", true => "Optimal"]),
			Color = (; categories = [
				AlgebraOfGraphics.NonNumeric{Int64}(1) => "Reward", AlgebraOfGraphics.NonNumeric{Int64}(-1) => "Punishment"
			])
		);
		axis = (; xautolimitmargin = (0.2, 0.2))
	)

	legend!(f[0,1], plt, tellwidth = false, orientation = :horizontal, titleposition = :left)

	# Save plot
	filepath = "results/workshop/generalization_PILT_1st_trial_optimality_valence.png"

	save(filepath, f)

	# upload_to_osf(
	# 		filepath,
	# 		proj,
	# 		osf_folder
	# )


	f


end

# ╔═╡ 4f4995a7-0582-43d8-899d-0ed899d8fa73
# Reliability of first trial in PILT
let  s = "2"
	fs = []

	# Run over sessions and parameters
	for s in unique(PILT_data_clean.session)

		# Select data and add half variable
		first_trial = DataFrames.transform!(
			filter(x -> !ismissing(x.repeating_chosen) && (x.trial == 1) && 
				(x.session == s),
				PILT_data_clean),
			:block => (x -> ifelse.(
				x .< (median(unique(x)) + 2),
				fill(1, length(x)),
				fill(2, length(x))
			)) => :half
		) 
		
		# Summarize repeating_chosen on 1st trial by participant, half, repeating_previous_optimal
		repeat_sum = combine(
			groupby(
				first_trial, 
				[:prolific_pid, :half, :repeating_previous_optimal]),
			:repeating_chosen => mean => :repeating_chosen
		)
	
		# Long to wide
		repeat_sum = unstack(
			repeat_sum,
			[:prolific_pid, :half],
			:repeating_previous_optimal,
			:repeating_chosen
		)
	
		repeat_sum.diff = repeat_sum[!, Symbol("true")] .- repeat_sum[!, Symbol("false")]
	
		# Long to wide
		repeat_sum = unstack(
			repeat_sum,
			[:prolific_pid],
			:half,
			:diff,
			renamecols = x -> "half_$x"
		)
	
		#Plot
		f = Figure(size = (19.47, 19.47) .* 36 ./ 2.54)
		workshop_reliability_scatter!(
			f[1, 1];
			df = dropmissing(repeat_sum),
			xcol = :half_1,
			ycol = :half_2,
			xlabel = "First half",
			ylabel = "Second half",
			subtitle = "Session $s",
			markersize = 5
		)

		# Save plot
		filepath = "results/workshop/generalization_PILT_sess$(s)_1st_trial_optimality_splithalf.png"
	
		save(filepath, f)
	
		# upload_to_osf(
		# 		filepath,
		# 		proj,
		# 		osf_folder
		# )

		push!(fs, f)
	end

	fs

end

# ╔═╡ 69fd4fa3-f5da-4d63-b132-e3b2903293dd
# Test-retest reliability of first trial in PILT
let
	# Select data and add half variable
	first_trial = DataFrames.transform!(
		filter(x -> !ismissing(x.repeating_chosen) && (x.trial == 1),
			PILT_data_clean)
	) 
	
	# Summarize repeating_chosen on 1st trial by participant, half, repeating_previous_optimal
	repeat_sum = combine(
		groupby(
			first_trial, 
			[:prolific_pid, :session, :repeating_previous_optimal]),
		:repeating_chosen => mean => :repeating_chosen
	)

	# Long to wide
	repeat_sum = unstack(
		repeat_sum,
		[:prolific_pid, :session],
		:repeating_previous_optimal,
		:repeating_chosen
	)
	
	repeat_sum.diff = repeat_sum[!, Symbol("true")] .- repeat_sum[!, Symbol("false")]

	# Save for correlation matrix
	CSV.write(
		"results/workshop/generalization_PILT_params.csv",
		select(
			repeat_sum,
			:prolific_pid,
			:session => (x -> parse.(Int, x)) => :session,
			:diff => :generalization_PILT_optimality_effect
		)
	)

	
	# Long to wide
	repeat_sum = unstack(
		repeat_sum,
		[:prolific_pid],
		:session,
		:diff,
		renamecols = x -> "sess_$x"
	)
	
	# Plot
	f = Figure(size = (19.47, 19.47) .* 36 ./ 2.54)
	workshop_reliability_scatter!(
		f[1, 1];
		df = dropmissing!(repeat_sum),
		xcol = :sess_1,
		ycol = :sess_2,
		xlabel = "Session 1",
		ylabel = "Session 2",
		subtitle = " ",
		markersize = 5
	)
	
	# Save plot
	filepath = "results/workshop/generalization_PILT_1st_trial_optimality_test_retest.png"
	
	save(filepath, f)
	
	# upload_to_osf(
	# 		filepath,
	# 		proj,
	# 		osf_folder
	# )

	f

end

# ╔═╡ dffb9490-15b3-46f3-9ad6-35dd95676998
# Reliability of first trial in PILT - optimal reward - optimal punishment
let  s = "2"
	fs = []

	# Run over sessions and parameters
	for s in unique(PILT_data_clean.session)

		# Select data and add half variable
		first_trial = DataFrames.transform!(
			filter(x -> !ismissing(x.repeating_chosen) && (x.trial == 1) && 
				(x.session == s) && x.repeating_previous_optimal,
				PILT_data_clean),
			:block => (x -> ifelse.(
				x .< (median(unique(x)) + 2),
				fill(1, length(x)),
				fill(2, length(x))
			)) => :half
		) 
		
		# Summarize repeating_chosen on 1st trial by participant, half, repeating_previous_optimal
		repeat_sum = combine(
			groupby(
				first_trial, 
				[:prolific_pid, :half, :previous_valence]),
			:repeating_chosen => mean => :repeating_chosen
		)
	
		# Long to wide
		repeat_sum = unstack(
			repeat_sum,
			[:prolific_pid, :half],
			:previous_valence,
			:repeating_chosen
		)
	
		repeat_sum.diff = repeat_sum[!, Symbol("1")] .- repeat_sum[!, Symbol("-1")]
	
		# Long to wide
		repeat_sum = unstack(
			repeat_sum,
			[:prolific_pid],
			:half,
			:diff,
			renamecols = x -> "half_$x"
		)
	
		#Plot
		f = Figure()
		workshop_reliability_scatter!(
			f[1, 1];
			df = dropmissing(repeat_sum),
			xcol = :half_1,
			ycol = :half_2,
			xlabel = "First half",
			ylabel = "Second half",
			subtitle = "Session $s reward - punishment generalization effect"
		)

		# Save plot
		filepath = "results/workshop/generalization_PILT_sess$(s)_1st_trial_optimality_by_valence_splithalf.png"
	
		save(filepath, f)
	
		# upload_to_osf(
		# 		filepath,
		# 		proj,
		# 		osf_folder
		# )

		push!(fs, f)
	end

	fs

end

# ╔═╡ e10aeb4a-1b33-40db-8ebb-3dbeb23a724e
# Test-retest reliability of first trial in PILT - optimal reward - optimal punishment
let
	# Select data and add half variable
	first_trial = DataFrames.transform!(
		filter(x -> !ismissing(x.repeating_chosen) && (x.trial == 1) &&
		x.repeating_previous_optimal,
			PILT_data_clean)
	) 
	
	# Summarize repeating_chosen on 1st trial by participant, half, repeating_previous_optimal
	repeat_sum = combine(
		groupby(
			first_trial, 
			[:prolific_pid, :session, :previous_valence]),
		:repeating_chosen => mean => :repeating_chosen
	)

	# Long to wide
	repeat_sum = unstack(
		repeat_sum,
		[:prolific_pid, :session],
		:previous_valence,
		:repeating_chosen
	)
	
	repeat_sum.diff = repeat_sum[!, Symbol("1")] .- repeat_sum[!, Symbol("-1")]
	
	# Long to wide
	repeat_sum = unstack(
		repeat_sum,
		[:prolific_pid],
		:session,
		:diff,
		renamecols = x -> "sess_$x"
	)
	
	#Plot
	f = Figure()
	workshop_reliability_scatter!(
		f[1, 1];
		df = dropmissing!(repeat_sum),
		xcol = :sess_1,
		ycol = :sess_2,
		xlabel = "Session 1",
		ylabel = "Session 2",
		subtitle = "Choice of previous optimal stimuli reward - punishment"
	)
	
	# Save plot
	filepath = "results/workshop/generalization_PILT_1st_trial_optimality_test_retest.png"
	
	save(filepath, f)
	
	# upload_to_osf(
	# 		filepath,
	# 		proj,
	# 		osf_folder
	# )

	f

end

# ╔═╡ Cell order:
# ╠═c4f778a8-a207-11ef-1db0-f57fc0a2a769
# ╠═d6f8130a-3527-4c89-aff2-0c0e64d494d9
# ╠═52ca98ce-1349-4d98-8e8b-8e8faa3aeba4
# ╠═0f1cf0ad-3a49-4c8e-8b51-607b7237e02f
# ╠═31f77ce8-9a2a-4ea4-b1d7-f4f843870286
# ╠═3b019f83-64f3-428a-96d6-42d9cc1969fd
# ╠═3cb8fb6f-7457-499d-9aad-47e0f2e8ec5c
# ╠═4f4995a7-0582-43d8-899d-0ed899d8fa73
# ╠═69fd4fa3-f5da-4d63-b132-e3b2903293dd
# ╠═dffb9490-15b3-46f3-9ad6-35dd95676998
# ╠═e10aeb4a-1b33-40db-8ebb-3dbeb23a724e
