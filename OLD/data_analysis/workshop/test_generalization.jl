### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ 926b1c86-a34a-11ef-1787-03cf4275cddb
begin
	cd("/home/jovyan/")
    import Pkg
    # activate the shared project environment
    Pkg.activate("relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	using Random, DataFrames, JSON, CSV, StatsBase, JLD2, HTTP, CairoMakie, Printf, Distributions, CategoricalArrays, AlgebraOfGraphics, Dates, Turing, SHA, HypothesisTests, GLM, MixedModels, StatsModels
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

# ╔═╡ 56cafcfb-90c3-4310-9b19-aac5ec231512
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

# ╔═╡ ea6eb668-de64-4aa5-b3ea-8a5bc0475250
# Set up saving to OSF
begin
	osf_folder = "Workshop figures/Generalization"
	proj = setup_osf("Task development")
end

# ╔═╡ 1a1eb012-16e2-4318-be51-89b2e6a3b55b
begin
	# Load data
	PILT_data, test_data, _, _, _, _,
		_, _ = load_pilot6_data()
	nothing
end

# ╔═╡ fcb0292e-8a86-40c7-a1da-b3f24fbbe492
function compute_optimality(data::AbstractDataFrame)
	
	# Select columns and reduce to task strcuture, which is the same across participants
	optimality = unique(data[!, [:session, :block, :stimulus_left, :stimulus_right, :optimal_right]])

	# Which was the optimal stimulus?
	optimality.optimal = ifelse.(
		optimality.optimal_right .== 1, 
		optimality.stimulus_right, 
		optimality.stimulus_left
	)

	# Which was the suboptimal stimulus?
	optimality.suboptimal = ifelse.(
		optimality.optimal_right .== 0, 
		optimality.stimulus_right, 
		optimality.stimulus_left
	)

	# Remove double appearances (right left permutation)
	optimality = unique(optimality[!, [:session, :block, 
		:optimal, :suboptimal]])

	# Wide to long
	optimality = DataFrame(
		stimulus = vcat(optimality.optimal, optimality.suboptimal),
		optimal = vcat(fill(true, nrow(optimality)), fill(false, nrow(optimality)))
	)

	return optimality
end


# ╔═╡ 120babf5-f4c4-4c43-aab4-b3537111d15d
# Prepare data
test_data_clean = let

	# Select post-PILT test
	test_data_clean = filter(x -> isa(x.block, Int64), test_data)

	# Remove people who didn't finish
	DataFrames.transform!(
		groupby(test_data_clean, [:session, :prolific_pid]),
		:trial => length => :n_trials
	)

	filter!(x -> x.n_trials == 40, test_data_clean)

	# Remove missing values
	filter!(x -> !isnothing(x.response), test_data_clean)

	# Clean PILT data
	PILT_data_clean = exclude_PLT_sessions(PILT_data, required_n_blocks = 20)
	PILT_data_clean = filter(x -> x.response != "noresp", PILT_data_clean)

	# Compute EV from PILT	
	empirical_EVs = combine(
		groupby(PILT_data_clean, [:prolific_pid, :session, :chosen_stimulus]),
		:chosen_feedback => mean => :EV
	)

	# Add empirical EVs to test data
	pre_join_nrow = nrow(test_data_clean)
	test_data_clean = leftjoin(
		test_data_clean,
		rename(
			empirical_EVs,
			:chosen_stimulus => :stimulus_left,
			:EV => :empirical_EV_left
		),
		on = [:session, :prolific_pid, :stimulus_left],
		order = :left
	)

	test_data_clean = leftjoin(
		test_data_clean,
		rename(
			empirical_EVs,
			:chosen_stimulus => :stimulus_right,
			:EV => :empirical_EV_right
		),
		on = [:session, :prolific_pid, :stimulus_right],
		order = :left
	)

	@assert nrow(test_data_clean) == pre_join_nrow "Problem with join operation"

	# Compute empirical EV diff
	test_data_clean.empirical_EV_diff = test_data_clean.empirical_EV_right .- 	
		test_data_clean.empirical_EV_left

	# Keep only test trials where stimulus was observed in PILT
	dropmissing!(test_data_clean, :empirical_EV_diff)

	# Compute optimality of each stimulus
	optimality = compute_optimality(PILT_data_clean)

	# Add to test data
	pre_join_nrow = nrow(test_data_clean)
	test_data_clean = leftjoin(
		test_data_clean,
		rename(
			optimality,
			:stimulus => :stimulus_left,
			:optimal => :optimal_left
		),
		on = :stimulus_left,
		order = :left
	)

	test_data_clean = leftjoin(
		test_data_clean,
		rename(
			optimality,
			:stimulus => :stimulus_right,
			:optimal => :optimal_right
		),
		on = :stimulus_right,
		order = :left
	)

	test_data_clean.optimality_diff = test_data_clean.optimal_right .- test_data_clean.optimal_left

	test_data_clean.optimality_diff_cat = CategoricalArray(test_data_clean.optimality_diff)

	@assert nrow(test_data_clean) == pre_join_nrow "Problem with join operation"
	test_data_clean

	# Compute valence from magnitude
	DataFrames.transform!(
		test_data_clean,
		:magnitude_left => ByRow(sign) => :valence_left,
		:magnitude_right => ByRow(sign) => :valence_right
	)

	test_data_clean.valence_diff = test_data_clean.valence_right .- test_data_clean.valence_left

	test_data_clean.valence_diff_cat = CategoricalArray(test_data_clean.valence_diff)

	# Create magnitude high and low varaibles
	test_data_clean.magnitude_high = maximum.(eachrow((hcat(
		test_data_clean.magnitude_left, test_data_clean.magnitude_right))))

	test_data_clean.magnitude_low = minimum.(eachrow((hcat(
		test_data_clean.magnitude_left, test_data_clean.magnitude_right))))

	# Create high_chosen variable
	test_data_clean.high_chosen = ifelse.(
		test_data_clean.right_chosen,
		test_data_clean.magnitude_right .== test_data_clean.magnitude_high,
		test_data_clean.magnitude_left .== test_data_clean.magnitude_high
	)

	test_data_clean.magnitude_diff = test_data_clean.magnitude_right .- test_data_clean.magnitude_left

	# Add even/odd variable
	test_data_clean.evenodd = ifelse.(
		iseven.(test_data_clean.trial),
		1,
		2
	)

	test_data_clean
	

end

# ╔═╡ 33811ec9-f7c1-499d-9d9d-1a83951004a0
# Splithalf EV
let

	fs = []

	for s in unique(test_data_clean.session)
	
		for (g, labs) in zip(
			[:block, :evenodd],
			[["First half", "Second half"], ["Even", "Odd"]]
		)

			# Select data
			forfit = filter(x -> x.session == s, test_data_clean)
	
			insertcols!(
				forfit,
				:group => CategoricalArray(forfit[!, g])
			)
	
			# Fit by EV and group
			mm_tests = [fit(
				MixedModel, 
				@formula(right_chosen ~ 1 + empirical_EV_diff + 
					(1 + empirical_EV_diff | prolific_pid)), 
				filter(x -> x.group == gg, forfit), 
				Bernoulli()
			) for gg in unique(forfit.group)]
	
			ranefs = (f -> DataFrame(raneftables(f).prolific_pid)).(mm_tests)
	
			ranefs = innerjoin(
				ranefs[1],
				ranefs[2],
				on = :prolific_pid,
				makeunique = true
			)
	
			# Plot
			f = Figure(size = (19.47, 19.47) .* 36 ./ 2.54)
			workshop_reliability_scatter!(
				f[1, 1];
				df = ranefs,
				xcol = :empirical_EV_diff,
				ycol = :empirical_EV_diff_1,
				xlabel = labs[1],
				ylabel = labs[2],
				subtitle = "Session $s",
				markersize = 5
			)
	
			# Save plot
			filepath = "results/workshop/test_PILT_sess$(s)_EV_sensitivity_splithalf_$(string(g)).png"
		
			save(filepath, f)
		
			# upload_to_osf(
			# 		filepath,
			# 		proj,
			# 		osf_folder
			# )
	
			push!(fs, f)
			
		end
	end
		
	fs
end

# ╔═╡ 375bffca-9073-4a3d-8b68-b1474bb2591c
# Fit EV sensitivity by session
mm_EV_sess = let
	# Fit by EV and group
	mm_EV_sess = [fit(
		MixedModel, 
		@formula(right_chosen ~ 1 + empirical_EV_diff + 
			(1 + empirical_EV_diff | prolific_pid)), 
		filter(x -> x.session == s, test_data_clean), 
		Bernoulli()
	) for s in unique(test_data_clean.session)]

end

# ╔═╡ b9b974ab-fc93-4d99-ad51-c61675147709
# Test retest EV
let
	ranefs = (f -> DataFrame(raneftables(f).prolific_pid)).(mm_EV_sess)

	# Join for plotting
	ranefs = innerjoin(
		ranefs[1],
		ranefs[2],
		on = :prolific_pid,
		makeunique = true
	)

	# Plot
	f = Figure(size = (19.47, 19.47) .* 36 ./ 2.54)
	workshop_reliability_scatter!(
		f[1, 1];
		df = ranefs,
		xcol = :empirical_EV_diff,
		ycol = :empirical_EV_diff_1,
		xlabel = "Session 1",
		ylabel = "Session 2",
		subtitle = " ",
		correct_r = false,
		markersize = 5
	)

	# Save plot
	filepath = "results/workshop/test_PILT_EV_sensitivity_test_retest.png"

	save(filepath, f)

	# upload_to_osf(
	# 		filepath,
	# 		proj,
	# 		osf_folder
	# )
					
	f
end

# ╔═╡ a4792246-baac-4972-98c7-1c20d7079046
# Splithalf optimality
let

	fs = []

	for s in unique(test_data_clean.session)
	
		for (g, labs) in zip(
			[:block, :evenodd],
			[["First half", "Second half"], ["Even", "Odd"]]
		)

			# Select data
			forfit = filter(x -> x.session == s, test_data_clean)
	
			insertcols!(
				forfit,
				:group => CategoricalArray(forfit[!, g])
			)
	
			# Fit by EV and group
			mm_tests = [fit(
				MixedModel, 
				@formula(right_chosen ~ 1 + empirical_EV_diff + optimality_diff + 
					(1 + empirical_EV_diff + optimality_diff | prolific_pid)), 
				filter(x -> x.group == gg, forfit), 
				Bernoulli()
			) for gg in unique(forfit.group)]
	
			ranefs = (f -> DataFrame(raneftables(f).prolific_pid)).(mm_tests)
	
			ranefs = innerjoin(
				ranefs[1],
				ranefs[2],
				on = :prolific_pid,
				makeunique = true
			)
	
			# Plot
			f = Figure(size = (19.47, 19.47) .* 36 ./ 2.54)
			workshop_reliability_scatter!(
				f[1, 1];
				df = ranefs,
				xcol = :optimality_diff,
				ycol = :optimality_diff_1,
				xlabel = labs[1],
				ylabel = labs[2],
				subtitle = "Session $s",
				markersize = 5
			)
	
			# Save plot
			filepath = "results/workshop/test_PILT_sess$(s)_optimality_bias_splithalf_$(string(g)).png"
		
			save(filepath, f)
		
			# upload_to_osf(
			# 		filepath,
			# 		proj,
			# 		osf_folder
			# )
	
			push!(fs, f)
			
		end
	end
		
	fs
end

# ╔═╡ 71471faf-8acc-48f9-8535-7ec9eaef0429
# Fit EV and optimality by session
mm_test_EV_optimality_sess = let
	# Fit by EV and optimality
	mm_tests = [fit(
		MixedModel, 
		@formula(right_chosen ~ 1 + empirical_EV_diff + optimality_diff +
			(1 + empirical_EV_diff + optimality_diff | prolific_pid)), 
		filter(x -> x.session == s, test_data_clean), 
		Bernoulli()
	) for s in unique(test_data_clean.session)]
end

# ╔═╡ 6fae17be-871b-42b4-8c8a-b7f959cf4d01
# Test retest optimality
let

	ranefs = (f -> DataFrame(raneftables(f).prolific_pid)).(mm_test_EV_optimality_sess)

	ranefs = innerjoin(
		ranefs[1],
		ranefs[2],
		on = :prolific_pid,
		makeunique = true
	)

	# Plot
	f = Figure(size = (19.47, 19.47) .* 36 ./ 2.54)
	workshop_reliability_scatter!(
		f[1, 1];
		df = ranefs,
		xcol = :optimality_diff,
		ycol = :optimality_diff_1,
		xlabel = "Session 1",
		ylabel = "Session 2",
		subtitle = " ",
		correct_r = false,
		markersize = 5
	)

	# Save plot
	filepath = "results/workshop/test_PILT_optimality_bias_test_retest.png"

	save(filepath, f)

	# upload_to_osf(
	# 		filepath,
	# 		proj,
	# 		osf_folder
	# )
					
	f
end

# ╔═╡ c54f34a1-c6bb-4236-a491-25e7a0b96da4
# Fit by EV
mm_EV = let
	mm_test = fit(MixedModel, @formula(right_chosen ~ 1 + empirical_EV_diff + (1 + empirical_EV_diff | prolific_pid)), test_data_clean, Bernoulli())

end

# ╔═╡ 50f853c8-8e24-4bd8-bb66-9f25e92d0b4b
# Plot by EV bin
let n_bins = 6
	# Quantile bin breaks
	EV_bins = quantile(test_data_clean.empirical_EV_diff, 
		range(0, 1, length = n_bins + 1))

	# Bin EV_diff
	test_data_clean.EV_diff_cut = 	
		cut(test_data_clean.empirical_EV_diff, EV_bins, extend = true)

	# Use mean of bin as label
	DataFrames.transform!(
		groupby(test_data_clean, :EV_diff_cut),
		:empirical_EV_diff => mean => :EV_diff_bin
	)
	
	# Summarize by participant and EV bin
	test_sum = combine(
		groupby(test_data_clean, [:prolific_pid, :EV_diff_bin]),
		:right_chosen => mean => :right_chosen
	)

	# Summarize by EV bin
	test_sum_sum = combine(
		groupby(test_sum, :EV_diff_bin),
		:right_chosen => mean => :right_chosen,
		:right_chosen => sem => :se
	)

	# Prediction line from model
	predicted = DataFrame(
		empirical_EV_diff = minimum(test_data_clean.empirical_EV_diff):0.01:maximum(test_data_clean.empirical_EV_diff),
		prolific_pid = "new",
		right_chosen = -1.
	)

	predicted.right_chosen = predict(mm_EV, predicted; new_re_levels=:population)

	# Plot mapping
	mp = data(test_sum_sum) * 
	mapping(
		:EV_diff_bin,
		:right_chosen,
		:se
	) * (visual(Errorbars) + visual(Scatter)) +
	data(predicted) * mapping(:empirical_EV_diff, :right_chosen) * visual(Lines)

	# Plot
	f = Figure(size = (30, 28) .* 36 ./ 2.54)
	
	draw!(f[1,1], mp; 
		axis = (; 
			xlabel = "Δ expected value\nright - left",
			ylabel = "Prop. right chosen"
		)
	)

	# Save
	filepath = "results/workshop/test_PILT_EV.png"

	save(filepath, f, pt_per_unit = 1)

	# upload_to_osf(
	# 	filepath,
	# 	proj,
	# 	osf_folder
	# )

	f

end

# ╔═╡ ea0f4939-d18f-4407-a13f-d5734cc608bb
# Fit by EV and optimality
mm_EV_optimality = let
	mm_test = fit(MixedModel, @formula(right_chosen ~ 1 + empirical_EV_diff * optimality_diff + (1 + empirical_EV_diff * optimality_diff | prolific_pid)), test_data_clean, Bernoulli(), contrasts = Dict(:optimality_diff_cat => EffectsCoding(), :valence_diff_cat => EffectsCoding()))

end

# ╔═╡ 3d3c637f-6278-4f54-acbb-9ff06e8b459b
# Plot by EV bin and optimality
let n_bins = 5

	test_opt_sum = []

	for op in unique(test_data_clean.optimality_diff)

		# Select data
		tdata = filter(x -> x.optimality_diff == op, test_data_clean)

		# Quantile bin breaks
		EV_bins = quantile(tdata.empirical_EV_diff, 
			range(0, 1, length = n_bins + 1))
	
		# Bin EV_diff
		tdata.EV_diff_cut = 	
			cut(tdata.empirical_EV_diff, EV_bins, extend = true)
	
		# Use mean of bin as label
		DataFrames.transform!(
			groupby(tdata, :EV_diff_cut),
			:empirical_EV_diff => mean => :EV_diff_bin
		)

		# Summarize by participant, EV bin, and optimality
		test_sum = combine(
			groupby(tdata, [:prolific_pid, :EV_diff_bin, :optimality_diff]),
			:right_chosen => mean => :right_chosen
		)
	
		# Summarize by EV bin
		test_sum_sum = combine(
			groupby(test_sum, [:EV_diff_bin, :optimality_diff]),
			:right_chosen => mean => :right_chosen,
			:right_chosen => sem => :se
		)

		push!(test_opt_sum, test_sum_sum)
		
	end

	# Combine into single data frame
	test_opt_sum = vcat(test_opt_sum...)

	# Prediction line from model
	predicted = combine(
		groupby(test_data_clean, :optimality_diff),
		:empirical_EV_diff => (x -> minimum(x):0.01:maximum(x)) => :empirical_EV_diff
	)

	predicted[!, :prolific_pid] .= "new"

	predicted[!, :right_chosen] .= .99

	predicted.right_chosen = predict(mm_EV_optimality, predicted; new_re_levels=:population)

	# Plot mapping
	mp = data(test_opt_sum) * 
	mapping(
		:EV_diff_bin,
		:right_chosen,
		:se,
		color = :optimality_diff => nonnumeric => "Optimal on:",
		group = :optimality_diff => nonnumeric => "Optimal on:"
	) * (visual(Errorbars) + visual(Scatter)) +
	data(predicted) * mapping(
		:empirical_EV_diff, 
		:right_chosen,
		color = :optimality_diff => nonnumeric => "Optimal on:",
		group = :optimality_diff => nonnumeric => "Optimal on:"
	) * visual(Lines)

	f = Figure(size = (30, 28) .* 36 ./ 2.54)
	
	plt = draw!(
		f[1,1], 
		mp,
		scales(
			Color = (; categories = [
				AlgebraOfGraphics.NonNumeric{Int64}(-1) => "Left", 
				AlgebraOfGraphics.NonNumeric{Int64}(0) => "Both / None", 
				AlgebraOfGraphics.NonNumeric{Int64}(1) => "Right"])
		);
		axis = (;
			xlabel = "Δ expected value\nright - left",
			ylabel = "Prop. right chosen"
		)
	)

	legend!(f[0, 1], plt, tellwdith = false, orientation = :horizontal, titleposition = :left)

	filepath = "results/workshop/test_PILT_EV_optimality.png"

	save(filepath, f, pt_per_unit = 1)

	# upload_to_osf(
	# 	filepath,
	# 	proj,
	# 	osf_folder
	# )


	f
end

# ╔═╡ c2d48fe0-d2a1-4c11-a5d7-95332dc78c70
let
	# Get predicted values
	fitted_vals = fitted(mm_EV_optimality)

	# Combine into real data DataFrame
	forplot = insertcols(
		test_data_clean,
		:simuated => ifelse.(
			abs.(test_data_clean.magnitude_right) .== 0.01,
			fitted_vals,
			1. .- fitted_vals
		)
	)

	# Choose only penny trial
	forplot = filter(x -> sum(abs.([x.magnitude_low, x.magnitude_high]) .== 0.01) == 1, forplot
	)

	# Magnitude of penny choice
	forplot.magnitude_penny = ifelse.(
		abs.(forplot.magnitude_low) .== 0.01,
		forplot.magnitude_low,
		forplot.magnitude_high
	)

	# Magnitude of other choice
	forplot.magnitude_other = ifelse.(
		abs.(forplot.magnitude_low) .!= 0.01,
		forplot.magnitude_low,
		forplot.magnitude_high
	)	

	# Whether penny was chosen
	forplot.penny_chosen = ifelse.(
		abs.(forplot.magnitude_high) .== 0.01,
		forplot.high_chosen,
		.!forplot.high_chosen
	)

	# Summarize by participant and magnitudes
	test_sum = combine(
		groupby(forplot, [:prolific_pid, :magnitude_penny, :magnitude_other]),
		:penny_chosen => mean => :penny_chosen,
		:simuated => mean => :sim_penny_chosen
	)

	# Summarize by magnitudes
	test_sum_sum = combine(
		groupby(test_sum, [:magnitude_penny, :magnitude_other]),
		:penny_chosen => mean => :penny_chosen,
		:penny_chosen => sem => :se,
		:sim_penny_chosen => mean => :sim_penny_chosen,
	)

	# Sort for plotting
	sort!(test_sum_sum, [:magnitude_penny, :magnitude_other])

	# Mapping
	mp_data = mapping(
			:magnitude_other => nonnumeric => "EV other choice",
			:penny_chosen => "Prop. penny chosen",
			:se,
			color = :magnitude_penny => nonnumeric => "EV penny choice:"
		) * visual(Errorbars) +
		mapping(
			:magnitude_other => nonnumeric => "EV other choice",
			:penny_chosen => "Prop. penny chosen",
			color = :magnitude_penny => nonnumeric => "EV penny choice:",
			marker = :magnitude_penny => nonnumeric => "EV penny choice:"
		) * visual(Scatter)

	mp_fit = mapping(
			:magnitude_other => nonnumeric => "EV other choice",
			:sim_penny_chosen => "Prop. penny chosen",
			color = :magnitude_penny => nonnumeric => "EV penny choice:",
			linestyle = :magnitude_penny => nonnumeric => "EV penny choice:"
		) * visual(Lines)

	mp_hline = mapping([0.5]) * visual(HLines, linestyle = :dash, color = :grey)
	
	# Plot broken penny data
	f1 = Figure(size = (30, 28) .* 36 ./ 2.54)

	plt1 = draw!(
		f1[1,1], 
		mp_hline + data(filter(x -> x.magnitude_penny == -0.01, test_sum_sum)) * (mp_data), 
		scales(
			Marker = (; palette = [:rect, :circle])
		))

	# Plot broken penny data and fit
	f2 = Figure(size = (30, 28) .* 36 ./ 2.54)

	plt2 = draw!(
		f2[1,1], 
		mp_hline + data(filter(x -> x.magnitude_penny == -0.01, test_sum_sum)) * (mp_data + mp_fit), 
		scales(
			Marker = (; palette = [:rect, :circle]),
			LineStyle = (; palette = [:dashdot, :dash])
		))

	# Plot broken penny data and fit and penny data
	f3 = Figure(size = (30, 28) .* 36 ./ 2.54)

	plt3 = draw!(
		f3[1,1], 
		mp_hline + data(filter(x -> x.magnitude_penny == -0.01, test_sum_sum)) * (mp_data + mp_fit) + data(filter(x -> x.magnitude_penny == 0.01, test_sum_sum)) * (mp_data), 
		scales(
			Marker = (; palette = [:rect, :circle]),
			LineStyle = (; palette = [:dashdot, :dash])
		))
	
	f3

	# Plot everything
	f4 = Figure(size = (30, 28) .* 36 ./ 2.54)

	plt4 = draw!(
		f4[1,1], 
		mp_hline + data(test_sum_sum) * (mp_data + mp_fit), 
		scales(
			Marker = (; palette = [:rect, :circle]),
			LineStyle = (; palette = [:dashdot, :dash])
		))
	
	# Set up legends
	add_legend! = (f, plt) -> legend!(
		f[0,1],
		plt,
		tellwidth = false,
		orientation = :horizontal,
		titleposition = :left
	)

	add_legend!(f1, plt4)
	add_legend!(f2, plt4)
	add_legend!(f3, plt4)
	add_legend!(f4, plt4)
	
	# Link axes
	linkaxes!(extract_axis.([f1[1,1], f2[1,1], f3[1,1], f4[1,1]])...)

	# Save
	filepaths = ["results/workshop/test_PILT_pennies_$(f).png" 
		for f in ["negative_data", "negative_all", "negative_all_positive_data", "all_all"]
	]

	for (fp, f) in zip(filepaths, [f1, f2, f3, f4])
		
		save(fp, f, pt_per_unit = 1)

		# upload_to_osf(
		# 	fp,
		# 	proj,
		# 	osf_folder
		# )
	end

	f1, f2, f3, f4

end

# ╔═╡ f59f5b19-b64a-474f-971e-bcbfca3f38f0
# Compute percent chosen penny vs broken penny
let

	# Get predicted values
	fitted_vals = fitted(mm_EV_optimality)

	# Combine into real data DataFrame
	forplot = insertcols(
		test_data_clean,
		:simuated => ifelse.(
			test_data_clean.magnitude_right .== test_data_clean.magnitude_high,
			fitted_vals,
			1. .- fitted_vals
		)
	)

	# Choose only penny trial
	forplot = filter(x -> all(abs.([x.magnitude_low, x.magnitude_high]) .== 0.01), forplot
	)

	# Summarize by participant and magnitudes
	test_sum = combine(
		groupby(forplot, :prolific_pid),
		:high_chosen => mean => :high_chosen,
		:simuated => mean => :sim_high_chosen
	)

	@info "Penny chosen %$(round(mean(test_sum.high_chosen) * 100, digits = 2)) of trials. Predicted: %$(round(mean(test_sum.sim_high_chosen) * 100, digits = 2))"

	# Summarize by participant and magnitudes
	test_sum = combine(
		groupby(forplot, [:prolific_pid, :session]),
		:high_chosen => mean => :high_chosen,
		:simuated => mean => :sim_high_chosen
	)

	test_sum = combine(
		groupby(test_sum, [:session]),
		:high_chosen => mean => :high_chosen,
		:sim_high_chosen => mean => :sim_high_chosen
	)

	for s in test_sum.session
		@info "Session $s: Penny chosen %$(round(mean(test_sum.high_chosen[test_sum.session .== s]) * 100, digits = 2)) of trials. Predicted: %$(round(mean(test_sum.sim_high_chosen[test_sum.session .== s]) * 100, digits = 2))"
	end

	
end

# ╔═╡ f5268ffc-ccd3-4e5e-ba4d-ceb52467741f
# Split half reliability of penny difference
let

	fs = []

	for s in unique(test_data_clean.session)

		for (sp, labs) in zip([:block, :evenodd], [["First half", "Second half"], ["Even trials", "Odd trials"]])
	
			pence = []
			
			for (p, n) in zip([0.01, -0.01], ["penny", "broken_penny"])
		
				# Select relevant trials
				penny = filter(x -> 
					(x.session == s) &&
					(p in [x.magnitude_right, x.magnitude_left]), 
					test_data_clean
				)
		
				# Compute penny chosen
				penny.penny_chosen = ifelse.(
					penny.magnitude_right .== p,
					penny.right_chosen,
					.!penny.right_chosen
				)
		
				# Summarize by participant
				penny_sum = combine(
					groupby(penny, [:prolific_pid, sp]),
					:penny_chosen => mean => Symbol("$(n)_chosen")
				)
		
				push!(pence, penny_sum)
			end
		
			# Combine into single DataFrame
			pence = innerjoin(
				pence[1],
				pence[2],
				on = [:prolific_pid, sp]
			)
		
			# Compute difference
			pence.diff = pence.broken_penny_chosen .- pence.penny_chosen
		
			# Long to wide
			pence = unstack(
				pence,
				:prolific_pid,
				sp,
				:diff,
				renamecols = x -> "diff_$x"
			)
		
			# Drop missing
			dropmissing!(pence)
		
			# Plot
			f = Figure(size = (19.47, 19.47) .* 36 ./ 2.54)
			
			workshop_reliability_scatter!(
				f[1, 1];
				df = pence,
				xcol = :diff_1,
				ycol = :diff_2,
				xlabel = labs[1],
				ylabel = labs[2],
				subtitle = "Session $s",
				markersize = 5
			)

			# Save plot
			filepath = "results/workshop/test_PILT_sess$(s)_penny_bias_splithalf_$(string(sp)).png"
		
			save(filepath, f)
		
			# upload_to_osf(
			# 		filepath,
			# 		proj,
			# 		osf_folder
			# )
		
			push!(fs, f)
	
		end
	end

	fs


end

# ╔═╡ 511ce7ed-0a7f-4069-b309-897a087f5411
# Test retest reliability of penny difference
f_pence, pence = let
	
	pence = []
	
	for (p, n) in zip([0.01, -0.01], ["penny", "broken_penny"])

		# Select relevant trials
		penny = filter(x -> 
			(p in [x.magnitude_right, x.magnitude_left]), 
			test_data_clean
		)

		# Compute penny chosen
		penny.penny_chosen = ifelse.(
			penny.magnitude_right .== p,
			penny.right_chosen,
			.!penny.right_chosen
		)

		# Summarize by participant
		penny_sum = combine(
			groupby(penny, [:prolific_pid, :session]),
			:penny_chosen => mean => Symbol("$(n)_chosen")
		)

		push!(pence, penny_sum)
	end

	# Combine into single DataFrame
	pence = innerjoin(
		pence[1],
		pence[2],
		on = [:prolific_pid, :session]
	)

	# Compute difference
	pence.diff = pence.broken_penny_chosen .- pence.penny_chosen

	# Long to wide
	pence_wide = unstack(
		pence,
		:prolific_pid,
		:session,
		:diff,
		renamecols = x -> "diff_$x"
	)

	# Drop missing
	dropmissing!(pence_wide)

	# Plot
	f = Figure(size = (19.47, 19.47) .* 36 ./ 2.54)
	
	workshop_reliability_scatter!(
		f[1, 1];
		df = pence_wide,
		xcol = :diff_1,
		ycol = :diff_2,
		xlabel = "Session 1",
		ylabel = "Session 2",
		subtitle = " ",
		markersize = 5
	)

	# Save plot
	filepath = "results/workshop/test_PILT_penny_bias_test_retest.png"

	save(filepath, f)

	# upload_to_osf(
	# 		filepath,
	# 		proj,
	# 		osf_folder
	# )

	f, pence

end

# ╔═╡ ca634214-2653-4622-bb89-d3e57233ba72
# Exprot params for correlation matrix
let

	# EV sensitivity
	ranefs = (f -> DataFrame(raneftables(f).prolific_pid)).(mm_EV_sess)
	params = vcat(
		DataFrames.select(
			ranefs[1],
			:prolific_pid,
			:prolific_pid => (x -> fill(1, length(x))) => :session,
			:empirical_EV_diff => :PILT_rest_EV_sensitivity,
		),
		DataFrames.select(
			ranefs[2],
			:prolific_pid,
			:prolific_pid => (x -> fill(2, length(x))) => :session,
			:empirical_EV_diff => :PILT_rest_EV_sensitivity
		)
	)

	# Optimality bias
	ranefs = (f -> DataFrame(raneftables(f).prolific_pid)).(mm_test_EV_optimality_sess)
	opt_bias = vcat(
		DataFrames.select(
			ranefs[1],
			:prolific_pid,
			:prolific_pid => (x -> fill(1, length(x))) => :session,
			:optimality_diff => :PILT_rest_learning_context_bias
		),
		DataFrames.select(
			ranefs[2],
			:prolific_pid,
			:prolific_pid => (x -> fill(2, length(x))) => :session,
			:empirical_EV_diff => :PILT_rest_learning_context_bias,
		)
	)

	pre_join_nrow = nrow(params)
	params = outerjoin(
		params,
		opt_bias,
		on = [:prolific_pid, :session]
	)

	@assert nrow(params) == pre_join_nrow

	# Broken penny bias
	pre_join_nrow = nrow(params)
	params = outerjoin(
		params,
		select(
			pence,
			:prolific_pid,
			:session => (x -> parse.(Int, x)) => :session,
			:diff => :PILT_test_broken_penny_bias
		),
		on = [:prolific_pid, :session]
	)
	
	@assert nrow(params) == pre_join_nrow

	CSV.write("results/workshop/PILT_test_params.csv", params)


end

# ╔═╡ Cell order:
# ╠═926b1c86-a34a-11ef-1787-03cf4275cddb
# ╠═56cafcfb-90c3-4310-9b19-aac5ec231512
# ╠═ea6eb668-de64-4aa5-b3ea-8a5bc0475250
# ╠═1a1eb012-16e2-4318-be51-89b2e6a3b55b
# ╠═120babf5-f4c4-4c43-aab4-b3537111d15d
# ╠═50f853c8-8e24-4bd8-bb66-9f25e92d0b4b
# ╠═3d3c637f-6278-4f54-acbb-9ff06e8b459b
# ╠═33811ec9-f7c1-499d-9d9d-1a83951004a0
# ╠═375bffca-9073-4a3d-8b68-b1474bb2591c
# ╠═b9b974ab-fc93-4d99-ad51-c61675147709
# ╠═a4792246-baac-4972-98c7-1c20d7079046
# ╠═71471faf-8acc-48f9-8535-7ec9eaef0429
# ╠═6fae17be-871b-42b4-8c8a-b7f959cf4d01
# ╠═c54f34a1-c6bb-4236-a491-25e7a0b96da4
# ╠═ea0f4939-d18f-4407-a13f-d5734cc608bb
# ╠═c2d48fe0-d2a1-4c11-a5d7-95332dc78c70
# ╠═f59f5b19-b64a-474f-971e-bcbfca3f38f0
# ╠═f5268ffc-ccd3-4e5e-ba4d-ceb52467741f
# ╠═511ce7ed-0a7f-4069-b309-897a087f5411
# ╠═ca634214-2653-4622-bb89-d3e57233ba72
# ╠═fcb0292e-8a86-40c7-a1da-b3f24fbbe492
