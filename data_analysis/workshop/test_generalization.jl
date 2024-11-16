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
	using Random, DataFrames, JSON, CSV, StatsBase, JLD2, HTTP, CairoMakie, Printf, Distributions, CategoricalArrays, AlgebraOfGraphics, Dates, Turing, SHA, HypothesisTests, GLM
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

# ╔═╡ fcafa95b-8d34-4221-8bd3-22f5cc5bf16f
using MixedModels, StatsModels

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
		groupby(test_data_clean, :prolific_pid),
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
		"even",
		"odd"
	)

	test_data_clean
	

end

# ╔═╡ 027ff42b-324f-4989-8952-4d119831796a
combine(
	groupby(test_data_clean, :evenodd),
	:trial => length
)

# ╔═╡ 33811ec9-f7c1-499d-9d9d-1a83951004a0
# Splithalf 
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
			f = Figure()
			workshop_reliability_scatter!(
				f[1, 1];
				df = ranefs,
				xcol = :empirical_EV_diff,
				ycol = :empirical_EV_diff_1,
				xlabel = labs[1],
				ylabel = labs[2],
				subtitle = "Session $s EV sensitivity"
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

# ╔═╡ b9b974ab-fc93-4d99-ad51-c61675147709
# ╠═╡ disabled = true
#=╠═╡
# Test retest 
let

	# Fit by EV and group
	mm_tests = [fit(
		MixedModel, 
		@formula(right_chosen ~ 1 + empirical_EV_diff + 
			(1 + empirical_EV_diff | prolific_pid)), 
		filter(x -> x.session == s, test_data_clean), 
		Bernoulli()
	) for s in unique(test_data_clean.session)]

	ranefs = (f -> DataFrame(raneftables(f).prolific_pid)).(mm_tests)

	ranefs = innerjoin(
		ranefs[1],
		ranefs[2],
		on = :prolific_pid,
		makeunique = true
	)

	# Plot
	f = Figure()
	workshop_reliability_scatter!(
		f[1, 1];
		df = ranefs,
		xcol = :empirical_EV_diff,
		ycol = :empirical_EV_diff_1,
		xlabel = "Session 1",
		ylabel = "Session 2",
		subtitle = "EV sensitivity"
	)

	# Save plot
	filepath = "results/workshop/test_PILT_sess$(s)_EV_sensitivity_test_retest.png"

	save(filepath, f)

	# upload_to_osf(
	# 		filepath,
	# 		proj,
	# 		osf_folder
	# )
					
	f
end
  ╠═╡ =#

# ╔═╡ 5bcb8ced-603a-461e-91ae-0347445c618e
test_data_clean |> describe

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
	f = Figure()
	
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

	f = Figure()
	
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

# ╔═╡ fcccb531-5d02-4391-b9f7-5c438da53da2
let

	fitted_vals = fitted(mm_test)
	fp = insertcols(
		test_data_clean,
		:simuated => ifelse.(
			test_data_clean.magnitude_right .> test_data_clean.magnitude_left,
			fitted_vals,
			1. .- fitted_vals
		)
	)

	high_chosen_sum = combine(
		groupby(fp, :prolific_pid),
		:high_chosen => mean => :acc,
		:simuated => mean => :sim_acc
	)

	@info "Proportion high magnitude chosen: 
		$(round(mean(high_chosen_sum.acc), digits = 2)), SE=$(round(sem(high_chosen_sum.acc), digits = 2))"

	@info "Simulated proportion high magnitude chosen: 
		$(round(mean(high_chosen_sum.sim_acc), digits = 2)), SE=$(round(sem(high_chosen_sum.sim_acc), digits = 2))"

	# Summarize by participant and magnitude
	test_sum = combine(
		groupby(fp, [:prolific_pid, :magnitude_low, :magnitude_high]),
		:high_chosen => mean => :acc,
		:simuated => mean => :sim_acc
	)

	test_sum_sum = combine(
		groupby(test_sum, [:magnitude_low, :magnitude_high]),
		:acc => mean => :acc,
		:acc => sem => :se,
		:sim_acc => mean => :sim_acc,
	)

	sort!(test_sum_sum, [:magnitude_low, :magnitude_high])

	mp = data(test_sum_sum) *
	mapping(
		:magnitude_high => nonnumeric => "High magntidue",
		:acc => "Prop. chosen high",
		:se,
		layout = :magnitude_low => nonnumeric
	) * (visual(Errorbars) + visual(ScatterLines)) +
	data(test_sum_sum) * mapping(
		:magnitude_high => nonnumeric => "High magntidue",
		:sim_acc => "Prop. chosen high",
		:se,
		layout = :magnitude_low => nonnumeric
	) * (visual(Scatter, marker = :+))

	draw(mp; axis = (; xticklabelrotation = 45))


end

# ╔═╡ Cell order:
# ╠═926b1c86-a34a-11ef-1787-03cf4275cddb
# ╠═56cafcfb-90c3-4310-9b19-aac5ec231512
# ╠═ea6eb668-de64-4aa5-b3ea-8a5bc0475250
# ╠═1a1eb012-16e2-4318-be51-89b2e6a3b55b
# ╠═120babf5-f4c4-4c43-aab4-b3537111d15d
# ╠═fcafa95b-8d34-4221-8bd3-22f5cc5bf16f
# ╠═50f853c8-8e24-4bd8-bb66-9f25e92d0b4b
# ╠═3d3c637f-6278-4f54-acbb-9ff06e8b459b
# ╠═027ff42b-324f-4989-8952-4d119831796a
# ╠═33811ec9-f7c1-499d-9d9d-1a83951004a0
# ╠═b9b974ab-fc93-4d99-ad51-c61675147709
# ╠═5bcb8ced-603a-461e-91ae-0347445c618e
# ╠═c54f34a1-c6bb-4236-a491-25e7a0b96da4
# ╠═ea0f4939-d18f-4407-a13f-d5734cc608bb
# ╠═fcccb531-5d02-4391-b9f7-5c438da53da2
# ╠═fcb0292e-8a86-40c7-a1da-b3f24fbbe492
