### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ 5255888c-7b4a-11ef-231e-e597b3580bd4
begin
	cd("/home/jovyan/")
    import Pkg
    # activate the shared project environment
    Pkg.activate("relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	using Random, DataFrames, JSON, CSV, StatsBase, JLD2, HTTP, CairoMakie, Printf, Distributions, CategoricalArrays, AlgebraOfGraphics
	using LogExpFunctions: logistic, logit
	include("fetch_preprocess_data.jl")
	include("sample_utils.jl")
	include("plotting_utils.jl")
	nothing
end

# ╔═╡ 71b2700e-62a9-4f19-a898-1ac6e11943f3
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

# ╔═╡ e6bd359a-5f0d-4b3a-bf08-96160db9d4a1
# Load data
begin
	PLT_data, test_data, vigour_data, jspsych_data = load_pilot2_data()
	nothing
end

# ╔═╡ 537aa0cb-3f3f-497b-b81e-5f271bfb247c
function bin_sum_EV(
	data::DataFrame;
	group::Union{Nothing, Symbol} = nothing,
	n_bins::Int64 = 5,
	col::Symbol = :empirical_EV_diff, # x axis data column,
	bin_group::Union{Nothing, Int64} = nothing
)
	# Copy data to avoid changing origianl DataFrame
	tdata = copy(data)

	# If no grouping is needed
	if isnothing(group)
		tdata[!, :group] .= 1
	else
		if !isnothing(bin_group)
			
			# Quantile bin breaks
			group_bins = quantile(tdata[!, col], 
				range(0, 1, length=bin_group + 1))

			# Bin group
			tdata.EV_group_cut = 	
				cut(tdata[!, group], group_bins, extend = true)
		
			# Use mean of bin as label
			transform!(
				groupby(tdata, :EV_group_cut),
				group => mean => :group
			)
		else
			rename!(tdata, group => :group)
		end
	end

	# Quantile bin breaks
	EV_bins = quantile(tdata[!, col], 
		range(0, 1, length=n_bins + 1))

	# Bin EV_diff
	tdata.EV_diff_cut = 	
		cut(tdata[!, col], EV_bins, extend = true)

	# Use mean of bin as label
	transform!(
		groupby(tdata, :EV_diff_cut),
		col => mean => :EV_diff_bin
	)

	# Summarize by participant and bin
	choice_EV_sum = combine(
		groupby(tdata, [:prolific_pid, :group, :EV_diff_bin]),
		:right_chosen => mean => :right_chosen
	) |> dropmissing

	# Summarize by bin
	choice_EV_sum = combine(
		groupby(choice_EV_sum, [:group, :EV_diff_bin]),
		:right_chosen => mean => :right_chosen,
		:right_chosen => sem => :se
	)


end

# ╔═╡ a71b8ea1-ba68-43f8-9597-d1b32c3a9413
# Bin and plot choice
function bin_EV_plot(
	f::GridPosition,
	data::AbstractDataFrame;
	group::Union{Nothing, Symbol} = nothing,
	n_bins::Int64 = 5,
	col::Symbol = :empirical_EV_diff, # x axis data column,
	group_label_f::Function = string,
	legend_title = "",
	colors::AbstractVector = Makie.wong_colors(),
	bin_group::Union{Nothing, Int64} = nothing
)

	choice_EV_sum = bin_sum_EV(
		data,
		group = group,
		n_bins = n_bins,
		col = col,
		bin_group = bin_group
	)
	
	# Plot
	ax_diff_choice = Axis(
		f,
		xlabel = "Diff. in EV (£)"
	)

	groups = sort(unique(choice_EV_sum.group))

	for g in groups

		t_sum = filter(x -> x.group == g, choice_EV_sum)

			scatter!(
				ax_diff_choice,
				t_sum.EV_diff_bin,
				t_sum.right_chosen,
				colormap = :viridis
			)
		
			errorbars!(
				ax_diff_choice,
				t_sum.EV_diff_bin,
				t_sum.right_chosen,
				t_sum.se
			)
	end

	if !isnothing(group)
		Legend(
			f,
			[MarkerElement(marker = :circle, color = colors[c]) for c in eachindex(groups)],
			group_label_f.(groups),
			legend_title,
			valign = :top,
			halign = :left,
			tellwidth = false,
			framevisible = false,
			nbanks = 2
		)
	end

	return ax_diff_choice
end

# ╔═╡ 572cf109-ca2c-4da2-950e-7a34a7c2eadd
function compute_optimality(data::AbstractDataFrame)
	
	# Select columns and reduce to task strcuture, which is the same across participants
	optimality = unique(data[!, [:session, :block, :stimulus_pair, :imageLeft, :imageRight, :optimalRight]])

	# Which was the optimal stimulus?
	optimality.optimal = replace.(ifelse.(
		optimality.optimalRight .== 1, 
		optimality.imageRight, 
		optimality.imageLeft
	), "imgs/" => "")

	# Which was the suboptimal stimulus?
	optimality.suboptimal = replace.(ifelse.(
		optimality.optimalRight .== 0, 
		optimality.imageRight, 
		optimality.imageLeft
	), "imgs/" => "")

	# Remove double appearances (right left permutation)
	optimality = unique(optimality[!, [:session, :block, :stimulus_pair, 
		:optimal, :suboptimal]])

	# Wide to long
	optimality = DataFrame(
		stimulus = vcat(optimality.optimal, optimality.suboptimal),
		optimal = vcat(fill(true, nrow(optimality)), fill(false, nrow(optimality)))
	)

	return optimality
end

# ╔═╡ 0260582c-2712-4692-a355-7e37de5af471
function extract_stimulus_magnitude()

	task = DataFrame(CSV.File("./results/pilot2.csv"))

	outcomes = filter(x -> x.feedback_common, task)

	outcomes = vcat(
		rename(
			outcomes[!, [:stimulus_right, :feedback_right]],
			:stimulus_right => :stimulus,
			:feedback_right => :feedback
		),
		rename(
			outcomes[!, [:stimulus_left, :feedback_left]],
			:stimulus_left => :stimulus,
			:feedback_left => :feedback
		)
	)

	outcomes = combine(
		groupby(outcomes, :stimulus),
		:feedback => (x -> mean(unique(x))) => :feedback
	)

	return outcomes

end

# ╔═╡ 78847e80-0600-4079-b32f-e97c2a9bb4ee
# Prepare test data
only_observed_test = let

	# Compute EV from PILT
	PLT_data.chosen_stim = replace.(PLT_data.chosenImg, "imgs/" => "")
	
	empirical_EVs = combine(
		groupby(PLT_data, [:session, :prolific_pid, :chosen_stim]),
		:chosenOutcome => mean => :EV
	)

	# Add empirical EVs to test data
	test_data = leftjoin(
		test_data,
		rename(
			empirical_EVs,
			:chosen_stim => :stimulus_left,
			:EV => :empirical_EV_left
		),
		on = [:session, :prolific_pid, :stimulus_left],
		order = :left
	)

	test_data = leftjoin(
		test_data,
		rename(
			empirical_EVs,
			:chosen_stim => :stimulus_right,
			:EV => :empirical_EV_right
		),
		on = [:session, :prolific_pid, :stimulus_right],
		order = :left
	)

	# Compute empirical EV diff
	test_data.empirical_EV_diff = test_data.empirical_EV_right .- 	
		test_data.empirical_EV_left

	# Keep only test trials where stimulus was observed in PILT
	only_observed_test = filter(x -> !ismissing(x.empirical_EV_diff), test_data)

	# Coarse stimulus magnitude
	magnitudes = extract_stimulus_magnitude()

	# Add to test data
	only_observed_test = leftjoin(
		only_observed_test,
		rename(
			magnitudes,
			:stimulus => :stimulus_left,
			:feedback => :magnitude_left
		),
		on = :stimulus_left,
		order = :left
	)

	only_observed_test = leftjoin(
		only_observed_test,
		rename(
			magnitudes,
			:stimulus => :stimulus_right,
			:feedback => :magnitude_right
		),
		on = :stimulus_right,
		order = :left
	)

	# Compute optimality of each stimulus
	optimality = compute_optimality(PLT_data)

	# Add to test data
	only_observed_test = leftjoin(
		only_observed_test,
		rename(
			optimality,
			:stimulus => :stimulus_left,
			:optimal => :optimal_left
		),
		on = :stimulus_left,
		order = :left
	)

	only_observed_test = leftjoin(
		only_observed_test,
		rename(
			optimality,
			:stimulus => :stimulus_right,
			:optimal => :optimal_right
		),
		on = :stimulus_right,
		order = :left
	)

	only_observed_test

end;

# ╔═╡ 7c47b391-13e4-450a-86a8-c3a0077a68c5
# Describe EV distributions
let
		# Plot distribution of EV difference
	f = Figure(size = (700, 350))

	ax_emp = Axis(
		f[1,1],
		xlabel = "Diff. in empirical EV"
	)

	hist!(ax_emp, 
		only_observed_test.empirical_EV_diff
	)

	ax_exp = Axis(
		f[1,2],
		xlabel = "Diff. in true EV"
	)

	hist!(ax_exp, only_observed_test.EV_diff)

	ax_scatt = Axis(
		f[1,3],
		xlabel = "Diff. in true EV",
		ylabel = "Diff. in empirical EV"
	)

	scatter!(ax_scatt, only_observed_test.EV_diff, only_observed_test.empirical_EV_diff)

	ablines!(ax_scatt, 0., 1., color = :grey, linestyle=:dash)

	f

end

# ╔═╡ 26253722-31a6-4977-973b-2f9d2a4db119
# EV effect on choice
let

	f = Figure(size = (700, 300))
	
	bin_EV_plot(f[1,1], only_observed_test)

	bin_EV_plot(f[1,2], only_observed_test; 
		group = :block,
		legend_title = "Block"
	)

	bin_EV_plot(f[1,3], only_observed_test; 
		group = :same_valence,
		legend_title = "Same original valence"
	)


	f
end

# ╔═╡ d254ed77-7386-4fec-b09d-ef7822e969ae
# Optimality and valence effect on choice
let

	f = Figure()

	# Drop skipped trials
	dat = dropmissing(
		only_observed_test[!, 
			[:prolific_pid, :magnitude_left, :magnitude_right, :optimal_left, :optimal_right, :right_chosen]])

	# Plot by valence ----------------------

	# Compute valence from magnitude
	transform!(
		dat,
		:magnitude_left => ByRow(sign) => :valence_left,
		:magnitude_right => ByRow(sign) => :valence_right
	)

	# Average by participant and valence
	val = combine(
		groupby(dat, [:prolific_pid, :valence_right, :valence_left]),
		:right_chosen => mean => :right_chosen
	)

	# Average by valence
	val = combine(
		groupby(val, [:valence_right, :valence_left]),
		:right_chosen => mean => :right_chosen,
		:right_chosen => sem => :se
	)

	# Create one valence variable
	val.type = 
		join.(
			eachrow(hcat(
				ifelse.(
					val.valence_left .> 0,
					fill("P", nrow(val)),
					fill("N", nrow(val))
				),
				ifelse.(
					val.valence_right .> 0,
					fill("P", nrow(val)),
					fill("N", nrow(val))
				)
			))
		)

	val.type = CategoricalArray(val.type, 
		levels = ["PP", "NN", "NP", "PN"]
			)


	# Plot bars
	bar = data(val) * mapping(:type, :right_chosen) * visual(BarPlot)

	# Plot error bars
	err = data(val) * mapping(:type, :right_chosen, :se => (x -> x*2)) * visual(Errorbars)

	# Plot chance
	hline = mapping([0.5]) * visual(HLines; color = :grey, linestyle = :dash)

	# Put together
	draw!(f[1,1], bar + err + hline; axis = (; xlabel = "Valence", ylabel = "Prop. right chosen"))

	# Plot by optimality
	# Average by participant and valence
	opt = combine(
		groupby(dat, [:prolific_pid, :optimal_right, :optimal_left]),
		:right_chosen => mean => :right_chosen
	)

	# Average by valence
	opt = combine(
		groupby(opt, [:optimal_right, :optimal_left]),
		:right_chosen => mean => :right_chosen,
		:right_chosen => sem => :se
	)

		# Create one valence variable
	opt.type = 
		join.(
			eachrow(hcat(
				ifelse.(
					opt.optimal_left,
					fill("O", nrow(val)),
					fill("S", nrow(val))
				),
				ifelse.(
					opt.optimal_right,
					fill("O", nrow(val)),
					fill("S", nrow(val))
				)
			))
		)

	opt.type = CategoricalArray(opt.type, 
		levels = ["OO", "SS", "SO", "OS"]
			)

	# Plot bars
	let
		bar = data(opt) * mapping(:type, :right_chosen) * visual(BarPlot)
	
		# Plot error bars
		err = data(opt) * mapping(:type, :right_chosen, :se => (x -> x*2)) * visual(Errorbars)
	
		# Plot chance
		hline = mapping([0.5]) * visual(HLines; color = :grey, linestyle = :dash)
	
		# Put together
		draw!(f[1,2], bar + err + hline; axis = (; xlabel = "Optimality", ylabel = "Prop. right chosen"))
	end

	# Plot positive chosen by optimality ----------------
	positive_chosen = filter(x -> x.valence_left != x.valence_right, dat)

	# DV: whether positive was chosen
	positive_chosen.positive_chosen = ifelse.(
		positive_chosen.right_chosen .== 1,
		positive_chosen.valence_right .> 0,
		positive_chosen.valence_left .> 0
	)

	# Create one optimality variable
	positive_chosen.type = 
		ifelse.(
			positive_chosen.optimal_right .&& positive_chosen.optimal_left,
			fill("Both", nrow(positive_chosen)),
			ifelse.(
				(.!positive_chosen.optimal_right) .&& 
					(.!positive_chosen.optimal_left),
				fill("Neither", nrow(positive_chosen)),
				ifelse.(
					(positive_chosen.optimal_right .&& (positive_chosen.valence_right .> 0)) .||(positive_chosen.optimal_left .&& (positive_chosen.valence_left .> 0)),
					fill("P", nrow(positive_chosen)),
					fill("N", nrow(positive_chosen))
				)
			)
		)

	positive_chosen.type = CategoricalArray(positive_chosen.type, 
		levels = ["Both", "Neither", "P", "N"]
			)

	# Average by participant and optimality
	positive_chosen = combine(
		groupby(positive_chosen, [:prolific_pid, :type]),
		:positive_chosen => mean => :positive_chosen
	)

	# Average by optimality
	positive_chosen = combine(
		groupby(positive_chosen, [:type]),
		:positive_chosen => mean => :positive_chosen,
		:positive_chosen => sem => :se
	)

	# Plot bars
	let
		bar = data(positive_chosen) * mapping(:type, :positive_chosen) * visual(BarPlot)
	
		# Plot error bars
		err = data(positive_chosen) * mapping(:type, :positive_chosen, :se => (x -> x*2)) * visual(Errorbars)
	
		# Plot chance
		hline = mapping([0.5]) * visual(HLines; color = :grey, linestyle = :dash)
	
		# Put together
		draw!(f[2,1], bar + err + hline; axis = (; xlabel = "Optimal", ylabel = "Prop. rewarding chosen"))
	end

	
	# Plot optimal chosen by valence ----------------
	optimal_chosen = filter(x -> x.optimal_right != x.optimal_left, dat)

	# DV: whether positive was chosen
	optimal_chosen.optimal_chosen = ifelse.(
		optimal_chosen.right_chosen .== 1,
		optimal_chosen.optimal_right,
		optimal_chosen.optimal_left
	)

	# Create one optimality variable
	optimal_chosen.type = 
		ifelse.(
			optimal_chosen.valence_right .> 0 .&& optimal_chosen.valence_left .> 0,
			fill("Both", nrow(optimal_chosen)),
			ifelse.(
				(optimal_chosen.valence_right .< 0) .&& 
					(optimal_chosen.valence_left .< 0),
				fill("Neither", nrow(optimal_chosen)),
				ifelse.(
					(optimal_chosen.optimal_right .&& (optimal_chosen.valence_right .> 0)) .|| (optimal_chosen.optimal_left .&& (optimal_chosen.valence_left .> 0)),
					fill("O", nrow(optimal_chosen)),
					fill("S", nrow(optimal_chosen))
				)
			)
		)

	# Average by participant and valence
	optimal_chosen = combine(
		groupby(optimal_chosen, [:prolific_pid, :type]),
		:optimal_chosen => mean => :optimal_chosen
	)

	# Average by valence
	optimal_chosen = combine(
		groupby(optimal_chosen, [:type]),
		:optimal_chosen => mean => :optimal_chosen,
		:optimal_chosen => sem => :se
	)

	# Plot bars
	let
		bar = data(optimal_chosen) * mapping(:type, :optimal_chosen) * visual(BarPlot)
	
		# Plot error bars
		err = data(optimal_chosen) * mapping(:type, :optimal_chosen, :se => (x -> x*2)) * visual(Errorbars)
	
		# Plot chance
		hline = mapping([0.5]) * visual(HLines; color = :grey, linestyle = :dash)
	
		# Put together
		draw!(f[2,2], bar + err + hline; axis = (; xlabel = "Rewarding", ylabel = "Prop. optimal chosen"))
	end

	f
end

# ╔═╡ d5967e11-51e7-40a2-ae4a-b5f071234357
# Magnitude effect on choice
let
	dat = dropmissing(only_observed_test[!, [:prolific_pid, :magnitude_left, :magnitude_right, :right_chosen]])

	dat.magnitude_low = minimum(hcat(dat.magnitude_right, dat.magnitude_left), dims = 2) |> vec

	dat.magnitude_high = maximum(hcat(dat.magnitude_right, dat.magnitude_left), dims = 2) |> vec

	dat.high_chosen = ifelse.(
		dat.magnitude_right .== dat.magnitude_high,
		dat.right_chosen,
		.!dat.right_chosen
	)
	
	dat_sum = combine(
		groupby(dat, [:prolific_pid, :magnitude_low, :magnitude_high]),
		:high_chosen => mean => :high_chosen,
		:high_chosen => length => :n
	)

	dat_sum = combine(
		groupby(dat_sum, [:magnitude_low, :magnitude_high]),
		:high_chosen => mean => :high_chosen,
		:high_chosen => sem => :se,
		:n => median => :n
	)

	filter!(x -> !(x.magnitude_low == x.magnitude_high), dat_sum)

	sort!(dat_sum, [:magnitude_low, :magnitude_high])

	dat_sum.high_optimal = (x -> x in [-0.255, -0.01, 0.75, 1.]).(dat_sum.magnitude_high)

	plt = data(dat_sum) * 
		visual(ScatterLines) * 
		mapping(:magnitude_low => nonnumeric, :high_chosen, 
			markersize = :n, layout = :magnitude_high => nonnumeric, 
			color = :high_optimal => nonnumeric)

	err = data(dat_sum) * 
		mapping(
			:magnitude_low => nonnumeric, 
			:high_chosen, 
			:se, 
			color = :high_optimal => nonnumeric,
			layout = :magnitude_high => nonnumeric) * 
		visual(Errorbars)
	hline = mapping(0.5) * visual(HLines, color = :grey, linestyle = :dash)

	spans = DataFrame(
		low = (1:7) .- 0.5,
		high = (1:7) .+ 0.5,
		optimal = [false, false, true, true, false, false, true]
	)

	color = [:red, :blue]

	vspans = data(spans) * mapping(:low, :high, color = :optimal => nonnumeric => AlgebraOfGraphics.scale(:secondary)) * visual(VSpan)


	draw(vspans + plt + err + hline, 
		scales(
			secondary = (; palette = [(:red, 0.2), (:green, 0.2)]), 
			Color = (; palette = [:red, :green])); 
		legend = (; show = false), 
		axis = (; xticklabelrotation=45.0, 
			xlabel = "Low magnitude",
			ylabel = "Prop. chosen high magnitude"
		)
	)
end

# ╔═╡ Cell order:
# ╠═5255888c-7b4a-11ef-231e-e597b3580bd4
# ╠═71b2700e-62a9-4f19-a898-1ac6e11943f3
# ╠═e6bd359a-5f0d-4b3a-bf08-96160db9d4a1
# ╠═78847e80-0600-4079-b32f-e97c2a9bb4ee
# ╠═7c47b391-13e4-450a-86a8-c3a0077a68c5
# ╠═26253722-31a6-4977-973b-2f9d2a4db119
# ╠═d254ed77-7386-4fec-b09d-ef7822e969ae
# ╠═d5967e11-51e7-40a2-ae4a-b5f071234357
# ╠═537aa0cb-3f3f-497b-b81e-5f271bfb247c
# ╠═a71b8ea1-ba68-43f8-9597-d1b32c3a9413
# ╠═572cf109-ca2c-4da2-950e-7a34a7c2eadd
# ╠═0260582c-2712-4692-a355-7e37de5af471
