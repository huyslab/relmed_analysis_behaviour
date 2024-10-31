### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ 122f9972-95ea-11ef-366f-2dd3a1fe1ff0
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
	nothing
end

# ╔═╡ bb220633-95e6-45b4-a4ac-3e725c6029dc
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

# ╔═╡ 571456f4-e2c8-436e-9c61-dffa2af8c778
# Set up saving to OSF
begin
	osf_folder = "Lab notebook images/PILT/"
	proj = setup_osf("Task development")
end

# ╔═╡ 4014436c-b206-4832-b9aa-ce854e428a0f
begin
	PILT_data, _, _, _ = load_pilot4x_data()
	filter!(x -> x.version == "4.1", PILT_data)
	PILT_data = exclude_PLT_sessions(PILT_data, required_n_blocks = 18)
	PILT_data_clean = filter(x -> x.choice != "noresp", PILT_data)
end

# ╔═╡ 20270f36-2a3f-4f64-9c1b-5509c69aeade
# Auxillary variables
begin
	# Make sure sorted
	sort!(PILT_data_clean, [:prolific_pid, :block, :trial])

	# Create appearance variable
	DataFrames.transform!(
		groupby(PILT_data_clean, [:prolific_pid, :block, :stimulus_pair]),
		:trial => (x -> 1:length(x)) => :appearance
	)
end

# ╔═╡ fd00327f-7529-4aa0-b81d-30a9e5821ea3
# Plot PILT accuracy curve
let

	f = Figure()

	ax = plot_group_accuracy!(
		f[1,1],
		group = :n_pairs,
		PILT_data_clean,
		legend = Dict(i => "$i" for i in sort(unique(PILT_data_clean.n_pairs))),
		legend_title = "Set size"
	)

	ax.xticks = vcat([1], 7:7:49)

	filpath = "results/pilot4.1_acc_curve.png"
	save(filpath, f, pt_per_unit = 1)

	upload_to_osf(
		filpath,
		proj,
		osf_folder
	)

	f

end

# ╔═╡ 540f5917-f14d-4d0b-8ec4-aa6f35fc819e
function plot_specific_trials_diff_by_participant!(
	f::GridPosition;
	PILT_data::DataFrame,
	selection_col::Symbol,
	selection_values::AbstractVector,
	x_levels::AbstractVector,
	subtitle::String = ""
)

	# Subset
	sum_epoch = filter(x -> (x[selection_col] in selection_values) & (x.n_pairs in x_levels), PILT_data)

	# Summarize accuracy by participant, set size
	sum_epoch = combine(
		groupby(
			sum_epoch,
			[:prolific_pid, :n_pairs]
		),
		:isOptimal => (x -> mean(filter(y -> !ismissing(y), x))) => :acc
	)

	mp = data(sum_epoch) * 
		mapping(:n_pairs => nonnumeric => "Set size", :acc => "Prop. optimal choice", group = :prolific_pid, color = :prolific_pid) * 
		(visual(Lines, alpha = 0.5, color = :grey) + visual(Scatter))

	draw!(f, mp; axis = (; subtitle = subtitle, 
		xautolimitmargin = (1 / length(x_levels), 1 / length(x_levels))))

end

# ╔═╡ 7c6ca554-5af7-4861-a06c-70d5bddb272b
function plot_specific_trials_reliability!(
	f::GridPosition;
	PILT_data::DataFrame,
	selection_col::Symbol,
	selection_values::AbstractVector,
	difference_function::Function,
	subtitle::String
)
	# Plot test retest of difference in beginning -----
	sum_epoch = filter(x -> x[selection_col] in selection_values, PILT_data)

	# Split blocks into two halves
	blocks = unique(sum_epoch[!, [:block, :n_pairs]])

	DataFrames.transform!(
		groupby(blocks, :n_pairs), 
		:block => (x -> 1:length(x)) => :reblock) 

	blocks.split = ifelse.(
		blocks.reblock .<= maximum(blocks.reblock) ÷ 2, 
		fill(1, nrow(blocks)),
		fill(2, nrow(blocks))
	)

	sum_epoch = leftjoin(
		sum_epoch,
		blocks[!, [:block, :split]],
		on = :block,
		order = :left
	)

	# Summarize accuracy by participant, set size, and split
	sum_epoch = combine(
		groupby(
			sum_epoch,
			[:prolific_pid, :n_pairs, :split]
		),
		:isOptimal => (x -> mean(filter(y -> !ismissing(y), x))) => :acc
	)

	# Long to wide
	sum_epoch = unstack(sum_epoch, 
		[:prolific_pid, :split], 
		:n_pairs,
		:acc,
		renamecols = x -> "n_pairs_$x"
	)

	# Compute difference
	sum_epoch.diff = difference_function(sum_epoch)

	# Long to wide over splits
	sum_epoch = unstack(
		sum_epoch,
		[:prolific_pid],
		:split,
		:diff,
		renamecols = x -> "split_$x"
	)

	# Remove missing due to missing data
	dropmissing!(sum_epoch)

	# Compute correlation
	sbr = spearman_brown(cor(sum_epoch.split_1, sum_epoch.split_2))

	mp = data(sum_epoch) * mapping(
		:split_1 => "First half",
		:split_2 => "Second half"
	) * (linear() + visual(Scatter)) +
		mapping([0], [1]) * visual(ABLines, color = :grey, linestyle = :dash)

	draw!(f, mp, axis = (; subtitle = subtitle))

	if sbr > 0
		Label(
			f,
			"SB r=$(round(sbr, digits = 2))",
			halign = 0.9,
			valign = 0.1,
			tellwidth = false,
			tellheight = false
		)
	end
end

# ╔═╡ 84624920-b1d8-46a2-92c5-a6763fabc82d
# Plot PILT accuracy curve reliability
let

	f = Figure(size = (500, 800))
	
	# Plot test retest of difference in the beginning -----
	plot_specific_trials_diff_by_participant!(
		f[1,1];
		PILT_data = PILT_data_clean,
		selection_col = :trial,
		selection_values = 2:6,
		x_levels = [1, 7],
		subtitle = "Trials 2-6"
	)
	
	plot_specific_trials_reliability!(
		f[1,2];
		PILT_data = PILT_data_clean,
		selection_col = :trial,
		selection_values = 2:6,
		difference_function = df -> df.n_pairs_1 .- 
			df.n_pairs_7,
		subtitle = "Set size 1 - 7, trials 2-6"
	)

	plot_specific_trials_diff_by_participant!(
		f[2,1];
		PILT_data = PILT_data_clean,
		selection_col = :trial,
		selection_values = 2:6,
		x_levels = [1, 3],
		subtitle = "Trials 2-6"
	)

	plot_specific_trials_reliability!(
		f[2,2];
		PILT_data = PILT_data_clean,
		selection_col = :trial,
		selection_values = 2:6,
		difference_function = df -> df.n_pairs_1 .- 
			df.n_pairs_3,
		subtitle = "Set size 1 - 3, trials 2-6"
	)

	plot_specific_trials_diff_by_participant!(
		f[3,1];
		PILT_data = PILT_data_clean,
		selection_col = :trial,
		selection_values = 9:15,
		x_levels = [3, 7],
		subtitle = "Trials 9-15"
	)

	plot_specific_trials_reliability!(
		f[3,2];
		PILT_data = PILT_data_clean,
		selection_col = :trial,
		selection_values = 9:15,
		difference_function = df -> df.n_pairs_3 .- 
			df.n_pairs_7,
		subtitle = "Set size 3 - 7, trials 9-15"
	)

	colsize!(f.layout, 1, Relative(0.4))
	rowgap!(f.layout, 1, 40)
	rowgap!(f.layout, 2, 40)

	# Save and push to OSF
	filepath = "results/pilot4.1_acc_curve_reliability.png"
	save(filepath, f, pt_per_unit = 1)

	upload_to_osf(
		filepath,
		proj,
		osf_folder
	)

	f

end

# ╔═╡ 33ef055d-1d52-4c59-8423-1e1d8042bb74
let
	# Plot PILT accuracy curve by appearance

	f = Figure(size = (700, 300))

	PILT_remmaped = copy(PILT_data_clean)

	DataFrames.transform!(
		groupby(PILT_remmaped, [:session, :prolific_pid, :exp_start_time, :block, :stimulus_pair]),
		:trial => (x -> 1:length(x)) => :trial
	)
	
	ax = plot_group_accuracy!(
		f[1,1],
		group = :n_pairs,
		PILT_remmaped,
		legend = Dict(i => "$i" for i in sort(unique(PILT_data_clean.n_pairs))),
		legend_title = "Set size"
	)

	ax.xlabel = "Appearance #"

	# Plot test retest of difference in the beginning -----
	plot_specific_trials_reliability!(
		f[1,2];
		PILT_data = PILT_data_clean,
		selection_col = :appearance,
		selection_values = 2:3,
		difference_function = (df -> df.n_pairs_1 .- 
			df.n_pairs_7),
		subtitle = "Set size 1 - 7, trials 2-4"
	)

	
	filepath = "results/pilot4.1_acc_curve_realigned.png"
	save(filepath, f, pt_per_unit = 1)

	upload_to_osf(
		filepath,
		proj,
		osf_folder
	)
	
	f

end

# ╔═╡ Cell order:
# ╠═122f9972-95ea-11ef-366f-2dd3a1fe1ff0
# ╠═bb220633-95e6-45b4-a4ac-3e725c6029dc
# ╠═571456f4-e2c8-436e-9c61-dffa2af8c778
# ╠═4014436c-b206-4832-b9aa-ce854e428a0f
# ╠═20270f36-2a3f-4f64-9c1b-5509c69aeade
# ╠═fd00327f-7529-4aa0-b81d-30a9e5821ea3
# ╠═84624920-b1d8-46a2-92c5-a6763fabc82d
# ╠═33ef055d-1d52-4c59-8423-1e1d8042bb74
# ╠═540f5917-f14d-4d0b-8ec4-aa6f35fc819e
# ╠═7c6ca554-5af7-4861-a06c-70d5bddb272b
