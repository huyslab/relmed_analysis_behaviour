### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ 2197c534-95d5-11ef-27a1-79dc7b442a4b
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

# ╔═╡ 9e09a52f-bba6-41fb-b86c-034d9d5b5187
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

# ╔═╡ 31f05093-bc6a-4c81-921e-e595f5cf9f98
# Set up saving to OSF
begin
	osf_folder = "Lab notebook images/PILT/"
	proj = setup_osf("Task development")
end

# ╔═╡ 23b197c9-99d8-46ec-b773-5f6a104aace7
begin
	PILT_data, _, _, _ = load_pilot4x_data()
	filter!(x -> x.version == "4.3", PILT_data)
	PILT_data = exclude_PLT_sessions(PILT_data, required_n_blocks = 18)
	PILT_data_clean = filter(x -> x.choice != "noresp", PILT_data)
end

# ╔═╡ 9eee81f7-8dc4-4586-9e5e-20953c7a8e61
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

# ╔═╡ 8c78ccca-2666-4255-8a48-7c0c7f82573e
# Plot PILT accuracy curve
resp = let

	f = Figure()

	ax = plot_group_accuracy!(
		f[1,1],
		group = :n_pairs,
		PILT_data_clean,
		legend = Dict(i => "$i" for i in sort(unique(PILT_data_clean.n_pairs))),
		legend_title = "Set size"
	)

	ax.xticks = [1, 10, 20, 30]

	filpath = "results/pilot4.3_acc_curve.png"
	save(filpath, f, pt_per_unit = 1)

	upload_to_osf(
		filpath,
		proj,
		osf_folder
	)

	f

end

# ╔═╡ c66df4e3-4654-42f3-a796-8ed1df618c9d
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

# ╔═╡ 38c10ee4-c288-4f2a-a9ce-950368c41dbb
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

# ╔═╡ 5c1e7335-05d5-4889-b9d0-796c81b1fe2e
# Plot PILT accuracy curve reliability
let

	f = Figure(size = (500, 800))
	
	# Plot test retest of difference in the beginning -----
	plot_specific_trials_diff_by_participant!(
		f[1,1];
		PILT_data = PILT_data_clean,
		selection_col = :trial,
		selection_values = 2:6,
		x_levels = [1, 5],
		subtitle = "Trials 2-6"
	)
	
	plot_specific_trials_reliability!(
		f[1,2];
		PILT_data = PILT_data_clean,
		selection_col = :trial,
		selection_values = 2:6,
		difference_function = df -> df.n_pairs_1 .- 
			df.n_pairs_5,
		subtitle = "Set size 1 - 5, trials 2-6"
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
		x_levels = [3, 5],
		subtitle = "Trials 9-15"
	)

	plot_specific_trials_reliability!(
		f[3,2];
		PILT_data = PILT_data_clean,
		selection_col = :trial,
		selection_values = 9:15,
		difference_function = df -> df.n_pairs_3 .- 
			df.n_pairs_5,
		subtitle = "Set size 3 - 5, trials 9-15"
	)

	colsize!(f.layout, 1, Relative(0.4))
	rowgap!(f.layout, 1, 40)
	rowgap!(f.layout, 2, 40)

	# Save and push to OSF
	filepath = "results/pilot4.3_acc_curve_reliability.png"
	save(filepath, f, pt_per_unit = 1)

	upload_to_osf(
		filepath,
		proj,
		osf_folder
	)

	f

end

# ╔═╡ Cell order:
# ╠═2197c534-95d5-11ef-27a1-79dc7b442a4b
# ╠═9e09a52f-bba6-41fb-b86c-034d9d5b5187
# ╠═31f05093-bc6a-4c81-921e-e595f5cf9f98
# ╠═23b197c9-99d8-46ec-b773-5f6a104aace7
# ╠═9eee81f7-8dc4-4586-9e5e-20953c7a8e61
# ╠═8c78ccca-2666-4255-8a48-7c0c7f82573e
# ╠═5c1e7335-05d5-4889-b9d0-796c81b1fe2e
# ╠═c66df4e3-4654-42f3-a796-8ed1df618c9d
# ╠═38c10ee4-c288-4f2a-a9ce-950368c41dbb
