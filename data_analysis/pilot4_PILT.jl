### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ b9941a88-9176-11ef-273e-21e7ccc577d7
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

# ╔═╡ be912151-01a2-4cc4-9075-0ecf3594c217
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

# ╔═╡ 6a851ca1-400f-4e89-808c-3da1e837d48b
begin
	PILT_data, _, _, _ = load_pilot4_data()
	PILT_data = exclude_PLT_sessions(PILT_data, required_n_blocks = 18)
	PILT_data_clean = filter(x -> x.choice != "noresp", PILT_data)
end

# ╔═╡ a949f606-2860-4129-bda4-041dcc095e15
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

# ╔═╡ 80a5c7c9-f54b-40e5-bc49-5e4318520b00
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

	ax.xticks = [1, 10, 20, 30]

	save("results/pilot4_acc_curve.png", f, pt_per_unit = 1)

	f

end

# ╔═╡ cd41d166-1299-4c82-b528-dd181b1f341d
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

# ╔═╡ bee3847c-fa5d-4344-b4ca-e7491c863722
# Plot average accuracy by set-size
let

	f = Figure(size = (600, 350))
	
	# Plot average accuracy per set size and participant
	acc_set_size = combine(
		groupby(PILT_data_clean, [:prolific_pid, :n_pairs]),
		:isOptimal => mean => :acc
	)

	# Plot
	mp1 = data(acc_set_size) * mapping(
		:n_pairs => nonnumeric => "Set size", 
		:acc => "Proportion optimal choice", 
		color = :n_pairs => nonnumeric) *
		visual(RainClouds)

	draw!(f[1,1], mp1)

	# Plot with lines between participants
	plot_specific_trials_diff_by_participant!(
		f[1,2];
		PILT_data = PILT_data_clean,
		selection_col = :trial,
		selection_values = 1:40,
		x_levels = [1, 3, 5]
	)


	save("results/pilot4_acc_set_size.png", f, pt_per_unit = 1)

	f
end

# ╔═╡ 51d4a3b8-f76e-4ac1-80fe-3ea11b5a67e8
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

# ╔═╡ 929549e1-acc7-4785-8015-17dd5ca6b51b
# Plot overall accuracy reliability
let

	f = Figure(size = (700, 250))

	plot_specific_trials_reliability!(
		f[1,1];
		PILT_data = PILT_data_clean,
		selection_col = :trial,
		selection_values = 1:40, # All
		difference_function = df -> df.n_pairs_1 .- 
			df.n_pairs_5,
		subtitle = "Set size 1 - 5"
	)

	plot_specific_trials_reliability!(
		f[1,2];
		PILT_data = PILT_data_clean,
		selection_col = :trial,
		selection_values = 1:40, # All
		difference_function = df -> df.n_pairs_3 .- 
			df.n_pairs_5,
		subtitle = "Set size 3 - 5"
	)

	plot_specific_trials_reliability!(
		f[1,3];
		PILT_data = PILT_data_clean,
		selection_col = :trial,
		selection_values = 1:40, # All
		difference_function = df -> df.n_pairs_1 .- 
			df.n_pairs_3,
		subtitle = "Set size 1 - 3"
	)

	save("results/pilot4_acc_set_size_reliability.png", f, pt_per_unit = 1)


	f

end

# ╔═╡ db24a086-cd2b-4306-bafd-2ab33d9063ce
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

	save("results/pilot4_acc_curve_reliability.png", f, pt_per_unit = 1)

	f

end

# ╔═╡ 6edf746f-3499-4429-b532-bf2c80dfb1f0
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
		selection_values = 2:4,
		difference_function = (df -> df.n_pairs_1 .- 
			df.n_pairs_5),
		subtitle = "Set size 1 - 5, trials 2-4"
	)

	save("results/pilot4_acc_curve_realigned.png", f, pt_per_unit = 1)

	f

end

# ╔═╡ Cell order:
# ╠═b9941a88-9176-11ef-273e-21e7ccc577d7
# ╠═be912151-01a2-4cc4-9075-0ecf3594c217
# ╠═6a851ca1-400f-4e89-808c-3da1e837d48b
# ╠═a949f606-2860-4129-bda4-041dcc095e15
# ╠═bee3847c-fa5d-4344-b4ca-e7491c863722
# ╠═929549e1-acc7-4785-8015-17dd5ca6b51b
# ╠═80a5c7c9-f54b-40e5-bc49-5e4318520b00
# ╠═db24a086-cd2b-4306-bafd-2ab33d9063ce
# ╠═cd41d166-1299-4c82-b528-dd181b1f341d
# ╠═51d4a3b8-f76e-4ac1-80fe-3ea11b5a67e8
# ╠═6edf746f-3499-4429-b532-bf2c80dfb1f0
