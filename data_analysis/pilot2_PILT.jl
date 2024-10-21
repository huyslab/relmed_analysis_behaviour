### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ ff8c6dd6-7b4a-11ef-3148-ad43f7d07801
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

# ╔═╡ 3f653d16-588f-4a30-8746-105c93305156
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

# ╔═╡ 2b7c2f4c-36cc-4008-bb44-0e34d40e3d92
# Load data
begin
	PLT_data, test_data, vigour_data, jspsych_data = load_pilot2_data()
	PLT_data = exclude_PLT_sessions(PLT_data, required_n_blocks = 18)
end

# ╔═╡ 21b3018f-4856-4b34-b7d2-f3549a6fc556
unique(jspsych_data.trialphase)

# ╔═╡ ec053072-8386-4000-b425-88496b82bb1d
# Plot average accuracy by set-size
let

	f = Figure(size = (700, 300))
	
	# Plot average accuracy per set size and participant
	acc_set_size = combine(
		groupby(PLT_data, [:prolific_pid, :n_pairs]),
		:isOptimal => mean => :acc
	)

	# Plot
	mp1 = data(acc_set_size) * mapping(
		:n_pairs => nonnumeric => "Set size", 
		:acc => "Proportion optimal choice", 
		color = :n_pairs => nonnumeric) *
		visual(RainClouds)

	draw!(f[1,1], mp1)

	# Summarize difference between set size 3 and 1 per paritipant and split
	set_size_diff_split = filter(x -> x.n_pairs != 2, PLT_data)

	# Split blocks into two halves
	blocks = unique(set_size_diff_split[!, [:block, :n_pairs]])
	
	DataFrames.transform!(
		groupby(blocks, :n_pairs), 
		:block => (x -> 1:length(x)) => :reblock) 

	blocks.split = ifelse.(
		isodd.(blocks.reblock), 
		fill(1, nrow(blocks)),
		fill(2, nrow(blocks))
	)

	set_size_diff_split = leftjoin(
		set_size_diff_split,
		blocks[!, [:block, :split]],
		on = :block,
		order = :left
	)

	# Summarize difference by participant, set size, and split
	set_size_diff_split = combine(
		groupby(set_size_diff_split, [:prolific_pid, :split, :n_pairs]),
		:isOptimal => (x -> mean(filter(y -> !ismissing(y), x))) => :acc
	)

	# Long to wide
	set_size_diff_split = unstack(set_size_diff_split, 
		[:prolific_pid, :split], 
		:n_pairs,
		:acc,
		renamecols = x -> "n_pairs_$x"
	)

	# Compute difference
	set_size_diff_split.diff = set_size_diff_split.n_pairs_3 - 
		set_size_diff_split.n_pairs_1

	# Long to wide over splits
	set_size_diff_split = unstack(
		set_size_diff_split,
		[:prolific_pid],
		:split,
		:diff,
		renamecols = x -> "split_$x"
	)

	# Remove missing due to missing data
	dropmissing!(set_size_diff_split)

	r = spearman_brown(cor(set_size_diff_split.split_1, set_size_diff_split.split_2))
	
	mp2 = 
		data(set_size_diff_split) * mapping(
		:split_1 => "Set size 3 - 1 odd blocks",
		:split_2 => "Set size 3 - 1 even blocks"
	) * (linear() + visual(Scatter)) +
		mapping([0], [1]) * visual(ABLines, color = :grey, linestyle = :dash)

	draw!(f[1,2], mp2, 
		axis = (; subtitle = "Spearman-Brown r=$(round(r, digits = 2))")
	)

	save("results/pilot2_acc_set_size.png", f, pt_per_unit = 1)

	f
end

# ╔═╡ 60559c55-840f-49cb-9537-1aceaaf5e658
# Plot PLT accuracy curve
let

	f = Figure()

	ax = plot_group_accuracy!(
		f[1,1],
		group = :n_pairs,
		PLT_data,
		legend = Dict(i => "$i" for i in 1:3),
		legend_title = "Set size"
	)

	ax.xticks = [1, 10, 20, 30]

	save("results/pilot2_acc_curve.png", f, pt_per_unit = 1)

	f

end

# ╔═╡ cb470684-4557-4cb4-a6c2-90a207d4dcc6
# Plot PLT accuracy curve
let

	PLT_remmaped = copy(PLT_data)

	DataFrames.transform!(
		groupby(PLT_remmaped, [:session, :prolific_pid, :exp_start_time, :block, :stimulus_pair]),
		:trial => (x -> 1:length(x)) => :trial
	)
	
	f = Figure()

	ax = plot_group_accuracy!(
		f[1,1],
		group = :n_pairs,
		PLT_remmaped,
		legend = Dict(i => "$i" for i in 1:3),
		legend_title = "Set size"
	)

	ax.xticks = [1, 10, 20, 30]

	save("results/pilot2_acc_curve_realigned.png", f, pt_per_unit = 1)

	f

end

# ╔═╡ 868309ad-7553-4414-b146-01e19e2815cd
# Sort by stimulus pair rather than block for naive fitting
PLT_remmaped = let
	PLT_remmaped = copy(PLT_data)

	# Make sure sorted
	sort!(
		PLT_remmaped,
		[:session, :prolific_pid, :exp_start_time, :block, :trial]
	)

	# Renumber trials
	DataFrames.transform!(
		groupby(PLT_remmaped, [:session, :prolific_pid, :exp_start_time, :block, :stimulus_pair]),
		:trial => (x -> 1:length(x)) => :trial
	)

	PLT_remmaped.block = PLT_remmaped.stimulus_pair_id

	# Sort by new block
	sort!(
		PLT_remmaped, 
		[:session, :prolific_pid, :exp_start_time, :block, :trial]
	)

	# Make sure sorted correctly
	@assert all(combine(groupby(PLT_remmaped, [:session, :prolific_pid, :exp_start_time, :block]), :trial => issorted => :sorted).sorted)

	PLT_remmaped

end

# ╔═╡ a54ab4d5-6b59-4fb7-a3d4-43deeb01d685


# ╔═╡ 22d12a3f-c8ca-4c46-8ff3-b7e1432a77d5
# Plot accuracy by switch / repeat
let
	f = Figure(
		#size = (700, 300)
	)
	
	DataFrames.transform!(
		groupby(PLT_data, [:prolific_pid, :block]),
		:stimulus_pair => (x -> [missing; x[1:end-1]]) => :prev_pair
	)

	switch_acc = filter(x -> !ismissing(x.prev_pair), PLT_data)
	
	switch_acc.switch = ifelse.(
		switch_acc.stimulus_pair .== switch_acc.prev_pair,
		fill("no", nrow(switch_acc)),
		fill("yes", nrow(switch_acc))
	)

	switch_acc_sum = combine(
		groupby(switch_acc, 
			[:prolific_pid, :switch]),
		:isOptimal => mean => :acc
	)


	mp1 = data(switch_acc_sum) * mapping(:switch => "Switch trial", :acc => "Proportion optimal choice", color = :switch) * visual(RainClouds)
	
	draw!(f[1,1], mp1)

	switch_acc.split = ifelse.(
		switch_acc.block .> 9, 
		fill(2, nrow(switch_acc)),
		fill(1, nrow(switch_acc))
	)

	switch_acc_split = combine(
		groupby(switch_acc, [:prolific_pid, :split, :switch]),
		:isOptimal => mean => :acc
	)

	# Long to wide
	switch_acc_split = unstack(
		switch_acc_split,
		[:prolific_pid, :split],
		:switch,
		:isOptimal
	)

	# r = spearman_brown(cor(switch_acc.yes, switch_acc.no)) 

	# mp2 = data(switch_acc) * mapping(:yes, :no) * (linear() + visual(Scatter)) +
	# 	mapping([0], [1]) * visual(ABLines, color = :grey, linestyle = :dash)

	# draw!(f[1,2], mp2; axis = (; subtitle = "Spearman-Brown r=$(round(r, digits = 2))"))

	# fs
end

# ╔═╡ Cell order:
# ╠═ff8c6dd6-7b4a-11ef-3148-ad43f7d07801
# ╠═3f653d16-588f-4a30-8746-105c93305156
# ╠═21b3018f-4856-4b34-b7d2-f3549a6fc556
# ╠═2b7c2f4c-36cc-4008-bb44-0e34d40e3d92
# ╠═ec053072-8386-4000-b425-88496b82bb1d
# ╠═60559c55-840f-49cb-9537-1aceaaf5e658
# ╠═cb470684-4557-4cb4-a6c2-90a207d4dcc6
# ╠═868309ad-7553-4414-b146-01e19e2815cd
# ╠═a54ab4d5-6b59-4fb7-a3d4-43deeb01d685
# ╠═22d12a3f-c8ca-4c46-8ff3-b7e1432a77d5
