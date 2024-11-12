### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ 8cf30b5e-a020-11ef-23b2-2da6e9116b54
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

# ╔═╡ 82ef300e-536f-40ce-9cde-72056e6f4b5e
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

# ╔═╡ 595c642e-32df-448e-81cc-6934e2152d70
# Set up saving to OSF
begin
	osf_folder = "Workshop figures/PILT"
	proj = setup_osf("Task development")
end

# ╔═╡ 14a292db-43d4-45d8-97a5-37ffc03bdc5c
begin
	# Load data
	PILT_data, _, _, _, _, _,
		_, _ = load_pilot6_data()
	nothing
end

# ╔═╡ 6ed82686-35ab-4afd-a1b2-6fa19ae67168
PILT_data_clean = let
	# Clean data
	PILT_data_clean = exclude_PLT_sessions(PILT_data, required_n_blocks = 20)
	PILT_data_clean = filter(x -> x.response != "noresp", PILT_data_clean)
end

# ╔═╡ c1f763df-382a-4320-a12b-9a6a981f213e


# ╔═╡ b5b75f4e-7b91-4287-a409-6f0ebdf20f4e
# Accuracy curveֿ
let

	# Sumarrize by participant and trial
	acc_curve = combine(
		groupby(PILT_data_clean, [:prolific_pid, :trial]),
		:response_optimal => mean => :acc
	)

	sort!(acc_curve, [:prolific_pid, :trial])

	# Summarize by trial
	acc_curve_sum = combine(
		groupby(acc_curve, :trial),
		:acc => mean => :acc,
		:acc => sem => :se
	)

	# Compute bounds
	insertcols!(
		acc_curve_sum,
		:lb => acc_curve_sum.acc .- acc_curve_sum.se,
		:ub => acc_curve_sum.acc .+ acc_curve_sum.se
	)

	# Create plot mapping
	mp = (
	# Error band
		mapping(
		:trial => "Trial #",
		:lb => "Prop. optimal choice",
		:ub => "Prop. optimal choice"
	) * visual(Band) +
	# Average line	
		mapping(
		:trial => "Trial #",
		:acc => "Prop. optimal choice"
	) * visual(Lines, linewidth = 4)
	)

	# Mathcing line
	matching = mapping([0.86]) * visual(HLines, linestyle = :dash)

	# Plot whole data
	f1 = Figure()
	
	draw!(f1, data(acc_curve_sum) * mp + matching;
		axis = (; xautolimitmargin = (0, 0))
	)

	# Plot up to trial 5
	f2 = Figure()
	
	draw!(
		f2, 
		data(filter(x -> x.trial <= 5, acc_curve_sum)) * mp;
		axis = (; xautolimitmargin = (0, 0))
	)


	# Plot with matching level
	f3 = Figure()
	
	draw!(
		f3, 
		data(filter(x -> x.trial <= 5, acc_curve_sum)) * 
		mp + 
		matching;
		axis = (; xautolimitmargin = (0, 0))
	)

	# Link axes
	linkaxes!(contents(f1[1,1])[1], contents(f2[1,1])[1], contents(f3[1,1])[1])

	# Save
	filepaths = [joinpath("results/workshop", "PILT_acc_curve_$k.png") for k in ["full", "partial", "partial_line"]]

	save.(filepaths, [f1, f2, f3])

	# for fp in filepaths
	# 	upload_to_osf(
	# 		fp,
	# 		proj,
	# 		osf_folder
	# 	)
	# end

	f1, f2, f3
	
end

# ╔═╡ 18b19cd7-8af8-44ad-8b92-d40a2cfff8b4
# Accuracy curveֿ by valence
let

	# Sumarrize by participant, valence, and trial
	acc_curve = combine(
		groupby(PILT_data_clean, [:prolific_pid, :valence, :trial]),
		:response_optimal => mean => :acc
	)

	sort!(acc_curve, [:prolific_pid, :valence, :trial])

	# Summarize by trial and valence
	acc_curve_sum = combine(
		groupby(acc_curve, [:valence, :trial]),
		:acc => mean => :acc,
		:acc => sem => :se
	)

	# Compute bounds
	insertcols!(
		acc_curve_sum,
		:lb => acc_curve_sum.acc .- acc_curve_sum.se,
		:ub => acc_curve_sum.acc .+ acc_curve_sum.se
	)

	# Labels for valence
	acc_curve_sum.val_lables = CategoricalArray(
		ifelse.(
			acc_curve_sum.valence .> 0,
			fill("Reward", nrow(acc_curve_sum)),
			fill("Punishment", nrow(acc_curve_sum))
		),
		levels = ["Reward", "Punishment"]
	)

	# Create plot mapping
	mp = (
	# Error band
		mapping(
		:trial => "Trial #",
		:lb => "Prop. optimal choice",
		:ub => "Prop. optimal choice",
		color = :val_lables => ""
	) * visual(Band, alpha = 0.5) +
	# Average line	
		mapping(
		:trial => "Trial #",
		:acc => "Prop. optimal choice",
		color = :val_lables => ""
	) * visual(Lines, linewidth = 4)
	)

	# Plot whole figure
	f1 = Figure()
	
	plt1 = draw!(f1[1,1], data(acc_curve_sum) * mp; 
		axis = (; xautolimitmargin = (0, 0)))

	legend!(
		f1[0,1], 
		plt1,
		tellwidth = false,
		framevisible = false,
		orientation = :horizontal
	)

	# Fix order of layers
	reorder_bands_lines!(f1[1,1])

	# Plot first five trials
	f2 = Figure()
	
	draw!(f2[1,1], data(filter(x -> x.trial <= 5, acc_curve_sum)) * mp; 
		axis = (; xautolimitmargin = (0, 0)))

	legend!(
		f2[0,1], 
		plt1,
		tellwidth = false,
		framevisible = false,
		orientation = :horizontal
	)


	# Fix order of layers
	reorder_bands_lines!(f2[1,1])

	
	# Link axes
	linkaxes!(extract_axis(f1[1,1]), extract_axis(f2[1,1]))

	# Save
	filepaths = [joinpath("results/workshop", "PILT_acc_curve_valence_$k.png") for k in ["full", "partial"]]

	save.(filepaths, [f1, f2])

	# for fp in filepaths
	# 	upload_to_osf(
	# 		fp,
	# 		proj,
	# 		osf_folder
	# 	)
	# end

	f1, f2
	
end

# ╔═╡ c40ea9ef-0d50-4889-a28a-778a14b0dec7
# Tell fitting functions the column names
pilt_columns = Dict(
	"block" => :cblock,
	"trial" => :trial,
	"feedback_optimal" => :feedback_optimal,
	"feedback_suboptimal" => :feedback_suboptimal,
	"choice" => :response_optimal
)

# ╔═╡ d1c6974f-462f-453f-bad1-536200b660fa
llb

# ╔═╡ 1e50c77d-fd05-46af-9f3f-eefcda7fa8f9
function sample_predictive(
	parameters::NamedTuple;
	model::Function,
	task::NamedTuple,
	n_samples::Int64 = 1,
	random_seed::Int64 = 0
)

	# Load task into model
	task_model = model(; 
		task...
	)

	# Condition on fitted values
	cond_model = condition(task_model, parameters)

	# Sample given values
	pp = sample(
		Xoshiro(random_seed),
		cond_model,
		Prior(),
		n_samples
	)

	choic_cols = filter(x -> occursin("choice", string(x)), names(pp))

	# Collect and return
	pp = collect(transpose(Array(pp[:, choic_cols, 1])))

	if n_samples == 1
		pp = vec(pp)
	end
	
	return pp
end

# ╔═╡ 0b6979e1-c4a6-47a8-b4bf-3ee123366ee1
function sample_predictive_multiple(
	fits::DataFrame;
	model::Function,
	task::DataFrame,
	task_unpack_function::Function,
	task_unpack_columns::Dict,
	dv_col::Symbol,
	grouping_cols::Vector{Symbol} = [:prolific_pid],
	n_samples::Int64 = 1,
	random_seed::Int64 = 0
)
	
	# Add dv column as missing values
	dtask = insertcols(task, dv_col => missing)

	# Convert to NamedTuple for model
	task_tuple = task_unpack_function(
		dtask;
		columns = task_unpack_columns
	)

	# Prepare for loop
	pps = []
	lk = ReentrantLock()

	# Run over groups
	groups = unique(fits[!, grouping_cols])
	
	
	Threads.@threads for i in 1:nrow(groups)

		# Select data
		conditions = Dict(col => groups[i, col] for col in names(groups))

		gdf = filter(row -> all(row[col] == conditions[col] for col in keys(conditions)), fits)

		# Extract parameter values
		param_values = NamedTuple(only(select(gdf, vcat(grouping_cols, [:lp]))))

		# Sample
		pp = sample_predictive(
			param_values;
			model = model,
			task = task_tuple,
			n_samples = n_samples,
			random_seed = random_seed
		)

		# DV pairs for inserting into DataFrame
		if n_samples == 1
			dv_pairs = [dv_col => pp]
		else
			dv_pairs = [Symbol("$(string(dv_col))_$i") => pp[:, i] 
				for i in 1:size(pp, 2)
			]
		end

		# Grouping column pairs for inserting into DataFrame
		grouping_pairs = [col => gdf[!, col][1] for col in grouping_cols]

		# Push DataFrame
		lock(lk) do
			push!(
				pps,
				insertcols(
					task,
					grouping_pairs...,
					dv_pairs...
				)
			)
		end

	end

	# Combine to single DataFrame
	return vcat(pps...)

end

# ╔═╡ 491111db-721d-424e-a989-747052bfd494
function early_stop_block(
	response_optimal::AbstractVector;
	n_groups::AbstractVector = fill(1, length(response_optimal)),
	group::AbstractVector = fill(1, length(response_optimal)), 
	criterion::Int64 = 5
)

	n_trials = length(n_groups)

	np = n_groups[1]
	
	# Initialize result vector
	exclude = fill(false, n_trials)

	# Consecutive optimal chioce counter
	consecutive_optimal = fill(0, np)

	for i in 1:n_trials

		# Check condition
		if i > 1
			exclude[i] = 
				exclude[i - 1] || # Already stopped
				all(consecutive_optimal .>= criterion) # Criterion met on previous trial	
		end

		# Update counter
		consecutive_optimal[group[i]] = 
			(response_optimal[i] == 1.) ? (consecutive_optimal[group[i]] + 1) : 0

	end

	return exclude
end

# ╔═╡ a1f3d0c3-4595-49c3-9ba1-e1a55019836a
# Prepare data for fit
function prepare_data(
	PILT_data_clean::DataFrame
)
	
	# Make sure sorted
	forfit = sort(PILT_data_clean, [:prolific_pid, :block, :trial])

	# Make sure block is Int64
	forfit.block = Int.(forfit.block)

	# Create feedback_optimal and feedback_suboptimal
	forfit.feedback_optimal = ifelse.(
		forfit.optimal_right .== 1,
		forfit.feedback_right,
		forfit.feedback_left
	)

	forfit.feedback_suboptimal = ifelse.(
		forfit.optimal_right .== 0,
		forfit.feedback_right,
		forfit.feedback_left
	)

	# Cumulative block number
	forfit.cblock = forfit.block .+ (parse.(Int, forfit.session) .- 1) .* maximum(forfit.block)

	# Split for reliability
	forfit.half = ifelse.(
		forfit.block .< median(unique(forfit.block)),
		fill(1, nrow(forfit)),
		fill(2, nrow(forfit))
	)

	return forfit

end

# ╔═╡ d26f4afb-d734-40af-97aa-9604db2a335a
function fit_by_factor(
	PILT_data_clean::DataFrame;
	model::Function,
	factor::Symbol,
	priors::Dict,
	unpack_function::Function,
	remap_columns::Dict
)

	fits = []

	levels = sort(unique(PILT_data_clean[!, factor]))

	for l in levels

		# Subset data
		gdf = filter(x -> x[factor] == l, PILT_data_clean)

		# Fit data
		fit = optimize_multiple(
			gdf;
			model = model,
			unpack_function = df -> unpack_function(df; columns = remap_columns),
		    priors = priors,
			grouping_col = :prolific_pid,
			n_starts = 10
		)

		# Add factor variable
		insertcols!(fit, factor => l)

		push!(fits, fit)
	end

	# Combine to single DataFrame
	fits = vcat(fits...)

	# Sort
	sort!(fits, [factor, :prolific_pid])

	return fits

end

# ╔═╡ a5b29872-3854-4566-887f-35d6e53479f6
fits_by_valence = let	
	# Fit data
	fits = fit_by_factor(
		prepare_data(PILT_data_clean);
		model = single_p_QL,
		factor = :valence,
		priors = Dict(
			:ρ => truncated(Normal(0., 5.), lower = 0.),
			:a => Normal(0., 2.)
		),
		unpack_function = unpack_single_p_QL,
		remap_columns = pilt_columns
	)
end

# ╔═╡ 4b9732c4-e74e-4775-8ff0-531be8576c30
# Plot parameters by valence
let fits = fits_by_valence
	# Perform t tests
	pvals = []
	for p in [:ρ, :a]
	
		ρ_wide = unstack(
			fits,
			:prolific_pid,
			:valence,
			p,
			renamecols = (x -> "valence_$x")
		) |> disallowmissing

		ttest = OneSampleTTest(ρ_wide.valence_1, ρ_wide[!, Symbol("valence_-1")])

		@info "$p: $ttest"
		push!(pvals, 
			pvalue(ttest)
		)
	end

	# Translate p_value to stars
	star_pval(p) = ["***", "**", "*", "ns."][findfirst(p .< [0.001, 0.01, 0.05, 1.])]

	# Tranform a to α
	fits.α = a2α.(fits.a)

	# Labels for valence
	fits.val_lables = CategoricalArray(
		ifelse.(
			fits.valence .> 0,
			fill("Reward", nrow(fits)),
			fill("Punishment", nrow(fits))
		),
		levels = ["Reward", "Punishment"]
	)

	# Plot
	f1 = Figure()

	for (i, (p, l)) in 
		enumerate(zip([:ρ, :α], ["Reward sensitivity", "Learning rate"]))
		
		mp = data(fits) * 
		(mapping(
			:valence => (x -> sign.(x) .* (abs.(x) .- 0.2)) => "",
			p => l,
			group = :prolific_pid
		) * visual(Lines, color = :grey, alpha = 0.1) +
		mapping(
			:valence => (x -> sign.(x) .* (abs.(x) .- 0.2)) => "",
			p => l,
			color = :valence => (x -> nonnumeric(sign.(x) .* (abs.(x) .- 0.2))) => "",
			group = :prolific_pid
		) * visual(Scatter, alpha = 0.3, markersize = 6) +
		mapping(
			:valence => (x -> sign.(x) .* (abs.(x) .+ 0.2)) => "",
			p => l,
			color = :valence => (x -> nonnumeric(sign.(x) .* (abs.(x) .- 0.2))) => "",
		) * visual(BoxPlot, width = 0.5, show_notch = true, outliercolor = :white))
	
	
		draw!(f1[1, i], mp,
			scales(Color = (; palette = Makie.wong_colors()[[2,1]])); 
			axis = (; 
				xticks = ([-1, 1],  ["Punishment", "Reward"]), 
				xreversed = true,
				subtitle = star_pval(pvals[i])
			)
		)
	end

	# Save
	filepath = "results/workshop/PILT_valence_parameters.png"

	save(filepath, f1)

	# upload_to_osf(
	# 		filepath,
	# 		proj,
	# 		osf_folder
	# )

	f1
end

# ╔═╡ 379b1456-06e9-4181-acca-25dfe2687ea4
ppc = let
	task = DataFrame(CSV.File("data/pilot6_pilt.csv"))

	# Cumulative block number
	task.cblock = task.block .+ (task.session .- 1) .* maximum(task.block)

	# Select relevant columns
	select!(task, 
		[:session, :cblock, :trial, :valence, :feedback_optimal, :feedback_suboptimal])
	
	# Disallow missing
	disallowmissing!(task, [:feedback_optimal, :feedback_optimal, :trial, :cblock])

	# Sort
	sort!(task, [:cblock, :trial])

	# Simulate
	ppc = vcat([sample_predictive_multiple(
		select(filter(x -> x.valence == v, fits_by_valence), [:prolific_pid, :a, :ρ, :lp]);
		model = single_p_QL,
		task = filter(x -> x.valence == v, task),
		task_unpack_function = unpack_single_p_QL,
		task_unpack_columns = pilt_columns,
		dv_col = :response_optimal,
		grouping_cols = [:prolific_pid],
		n_samples = 10
	) for v in unique(task.valence)]...)

	# Checks
	@assert all(combine(
		groupby(ppc, [:prolific_pid, :valence]),
		:cblock => issorted => :block_sorted
	).block_sorted)

	@assert all(combine(
		groupby(ppc, [:prolific_pid, :cblock]),
		:trial => issorted => :trial_sorted
	).trial_sorted)

	# Wide to long
	ppc = stack(
		ppc,
		filter(x -> occursin("response_optimal", x), names(ppc)),
		[:prolific_pid, :cblock, :trial, :valence],
		variable_name = :draw,
		value_name = :response_optimal
	)

	# Pretty draw index
	ppc.draw = (s -> parse(Int, replace(s, "response_optimal_" => ""))).(ppc.draw)

	# Apply early stopping
	sort!(ppc, [:draw, :prolific_pid, :valence, :cblock, :trial])
	DataFrames.transform!(
		groupby(ppc, [:prolific_pid, :draw, :cblock]),
		:response_optimal => early_stop_block => :early_stop_exclude
	)

	filter!(x -> !x.early_stop_exclude, ppc)

	ppc
end

# ╔═╡ 3f44cf3e-e2dc-4f8b-a592-a73a9f39f7b2
let
	# Summarize ppc by draw, participant, valence, trial
	ppc_curve = combine(
		groupby(ppc, [:draw, :prolific_pid, :valence, :trial]),
		:response_optimal => mean => :acc
	)

	# Summarize by draw, trial and valence
	ppc_curve = combine(
		groupby(ppc_curve, [:draw, :valence, :trial]),
		:acc => mean => :acc
	)

	# Summarize by and valence
	ppc_curve = combine(
		groupby(ppc_curve, [:valence, :trial]),
		:acc => mean => :acc,
		:acc => llb => :llb,
		:acc => lb => :lb,
		:acc => uub => :uub,
		:acc => ub => :ub,
	)

	# Labels for valence
	ppc_curve.val_lables = CategoricalArray(
		ifelse.(
			ppc_curve.valence .> 0,
			fill("Reward", nrow(ppc_curve)),
			fill("Punishment", nrow(ppc_curve))
		),
		levels = ["Reward", "Punishment"]
	)


	# Create plot mapping
	mp = data(ppc_curve) * (
	# Error band
		mapping(
			:trial => "Trial #",
			:llb => "Prop. optimal choice",
			:uub => "Prop. optimal choice",
			color = :val_lables => ""
	) * visual(Band, alpha = 0.1) +
		mapping(
			:trial => "Trial #",
			:lb => "Prop. optimal choice",
			:ub => "Prop. optimal choice",
			color = :val_lables => ""
	) * visual(Band, alpha = 0.5) +
	# Average line	
		mapping(
			:trial => "Trial #",
			:acc => "Prop. optimal choice",
			color = :val_lables => ""
	) * visual(Lines, linewidth = 4)
	)

	draw(mp)

end

# ╔═╡ Cell order:
# ╠═8cf30b5e-a020-11ef-23b2-2da6e9116b54
# ╠═82ef300e-536f-40ce-9cde-72056e6f4b5e
# ╠═595c642e-32df-448e-81cc-6934e2152d70
# ╠═14a292db-43d4-45d8-97a5-37ffc03bdc5c
# ╠═6ed82686-35ab-4afd-a1b2-6fa19ae67168
# ╠═c1f763df-382a-4320-a12b-9a6a981f213e
# ╠═b5b75f4e-7b91-4287-a409-6f0ebdf20f4e
# ╠═18b19cd7-8af8-44ad-8b92-d40a2cfff8b4
# ╠═c40ea9ef-0d50-4889-a28a-778a14b0dec7
# ╠═a5b29872-3854-4566-887f-35d6e53479f6
# ╠═4b9732c4-e74e-4775-8ff0-531be8576c30
# ╠═379b1456-06e9-4181-acca-25dfe2687ea4
# ╠═3f44cf3e-e2dc-4f8b-a592-a73a9f39f7b2
# ╠═d1c6974f-462f-453f-bad1-536200b660fa
# ╠═1e50c77d-fd05-46af-9f3f-eefcda7fa8f9
# ╠═0b6979e1-c4a6-47a8-b4bf-3ee123366ee1
# ╠═491111db-721d-424e-a989-747052bfd494
# ╠═a1f3d0c3-4595-49c3-9ba1-e1a55019836a
# ╠═d26f4afb-d734-40af-97aa-9604db2a335a
