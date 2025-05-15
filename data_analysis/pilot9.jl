### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ f4428174-30cc-11f0-29db-4d087f316fe5
begin
	cd("/home/jovyan/")
    import Pkg
    # activate the shared project environment
    Pkg.activate("relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	using Random, DataFrames, JSON, CSV, StatsBase, JLD2, HTTP, CairoMakie, Printf, Distributions, CategoricalArrays, AlgebraOfGraphics, Dates, Turing, SHA, HypothesisTests, Tidier
	using Tidier, GLM, MixedModels, PlutoUI, LaTeXStrings, ColorSchemes
	using LogExpFunctions: logistic, logit
	import OpenScienceFramework as OSF
	include("fetch_preprocess_data.jl")
	include("sample_utils.jl")
	include("plotting_utils.jl")
	include("stats_utils.jl")
	include("osf_utils.jl")
	include("model_utils.jl")
	include("PILT_models.jl")
	nothing
end

# ╔═╡ 5e6cb37e-7b4d-4188-8d2f-58362a3cf981
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

	spearman_brown(
	r;
	n = 2 # Number of splits
	) = (n * r) / (1 + (n - 1) * r)
end

# ╔═╡ 7f61cd3a-0e9a-47ba-8708-ca4e5816262d
begin
	PILT_data, test_data, _, _, jspsych_data = load_pilot9_data(;
		force_download=true)
	nothing
end

# ╔═╡ 23f77f57-7880-4415-9f97-59178d0d4b4d
PILT_data_clean = let
	# Clean data
	PILT_data_clean = exclude_PLT_sessions(PILT_data, required_n_blocks = 21)
	filter!(x -> x.response != "noresp", PILT_data_clean)

	# Remove empty columns
	select!(
		PILT_data_clean,
		Not([:EV_right, :EV_left])
	)
end

# ╔═╡ 46eb5012-f5b5-4174-bf5a-32fc305bd4e9
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

	# Plot
	mp = (data(acc_curve) * mapping(
		:trial => "Trial #",
		:acc => "Prop. optimal choice",
		group = :prolific_pid,
		color = :prolific_pid
	) * visual(Lines, linewidth = 1, alpha = 0.7)) +
	(data(acc_curve_sum) * 
	mapping(
		:trial => "Trial #",
		:acc => "Prop. optimal choice"
	) * visual(Lines, linewidth = 4))
	
	
	draw(mp; legend = (; show = false))
end

# ╔═╡ 93ef440c-4c75-4239-84d2-ba1fe3c9a2ea
let

	# Extract trial plan from data
	unique_feedback = unique(select(PILT_data_clean, [:session, :block, :trial, :stimulus_right, :stimulus_left, :feedback_right, :feedback_left]))

	# Make sure it is the same for all participants
	@assert nrow(unique_feedback) == maximum(PILT_data_clean.block) * 10 "Trial plan not the same for all participants"

	# Feedback per stimulus, regardless of presentations side
	unique_feedback = vcat(
		[select(
			unique_feedback,
			:session,
			:block,
			:trial,
			Symbol("stimulus_$side") => :stimulus,
			Symbol("feedback_$side") => :feedback
		) for side in ["right", "left"]]...
	)

	# Average for EV
	EVs = combine(
		groupby(unique_feedback, [:session, :block, :stimulus]),
		:feedback => mean => :EV
	)

	# Merge back into data
	for side in ["right", "left"]
		leftjoin!(
			PILT_data_clean,
			select(
				EVs,
				:session,
				:block,
				:stimulus => Symbol("stimulus_$side"),
				:EV => Symbol("EV_$side")
			),
			on = [:session, :block, Symbol("stimulus_$side")]
		)
	end

	# EV difference variable
	PILT_data_clean.EV_diff = PILT_data_clean.EV_right .- PILT_data_clean.EV_left
end

# ╔═╡ 44471eb1-69ff-454e-9ac9-217f080bca7c
PILT_data_clean.types = let df = copy(PILT_data_clean)
		task = DataFrame(CSV.File("data/pilot9_PILT.csv"))

	stimuli = unique(task[!, [:session, :block, :stimulus_A, :primary_outcome_A, :stimulus_B, :primary_outcome_B]])

	stimuli = vcat(
		[select(
			stimuli,
			:session,
			:block,
			Symbol("stimulus_$ltr") => :stimulus,
			Symbol("primary_outcome_$ltr") => :primary_outcome
		) for ltr in ["A", "B"]]...
	)

	stimuli.stimulus = (x -> "imgs/PILT_stims/$x").(stimuli.stimulus)

	
	side = "right"
	for side in ["right", "left"]
		leftjoin!(
			df,
			select(
				stimuli,
				:stimulus => Symbol("stimulus_$side"),
				:primary_outcome => Symbol("primary_outcome_$side")
			),
			on = Symbol("stimulus_$side")
		)
	end

	df.primary_same = df.primary_outcome_right .== df.primary_outcome_left

	df.primary_same_valence = sign.(df.primary_outcome_right) .== sign.(df.primary_outcome_left)

	df.types = ifelse.(
		df.primary_same,
		"Identical",
		ifelse.(
			df.primary_same_valence,
			"Same valence",
			"Opposing valence"
		)
	)

	df.types = CategoricalArray(df.types, levels = ["Identical", "Same valence", "Opposing valence"])
end

# ╔═╡ 213393d4-c02a-41e3-a798-0130ca4bd500
let df = copy(PILT_data_clean)

	# Sumarrize by participant and trial
	acc_curve = combine(
		groupby(df, [:prolific_pid, :types, :trial]),
		:response_optimal => mean => :acc
	)

	# Summarize by trial
	acc_curve_sum = combine(
		groupby(acc_curve, [:types, :trial]),
		:acc => mean => :acc,
		:acc => sem => :se
	)

	# Error bands
	acc_curve_sum.lb = acc_curve_sum.acc - acc_curve_sum.se
	acc_curve_sum.ub = acc_curve_sum.acc + acc_curve_sum.se

	# Sort
	sort!(acc_curve_sum, [:types, :trial])

	# Plot
	mp = data(acc_curve_sum) * 
	(
		mapping(
			:trial => "Trial #",
			:acc => "Prop. optimal choice",
			color = :types => "Common outcomes"
		) * visual(Lines, linewidth = 4) +
		mapping(
			:trial => "Trial #",
			:lb => "Prop. optimal choice",
			:ub => "Prop. optimal choice",
			color = :types => "Common outcomes"
		) * visual(Band, alpha = 0.5)
	) + mapping([5]) * visual(VLines, color = :grey, linestyle = :dash) +
	mapping([0.5]) * visual(HLines, color = :grey, linestyle = :dash)
	
	f = Figure()
	plt = draw!(f[1,1], mp; axis = (;yticks = 0.5:0.1:0.9))
	legend!(
		f[0,1], 
		plt,
		tellwidth = false,
		framevisible = false,
		orientation = :horizontal
	)

	f

end

# ╔═╡ 7cc07d9e-c6f2-4218-a8a5-d00647dd0caa
# Tell fitting functions the column names
pilt_columns = Dict(
	"block" => :cblock,
	"trial" => :trial,
	"feedback_optimal" => :feedback_optimal,
	"feedback_suboptimal" => :feedback_suboptimal,
	"choice" => :response_optimal
)

# ╔═╡ c41ef11f-8b7a-4265-b62f-49a0f9e94f4f
# Split half
fits_by_types = let df = copy(PILT_data_clean)

	# Create feedback_optimal and feedback_suboptimal
	df.feedback_optimal = ifelse.(
		df.optimal_right .== 1,
		df.feedback_right,
		df.feedback_left
	)

	df.feedback_suboptimal = ifelse.(
		df.optimal_right .== 0,
		df.feedback_right,
		df.feedback_left
	)

	df.abs_EV_diff = abs.(df.EV_diff)

	# Make sure block is Int64
	df.block = Int.(disallowmissing(df.block))

	fits = Dict()
	for t in unique(df.types)
		gdf = filter(x -> x.types == t, df)
		
		blocks = unique(gdf[!, [:session, :block, :abs_EV_diff]])
		
		blocks.half = vcat(fill(1, nrow(blocks) ÷ 2), fill(2, nrow(blocks) ÷ 2 + rem(nrow(blocks), 2)))
	
		sort!(blocks, [:session, :half, :block])
	
		DataFrames.transform!(
			groupby(blocks, :half),
			:block => (x -> 1:length(x)) => :cblock
		)

		@info t
		@info combine(groupby(blocks, [:half]), 
			:abs_EV_diff => mean,
			:abs_EV_diff => std,
			:abs_EV_diff => minimum,
			:abs_EV_diff => maximum,
			:abs_EV_diff => length
		)
	
		leftjoin!(
			gdf,
			select(blocks, [:block, :cblock, :half]),
			on = :block
		)
		
		# Make sure block is Int64
		gdf.cblock = Int.(disallowmissing(gdf.cblock))
	
		sort!(gdf, [:prolific_pid, :half, :cblock, :trial])
	
		select(gdf, pushfirst!(collect(values(pilt_columns)), :prolific_pid, :block, :half))
		
		fits[only(unique(gdf.types))] = optimize_multiple_by_factor(
			gdf;
			model = single_p_QL_recip,
			factor = :half,
			priors = Dict(
				:ρ => truncated(Normal(0, 5.), lower = 0.),
				:a => Normal(0, 2.)
			),
			unpack_function = unpack_single_p_QL,
			remap_columns = pilt_columns
		)
	end

	vcat([insertcols(v, :types => k) for (k, v) in fits]...)
		
end

# ╔═╡ 5a2a693e-3b5b-4646-b75e-2e14cc14bd45
# Test retest of parameters
let
	f = Figure(size = (700, 900))

	for (j, t) in enumerate(unique(fits_by_types.types))

		gdf = filter(x -> x.types == t, fits_by_types)

		# Run over parameters
		for (i, (p, st, tf)) in enumerate(zip(
			[:a, :ρ], 
			["Learning rate", "Reward sensitivity"],
			[x -> string.(round.(a2α.(x), digits = 1)), Makie.automatic]
		))
	
	
			# Long to wide
			this_retest = unstack(
				gdf,
				:prolific_pid,
				:half,
				p,
				renamecols = (x -> "$(p)_$x")
			)
	
			# Plot
			workshop_reliability_scatter!(
				f[j, i];
				df = dropmissing!(this_retest),
				xcol = Symbol("$(p)_1"),
				ycol = Symbol("$(p)_2"),
				xlabel = "Split 1",
				ylabel = ["Split 2", ""][i],
				subtitle = "$t\n$st",
				tickformat = tf,
				correct_r = true,
				markersize = 5
			)
	
		end
	end
	
	f
end

# ╔═╡ 928661b2-eda5-4ac5-9db8-062f87257ca3
# Split half
fits_splithalf = let df = copy(PILT_data_clean)

	# Split into halves stratified on EV_diff
	df.abs_EV_diff = abs.(df.EV_diff)

	filter!(x -> x.abs_EV_diff > 1., df)

	blocks = unique(df[!, [:session, :block, :abs_EV_diff]])
	
	blocks.half = vcat(fill(1, nrow(blocks) ÷ 2), fill(2, nrow(blocks) ÷ 2 + rem(nrow(blocks), 2)))

	sort!(blocks, [:session, :half, :block])

	DataFrames.transform!(
		groupby(blocks, :half),
		:block => (x -> 1:length(x)) => :cblock
	)

	@info combine(groupby(blocks, :half), 
		:abs_EV_diff => mean,
		:abs_EV_diff => std,
		:abs_EV_diff => minimum,
		:abs_EV_diff => maximum,
		:abs_EV_diff => length
	)

	leftjoin!(
		df,
		select(blocks, [:block, :cblock, :half]),
		on = :block
	)

	# Create feedback_optimal and feedback_suboptimal
	df.feedback_optimal = ifelse.(
		df.optimal_right .== 1,
		df.feedback_right,
		df.feedback_left
	)

	df.feedback_suboptimal = ifelse.(
		df.optimal_right .== 0,
		df.feedback_right,
		df.feedback_left
	)

	# Make sure block is Int64
	df.cblock = Int.(df.cblock)

	sort!(df, [:prolific_pid, :half, :cblock, :trial])

	select(df, pushfirst!(collect(values(pilt_columns)), :prolific_pid, :block, :half))
	
	fits = optimize_multiple_by_factor(
		df;
		model = single_p_QL_recip,
		factor = :half,
		priors = Dict(
			:ρ => truncated(Normal(0, 5.), lower = 0.),
			:a => Normal(0, 2.)
		),
		unpack_function = unpack_single_p_QL,
		remap_columns = pilt_columns
	)
end

# ╔═╡ 672153f5-f2bd-481a-aa76-c1394ecdf681
let df = copy(PILT_data_clean)

	# Split into halves stratified on EV_diff
	df.abs_EV_diff = abs.(df.EV_diff)

	filter!(x -> x.abs_EV_diff > 1., df)

	blocks = unique(df[!, [:session, :block, :abs_EV_diff]])
	
	blocks.half = vcat(fill(1, nrow(blocks) ÷ 2), fill(2, nrow(blocks) ÷ 2 + rem(nrow(blocks), 2)))

	sort!(blocks, [:session, :half, :block])

	DataFrames.transform!(
		groupby(blocks, :half),
		:block => (x -> 1:length(x)) => :cblock
	)

	@info combine(groupby(blocks, :half), 
		:abs_EV_diff => mean,
		:abs_EV_diff => std,
		:abs_EV_diff => minimum,
		:abs_EV_diff => maximum,
		:abs_EV_diff => length
	)

	leftjoin!(
		df,
		select(blocks, [:block, :cblock, :half]),
		on = :block
	)

	df_sum = combine(
		groupby(df, [:prolific_pid, :half, :block]),
		:response_optimal => mean => :response_optimal
	)

	df_sum = combine(
		groupby(df_sum, [:prolific_pid, :half]),
		:response_optimal => mean => :response_optimal
	)

	# Long to wide
	retest = unstack(
		df_sum,
		:prolific_pid,
		:half,
		:response_optimal
	)

	@info cor(retest[!, Symbol("1")], retest[!, Symbol("2")])

	draw(data(retest) * mapping(Symbol("1"), Symbol("2")) * visual(Scatter))

end


# ╔═╡ c44884ce-c91f-4651-b9b3-73d473ac7c61
# Test retest of parameters
let
	f = Figure(
	)

	# Run over parameters
	for (i, (p, st, tf)) in enumerate(zip(
		[:a, :ρ], 
		["Learning rate", "Reward sensitivity"],
		[x -> string.(round.(a2α.(x), digits = 1)), Makie.automatic]
	))


		# Long to wide
		this_retest = unstack(
			fits_splithalf,
			:prolific_pid,
			:half,
			p,
			renamecols = (x -> "$(p)_$x")
		)

		# Plot
		RLDM_reliability_scatter!(
			f[1, i];
			df = dropmissing!(this_retest),
			xcol = Symbol("$(p)_1"),
			ycol = Symbol("$(p)_2"),
			xlabel = "Split 1",
			ylabel = ["Split 2", ""][i],
			subtitle = "$st",
			tickformat = tf,
			correct_r = false,
			markersize = 2
		)

	end

	colgap!(f.layout, 10)
	
	f
end

# ╔═╡ e408ac2e-6767-4d9d-b8ce-b8efced23288
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

	sort!(forfit, [:prolific_pid, :cblock, :trial])

	return forfit

end

# ╔═╡ 1d5ad1bc-2e93-4211-a08e-f5d5be23ccc1
"""
    quantile_bin_centers(x::AbstractVector, nbins::Int)

Bins the vector `x` into `nbins` equal-probability (quantile) bins and labels each value by its bin center.

# Arguments
- `x::AbstractVector`: Input data to bin.
- `nbins::Int`: Number of quantile bins.

# Returns
- `centers::Vector{Float64}`: Vector of bin center values, same length as `x`.

# Example
```julia
x = randn(100)
centers = quantile_bin_centers(x, 4)
```
"""
function quantile_bin_centers(x::AbstractVector, nbins::Int)
    # Compute quantile edges
    edges = quantile(x, range(0, 1; length=nbins+1))
    # Assign each value to a bin
    bin_idx = map(v -> searchsortedlast(edges, v), x)
    # Fix rightmost edge
    bin_idx = map(i -> min(i, nbins), bin_idx)
    # Compute bin centers
    centers = [round((edges[i] + edges[i+1]) / 2, digits = 2) for i in 1:nbins]
    # Label each value by its bin center
    return [centers[i] for i in bin_idx]
end

# ╔═╡ 2d8e283c-9215-4eaa-8447-2772b7e26645
let df = copy(PILT_data_clean)

	# Absolute value of EV difference
	df.abs_EV_diff = abs.(df.EV_diff)

	# Bin variable
	df.EV_bin = quantile_bin_centers(df.abs_EV_diff, 4)

	
	# Sumarrize by participant and trial
	acc_curve = combine(
		groupby(df, [:prolific_pid, :EV_bin, :trial]),
		:response_optimal => mean => :acc
	)

	# Summarize by trial
	acc_curve_sum = combine(
		groupby(acc_curve, [:EV_bin, :trial]),
		:acc => mean => :acc,
		:acc => sem => :se
	)

	# Error bands
	acc_curve_sum.lb = acc_curve_sum.acc - acc_curve_sum.se
	acc_curve_sum.ub = acc_curve_sum.acc + acc_curve_sum.se

	# Sort
	sort!(acc_curve_sum, [:EV_bin, :trial])

	# Plot
	mp = data(acc_curve_sum) * 
	(
		mapping(
			:trial => "Trial #",
			:acc => "Prop. optimal choice",
			color = :EV_bin => nonnumeric => "Abs. EV difference"
		) * visual(Lines, linewidth = 4) +
		mapping(
			:trial => "Trial #",
			:lb => "Prop. optimal choice",
			:ub => "Prop. optimal choice",
			color = :EV_bin => nonnumeric => "Abs. EV difference"
		) * visual(Band, alpha = 0.5)
	) + mapping([5]) * visual(VLines, color = :grey, linestyle = :dash) +
	mapping([0.5]) * visual(HLines, color = :grey, linestyle = :dash)
	
	f = Figure()
	plt = draw!(f[1,1], mp; axis = (;yticks = 0.5:0.1:0.9))
	legend!(
		f[0,1], 
		plt,
		tellwidth = false,
		framevisible = false,
		orientation = :horizontal,
		titleposition = :left
	)

	f

end

# ╔═╡ f6ccb84b-d67d-4b69-b44b-7ae6540cd1b1
"""
    add_stratified_split_group(df::DataFrame, col::Symbol; prop::Float64=0.5, nbins::Int=5, rng=Random.GLOBAL_RNG, group_col::Symbol=:split_group)

Adds a column to `df` assigning each row to a group (1 or 2) such that the groups have similar distributions of `col`.
- `prop`: Proportion in group 1 (default 0.5).
- `nbins`: Number of bins for stratification (default 5).
- `group_col`: Name of the new group column (default :split_group).
Returns the modified DataFrame.
"""
function stratified_split_group(df::DataFrame, col::Symbol; prop::Float64=0.5, nbins::Int=5, rng=Xoshiro(0), group_col::Symbol=:split_group)
    bins = quantile_bin_centers(df[!, col], nbins)
    group_assignments = similar(bins, Int)
    for b in unique(bins)
        inds = findall(bins .== b)
        n1 = round(Int, length(inds) * prop)
        inds = shuffle(rng, inds)
        group_assignments[inds[1:n1]] .= 1
        group_assignments[inds[n1+1:end]] .= 2
    end
    return group_assignments
end

# ╔═╡ fe7479df-fde4-41de-936a-af4f88b8cf8a
describe(PILT_data_clean)

# ╔═╡ Cell order:
# ╠═f4428174-30cc-11f0-29db-4d087f316fe5
# ╠═5e6cb37e-7b4d-4188-8d2f-58362a3cf981
# ╠═7f61cd3a-0e9a-47ba-8708-ca4e5816262d
# ╠═23f77f57-7880-4415-9f97-59178d0d4b4d
# ╠═46eb5012-f5b5-4174-bf5a-32fc305bd4e9
# ╠═93ef440c-4c75-4239-84d2-ba1fe3c9a2ea
# ╠═44471eb1-69ff-454e-9ac9-217f080bca7c
# ╠═213393d4-c02a-41e3-a798-0130ca4bd500
# ╠═c41ef11f-8b7a-4265-b62f-49a0f9e94f4f
# ╠═5a2a693e-3b5b-4646-b75e-2e14cc14bd45
# ╠═2d8e283c-9215-4eaa-8447-2772b7e26645
# ╠═f6ccb84b-d67d-4b69-b44b-7ae6540cd1b1
# ╠═7cc07d9e-c6f2-4218-a8a5-d00647dd0caa
# ╠═928661b2-eda5-4ac5-9db8-062f87257ca3
# ╠═672153f5-f2bd-481a-aa76-c1394ecdf681
# ╠═c44884ce-c91f-4651-b9b3-73d473ac7c61
# ╠═e408ac2e-6767-4d9d-b8ce-b8efced23288
# ╠═1d5ad1bc-2e93-4211-a08e-f5d5be23ccc1
# ╠═fe7479df-fde4-41de-936a-af4f88b8cf8a
