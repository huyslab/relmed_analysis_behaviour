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