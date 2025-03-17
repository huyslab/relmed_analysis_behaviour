### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ 2d7211b4-b31e-11ef-3c0b-e979f01c47ae
begin
	cd("/home/jovyan")
	import Pkg
	
	# activate the shared project environment
    Pkg.activate("$(pwd())/relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	using CairoMakie, Random, DataFrames, Distributions, StatsBase,
		ForwardDiff, LinearAlgebra, JLD2, FileIO, CSV, Dates, JSON, RCall, Turing, Printf, Combinatorics, JuMP, HiGHS, AlgebraOfGraphics, CategoricalArrays
	using LogExpFunctions: logistic, logit
	using IterTools: product

	Turing.setprogress!(false)

	include("$(pwd())/PILT_models.jl")
	include("$(pwd())/sample_utils.jl")
	include("$(pwd())/model_utils.jl")
	include("$(pwd())/plotting_utils.jl")
	include("$(pwd())/fetch_preprocess_data.jl")
	include("$(pwd())/generate_experimental_sequences/sequence_utils.jl")
	nothing
end

# ╔═╡ 114f2671-1888-4b11-aab1-9ad718ababe6
begin
	# Set theme	
	th = merge(theme_minimal(), Theme(
		font = "Helvetica",
		fontsize = 16,
		Axis = (
			xticklabelsize = 14,
			yticklabelsize = 14,
			spinewidth = 1.5,
			xtickwidth = 1.5,
			ytickwidth = 1.5
		)
	))
	
	set_theme!(th)
end

# ╔═╡ 3cb53c43-6b38-4c95-b26f-36697095f463
# General parameters
begin
	sessions = [
		"screening",
		"wk0",
		"wk2",
		"wk4",
		"wk24",
		"wk28"
	]

end

# ╔═╡ de74293f-a452-4292-b5e5-b4419fb70feb
categories = let

	categories = DataFrame(CSV.File("generate_experimental_sequences/trial1_stimuli/stimuli.csv")).concept |> unique

	@info "Found $(length(categories)) categories"

	categories
end

# ╔═╡ 56abf8a4-acad-4408-a86c-aad2d5aa3cd7
md"""# PILT"""

# ╔═╡ 1f791569-cfad-4bc8-9aef-ea324ea7be23
# PILT parameters
begin
	# PILT Parameters
	PILT_blocks_per_valence = 10
	PILT_trials_per_block = 10
	
	PILT_total_blocks = PILT_blocks_per_valence * 2
	PILT_n_confusing = vcat([0, 0, 1, 1], fill(2, PILT_total_blocks ÷ 2 - 4)) # Per valence
		
	# Post-PILT test parameters
	PILT_test_n_blocks = 5
end

# ╔═╡ 2c75cffc-1adc-44b6-bed3-12ed0c7025b7
function has_consecutive_repeats(vec::Vector, n::Int = 3)
    count = 1
    for i in 2:length(vec)
        if vec[i] == vec[i - 1]
            count += 1
            if count > n
                return true
            end
        else
            count = 1
        end
    end
    return false
end

# ╔═╡ e95a48f1-54e0-471b-b22d-5dd65569329e
# Assign valence and set size per block
PILT_block_attr = let random_seed = 5
	
	# # All combinations of set sizes and valence
	block_attr = DataFrame(
		block = repeat(1:PILT_total_blocks),
		valence = repeat([1, -1], inner = PILT_blocks_per_valence),
		fifty_high = fill(true, PILT_total_blocks)
	)

	# Shuffle set size and valence, making sure valence is varied, and positive in the first block and any time noise is introduced, and shaping doesn't extend too far into the task
	rng = Xoshiro(random_seed)

	first_three_same = true
	first_block_punishement = true
	too_many_repeats = true
	first_confusing_punishment = true
	shaping_too_long = true
	while first_three_same || first_block_punishement || too_many_repeats ||
		first_confusing_punishment || shaping_too_long

		DataFrames.transform!(
			block_attr,
			:block => (x -> shuffle(rng, x)) => :block
		)
		
		sort!(block_attr, :block)

		# Add n_confusing
		DataFrames.transform!(
			groupby(block_attr, :valence),
			:block => (x -> PILT_n_confusing) => :n_confusing
		)

		# Compute criterion variables
		first_three_same = allequal(block_attr[1:3, :valence])
		
		first_block_punishement = block_attr.valence[1] == -1

		too_many_repeats = has_consecutive_repeats(block_attr.valence)

		first_confusing_punishment = 
			(block_attr.valence[findfirst(block_attr.n_confusing .== 1)] == -1) |
			(block_attr.valence[findfirst(block_attr.n_confusing .== 2)] == -1)

		shaping_too_long = 
			!all(block_attr.n_confusing[11:end] .== maximum(PILT_n_confusing))
	end

	# Return
	block_attr
end

# ╔═╡ 211829a2-5afe-426d-b3b8-03ef04309b57
# Create feedback sequences per pair
PILT_sequence, common_per_pos, EV_per_pos = 
	let random_seed = 0
	
	# Compute how much we need of each sequence category
	n_confusing_wanted = combine(
		groupby(PILT_block_attr, [:n_confusing, :fifty_high]),
		:block => length => :n
	)
	
	# Generate all sequences and compute FI
	FI_seqs = [compute_save_FIs_for_all_seqs(;
		n_trials = 10,
		n_confusing = r.n_confusing,
		fifty_high = r.fifty_high,
		model = single_p_QL_recip,
		model_name = "QL_recip",
		unpack_function = unpack_single_p_QL,
		prop_fifty = 0.2,
	) for r in eachrow(n_confusing_wanted)]

	# Unpack results
	common_seqs = [x[2] for x in FI_seqs]
	magn_seqs = [x[3] for x in FI_seqs]

	# # Choose sequences optimizing FI under contraints
	chosen_idx, common_per_pos, EV_per_pos = optimize_FI_distribution(
		n_wanted = n_confusing_wanted.n,
		FIs = [x[1] for x in FI_seqs],
		common_seqs = common_seqs,
		magn_seqs = magn_seqs,
		ω_FI = .15,
		constrain_pairs = false,
		filename = "results/exp_sequences/pilot7_opt.jld2"
	)

	@assert length(vcat(chosen_idx...)) == nrow(PILT_block_attr) "Number of saved optimize sequences does not match number of sequences needed. Delete file and rerun."

	# Shuffle chosen sequences
	rng = Xoshiro(random_seed)
	shuffle!.(rng, chosen_idx)

	# Unpack chosen sequences
	chosen_common = [[common_seqs[s][idx[1]] for idx in chosen_idx[s]]
		for s in eachindex(common_seqs)]

	chosen_magn = [[magn_seqs[s][idx[2]] for idx in chosen_idx[s]]
		for s in eachindex(magn_seqs)]

	# Repack into DataFrame	
	n_sequences = sum(length.(chosen_common))
	task = DataFrame(
		idx = repeat(1:n_sequences, inner = PILT_trials_per_block),
		sequence = repeat(vcat([1:length(x) for x in chosen_common]...), 
			inner = PILT_trials_per_block),
		trial = repeat(1:PILT_trials_per_block, n_sequences),
		feedback_common = vcat(vcat(chosen_common...)...),
		variable_magnitude = vcat(vcat(chosen_magn...)...)
	)

	# Create n_confusing and fifty_high varaibles
	DataFrames.transform!(
		groupby(task, :idx),
		:feedback_common => (x -> PILT_trials_per_block - sum(x)) => :n_confusing,
		:variable_magnitude => (x -> 1. in x) => :fifty_high
	)

	# Add sequnces variable to PILT_block_attr
	DataFrames.transform!(
		groupby(PILT_block_attr, [:n_confusing, :fifty_high]),
		:block => (x -> shuffle(rng, 1:length(x))) => :sequence
	)


	# Combine with block attributes
	task = innerjoin(
		task,
		PILT_block_attr,
		on = [:n_confusing, :fifty_high, :sequence],
		order = :left
	)


	@assert nrow(task) == length(vcat(vcat(chosen_common...)...)) "Problem with join operation"
	@assert nrow(unique(task[!, [:block]])) == PILT_total_blocks "Problem with join operation"
		
	@assert mean(task.fifty_high) == 1. "Proportion of blocks with 50 pence in high magnitude option expected to be 1."

	# Sort by block
	sort!(task, [:block, :trial])

	# Remove auxillary variables
	select!(task, Not([:sequence, :idx]))

	# Compute low and high feedback
	task.feedback_high = ifelse.(
		task.valence .> 0,
		ifelse.(
			task.fifty_high,
			task.variable_magnitude,
			fill(1., nrow(task))
		),
		ifelse.(
			task.fifty_high,
			fill(-0.01, nrow(task)),
			.- task.variable_magnitude
		)
	)

	task.feedback_low = ifelse.(
		task.valence .> 0,
		ifelse.(
			.!task.fifty_high,
			task.variable_magnitude,
			fill(0.01, nrow(task))
		),
		ifelse.(
			.!task.fifty_high,
			fill(-1, nrow(task)),
			.- task.variable_magnitude
		)
	)

	# Compute feedback optimal and suboptimal
	task.feedback_optimal = ifelse.(
		task.feedback_common,
		task.feedback_high,
		task.feedback_low
	)

	task.feedback_suboptimal = ifelse.(
		.!task.feedback_common,
		task.feedback_high,
		task.feedback_low
	)

	task, common_per_pos, EV_per_pos
end

# ╔═╡ 6d0e251a-bca3-4f5a-a0ab-f6ef9408cb70
PILT_sessions = let rng = Xoshiro(0)

	# Duplicate over sessions (excluding screening)
	PILT_sessions = vcat(
		[
			insertcols(PILT_sequence, 1, :session => s) for s in sessions[2:end]
		]...
	)

	@info "Categories to begin with: $(length(categories))"

	# Assign stimulus pairs
	stimuli = vcat(
		[
			insertcols(assign_stimuli_and_optimality(;
				n_phases = 1,
				n_pairs = fill(1, PILT_total_blocks),
				categories = categories,
				rng = rng
		), 1, :session => s) for s in sessions[2:end]
		]...
	)

	@info "Categories left after PILT assignment: $(length(categories))"

	@info "Proportion of blocks on which the novel category is optimal : $(mean(stimuli.optimal_A))"


	# Merge
	leftjoin!(
		PILT_sessions,
		select(
			stimuli,
			:session,
			:block,
			:stimulus_A,
			:stimulus_B,
			:optimal_A
		),
		on = [:session, :block]
	)

end

# ╔═╡ c05db5b9-3f17-408f-8553-bd0c162fb0e7
PILT = let task = PILT_sessions,
	rng = Xoshiro(0)

	DataFrames.transform!(
		groupby(task, [:session, :block]),
		:block => 
			(x -> shuffled_fill([true, false], length(x); rng = rng)) =>
			:A_on_right
	)

	# Create stimulus_right and stimulus_left variables
	task.stimulus_right = ifelse.(
		task.A_on_right,
		task.stimulus_A,
		task.stimulus_B
	)

	task.stimulus_left = ifelse.(
		.!task.A_on_right,
		task.stimulus_A,
		task.stimulus_B
	)

	# Create optimal_right variable
	task.optimal_right = (task.A_on_right .& task.optimal_A) .| (.!task.A_on_right .& .!task.optimal_A)

	# Create feedback_right and feedback_left variables
	task.feedback_right = ifelse.(
		task.optimal_right,
		task.feedback_optimal,
		task.feedback_suboptimal
	)

	task.feedback_left = ifelse.(
		.!task.optimal_right,
		task.feedback_optimal,
		task.feedback_suboptimal
	)

	# Add empty variables needed for experiment script
	insertcols!(
		task,
		:n_stimuli => 2,
		:n_groups => 1,
		:stimulus_group => 1,
		:stimulus_group_id => task.block,
		:stimulus_middle => "",
		:feedback_middle => "",
		:optimal_side => "",
		:present_pavlovian => true,
		:early_stop => false
	)

	task

end

# ╔═╡ bd352949-cfa2-4979-bdc8-9094b0e5eaa8
# Validate task DataFrame
let task = PILT
	@assert maximum(task.block) == length(unique(task.block)) "Error in block numbering"

	@assert all(combine(groupby(task, :session), 
		:block => issorted => :sorted).sorted) "Task structure not sorted by block"

	@assert all(combine(groupby(task, [:session, :block]), 
		:trial => issorted => :sorted).sorted) "Task structure not sorted by trial number"

	@assert all(sign.(task.valence) == sign.(task.feedback_right)) "Valence doesn't match feedback sign"

	@assert all(sign.(task.valence) == sign.(task.feedback_left)) "Valence doesn't match feedback sign"

	@assert sum(unique(task[!, [:session, :block, :valence]]).valence) == 0 "Number of reward and punishment blocks not equal"

	@info "Overall proportion of common feedback: $(round(mean(task.feedback_common), digits = 2))"

	@assert all((task.variable_magnitude .== abs.(task.feedback_right)) .| 
		(task.variable_magnitude .== abs.(task.feedback_left))) ":variable_magnitude, which is used for sequnece optimization, doesn't match end result column :feedback_right no :feedback_left"

	# Count losses to allocate coins in to safe for beginning of task
	worst_loss = filter(x -> x.valence == -1, task) |> 
		df -> ifelse.(
			df.feedback_right .< df.feedback_left, 
			df.feedback_right, 
			df.feedback_left) |> 
		countmap

	@info "Worst possible loss in this task is of these coin numbers: $worst_loss"

end

# ╔═╡ e524bd61-1c54-491d-a904-888916f7561a
let
	save_to_JSON(PILT, "results/trial1_PILT.json")
	CSV.write("results/trial1_PILT.csv", PILT)
end

# ╔═╡ 11a926d3-7983-4fc0-bd9c-caa72b830b33
# Visualize PILT seuqnce
let task = PILT

	f = Figure(size = (700, 300))

	# Proportion of confusing by trial number
	confusing_location = combine(
		groupby(task, [:session, :trial]),
		:feedback_common => (x -> mean(.!x)) => :feedback_confusing
	)

	mp1 = data(confusing_location) * mapping(
		:trial => "Trial", 
		:feedback_confusing => "Prop. confusing feedback",
		color = :session => nonnumeric => "Session",
		group = :session => nonnumeric
	) * visual(ScatterLines)

	plt1 = draw!(f[1,1], mp1)

	legend!(
		f[1,1], 
		plt1,
		tellwidth = false,
		tellheight = false,
		valign = 1.2,
		halign = 0.,
		framevisible = false
	)

	# Plot confusing trials by block
	fp = insertcols(
		task,
		:color => ifelse.(
			task.feedback_common,
			(task.valence .+ 1) .÷ 2,
			fill(3, nrow(task))
		)
	)

	for (i, s) in enumerate(unique(fp.session))
		mp = data(filter(x -> x.session == s, fp)) * mapping(
			:trial => "Trial",
			:block => "Block",
			:color
		) * visual(Heatmap)

		draw!(f[1,i+1], mp, axis = (; yreversed = true, subtitle = "Session $i"))
	end
	


	save("results/trial1_pilt_trial_plan.png", f, pt_per_unit = 1)

	f

end

# ╔═╡ b13e8f5f-7522-497e-92d1-51d782fca33b
md"""## Post-PILT test"""

# ╔═╡ c7d66e4b-6326-4edb-8761-b41f6eebb4f3
function create_test_sequence(
	pilt_task::DataFrame;
	random_seed::Int64, 
	same_weight::Float64 = 6.5,
	test_n_blocks::Int64 = PILT_test_n_blocks
) 
	
	rng = Xoshiro(random_seed)

	# Extract stimuli and their common feedback from task structure
	stimuli = vcat([rename(
		pilt_task[pilt_task.feedback_common, [:session, :block, Symbol("stimulus_$s"), Symbol("feedback_$s")]],
		Symbol("stimulus_$s") => :stimulus,
		Symbol("feedback_$s") => :feedback
	) for s in ["right", "left"]]...)

	# Summarize magnitude per stimulus
	stimuli = combine(
		groupby(stimuli, [:session, :block, :stimulus]),
		:feedback => (x -> mean(unique(x))) => :magnitude
	)

	# Step 1: Identify unique stimuli
	unique_stimuli = unique(stimuli.stimulus)

	# Step 2: Define existing pairs
	create_pair_list(d) = [filter(x -> x.block == p, d).stimulus 
		for p in unique(stimuli.block)]

	existing_pairs = create_pair_list(stimuli)

	# Step 3: Generate all possible pairs
	all_possible_pairs = unique(sort.(collect(combinations(unique_stimuli, 2))))

	# Step 6: Select pairs ensuring each stimulus is used once and magnitudes are balanced
	final_pairs = []
	used_stimuli = Set{String}()

	# Create a priority queue for balanced selection based on pair counts
	pair_counts = Dict{Vector{Float64}, Int}()

	# Function to retrieve attribute of stimulus
	stim_attr(s, attr) = stimuli[stimuli.stimulus .== s, :][!, attr][1]

	for b in 1:test_n_blocks

		# Step 4: Filter valid pairs: were not paired in PILT, ano same category
		valid_pairs = 
			filter(pair -> 
				!(pair in existing_pairs) && 
				!(reverse(pair) in existing_pairs) && 
				(pair[1][1:(end-5)] != pair[2][1:(end-5)]), 
			all_possible_pairs)
	
		# Step 5: Create a mapping of pairs to their magnitudes
		magnitude_pairs = Dict{Vector{Float64}, Vector{Vector{String}}}()
		
		for pair in valid_pairs
		    mag1 = stimuli[stimuli.stimulus .== pair[1], :].magnitude[1]
		    mag2 = stimuli[stimuli.stimulus .== pair[2], :].magnitude[1]
		    key = sort([mag1, mag2])
		    if !haskey(magnitude_pairs, key)
		        magnitude_pairs[key] = []
		    end
		    push!(magnitude_pairs[key], pair)
		end
	
		@assert sum(length(vec) for vec in values(magnitude_pairs)) == length(valid_pairs)
	
		# Step 5.5 - Shuffle order within each magnitude
		for (k, v) in magnitude_pairs
			magnitude_pairs[k] = shuffle(rng, v)
		end

		# Initialize counts
		if b == 1
			for key in keys(magnitude_pairs)
			    pair_counts[key] = 0
			end
		end
		
		block_pairs = []
		
		while true
		    found_pair = false
	
		    # Select pairs while balancing magnitudes
		    for key in sort(collect(keys(magnitude_pairs)), by = x -> pair_counts[x] + same_weight * (x[1] == x[2])) # Sort by count, putting equal magnitude las
		        pairs = magnitude_pairs[key]
	
				# First try to find a same block pair
		        for pair in pairs
		            if !(pair[1] in used_stimuli) && !(pair[2] in used_stimuli)  && 
						stim_attr(pair[1], "block") == stim_attr(pair[2], "block")
					
		                push!(block_pairs, pair)
		                push!(used_stimuli, pair[1])
		                push!(used_stimuli, pair[2])
		                pair_counts[key] += 1
		                found_pair = true
		                break  # Stop going over pairs
		            end
		        end
	
				# Then try different block pair
		        for pair in pairs
		            if !found_pair &&!(pair[1] in used_stimuli) && 
						!(pair[2] in used_stimuli) 
					
		                push!(block_pairs, pair)
		                push!(used_stimuli, pair[1])
		                push!(used_stimuli, pair[2])
		                pair_counts[key] += 1
		                found_pair = true
		                break  # Stop going over pairs
		            end
		        end
		        
		        if found_pair
		            break  # Restart the outer loop if a pair was found
		        end
		    end

			# Alert bad seed
			if !found_pair
				return DataFrame(), NaN, NaN, NaN
			end
		
		    if length(used_stimuli) == length(unique_stimuli)
		        break  # Exit if all stimuli are used or no valid pairs remain
		    end
		end

		# Step 7 - Shuffle pair order
		shuffle!(rng, block_pairs)

		# Add block pairs to final pairs
		append!(final_pairs, block_pairs)

		# Add block pairs to existing pairs
		append!(existing_pairs, block_pairs)

		# Empty used_stimuli
		used_stimuli = []
	end

	# Shuffle order within each pair
	shuffle!.(rng, final_pairs)

	# Step 8 - Form DataFrame
	pairs_df = DataFrame(
		block = repeat(1:test_n_blocks, inner = length(unique_stimuli) ÷ 2),
		trial = repeat(1:(length(unique_stimuli) ÷ 2) * test_n_blocks),
		stimulus_right = [p[2] for p in final_pairs],
		stimulus_left = [p[1] for p in final_pairs],
		magnitude_right = [stimuli[stimuli.stimulus .== p[2], :].magnitude[1] for p in final_pairs],
		magnitude_left = [stimuli[stimuli.stimulus .== p[1], :].magnitude[1] for p in final_pairs],
		original_block_right = [stimuli[stimuli.stimulus .== p[2], :].block[1] for p in final_pairs],
		original_block_left = [stimuli[stimuli.stimulus .== p[1], :].block[1] for p in final_pairs]
	)

	# Same / different block variable
	pairs_df.same_block = pairs_df.original_block_right .== pairs_df.original_block_left

	# Valence variables
	pairs_df.valence_left = sign.(pairs_df.magnitude_left)
	pairs_df.valence_right = sign.(pairs_df.magnitude_right)
	pairs_df.same_valence = pairs_df.valence_left .== pairs_df.valence_right

	# Compute sequence stats
	prop_same_block = (mean(pairs_df.same_block)) 
	prop_same_valence = (mean(pairs_df.same_valence))
	n_same_magnitude = sum(pairs_df.magnitude_right .== pairs_df.magnitude_left)
	
	pairs_df, prop_same_block, prop_same_valence, n_same_magnitude
end

# ╔═╡ 44d302d2-30c9-4157-a0ed-e8784f1ccb9b
# Choose test sequence with best stats
function find_best_test_sequence(
	task::DataFrame; # PILT task structure
	n_seeds::Int64 = 100, # Number of random seeds to try
	same_weight::Float64 = 4.1 # Weight reducing the number of same magntiude pairs
) 

	# Initialize stats variables
	prop_block = []
	prop_valence = []
	n_magnitude = []

	# Run over seeds
	for s in 1:n_seeds
		_, pb, pv, nm = create_test_sequence(task, random_seed = s, same_weight = same_weight)

		push!(prop_block, pb)
		push!(prop_valence, pv)
		push!(n_magnitude, nm)
	end

	# First, choose a sequence with the minimal number of same-magnitude pairs
	pass_magnitude = (1:n_seeds)[n_magnitude .== 
		minimum(filter(x -> !isnan(x), n_magnitude))]

	@assert !isempty(pass_magnitude)

	# Apply magnitude selection
	prop_block = prop_block[pass_magnitude]
	prop_valence = prop_valence[pass_magnitude]

	# Compute deviation from goal
	dev_block = abs.(prop_block .- 1/3)
	dev_valence = abs.(prop_block .- 0.5)

	# Choose best sequence
	chosen = pass_magnitude[argmin(dev_valence)]

	# Return sequence and stats
	return create_test_sequence(task, random_seed = chosen, same_weight = same_weight)
end

# ╔═╡ e51b4cec-6fd2-49d3-a9ec-ebbe960a7f49
PILT_test_template = let s = "wk0",
	task = filter(x -> x.session == s, PILT)
	
	# Find test sequence for each session
	test, pb, pv, nm = find_best_test_sequence(
		task,
		n_seeds = 100, # Number of random seeds to try
		same_weight = 25. # Weight reducing the number of same magntiude pairs
	) 

	# Add session variable
	insertcols!(test, 1, :session => s)

	@info "Session $s: proportion of same block pairs: $pb"
	@info "Session $s: proportion of same valence pairs: $pv"
	@info "Session $s: number of same magnitude pairs: $nm"

	# Create magnitude_pair variable
	test.magnitude_pair = [sort([r.magnitude_left, r.magnitude_right]) for r in eachrow(test)]

	# Create feedback_right and feedback_left variables - these determine coins given on this trial
	test.feedback_left = (x -> abs(x) == 0.01 ? x : sign(x)).(test.magnitude_left)

	test.feedback_right = (x -> abs(x) == 0.01 ? x : sign(x)).(test.magnitude_right)

	test
end

# ╔═╡ 8700a65a-3117-4d62-98f9-26f2839ba6a2
PILT_test = let PILT_test_template = copy(PILT_test_template)

	# Create stimulus dict to replace equivalent stimuli
	stimuli_dict = unique(
		select(
			PILT,
			:session,
			:block,
			:stimulus_A,
			:stimulus_B
		)
	)

	# Wide to long
	stimuli_dict = stack(
		stimuli_dict,
		[:stimulus_A, :stimulus_B],
		value_name = :stimulus
	)

	# Variable capturing stimulus isometry
	stimuli_dict.stimulus_essence = (r -> "$(r.block)_$(r.variable[end])").(eachrow(stimuli_dict))

	# Join add stimulus_essence
	leftjoin!(
		PILT_test_template,
		select(
			stimuli_dict,
			:stimulus => :stimulus_left,
			:stimulus_essence => :stimulus_essence_left
		),
		on = :stimulus_left
	)

	leftjoin!(
		PILT_test_template,
		select(
			stimuli_dict,
			:stimulus => :stimulus_right,
			:stimulus_essence => :stimulus_essence_right
		),
		on = :stimulus_right
	)

	function find_stim(e, s)
		println(e)
		println(s)
		

	end


	# Create multisession
	PILT_test = vcat(
		[
			DataFrames.transform(
				select(
					PILT_test_template,
					Not([:session, :stimulus_right, :stimulus_left])
				),
				:trial => (x -> s) => :session,
				:stimulus_essence_right => ByRow(e -> only(filter(
					x -> (x.stimulus_essence == e) & (x.session == s), stimuli_dict).stimulus)) => :stimulus_right,
				:stimulus_essence_left => ByRow(e -> only(filter(
						x -> (x.stimulus_essence == e) & (x.session == s), stimuli_dict).stimulus)) => :stimulus_left
			)
		for s in sessions[2:end]]...
	)

end

# ╔═╡ dd7112c9-35ac-4d02-a9c4-1e19efad0f31
# Test test sequence
let
	@assert all(PILT_test_template.stimulus_left .== filter(x -> x.session == "wk0", PILT_test).stimulus_left) "Final sequence for wk0 does not match template"

	@assert all(PILT_test_template.stimulus_right .== filter(x -> x.session == "wk0", PILT_test).stimulus_right) "Final sequence for wk0 does not match template"


end

# ╔═╡ 9f7b362c-5a60-4af1-a7e1-64b9665eee1e
md"""# RLX"""

# ╔═╡ 54f6f217-3ae5-49c7-9456-a5abcbbdc62f
# RLX Parameters
begin
	RLX_prop_fifty = 0.2
	RLX_shaping_n = 20
	RLX_test_n_blocks = 2
end

# ╔═╡ 9f300301-b018-4bea-8fc4-4bc889b11afd
triplet_order = let
	triplet_order = DataFrame(CSV.File(
		"generate_experimental_sequences/pilot8_wm_stimulus_sequence.csv"))

	select!(
		triplet_order, 
		:stimset => :stimulus_group,
		:delay
	)
end

# ╔═╡ 184a054c-5a88-44f8-865e-da75a10191ec
md"""## RLWM"""

# ╔═╡ 60c50147-708a-46f8-a813-7667116fc8d2
md"""### Post-WM test"""

# ╔═╡ ffe06202-d829-4145-ae26-4a95449d64e6
md"""# RLLTM"""

# ╔═╡ 3fa8c293-ac47-4acd-bdb7-9313286ee464
function assign_triplet_stimuli_RLLTM(
	categories::AbstractVector,
	n_triplets::Int64;
	rng::AbstractRNG = Xoshiro(0)
)

	dicts = [
		Dict(
			:stimulus_group => i,
			:stimulus_A => popat!(
				categories, 
				rand(rng, 1:length(categories))
			) * "_1.jpg",
			:stimulus_B => popat!(
				categories, 
				rand(rng, 1:length(categories))
			) * "_1.jpg",
			:stimulus_C => popat!(
					categories, 
					rand(rng, 1:length(categories))
			) * "_1.jpg"
		)
		for i in 1:n_triplets
	]

	return select(
		DataFrame(dicts),
		:stimulus_group,
		:stimulus_A,
		:stimulus_B,
		:stimulus_C
	)
	
end

# ╔═╡ f89e88c9-ebfc-404f-964d-acff5c7f8985
function integer_allocation(p::Vector{Float64}, n::Int)
    i = floor.(Int, p * n)  # Floor to ensure sum does not exceed n
    diff = n - sum(i)       # Remaining to distribute
    indices = sortperm(p * n .- i, rev=true)  # Sort by largest remainder
    i[indices[1:diff]] .+= 1  # Distribute the remainder
    return i
end

# ╔═╡ 68873d3e-054d-4ab4-9d89-73586bb0370e
function prop_fill_shuffle(
	values::AbstractVector,
	props::Vector{Float64},
	n::Int64;
	rng::AbstractRNG = Xoshiro(1)
)
	# Find integers
	ints = integer_allocation(props, n)
	
	# Fill
	res = [fill(v, i) for (v, i) in zip(values, ints)]
	
	# Return shuffled
	shuffle(rng, vcat(res...))
end

# ╔═╡ b9134153-d9e9-4e35-bfc4-2c5c5a4329ee
RLX_block = let rng = Xoshiro(0)

	n_trials = nrow(triplet_order)

	# Basic variables
	det_block = DataFrame(
		block = fill(1, n_trials),
		trial = 1:n_trials,
		stimulus_group = triplet_order.stimulus_group,
		delay = triplet_order.delay
	)

	# Draw optimal feedback
	DataFrames.transform!(
		groupby(det_block, :stimulus_group),
		:trial => (x -> prop_fill_shuffle(
			[1., 0.5],
			[1 - RLX_prop_fifty, RLX_prop_fifty],
			length(x),
			rng = rng
			)
		) => :feedback_optimal
	) 

	@info "Proportion fifty pence: $(mean(det_block.feedback_optimal .== 0.5))"

	# Copy over multiple sessions, skipping screening ------------
	det_block = vcat(
		[insertcols(
			det_block,
			1,
			:session => s
		) for s in sessions[2:end]]...
	)


	det_block

end

# ╔═╡ 6417ad94-1852-4cce-867e-a856295ec782
# Create deterministic block
RLWM = let RLWM = copy(RLX_block),
	rng = Xoshiro(0)


	# Assign stimuli --------

	stimuli = unique(select(RLWM, :session, :stimulus_group))

	stimuli.stimulus_left = [
		popat!(categories, rand(rng, 1:length(categories))) * "_1.jpg" for _ in 1:nrow(stimuli)
	]
		
	leftjoin!(
		RLWM,
		stimuli,
		on = [:session, :stimulus_group]
	)

	# Replicate - for RLWM there is only one stimulus, but this is requirement of js script
	RLWM.stimulus_middle = RLWM.stimulus_left
	RLWM.stimulus_right = RLWM.stimulus_left
	
	# Assign stimuli locations -----------------------------
	# Count appearances per stimulus_group
	stimulus_ordering = combine(
		groupby(RLWM, [:session, :stimulus_group]),
		:stimulus_group => length => :n
	)

	# Sort by descending n to distribute largest trials first
	shuffle!(rng, stimulus_ordering)
	sort!(stimulus_ordering, [:session, :n], rev=true)

	for gdf in groupby(stimulus_ordering, :session)

		# Track total counts per action
		action_sums = Dict(1 => 0, 2 => 0, 3 => 0)
	
		# Placeholder for optimal action
		gdf.optimal_action .= 99
		
		# Assign actions to balance total n
		for row in eachrow(gdf)
		    # Pick the action with the smallest current total
		    best_action = argmin(action_sums)
		    row.optimal_action = best_action
		    action_sums[best_action] += row.n
		end

	end

	# Join with data frame
	leftjoin!(
		RLWM,
		select(stimulus_ordering, [:session, :stimulus_group, :optimal_action]),
		on = [:session, :stimulus_group]
	)

	@info "Proportion optimal in each location: $([round(mean((x -> x == i).(RLWM.optimal_action)), digits = 3) for i in 1:3])"

	# Additional variables --------

	# Assign feedback to action
	for (i, side) in enumerate(["left", "middle", "right"])
		RLWM[!, Symbol("feedback_$side")] = ifelse.(
			(x -> x == i).(RLWM.optimal_action),
			RLWM.feedback_optimal,
			fill(0.01, nrow(RLWM))
		)
	end

	# For compatibility with probabilistic block
	RLWM.feedback_common .= true

	# Valence variable
	RLWM.valence .= 1

	# Apperance variable
	DataFrames.transform!(
		groupby(RLWM, [:session, :block, :stimulus_group]),
		:trial => (x -> 1:length(x)) => :appearance
	)

	# Cumulative triplet index
	RLWM.stimulus_group_id = RLWM.stimulus_group .+ (maximum(RLWM.stimulus_group) .* (RLWM.block .- 1))

	# Create optimal_side variable
	RLWM.optimal_side = (x -> ["left", "middle", "right"][x]).(RLWM.optimal_action)
		

	# Add variables needed for experiment code
	insertcols!(
		RLWM,
		:n_stimuli => 1,
		:optimal_right => "",
		:present_pavlovian => false,
		:n_groups => maximum(RLWM.stimulus_group),
		:early_stop => false
	)

	RLWM

end

# ╔═╡ e6f984aa-20dc-4a7d-8a3b-75728995a1f7
# Checks
let task = RLWM
	@assert all(combine(groupby(task, [:session]), 
		:block => issorted => :sorted).sorted) "Task structure not sorted by block"

	@assert all(combine(groupby(task, [:session, :block]), 
		:trial => issorted => :sorted).sorted) "Task structure not sorted by trial number"
	
	@assert all(sign.(task.valence) == sign.(task.feedback_right)) "Valence doesn't match feedback sign"

	@assert all(sign.(task.valence) == sign.(task.feedback_left)) "Valence doesn't match feedback sign"

	@info "Overall proportion of common feedback: $(round(mean(task.feedback_common), digits = 2))"

end

# ╔═╡ efdfdeb0-2b56-415e-acf7-d6236ee7b199
let
	save_to_JSON(RLWM, "results/trial1_WM.json")
	CSV.write("results/trial1_WM.csv", RLWM)
end

# ╔═╡ 1491f0f9-0c40-41ca-b7a9-055259f66eb3
RLWM_test = let

	RLWM_test = DataFrame()

	for gdf in groupby(RLWM, :session)

		# Reset random seed so that randomization is the same across sessions
		rng = Xoshiro(0)

		# Get unique stimuli
		RLWM_stimuli = unique(gdf.stimulus_left)
	
		# Get all combinations
		RLWM_pairs = collect(combinations(RLWM_stimuli, 2))
	
		# Shuffle order within pair
		shuffle!.(rng, RLWM_pairs)
	
		# Repeat flipped
		RWLM_blocks = [iseven(i) ? reverse.(RLWM_pairs) : RLWM_pairs for i in 1:RLX_test_n_blocks]
	
		# Assemble into DataFrame
		RLWM_test_session = vcat([DataFrame(
			block = fill(i, length(stims)),
			stimulus_left = [x[1] for x in stims],
			stimulus_right = [x[2] for x in stims]
		) for (i, stims) in enumerate(RWLM_blocks)]...)
	
		# Shuffle trial order within block
		DataFrames.transform!(
			groupby(RLWM_test_session, :block),
			:block => (x -> shuffle(rng, 1:length(x))) => :trial
		)
	
		sort!(RLWM_test_session, [:block, :trial])

		# Add session variable
		insertcols!(
			RLWM_test_session,
			1,
			:session => gdf.session[1],
		)
	
		# Add variables needed for JS ------------------
		insertcols!(
			RLWM_test_session,
			:feedback_left => 1.,
			:feedback_right => 1.,
			:magnitude_left => 1.,
			:magnitude_right => 1.,
			:same_valence => true,
			:same_block => true,
			:original_block_left => 1,
			:original_block_right => 1
		)

		RLWM_test = vcat(RLWM_test, RLWM_test_session)

	end

	RLWM_test
	
end

# ╔═╡ b28f57a2-8aab-45e9-9d16-4c3b9fcf3828
let
	save_to_JSON(RLWM_test, "results/trial1_WM_test.json")
	CSV.write("results/trial1_WM_test.csv", RLWM_test)
end

# ╔═╡ 1a6d525f-5317-4b2b-a631-ea646ee20c9f
# Tests for RLWM_test
let

	# Make sure all stimuli are in RLWM
	test_stimuli = unique(
		vcat(
			RLWM_test.stimulus_left,
			RLWM_test.stimulus_right
		)
	)

	RLWM_stimuli =  unique(RLWM.stimulus_left)

	@assert all((x -> x in RLWM_stimuli).(test_stimuli)) "Test stimuli not in RLWM sequence"

	@assert all((x -> x in test_stimuli).(RLWM_stimuli)) "Not all RLWM stimuli appear in test"


end

# ╔═╡ f5916a9f-ddcc-4c03-9328-7dd76c4c74b2
# Create deterministic block
RLLTM_det_block = let det_block = copy(RLX_block),
	rng = Xoshiro(1)
	
	# Assign stimuli categories
	stimuli = vcat([insertcols(
		assign_triplet_stimuli_RLLTM(categories,
			maximum(det_block.stimulus_group);
			rng = rng
		),
		1,
		:session => s
	) for s in unique(det_block.session)]...)

	# Merge with trial structure
	det_block = innerjoin(
		det_block,
		stimuli,
		on = [:session, :stimulus_group],
		order = :left
	)

	# Assign stimuli locations -----------------------------
	# Count appearances per stimulus_group
	stimulus_ordering = combine(
		groupby(det_block, [:session, :stimulus_group]),
		:stimulus_group => length => :n
	)

	# Sort by descending n to distribute largest trials first
	shuffle!(rng, stimulus_ordering)
	sort!(stimulus_ordering, [:session, :n], rev=true)

	for gdf in groupby(stimulus_ordering, :session)

		# Track total counts per action
		action_sums = Dict(1 => 0, 2 => 0, 3 => 0)
	
		# Place holder for optimal action
		gdf.optimal_action .= 99
		
		# Assign actions to balance total n
		for row in eachrow(gdf)
		    # Pick the action with the smallest current total
		    best_action = argmin(action_sums)
		    row.optimal_action = best_action
		    action_sums[best_action] += row.n
		end
	end

	# Function to translate to "ABC" strings
	function stim_ordering(pos::Int64)
		s = join(shuffle(collect("BC")))
		return s[1:pos-1] * "A" * s[pos:end]
	end

	# Translate to "ABC" orderings
	stimulus_ordering.stimulus_ordering = stim_ordering.(stimulus_ordering.optimal_action)

	# Join with data frame
	leftjoin!(
		det_block,
		select(stimulus_ordering, [:session, :stimulus_group, :stimulus_ordering]),
		on = [:session, :stimulus_group]
	)

	@info "Proportion optimal in each location: $([round(mean((x -> x[i] == 'A').(det_block.stimulus_ordering)), digits = 3) for i in 1:3])"

	# Additional variables --------
	# Assign stimulus identity to location
	DataFrames.transform!(
		det_block,
		[:stimulus_ordering, :stimulus_A, :stimulus_B, :stimulus_C] =>
		((o, a, b, c) -> [[a b c][i, (Int(o[i][1]) - Int('A') + 1)] for i in 1:length(o)]) => :stimulus_left,
		[:stimulus_ordering, :stimulus_A, :stimulus_B, :stimulus_C] =>
		((o, a, b, c) -> [[a b c][i, (Int(o[i][2]) - Int('A') + 1)] for i in 1:length(o)]) => :stimulus_middle,
		[:stimulus_ordering, :stimulus_A, :stimulus_B, :stimulus_C] =>
		((o, a, b, c) -> [[a b c][i, (Int(o[i][3]) - Int('A') + 1)] for i in 1:length(o)]) => :stimulus_right		
	)

	# Assign feedback to stimulus essence, by covention stimulus_A is optimal
	det_block.feedback_A = det_block.feedback_optimal
	
	det_block.feedback_B = fill(0.01, nrow(det_block))

	det_block.feedback_C = fill(0.01, nrow(det_block))

	# Assign feedback to location, by covention stimulus_A is optimal
	det_block.feedback_left = ifelse.(
		(x -> x[1] == 'A').(det_block.stimulus_ordering),
		det_block.feedback_optimal,
		fill(0.01, nrow(det_block))
	)
	
	det_block.feedback_middle = ifelse.(
		(x -> x[2] == 'A').(det_block.stimulus_ordering),
		det_block.feedback_optimal,
		fill(0.01, nrow(det_block))
	)

	det_block.feedback_right = ifelse.(
		(x -> x[3] == 'A').(det_block.stimulus_ordering),
		det_block.feedback_optimal,
		fill(0.01, nrow(det_block))
	)

	# For compatibility with probabilistic block
	det_block.feedback_common .= true

	det_block

end

# ╔═╡ e3bff0b9-306a-4bf9-8cbd-fe0e580bd118
RLLTM = let RLLTM = RLLTM_det_block

	# Valence variable
	RLLTM.valence .= 1

	# Apperance variable
	DataFrames.transform!(
		groupby(RLLTM, [:block, :stimulus_group]),
		:trial => (x -> 1:length(x)) => :appearance
	)

	# Cumulative triplet index
	RLLTM.stimulus_group_id = RLLTM.stimulus_group .+ (maximum(RLLTM.stimulus_group) .* (RLLTM.block .- 1))

	# Create optimal_side variable
	RLLTM.optimal_side = [["left", "middle", "right"][findfirst('A', o)] for o in RLLTM.stimulus_ordering]


	# Add variables needed for experiment code
	insertcols!(
		RLLTM,
		:n_stimuli => 3,
		:optimal_right => "",
		:present_pavlovian => false,
		:n_groups => maximum(RLLTM.stimulus_group),
		:early_stop => false
	)

	RLLTM
end

# ╔═╡ f9be1490-8e03-445f-b36e-d8ceff894751
# Checks
let task = RLLTM
	@assert all(combine(groupby(task, [:session]), 
		:block => issorted => :sorted).sorted) "Task structure not sorted by block"

	@assert all(combine(groupby(task, [:session, :block]), 
		:trial => issorted => :sorted).sorted) "Task structure not sorted by trial number"
	
	@assert all(sign.(task.valence) == sign.(task.feedback_right)) "Valence doesn't match feedback sign"

	@assert all(sign.(task.valence) == sign.(task.feedback_left)) "Valence doesn't match feedback sign"

	@info "Overall proportion of common feedback: $(round(mean(task.feedback_common), digits = 2))"

	@assert !all(RLWM.feedback_right .== RLLTM.feedback_right) "Optimal actions should be different on RLWM and RLLTM"

	@assert all(RLWM.stimulus_group .== RLLTM.stimulus_group) "Stimuli sequence not identical in RLWM and RLLTM"

end

# ╔═╡ eecaac0c-e051-4543-988c-e969de3a8567
let
	save_to_JSON(RLLTM, "results/trial1_LTM.json")
	CSV.write("results/trial1_LTM.csv", RLLTM)
end

# ╔═╡ bcb73e19-8bf9-4e75-a9f7-8152e3d23201
md"""## Post-LTM test"""

# ╔═╡ 8d8e4026-2185-4d67-a259-4cdebaab0b94
function prepare_for_finding_ltm_test_sequence(
	task::DataFrame;
)

	# Extract stimuli and their common feedback from task structure
	stimuli = vcat([select(
		filter(x -> x.feedback_common, task),
		:session,
		:block,
		:stimulus_group_id,
		:stimulus_group_id => ByRow(b -> "$(b)_$s") => :stimulus_essence,
		Symbol("feedback_$s") => :feedback
	) for s in ["A", "B", "C"]]...)

	# Summarize magnitude per stimulus
	stimuli = combine(
		groupby(stimuli, [:session, :block, :stimulus_group_id, :stimulus_essence]),
		:feedback => (x -> mean(unique(x))) => :magnitude
	)

	# Create Dict of unique stimuli by magnitude
	magnitudes = unique(stimuli.magnitude)
	stimuli_magnitude = Dict(m => unique(filter(x -> x.magnitude == m, stimuli).stimulus_essence) for m in magnitudes)

	#  Define existing pairs
	pairer(v) = [[v[i], v[j]] for i in 1:length(v) for j in i+1:length(v)]
	create_pair_list(d) = vcat([pairer(filter(x -> x.stimulus_group_id == p, d).stimulus_essence) 
		for p in unique(stimuli.stimulus_group_id)]...)

	existing_pairs = create_pair_list(stimuli)

	existing_pairs = sort.(existing_pairs)

	return stimuli_magnitude, existing_pairs

end

# ╔═╡ 391f3f2a-74ae-463b-b109-a47eaaafdf97
function create_ltm_test_pairs(stimuli::Dict{Float64, Vector{String}}, 
                            existing_pairs::Vector{Vector{String}}, 
                            target_n_different::Int64, 
                            target_n_same::Int64)

	@assert all(issorted.(existing_pairs)) "Pairs not sorted within `existing_pairs`"
        
    # Extract unique magnitude values
    magnitudes = collect(keys(stimuli))
    
    # Initialize storage for results
    new_pairs = DataFrame(
		stimulus_essence_א = String[], 
		stimulus_essence_ב = String[],
		magnitude_א = Float64[], 
		magnitude_ב = Float64[])
    
    # Track stimulus usage
    stim_usage = Dict(s => 0 for m in magnitudes for s in stimuli[m])
    
    # Function to sample a pair of stimuli from given magnitude categories
    function sample_pair(mag1, mag2)
        available_stim1 = sort(stimuli[mag1], by=s -> stim_usage[s])
        available_stim2 = sort(stimuli[mag2], by=s -> stim_usage[s])
        
        for stim1 in available_stim1, stim2 in available_stim2
            if stim1 != stim2 && !(sort([stim1, stim2]) in existing_pairs)
                return stim1, stim2
            end
        end
        return rand(available_stim1), rand(available_stim2)
    end
    
    # Generate different-magnitude pairs
    for (i, mag1) in enumerate(magnitudes), mag2 in magnitudes[i+1:end]
        for _ in 1:target_n_different
            stim1, stim2 = sample_pair(mag1, mag2)
            push!(new_pairs, (stim1, stim2, mag1, mag2))
            push!(existing_pairs, sort([stim1, stim2]))
            stim_usage[stim1] += 1
            stim_usage[stim2] += 1
        end
    end
    
    # Generate same-magnitude pairs
    for mag in magnitudes
        if length(stimuli[mag]) > 1  # Ensure at least two stimuli exist
            for _ in 1:target_n_same
                stim1, stim2 = sample_pair(mag, mag)
                push!(new_pairs, (stim1, stim2, mag, mag))
                push!(existing_pairs, sort([stim1, stim2]))
                stim_usage[stim1] += 1
                stim_usage[stim2] += 1
            end
        end
    end
    
    return new_pairs
end


# ╔═╡ c65d7f1f-224d-4144-aa46-d48a482db95a
# Create LTM test sequence
RLLTM_test_session = let rng = Xoshiro(0)

	# Process LTM sequence to be able to find test pairs
	stimuli_magnitude, existing_pairs = 
		prepare_for_finding_ltm_test_sequence(
		filter(x -> x.session == RLLTM.session[1], RLLTM)
	)
	

	# Find test pairs
	RLLTM_test = create_ltm_test_pairs(stimuli_magnitude, 
						existing_pairs, 
						60, 
						20)

	# Shuffle
	RLLTM_test.trial = shuffle(rng, 1:nrow(RLLTM_test))

	sort!(RLLTM_test, :trial)

	# Add needed variables
	insertcols!(
		RLLTM_test,
		1,
		:block => 1,
		:original_block_right => 1,
		:original_block_left => 1,
		:same_block => true,
		:valence_left => 1,
		:valence_right => 1,
		:same_valence => true
	)

	# Add magnitude pair variable
	RLLTM_test.magnitude_pair = [sort([r.magnitude_א, r.magnitude_ב]) for r in eachrow(RLLTM_test)]


	# Assign left / right
	DataFrames.transform!(
		groupby(RLLTM_test, :magnitude_pair),
		:trial => (x -> shuffled_fill([true, false], length(x), rng = rng)) => :א_on_right
	)

	RLLTM_test.magnitude_right = ifelse.(
		RLLTM_test.א_on_right,
		RLLTM_test.magnitude_א,
		RLLTM_test.magnitude_ב
	)

	RLLTM_test.magnitude_left = ifelse.(
		.!RLLTM_test.א_on_right,
		RLLTM_test.magnitude_א,
		RLLTM_test.magnitude_ב
	)

	# Add feedback_right and feedback_left variables - these determine the coins added to the safe for the trial
	RLLTM_test.feedback_right = ifelse.(
		RLLTM_test.magnitude_right .== 0.75,
		fill(1., nrow(RLLTM_test)),
		fill(0.01, nrow(RLLTM_test))
	)

	RLLTM_test.feedback_left = ifelse.(
		RLLTM_test.magnitude_left .== 0.75,
		fill(1., nrow(RLLTM_test)),
		fill(0.01, nrow(RLLTM_test))
	)

	RLLTM_test.block .= 1


	RLLTM_test

end

# ╔═╡ ca34e152-6a53-4e25-a3d2-964c75d70fd5
RLLTM_test = let

	# Get all stimuli
	all_stimuli = unique(vcat([select(
		RLLTM,
		:session,
		:block,
		:stimulus_group_id,
		:stimulus_group_id => ByRow(b -> "$(b)_$stim") => :stimulus_essence,
		Symbol("stimulus_$stim") => :stimulus
	) for stim in ["A", "B", "C"]]...))


	# Join stimulus with stimulus essence
	RLLTM_test = innerjoin(
		RLLTM_test_session,
		select(
			all_stimuli,
			:session,
			:stimulus_essence => :stimulus_essence_א,
			:stimulus => :stimulus_א
		),
		on = :stimulus_essence_א
	)

	RLLTM_test = innerjoin(
		RLLTM_test,
		select(
			all_stimuli,
			:session,
			:stimulus_essence => :stimulus_essence_ב,
			:stimulus => :stimulus_ב
		),
		on = [:session, :stimulus_essence_ב]
	)

	# Sort
	sort!(RLLTM_test, [:session, :trial])

	# Compute stimulus on the right and left
	RLLTM_test.stimulus_right = ifelse.(
		RLLTM_test.א_on_right,
	    RLLTM_test.stimulus_א,
		RLLTM_test.stimulus_ב
	)

	RLLTM_test.stimulus_left = ifelse.(
		.!RLLTM_test.א_on_right,
	    RLLTM_test.stimulus_א,
		RLLTM_test.stimulus_ב
	)

	RLLTM_test

end

# ╔═╡ 30f86e30-e7e9-43c4-9001-1f2f0c6c2bea
# Tests for RLLtM_tests
let
	# Test even distribution of stimuli appearances
	long_test = vcat(
		select(
			RLLTM_test,
			:stimulus_א => :stimulus,
			:magnitude_א => :magnitude
		),
		select(
			RLLTM_test,
			:stimulus_ב => :stimulus,
			:magnitude_ב => :magnitude
		)
	)

	test_n = combine(
		groupby(
			long_test,
			:stimulus
		),
		:stimulus => length => :n,
		:magnitude => unique => :magnitude
	)

	@assert all(combine(
		groupby(
			test_n,
			:magnitude
		),
		:n => (x -> (maximum(x) - minimum(x)) == 1) => :n_diff
	).n_diff) "Number of appearances for each stimulus not balanced"

	# Test left right counterbalancing of magnitude

	@assert mean(RLLTM_test.magnitude_left) ≈ mean(RLLTM_test.magnitude_right) "Different average magnitudes for stimuli presented on left and right"

	# Make sure all stimuli are in RLLTM
	test_stimuli = unique(
		vcat(
			RLLTM_test.stimulus_left,
			RLLTM_test.stimulus_right
		)
	)

	RLLTM_stimuli =  unique(
		vcat(
			RLLTM.stimulus_left,
			RLLTM.stimulus_right,
			RLLTM.stimulus_middle
		)
	)

	@assert all((x -> x in RLLTM_stimuli).(test_stimuli)) "Test stimuli not in RLLTM sequence"

	@assert all((x -> x in test_stimuli).(RLLTM_stimuli)) "Not all RLLTM stimuli appear in test"

	# Test assignment of stimuli to stimuli_essence ---------------
	# Extract stimuli and their common feedback from task structure
	stimuli_magntiude = vcat([select(
		filter(x -> x.feedback_common, RLLTM),
		:session,
		:stimulus_group_id,
		Symbol("stimulus_$s") => :stimulus,
		Symbol("feedback_$s") => :feedback
	) for s in ["A", "B", "C"]]...)

	# Summarize magnitude per stimulus
	stimuli_magntiude = combine(
		groupby(stimuli_magntiude, [:session, :stimulus]),
		:feedback => (x -> mean(unique(x))) => :magnitude
	)

	# Extract stimuli and their common feedback from test structure
	test_magntiude = vcat([select(
		RLLTM_test,
		:session,
		Symbol("stimulus_$s") => :stimulus,
		Symbol("magnitude_$s") => :magnitude_test
	) for s in ["א", "ב"]]...)

	magnitdue_compare =
		leftjoin(
			stimuli_magntiude,
			test_magntiude,
			on = [:session, :stimulus]
		)

	@assert all(magnitdue_compare.magnitude .== magnitdue_compare.magnitude_test) "Test structure magnitude does not match learning phase structure magnitude"

end

# ╔═╡ 832ccb61-b588-4614-9f5a-efa0f9a6087d
let
	save_to_JSON(RLLTM_test, "results/trial1_LTM_test.json")
	CSV.write("results/trial1_LTM_test.csv", RLLTM_test)
end

# ╔═╡ Cell order:
# ╠═2d7211b4-b31e-11ef-3c0b-e979f01c47ae
# ╠═114f2671-1888-4b11-aab1-9ad718ababe6
# ╠═3cb53c43-6b38-4c95-b26f-36697095f463
# ╠═de74293f-a452-4292-b5e5-b4419fb70feb
# ╟─56abf8a4-acad-4408-a86c-aad2d5aa3cd7
# ╠═1f791569-cfad-4bc8-9aef-ea324ea7be23
# ╠═e95a48f1-54e0-471b-b22d-5dd65569329e
# ╠═211829a2-5afe-426d-b3b8-03ef04309b57
# ╠═6d0e251a-bca3-4f5a-a0ab-f6ef9408cb70
# ╠═c05db5b9-3f17-408f-8553-bd0c162fb0e7
# ╠═bd352949-cfa2-4979-bdc8-9094b0e5eaa8
# ╠═e524bd61-1c54-491d-a904-888916f7561a
# ╠═11a926d3-7983-4fc0-bd9c-caa72b830b33
# ╠═2c75cffc-1adc-44b6-bed3-12ed0c7025b7
# ╟─b13e8f5f-7522-497e-92d1-51d782fca33b
# ╠═e51b4cec-6fd2-49d3-a9ec-ebbe960a7f49
# ╠═8700a65a-3117-4d62-98f9-26f2839ba6a2
# ╠═dd7112c9-35ac-4d02-a9c4-1e19efad0f31
# ╠═44d302d2-30c9-4157-a0ed-e8784f1ccb9b
# ╠═c7d66e4b-6326-4edb-8761-b41f6eebb4f3
# ╟─9f7b362c-5a60-4af1-a7e1-64b9665eee1e
# ╠═54f6f217-3ae5-49c7-9456-a5abcbbdc62f
# ╠═9f300301-b018-4bea-8fc4-4bc889b11afd
# ╠═b9134153-d9e9-4e35-bfc4-2c5c5a4329ee
# ╟─184a054c-5a88-44f8-865e-da75a10191ec
# ╠═6417ad94-1852-4cce-867e-a856295ec782
# ╠═e6f984aa-20dc-4a7d-8a3b-75728995a1f7
# ╠═efdfdeb0-2b56-415e-acf7-d6236ee7b199
# ╟─60c50147-708a-46f8-a813-7667116fc8d2
# ╠═1491f0f9-0c40-41ca-b7a9-055259f66eb3
# ╠═1a6d525f-5317-4b2b-a631-ea646ee20c9f
# ╠═b28f57a2-8aab-45e9-9d16-4c3b9fcf3828
# ╟─ffe06202-d829-4145-ae26-4a95449d64e6
# ╠═f5916a9f-ddcc-4c03-9328-7dd76c4c74b2
# ╠═e3bff0b9-306a-4bf9-8cbd-fe0e580bd118
# ╠═f9be1490-8e03-445f-b36e-d8ceff894751
# ╠═eecaac0c-e051-4543-988c-e969de3a8567
# ╠═3fa8c293-ac47-4acd-bdb7-9313286ee464
# ╠═68873d3e-054d-4ab4-9d89-73586bb0370e
# ╠═f89e88c9-ebfc-404f-964d-acff5c7f8985
# ╟─bcb73e19-8bf9-4e75-a9f7-8152e3d23201
# ╠═c65d7f1f-224d-4144-aa46-d48a482db95a
# ╠═ca34e152-6a53-4e25-a3d2-964c75d70fd5
# ╠═30f86e30-e7e9-43c4-9001-1f2f0c6c2bea
# ╠═832ccb61-b588-4614-9f5a-efa0f9a6087d
# ╠═8d8e4026-2185-4d67-a259-4cdebaab0b94
# ╠═391f3f2a-74ae-463b-b109-a47eaaafdf97
