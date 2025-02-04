### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ dd014274-d7f6-11ef-14ba-ab2a11dedc23
begin
	cd("/home/jovyan")
	import Pkg
	
	# activate the shared project environment
    Pkg.activate("$(pwd())/relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	using CairoMakie, Random, DataFrames, Distributions, StatsBase,
		ForwardDiff, LinearAlgebra, JLD2, FileIO, CSV, Dates, JSON, RCall, Turing, Printf, Combinatorics, JuMP, HiGHS, AlgebraOfGraphics
	using LogExpFunctions: logistic, logit

	Turing.setprogress!(false)

	include("$(pwd())/PILT_models.jl")
	include("$(pwd())/sample_utils.jl")
	include("$(pwd())/model_utils.jl")
	include("$(pwd())/plotting_utils.jl")
	include("$(pwd())/fetch_preprocess_data.jl")
	include("$(pwd())/generate_experimental_sequences/sequence_utils.jl")
	nothing
end

# ╔═╡ b936ea97-e296-49f9-bc7e-59abad579c02
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

# ╔═╡ 7191fee3-ac77-47df-9270-0928e6304847
# PILT parameters
begin
	# PILT Parameters
	PILT_blocks_per_valence = 21
	PILT_trials_per_block = 10
	
	PILT_total_blocks = PILT_blocks_per_valence * 2
	PILT_n_confusing = vcat([0, 0, 1, 1], fill(2, PILT_total_blocks ÷ 2 - 4)) # Per valence
		
	# Post-PILT test parameters
	PILT_test_n_blocks = 2
end

# ╔═╡ 98d3fabd-edb0-485c-bfa6-3cf6c22ea121
# ╠═╡ disabled = true
#=╠═╡

  ╠═╡ =#

# ╔═╡ eb76c49f-2ba7-4782-890c-de1ecc6dcd80
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

# ╔═╡ 693cebb1-c85f-4068-87ca-12567d0a9981
function reorder_with_fixed(v::AbstractVector, fixed::AbstractVector; rng::Xoshiro = Xoshiro(0))
	v = collect(v)
	
    # Ensure fixed vector is of the same length as v, filling with `missing` if needed
    fixed = vcat(fixed, fill(missing, length(v) - length(fixed)))[1:length(v)]
    
    # Identify indices where `fixed` is not `missing`
    fixed_indices = findall(!ismissing, fixed)
    
    # Create a pool of remaining elements by removing one instance of each fixed value
    remaining = v[:]
    for idx in fixed_indices
        value = fixed[idx]
        first_match = findfirst(==(value), remaining)
        if first_match !== nothing
            deleteat!(remaining, first_match)
        else
            error("Fixed value $value not found in v")
        end
    end
    
    # Shuffle the remaining elements
    shuffled_remaining = shuffle(rng, remaining)
    
    # Create the result with the same type as `v` but allow `missing`
    result = Vector{Union{eltype(v), Missing}}(undef, length(v))
    result .= missing
    
    # Place fixed elements in their positions
    for idx in fixed_indices
        result[idx] = fixed[idx]
    end
    
    # Fill the rest with shuffled elements
    remaining_idx = setdiff(1:length(v), fixed_indices)
    result[remaining_idx] .= shuffled_remaining
    
    return result
end

# ╔═╡ 113802e4-1b44-4c74-b025-2809956e6903
function count_valence_transitions(valence::AbstractVector)
	# Preallocate Dict
	transitions = Dict((i, j) => 0 for i in [1, -1] for j in [1, -1])

	# Loop and count
	for i in eachindex(valence)[2:end]
		transitions[(valence[i-1], valence[i])] += 1
	end

	return transitions
end

# ╔═╡ 0b38f306-d6c5-4c07-a477-2b4881394aed
# Assign valence and set size per block
block_score, PILT_block_attr = let random_seed = 5
	
	# # All combinations of valence
	initial_block_attr = DataFrame(
		block = repeat(1:PILT_total_blocks),
		valence = repeat([1, -1], inner = PILT_blocks_per_valence),
		fifty_high = fill(true, PILT_total_blocks)
	)

	# Shuffle set size and valence, making sure valence is varied, and positive in the first block and any time noise is introduced, and shaping doesn't extend too far into the task
	rng = Xoshiro(random_seed)

	best_score = Inf
	best_block_order = DataFrame()


	for _ in 1:10
		block_attr = DataFrame()
		first_three_same = true
		first_block_punishement = true
		too_many_repeats = true
		first_confusing_punishment = true
		shaping_too_long = true
		last_six_unbalanced = true
		last_six_transitions = true
		while first_three_same || first_block_punishement || too_many_repeats ||
			first_confusing_punishment || shaping_too_long || last_six_unbalanced ||
			last_six_transitions
	
			block_attr = DataFrames.transform(
				initial_block_attr,
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

			last_six_unbalanced = sum(block_attr.valence[37:42]) != 0

			last_six_transitions = 0 in values(count_valence_transitions(block_attr.valence[37:42]))
			
			first_block_punishement = block_attr.valence[1] == -1
	
			too_many_repeats = has_consecutive_repeats(block_attr.valence)
	
			first_confusing_punishment = 
				(block_attr.valence[findfirst(block_attr.n_confusing .== 1)] == -1) |
				(block_attr.valence[findfirst(block_attr.n_confusing .== 2)] == -1)
	
			shaping_too_long = 
				!all(block_attr.n_confusing[11:end] .== maximum(PILT_n_confusing))
		end
	
		# Compute number transitions from one valence to next
		transitions = count_valence_transitions(block_attr.valence[1:36])
	
		# Calculate deviation from uniform
		score = sum(abs.(values(transitions) .- (nrow(block_attr) / 4)))

		# Choose best score
		if score < best_score
			best_score = score
			best_block_order = block_attr
		end

	end

	@info(count_valence_transitions(best_block_order.valence[1:36]))
	

	best_score, best_block_order
end

# ╔═╡ 274d49b7-e48d-48e3-9778-a844b8ae7902
0 in values(count_valence_transitions((shuffle([1,1, 1, -1, -1, -1]))))

# ╔═╡ e50d1e10-f5b6-478c-9493-83bd93bc71b1
# Create feedback sequences per pair
PILT_sequences, common_per_pos, EV_per_pos = 
	let random_seed = 2
	
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
		ω_FI = .5,
		constrain_pairs = false,
		filename = "results/exp_sequences/eeg2.5_opt.jld2"
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

# ╔═╡ a64696c0-80e7-48a7-97bc-3d01bc41ae20
categories = ["airplane", "bagel", "beanbag", "bed", "boot", "bottle", "bowtie", "bucket", "calculator", "chessboard", "compass", "cookpot", "cup", "cushion", "easteregg", "frisbee", "gift", "glove", "hairbrush", "hat", "headphones", "jacket", "lamp", "lantern", "lawnmower", "necklace", "pen", "ring", "rug", "scale", "snowglobe", "socks", "sofa", "speakers", "spoon", "suitcase", "tent", "tongs", "toyrabbit", "trumpet", "umbrella", "watch", "wineglass"]

# ╔═╡ 353ed0bd-41ec-4146-8561-34c57b1765fe
# Assign stimulus images
PILT_stimuli = let random_seed = 0

	# Shuffle categories
	shuffle!(Xoshiro(random_seed), categories)

	# Assign stimulus pairs
	stimuli = assign_stimuli_and_optimality(;
		n_phases = 1,
		n_pairs = fill(1, PILT_total_blocks),
		categories = copy(categories),
		random_seed = random_seed
	)

	@info "Proportion of blocks on which the novel category is optimal : $(mean(stimuli.optimal_A))"

	rename!(stimuli, :phase => :session) # For compatibility with multi-phase sessions

	stimuli
end

# ╔═╡ 749f56e1-fcae-4497-a08e-ab40c3ec66a0
stimuli = let random_seed = 0

    shuffle!(Xoshiro(0), categories)  # Shuffle categories with RNG

    n_trials = 1000##20_000_000
    n_threads = Threads.nthreads()

    best_stimuli = nothing
    best_score = Inf

    # Allocate thread-local storage for best results
    local_best_stimuli = Vector{DataFrame}(undef, n_threads)
    local_best_scores = fill(Inf, n_threads)

	rngs = (x -> Xoshiro(x)).(1:n_threads)

    Threads.@threads for t in 1:n_trials

        # Assign stimulus pairs
		trial_stimuli = assign_stimuli_and_optimality(;
			n_phases = 1,
			n_pairs = fill(1, PILT_total_blocks),
			categories = copy(categories),
			random_seed = random_seed
		)

        rename!(trial_stimuli, :phase => :session)

        # Compute repeating_prev_optimal
        trial_stimuli.repeating_optimal = (trial_stimuli.block .> 1) .&& .!trial_stimuli.optimal_A

		# Compute repeating_prev_optimal
		trial_stimuli.repeating_prev_optimal = 
			vcat([missing], trial_stimuli.optimal_A[1:(end-1)])

        # Add valence and prev_valence
        trial_stimuli = leftjoin(
            trial_stimuli,
            insertcols(
                PILT_block_attr[!, [:block, :valence]],
                :prev_valence => vcat([missing], PILT_block_attr.valence[1:(end-1)])
            ),
            on = :block,
            order = :left
        )

        # Compute generalization score
        generalization_ns = combine(
            groupby(
                dropmissing(trial_stimuli, [:repeating_optimal, :repeating_prev_optimal]),
                [:repeating_optimal, :repeating_prev_optimal, :valence, :prev_valence]
            ),
            :repeating_optimal => length => :n
        )

        score = std(generalization_ns.n) +
                1000 * (nrow(generalization_ns) != 16) -
                100 * minimum(generalization_ns.n)

        # Update thread-local best score and stimuli
        thread_id = Threads.threadid()
        if score < local_best_scores[thread_id]
            local_best_scores[thread_id] = score
            local_best_stimuli[thread_id] = trial_stimuli
        end
    end

    # Find global best score across all threads
    global_best_index = argmin(local_best_scores)
    best_stimuli = local_best_stimuli[global_best_index]
    best_score = local_best_scores[global_best_index]

    best_stimuli
end

# ╔═╡ 3fbe11d0-e8ab-4080-a326-ba616a38ed0b
# Add stimulus assignments to sequences DataFrame, and assign right / left
PILT_task = let random_seed = 1

	# Join stimuli and sequences
	task = innerjoin(
		vcat(
			insertcols(PILT_sequences, 1, :session => 1), 
			insertcols(PILT_sequences, 1, :session => 2)),
		PILT_stimuli,
		on = [:session, :block],
		order = :left
	)

	@assert nrow(task) == nrow(PILT_sequences)  "Problem in join operation"

	# Assign right / left, equal proportions within each pair
	rng = Xoshiro(random_seed)

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
		:present_pavlovian => true
	)

	task
end

# ╔═╡ 8c6bca1d-1f69-4a87-a09a-71400f11a99a
# Validate task DataFrame
let task = PILT_task
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

# ╔═╡ 015b7b15-3871-4dc4-b79b-c98231e45a70
let
	save_to_JSON(PILT_task, "results/eeg_pilot_2.5_PILT.json")
	CSV.write("results/eeg_pilot_2.5_PILT.csv", PILT_task)
end

# ╔═╡ b56f3b2a-beb5-4870-9cbb-7921c6d0e890
# Visualize PILT seuqnce
let task = PILT_task

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
	


	save("results/eeg2.5_pilt_trial_plan.png", f, pt_per_unit = 1)

	f

end

# ╔═╡ 0b460b18-9111-44fc-a34d-6f4fc4bcfa3f
filter(x -> (x.session == 1) & (x.block > 36), PILT_task)

# ╔═╡ 99100834-6763-486a-ba33-e1aa4e8eb2ba
PILT_task.block .> 36

# ╔═╡ 7b06d63f-a7e8-4561-9a11-38e4e8123233
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

# ╔═╡ e704ce3a-992f-4620-bbf0-dc17f104e2e8
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

# ╔═╡ 8f6e4026-2f19-4b98-8d72-155fd7941252
PILT_test = let task = PILT_task
	
	# Find test sequence for each session
	PILT_test = []
	for s in 1
		test, pb, pv, nm = find_best_test_sequence(
			filter(x -> x.session == s & x.block <= 36, task),
			n_seeds = 100, # Number of random seeds to try
			same_weight = 10. # Weight reducing the number of same magntiude pairs
		) 

		# Add session variable
		insertcols!(test, 1, :session => s)

		push!(PILT_test, test)
	
		@info "Session $s: proportion of same block pairs: $pb"
		@info "Session $s: proportion of same valence pairs: $pv"
		@info "Session $s: number of same magnitude pairs: $nm"
	end

	# Concatenate together
	PILT_test = vcat(PILT_test...)

	# Create magnitude_pair variable
	PILT_test.magnitude_pair = [sort([r.magnitude_left, r.magnitude_right]) for r in eachrow(PILT_test)]

	PILT_test
end

# ╔═╡ 66847a50-07f8-4e32-9061-49eb30e12787
let
	save_to_JSON(PILT_test, "results/eeg_pilot2.5_PILT_test.json")
	CSV.write("results/eeg_pilot2.5_PILT_test.csv", PILT_test)
end

# ╔═╡ 6fd3882c-7669-4705-91e9-31dd08525e2f
PILT_test_extra = let task = PILT_task
	
	# Find test sequence for each session
	PILT_test = []
	for s in 1
		test, pb, pv, nm = find_best_test_sequence(
			filter(x -> (x.session == s) & (x.block > 36), task),
			n_seeds = 100, # Number of random seeds to try
			same_weight = 10. # Weight reducing the number of same magntiude pairs
		) 

		# Add session variable
		insertcols!(test, 1, :session => s)

		push!(PILT_test, test)
	
		@info "Session $s: proportion of same block pairs: $pb"
		@info "Session $s: proportion of same valence pairs: $pv"
		@info "Session $s: number of same magnitude pairs: $nm"
	end

	# Concatenate together
	PILT_test = vcat(PILT_test...)

	# Create magnitude_pair variable
	PILT_test.magnitude_pair = [sort([r.magnitude_left, r.magnitude_right]) for r in eachrow(PILT_test)]

	PILT_test
end

# ╔═╡ d40b6873-f966-4992-8966-9367953c0384
let
	save_to_JSON(PILT_test_extra, "results/eeg_pilot2.5_PILT_test_extra.json")
	CSV.write("results/eeg_pilot2.5_PILT_test_extra.csv", PILT_test_extra)
end

# ╔═╡ Cell order:
# ╠═dd014274-d7f6-11ef-14ba-ab2a11dedc23
# ╠═b936ea97-e296-49f9-bc7e-59abad579c02
# ╠═7191fee3-ac77-47df-9270-0928e6304847
# ╠═98d3fabd-edb0-485c-bfa6-3cf6c22ea121
# ╠═eb76c49f-2ba7-4782-890c-de1ecc6dcd80
# ╠═0b38f306-d6c5-4c07-a477-2b4881394aed
# ╠═274d49b7-e48d-48e3-9778-a844b8ae7902
# ╠═693cebb1-c85f-4068-87ca-12567d0a9981
# ╠═113802e4-1b44-4c74-b025-2809956e6903
# ╠═e50d1e10-f5b6-478c-9493-83bd93bc71b1
# ╠═a64696c0-80e7-48a7-97bc-3d01bc41ae20
# ╠═353ed0bd-41ec-4146-8561-34c57b1765fe
# ╠═749f56e1-fcae-4497-a08e-ab40c3ec66a0
# ╠═3fbe11d0-e8ab-4080-a326-ba616a38ed0b
# ╠═8c6bca1d-1f69-4a87-a09a-71400f11a99a
# ╠═015b7b15-3871-4dc4-b79b-c98231e45a70
# ╠═b56f3b2a-beb5-4870-9cbb-7921c6d0e890
# ╠═8f6e4026-2f19-4b98-8d72-155fd7941252
# ╠═66847a50-07f8-4e32-9061-49eb30e12787
# ╠═6fd3882c-7669-4705-91e9-31dd08525e2f
# ╠═d40b6873-f966-4992-8966-9367953c0384
# ╠═0b460b18-9111-44fc-a34d-6f4fc4bcfa3f
# ╠═99100834-6763-486a-ba33-e1aa4e8eb2ba
# ╠═7b06d63f-a7e8-4561-9a11-38e4e8123233
# ╠═e704ce3a-992f-4620-bbf0-dc17f104e2e8
