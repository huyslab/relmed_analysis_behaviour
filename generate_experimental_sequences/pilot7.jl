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

# ╔═╡ de74293f-a452-4292-b5e5-b4419fb70feb
categories = let
	categories = (s -> replace(s, ".jpg" => "")[1:(end-2)]).(readdir("generate_experimental_sequences/pilot7_stims"))

	# Keep only categories where we have two files exactly
	keeps = filter(x -> last(x) == 2, countmap(categories))

	filter(x -> x in keys(keeps), unique(categories))
end

# ╔═╡ c05d90b6-61a7-4f9e-a03e-3e11791da6d0
# RLWM Parameters
begin
	WM_set_sizes = [2, 4, 6]
	WM_blocks_per_cell = [fill(3, 4), [1, 2, 2, 1], fill(1, 4)] # Det. reward - det. punishment - prob. reward - prob. punishment
	WM_trials_per_triplet = 10

	# Total number of blocks
	WM_n_total_blocks = sum(sum.(WM_blocks_per_cell))

	# Total number of triplets
	WM_n_total_tiplets = sum(sum.(WM_set_sizes .* WM_blocks_per_cell))

	# Post-PILT test parameters
	WM_test_n_blocks = 1
end

# ╔═╡ 85deb936-2204-4fe8-a0dd-a23f527f813d
md"""## Post-PILT test"""

# ╔═╡ db7b8c41-7160-4a77-a058-26086d09b7a4
function prepare_for_finding_test_sequence(
	pilt_task::DataFrame;
	stimulus_locations::Vector{String} = ["right", "middle", "left"]
)

	# Extract stimuli and their common feedback from task structure
	stimuli = vcat([rename(
		pilt_task[pilt_task.feedback_common, [:session, :block, :n_groups, :stimulus_group_id,  Symbol("stimulus_$s"), Symbol("feedback_$s")]],
		Symbol("stimulus_$s") => :stimulus,
		Symbol("feedback_$s") => :feedback
	) for s in stimulus_locations]...)

	# Summarize magnitude per stimulus
	stimuli = combine(
		groupby(stimuli, [:session, :block, :n_groups, :stimulus_group_id, :stimulus]),
		:feedback => (x -> mean(unique(x))) => :magnitude
	)

	# Step 1: Identify unique stimuli
	unique_stimuli = unique(stimuli.stimulus)

	# Step 2: Define existing pairs
	pairer(v) = [[v[i], v[j]] for i in 1:length(v) for j in i+1:length(v)]
	create_pair_list(d) = vcat([pairer(filter(x -> x.stimulus_group_id == p, d).stimulus) 
		for p in unique(stimuli.stimulus_group_id)]...)

	existing_pairs = create_pair_list(stimuli)

	# Step 3: Generate all possible pairs
	all_possible_pairs = unique(sort.(collect(combinations(unique_stimuli, 2))))

	return stimuli, unique_stimuli, existing_pairs, all_possible_pairs

end

# ╔═╡ 62ca4f41-0b0d-4125-a85b-0a9752714d64
function create_test_sequence(;
	stimuli::DataFrame,
	unique_stimuli::AbstractVector,
	existing_pairs::AbstractVector,
	all_possible_pairs::AbstractVector,
	random_seed::Int64, 
	same_weight::Float64 = 6.5,
	test_n_blocks::Int64 = WM_test_n_blocks
) 

	rng = Xoshiro(random_seed)

	# Function to summarize used pairs
	pairer(v) = [[v[i], v[j]] for i in 1:length(v) for j in i+1:length(v)]
	create_pair_list(d) = vcat([pairer(filter(x -> x.stimulus_group_id == p, d).stimulus) 
		for p in unique(stimuli.stimulus_group_id)]...)

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
				return DataFrame(), NaN, NaN, NaN, NaN
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
		original_block_left = [stimuli[stimuli.stimulus .== p[1], :].block[1] for p in final_pairs],
		original_n_groups_right = [stimuli[stimuli.stimulus .== p[2], :].n_groups[1] for p in final_pairs],
		original_n_groups_left = [stimuli[stimuli.stimulus .== p[1], :].n_groups[1] for p in final_pairs]
	)

	# Same / different block variable
	pairs_df.same_block = pairs_df.original_block_right .== pairs_df.original_block_left

	# Valence variables
	pairs_df.valence_left = sign.(pairs_df.magnitude_left)
	pairs_df.valence_right = sign.(pairs_df.magnitude_right)
	pairs_df.same_valence = pairs_df.valence_left .== pairs_df.valence_right

	# Set size variables
	pairs_df.set_sizes = [sort([r.original_n_groups_left, r.original_n_groups_right]) for r in eachrow(pairs_df)]

	set_size_pairings = combine(
		groupby(pairs_df, :set_sizes),
		:set_sizes => length => :n
	)

	# Compute sequence stats
	prop_same_block = mean(pairs_df.same_block)
	prop_same_valence = mean(pairs_df.same_valence)
	std_set_sizes = std(set_size_pairings.n)
	prop_same_magnitude = mean(pairs_df.magnitude_right .== pairs_df.magnitude_left)
	
	pairs_df, prop_same_block, prop_same_valence, prop_same_magnitude, std_set_sizes
end

# ╔═╡ 6dff1b52-b0f0-4895-89d3-f732791e11c5
# Choose test sequence with best stats
function find_best_test_sequence(
	task::DataFrame; # PILT task structure
	n_seeds::Int64 = 10, # Number of random seeds to try
	same_weight::Float64 = 4.1 # Weight reducing the number of same magntiude pairs
) 

	# Prepare for finding sequences
	stimuli, unique_stimuli, existing_pairs, all_possible_pairs =
		prepare_for_finding_test_sequence(task)

	
	best_score = Inf
	chosen_test = DataFrame()
	best_pb = Inf
	best_pv = Inf
	best_pm = Inf 
	best_sss = Inf

	# Run over seeds
	for s in 1:n_seeds
		test, pb, pv, pm, sss = create_test_sequence(; 
			stimuli = stimuli,  
			existing_pairs = existing_pairs,
			unique_stimuli = unique_stimuli,
			all_possible_pairs = all_possible_pairs,
			random_seed = s, 
			same_weight = same_weight
		)

		
		# Compute deviation from goal
		dev_block = abs(pb - 1/3)
		dev_valence = abs(pv - 0.5)

		# Compute score for seed
		score = dev_block + dev_valence + sss

		if (!isnan(score)) && score < best_score
			best_score = score
			best_pb = pb
			best_pv = pv
			best_pm = pm
			best_sss = sss
			chosen_test = test
		end

	end

	# Return sequence and stats
	return chosen_test, best_pb, best_pv, best_pm, best_sss
end

# ╔═╡ 70049c30-ca94-421d-80fe-61c5af5d404f
countmap(1:10)

# ╔═╡ 5b37feb9-30c2-4e72-bba9-08f3b4e1c499
function assign_triplet_stimuli_and_optimality(;
	n_phases::Int64,
	n_groups::Vector{Int64}, # Number of groups in each block. Assume same for all phases
	categories::Vector{String} = [('A':'Z')[div(i - 1, 26) + 1] * ('a':'z')[rem(i - 1, 26)+1] 
		for i in 1:(sum(n_groups) * 3 * n_phases + n_phases)],
	rng::AbstractRNG
)

	# Copy categories so that it is not changed
	this_cats = copy(categories)

	total_n_groups = sum(n_groups) # Number of pairs needed
	
	@assert rem(length(n_groups), 2) == 0 "Code only works for even number of blocks per sesion"

	# Compute how many repeating categories we will have
	n_repeating = sum(min.(n_groups[2:end], n_groups[1:end-1]))

	# rng = Xoshiro(random_seed)

	# Assign whether repeating is optimal and shuffle
	repeating_optimal = vcat([shuffled_fill([true, false, false], n_repeating, rng = rng) for p in 1:n_phases]...)

	# Assign whether categories that cannot repeat are optimal
	rest_optimal = vcat([shuffled_fill([true, false, false], total_n_groups - n_repeating, rng = rng) for p in 1:n_phases]...)

	# Initialize vectors for stimuli. A novel to be repeated, B just novel, C may be repeating
	stimulus_A = []
	stimulus_B = []
	stimulus_C = []
	optimal_C = []
	repeating_C = []
	
	for j in 1:n_phases
		for (i, p) in enumerate(n_groups)

	
			# Choose repeating categories for this block
			n_repeating = ((i > 1) && minimum([p, n_groups[i - 1]])) * 1
			append!(
				stimulus_C,
				stimulus_A[(end - n_repeating + 1):end]
			)

			# Update repeating_C variable
			append!(
				repeating_C,
				fill(true, n_repeating)
			)
	
			# Fill up stimulus_repeating with novel categories if not enough to repeat
			if (p - n_repeating) > 0
				for _ in 1:(p - n_repeating)
					push!(
						stimulus_C,
						popfirst!(this_cats)
					)
				end
			end

			# Update repeating_C variable
			append!(
				repeating_C,
				fill(false, p - n_repeating)
			)
			
			# Choose novel categories for this block
			for _ in 1:p
				push!(
					stimulus_A,
					popfirst!(this_cats)
				)

				push!(
					stimulus_B,
					popfirst!(this_cats)
				)
			end

			# Populate who is optimal vector
			for _ in 1:(n_repeating)
				push!(
					optimal_C,
					popfirst!(repeating_optimal)
				)
			end

			if (p - n_repeating) > 0
				for _ in 1:(p - n_repeating)
					push!(
						optimal_C,
						popfirst!(rest_optimal)
					)
				end
			end
		end
	end

	stimulus_A = (x -> x * "_1.jpg").(stimulus_A)
	stimulus_B = (x -> x * "_1.jpg").(stimulus_B)
	stimulus_C = (x -> x * "_2.jpg").(stimulus_C)

	optimal_stimulus = ifelse.(
		optimal_C,
		"C",
		"X"
	)

	optimal_stimulus[optimal_stimulus .== "X"] = shuffled_fill(["A","B"], sum(.!optimal_C), rng = rng)

	return DataFrame(
		phase = repeat(1:n_phases, inner = total_n_groups),
		block = repeat(
			vcat([fill(i, p) for (i, p) in enumerate(n_groups)]...), n_phases),
		stimulus_group = repeat(
			vcat([1:p for p in n_groups]...), n_phases),
		stimulus_A = stimulus_A,
		stimulus_B = stimulus_B,
		stimulus_C = stimulus_C,
		optimal_stimulus = optimal_stimulus,
		repeating_C = repeating_C
	)

end


# ╔═╡ 69afd881-0c45-48c9-8db9-699f9ae23bec
function count_delays(base_sequence::AbstractVector)

	stimuli = unique(base_sequence)

	# Calculate delays between successive appearances
	delays = Dict(stim => Int[] for stim in stimuli)
	last_position = Dict(stim => -99 for stim in stimuli)

	for (i, stim) in enumerate(base_sequence)
		if last_position[stim] != -99
			push!(delays[stim], i - last_position[stim])
		end
		last_position[stim] = i
	end

	return vcat(values(delays)...)
end

# ╔═╡ b4a3c42d-ebc6-4d7f-a451-271fc3a5132d
function randomize_triplets(
	ns::Int64, 
	n_repeats::Int64; 
	max_iters::Int64 = 2*10^(ns),
	rng::AbstractRNG = Random.default_rng()
)

	stimuli = 1:ns
    target_uniform = [1:(2 * ns - 1);]
    ideal_freq = n_repeats * ns / length(target_uniform)

    best_sequence = []
    best_score = Inf
	best_hist = []
	best_delays = []

    # Generate the initial sequence
    base_sequence = repeat(stimuli, outer = n_repeats)

    for _ in 1:max_iters
        # Shuffle the sequence in miniblocks of 2*ns
		for i in 1:(2*ns):(n_repeats * ns)
        	base_sequence[i:(i+2*ns-1)] = shuffle(rng, base_sequence[i:(i+2*ns-1)])
		end
        
        # Calculate delays between successive appearances
        all_delays = count_delays(base_sequence)
		
        # Calculate a score
        histogram = counts(all_delays, 1:(2 * ns - 1))
        score = sum(abs.(histogram .- ideal_freq)) + 
			1000 * (maximum(all_delays) > (2 * ns - 1))

        if score < best_score
            best_sequence = copy(base_sequence)
			best_score = score
            best_hist = counts(all_delays)
			best_delays = sort(unique(all_delays))
        end
    end

    return best_sequence
end

# ╔═╡ fdbe5c4e-29cd-4d24-bbe5-40d24d5f98f4
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

# ╔═╡ 3952d2ef-a5c6-40a1-9373-6c4c9ff5ec2b
function count_valence_transitions(valence::AbstractVector)
	# Preallocate Dict
	transitions = Dict((i, j) => 0 for i in [1, -1] for j in [1, -1])

	# Loop and count
	for i in eachindex(valence)[2:end]
		transitions[(valence[i-1], valence[i])] += 1
	end

	return transitions
end

# ╔═╡ 699245d7-1493-4f94-bcfc-83184ca521eb
# Assign valence, probabilistic/determinitistic, and set size per block
block_order = let random_seed = 1

	# Replicate set sizes
	block_order = DataFrame(
		set_size = vcat([fill(s, n) for (i, s) in enumerate(WM_set_sizes) for n in WM_blocks_per_cell[i]]...),
	)

	# Set deterministic/probabilistic, valence
	DataFrames.transform!(
		groupby(block_order, :set_size),
		:set_size => (x -> vcat(
			fill("det", length(x) ÷ 2), 
			fill("prob", length(x) ÷ 2)
		)) => :det_prob,
		:set_size => (x -> [y for pair in zip(
			fill(-1, length(x) ÷ 2), 
			fill(1, length(x) ÷ 2)
		) for y in pair]) => :valence
	)

	# Shuffle with constraints
	rng = Xoshiro(random_seed)

	best_score = Inf
	best_block_order = DataFrame()

	# Shuffle trying to keep transitions between valence block uniform
	for _ in 1:10
		# Shuffle set size order making sure we start with low set sizes
		torder = DataFrames.transform(
			groupby(block_order, [:det_prob, :valence]),
			:set_size => (x -> reorder_with_fixed(x, [2, 4], rng = rng)) => :set_size
		)
	
		sort!(torder, :det_prob)
	
		# Shuffle valence, set size order, making sure we start with rewards and low set sizes
		torder.block = vcat(
			invperm(reorder_with_fixed(1:(nrow(torder) ÷ 2), [2, 4, 1], rng = rng)),
			invperm(reorder_with_fixed(1:(nrow(torder) ÷ 2), [2, 4, 1], rng = rng)) .+ (nrow(torder) ÷ 2)
		)
	
		sort!(torder, :block)

		# Index of first probabilistic block
		first_prob = findfirst(torder.det_prob .== "prob")

		# Compute number transitions from one valence to next
		transitions = count_valence_transitions(torder.valence)

		# Adjust for transition from det to prob
		transitions[(torder.valence[first_prob-1], torder.valence[first_prob])] -= 1

		# Calculate deviation from uniform
		score = sum(abs.(values(transitions) .- (nrow(torder) / 4)))

		# Choose best score
		if score < best_score
			best_score = score
			best_block_order = torder
		end
	end

	block_order = best_block_order

	# Assign n_confusing
	DataFrames.transform!(
		groupby(block_order, [:det_prob, :valence]),
		:det_prob => (x -> ifelse(
				x[1] == "det",
				fill(0, length(x)),
				vcat([1, 1], fill(2, length(x) - 2))
			)) => :n_confusing
	)

	# Reorder columns
	select!(
		block_order,
		[:block, :det_prob,  :set_size, :n_confusing, :valence]
	)

	# Checks
	@assert nrow(block_order) == WM_n_total_blocks "Number of blocks different from desired"

	@assert sort(combine(
		groupby(block_order, [:set_size, :det_prob, :valence]),
		:block => length => :n
	), [:set_size, :det_prob]).n == [3, 3, 3, 3, 1, 2, 2, 1, 1, 1, 1, 1] "Block numbers don't match desired"

	block_order
	
end

# ╔═╡ 7e078cb5-c615-4dc8-9060-3b69c86648b6
# Create feedback sequences per pair
sequence, common_per_pos, EV_per_pos =
let random_seed = 2
	
	# Compute how much we need of each sequence category
	n_confusing_wanted = combine(
		groupby(block_order, :n_confusing),
		:set_size => sum => :n
	)
	
	# Generate all sequences and compute FI
	FI_seqs = [compute_save_FIs_for_all_seqs(;
		n_trials = 10,
		n_confusing = n,
		fifty_high = true,
		prop_fifty = 0.2,
		model = single_p_QL_recip,
		model_name = "QL_recip",
		unpack_function = unpack_single_p_QL
	) for n in n_confusing_wanted.n_confusing]

	# Unpack results
	common_seqs = [x[2] for x in FI_seqs]
	magn_seqs = [x[3] for x in FI_seqs]

	# # Choose sequences optimizing FI under contraints
	chosen_idx, common_per_pos, EV_per_pos = optimize_FI_distribution(
		n_wanted = n_confusing_wanted.n,
		FIs = [x[1] for x in FI_seqs],
		common_seqs = common_seqs,
		magn_seqs = magn_seqs,
		ω_FI = 0.07,
		constrain_pairs = false,
		filename = "results/exp_sequences/pilot7_opt.jld2"
	)

	@assert length(vcat(chosen_idx...)) == sum(block_order.set_size) "Number of saved optimize sequences does not match number of sequences needed. Delete file and rerun."

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
		idx = repeat(1:n_sequences, inner = WM_trials_per_triplet),
		sequence = repeat(vcat([1:length(x) for x in chosen_common]...), 
			inner = WM_trials_per_triplet),
		appearance = repeat(1:WM_trials_per_triplet, n_sequences),
		feedback_common = vcat(vcat(chosen_common...)...),
		variable_magnitude = vcat(vcat(chosen_magn...)...)
	)

	# Create n_confusing varaible
	DataFrames.transform!(
		groupby(task, :idx),
		:feedback_common => (x -> WM_trials_per_triplet - sum(x)) => :n_confusing
	)

	# Expand block_order to triplet_order
	triplet_order = combine(
		groupby(block_order, [:block, :set_size, :n_confusing, :valence]),
		:set_size => (x -> 1:x[1]) => :stimulus_group
	)

	# Add sequnces variable to block_order
	DataFrames.transform!(
		groupby(triplet_order, :n_confusing),
		:stimulus_group => (x -> shuffle(rng, 1:length(x))) => :sequence
	)

	# Combine with block attributes
	task = innerjoin(
		task,
		triplet_order,
		on = [:n_confusing, :sequence],
		order = :left
	)

	@assert nrow(task) == length(vcat(vcat(chosen_common...)...)) "Problem with join operation"
	@assert nrow(unique(task[!, [:block]])) == WM_n_total_blocks "Problem with join operation"
		
	# Sort by block
	sort!(task, [:block, :stimulus_group, :appearance])

	# Remove auxillary variables
	select!(task, Not([:sequence, :idx]))

	# Shuffle triplet order
	shuffled_triplet_order = DataFrame(
		block = vcat([fill(r.block, r.set_size * WM_trials_per_triplet) for r in eachrow(block_order)]...),
		stimulus_group = vcat([randomize_triplets(ns, WM_trials_per_triplet; rng = rng) for ns in block_order.set_size]...)
	)

	DataFrames.transform!(
		groupby(shuffled_triplet_order, [:block, :stimulus_group]),
		:block => (x -> 1:length(x)) => :appearance
	)

	task = innerjoin(
		task,
		shuffled_triplet_order,
		on = [:block, :stimulus_group, :appearance],
		order = :right
	)

	# Compute trial varaible
	DataFrames.transform!(
		groupby(task, :block),
		:block => (x -> 1:length(x)) => :trial
	)

	# Compute triplet counter
	group_ids = sort(unique(task[!, [:block, :stimulus_group]]), [:block, :stimulus_group])

	group_ids.stimulus_group_id = 1:nrow(group_ids)

	task = leftjoin(
		task,
		group_ids,
		on = [:block, :stimulus_group],
		order = :left
	)

	# Compute low and high feedback
	task.feedback_high = ifelse.(
		task.valence .> 0,
		task.variable_magnitude,
		ifelse.(
			task.variable_magnitude .== 1.,
			fill(-0.01, nrow(task)),
			.- task.variable_magnitude
		)
	)

	task.feedback_low = ifelse.(
		task.valence .> 0,
		fill(0.01, nrow(task)),
		fill(-1.0, nrow(task))
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

# ╔═╡ 6469c3ec-7e45-4c48-8621-75b17ac347d0
stimuli = let random_seed = 0

    shuffle!(Xoshiro(0), categories)  # Shuffle categories with RNG

    n_trials = 2000#20_000_000
    n_threads = Threads.nthreads()

    best_stimuli = nothing
    best_score = Inf

    # Allocate thread-local storage for best results
    local_best_stimuli = Vector{DataFrame}(undef, n_threads)
    local_best_scores = fill(Inf, n_threads)

	rngs = (x -> Xoshiro(x)).(1:n_threads)

    Threads.@threads for t in 1:n_trials

        # Assign stimulus pairs
        trial_stimuli = assign_triplet_stimuli_and_optimality(
            n_phases = 1,
            n_groups = disallowmissing(block_order.set_size),
			categories = categories,
            rng = rngs[Threads.threadid()]
        )

        rename!(trial_stimuli, :phase => :session)

        # Compute repeating_prev_optimal
        trial_stimuli.repeating_optimal = ifelse.(
            trial_stimuli.repeating_C,
            trial_stimuli.optimal_stimulus .== "C",
            missing
        )

        repeating_prev_optimal::Vector{Union{Missing, Bool}} = fill(missing, nrow(trial_stimuli))

        for (i, r) in enumerate(eachrow(trial_stimuli))
            if r.repeating_C
                repeating_prev_optimal[i] = only(
                    trial_stimuli.optimal_stimulus[
                        trial_stimuli.stimulus_A .== replace(r.stimulus_C, "_2" => "_1")
                    ]
                ) == "A"
            end
        end

        trial_stimuli.repeating_prev_optimal = repeating_prev_optimal

        # Add valence and prev_valence
        trial_stimuli = leftjoin(
            trial_stimuli,
            insertcols(
                block_order[!, [:block, :valence]],
                :prev_valence => vcat([missing], block_order.valence[1:(end-1)])
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

# ╔═╡ 6ade28b0-34c9-483f-ba23-895f4302bd0f
# Combine stimuli and sequences
task = let random_seed = 1

	rng = Xoshiro(random_seed)

	# Join stimuli and sequence
	task = leftjoin(
		sequence,
		stimuli,
		on = [:block, :stimulus_group, :valence],
		order = :left
	)

	# Randomize stimuli location
	DataFrames.transform!(
		groupby(task, :stimulus_group_id),
		:trial => (x -> shuffled_fill(
			collect(permutations(["A", "B", "C"])), 
			length(x);
			rng = rng
		)) => :stimulus_locations
	)

	# Create stimulus_right, stimulus_middle, and stimulus_left variables
	task.stimulus_left = [task[i, Symbol("stimulus_$(x[1])")] 
		for (i,x) in enumerate(task.stimulus_locations)]
	
	task.stimulus_middle = [task[i, Symbol("stimulus_$(x[2])")] 
		for (i,x) in enumerate(task.stimulus_locations)]

	
	task.stimulus_right = [task[i, Symbol("stimulus_$(x[3])")] 
		for (i,x) in enumerate(task.stimulus_locations)]

	# Create optimal_side variable
	task.optimal_side = [["left", "middle", "right"][findfirst(task.stimulus_locations[i] .== x)] for (i,x) in enumerate(task.optimal_stimulus)]

	# Create feedback_right, feedback_middle, feedback_left variables
	task.feedback_left = ifelse.(
		task.optimal_side .== "left",
		task.feedback_optimal,
		task.feedback_suboptimal
	)

	task.feedback_middle = ifelse.(
		task.optimal_side .== "middle",
		task.feedback_optimal,
		task.feedback_suboptimal
	)

	task.feedback_right = ifelse.(
		task.optimal_side .== "right",
		task.feedback_optimal,
		task.feedback_suboptimal
	)

	# Add variables needed for experiment code
	insertcols!(
		task,
		:n_stimuli => 3,
		:optimal_right => "",
		:present_pavlovian => true
	)

	rename!(
		task,
		:set_size => :n_groups,
	)

	# Reorder columns
	select!(
		task,
		:session,
		:block,
		:valence,
		:trial,
		:stimulus_group,
		:stimulus_group_id,
		:appearance,
		names(task)
	)


end

# ╔═╡ f4dd2e9b-a500-406f-b2f0-3ec4d9611d8b
let
	save_to_JSON(task, "results/pilot7_PILT.json")
	CSV.write("results/pilot7_PILT.csv", task)
end

# ╔═╡ 87035e3e-e7ce-4320-a440-c150c4547c02
# Visualize seuqnce
let 

	f = Figure(size = (700, 300))

	# Proportion of confusing by trial number
	confusing_location = combine(
		groupby(task, :appearance),
		:feedback_common => (x -> mean(.!x)) => :feedback_confusing
	)

	mp1 = data(confusing_location) * mapping(
		:appearance => "Appearance", 
		:feedback_confusing => "Prop. confusing feedback"
	) * visual(ScatterLines)

	plt1 = draw!(f[1,1], mp1)

	# Plot confusing trials by block
	fp = insertcols(
		task,
		:color => ifelse.(
			task.feedback_common,
			(task.valence .+ 1) .÷ 2,
			fill(3, nrow(task))
		)
	)

	mp = data(fp) * mapping(
		:trial => "Trial",
		:block => "Block",
		:color
	) * visual(Heatmap)

	draw!(f[1,2], mp, axis = (; yreversed = true))

	# Plot confusing appearnce by triplet
	mp = data(fp) * mapping(
		:appearance => "Appearance",
		:stimulus_group_id => "Triplet",
		:color
	) * visual(Heatmap)

	draw!(f[1,3], mp, axis = (; yreversed = true))

	f

end

# ╔═╡ 2b7204c6-4fc6-41d2-b446-1c6bf75750b7
test = let
	
	# Find test sequence
	test, pb, pv, pm, sss = find_best_test_sequence(
		task,
		n_seeds = 100, # Number of random seeds to try
		same_weight = 8. # Weight reducing the number of same magntiude pairs
	) 

	# Add session variable
	insertcols!(test, 1, :session => 1)

	@info "Proportion of same block pairs: $pb"
	@info "Proportion of same valence pairs: $pv"
	@info "Proportion of same magnitude pairs: $pm"
	@info "SD of set size pair counts: $sss"

	# Create magnitude_pair variable
	test.magnitude_pair = [sort([r.magnitude_left, r.magnitude_right]) for r in eachrow(test)]

	@info "# of pairs per magnitude: $(sort(countmap((test.magnitude_pair))))"

	test
end

# ╔═╡ 0089db22-38ad-4d9c-88a2-12b82361384f
let
	save_to_JSON(test, "results/pilot7_PILT_test.json")
	CSV.write("results/pilot7_PILT_test.csv", test)
end

# ╔═╡ 8e9ffd82-89ec-4a63-83a8-54dfde7192a0
# Checks
let
	delays = combine(
		groupby(task, [:block, :n_groups]),
		:stimulus_group => (x -> length(unique(count_delays(x)))) => :n_unique_delays,
		[:stimulus_group, :n_groups] => ((t, s) -> Ref(counts(count_delays(t), 1:(2*s[1]-1)))) => :distribution
	)

	@assert all(delays.n_unique_delays .== 2 .* delays.n_groups .- 1) "Number of unique delay values between triplets should be exactly 2*ns-1"

	@assert all(all.((x -> 3 .≤ x .≤ 6).(delays.distribution))) "Distribution of delays far from uniform"

	@info "Each delay value appears $(minimum(minimum.(delays.distribution))) - $(maximum(maximum.(delays.distribution))) times"

	generaliation_ns = combine(
		groupby(
			dropmissing(stimuli, [:repeating_optimal, :repeating_prev_optimal]), [:repeating_optimal, :repeating_prev_optimal, :valence, :prev_valence]),
		:repeating_optimal => length => :n
	)

	@assert nrow(generaliation_ns) == 16 "There should be 16 conditions for generalization"

	@info "Each cell for generalization is repeated $(minimum(generaliation_ns.n))-$(maximum(generaliation_ns.n)) times"
	
	@assert maximum(task.block) == length(unique(task.block)) "Error in block numbering"

	@assert all(combine(groupby(task, [:session]), 
		:block => issorted => :sorted).sorted) "Task structure not sorted by block"

	@assert all(combine(groupby(task, [:session, :block]), 
		:trial => issorted => :sorted).sorted) "Task structure not sorted by trial number"
	
	@assert all(sign.(task.valence) == sign.(task.feedback_right)) "Valence doesn't match feedback sign"

	@assert all(sign.(task.valence) == sign.(task.feedback_left)) "Valence doesn't match feedback sign"

	@assert sum(unique(task[!, [:session, :block, :valence]]).valence) == 0 "Number of reward and punishment blocks not equal"

	@info "Overall proportion of common feedback: $(round(mean(task.feedback_common), digits = 2))"

	@assert all(combine(groupby(task, :stimulus_group_id),
		:appearance => maximum => :max_appear
	).max_appear .== WM_trials_per_triplet) "Didn't find exactly $WM_trials_per_triplet apperances per pair"

	# Count losses to allocate coins in to safe for beginning of task
	worst_loss = filter(x -> x.valence < 0, task) |> 
		df -> ifelse.(
			df.feedback_right .< df.feedback_left, 
			df.feedback_right, 
			df.feedback_left) |> 
		countmap

	@info "Worst possible loss in this task is of these coin numbers: $worst_loss"


end

# ╔═╡ Cell order:
# ╠═2d7211b4-b31e-11ef-3c0b-e979f01c47ae
# ╠═114f2671-1888-4b11-aab1-9ad718ababe6
# ╠═de74293f-a452-4292-b5e5-b4419fb70feb
# ╠═c05d90b6-61a7-4f9e-a03e-3e11791da6d0
# ╠═699245d7-1493-4f94-bcfc-83184ca521eb
# ╠═7e078cb5-c615-4dc8-9060-3b69c86648b6
# ╠═6469c3ec-7e45-4c48-8621-75b17ac347d0
# ╠═6ade28b0-34c9-483f-ba23-895f4302bd0f
# ╠═f4dd2e9b-a500-406f-b2f0-3ec4d9611d8b
# ╠═8e9ffd82-89ec-4a63-83a8-54dfde7192a0
# ╠═87035e3e-e7ce-4320-a440-c150c4547c02
# ╠═85deb936-2204-4fe8-a0dd-a23f527f813d
# ╠═2b7204c6-4fc6-41d2-b446-1c6bf75750b7
# ╠═0089db22-38ad-4d9c-88a2-12b82361384f
# ╠═6dff1b52-b0f0-4895-89d3-f732791e11c5
# ╠═db7b8c41-7160-4a77-a058-26086d09b7a4
# ╠═62ca4f41-0b0d-4125-a85b-0a9752714d64
# ╠═70049c30-ca94-421d-80fe-61c5af5d404f
# ╠═5b37feb9-30c2-4e72-bba9-08f3b4e1c499
# ╠═b4a3c42d-ebc6-4d7f-a451-271fc3a5132d
# ╠═69afd881-0c45-48c9-8db9-699f9ae23bec
# ╠═fdbe5c4e-29cd-4d24-bbe5-40d24d5f98f4
# ╠═3952d2ef-a5c6-40a1-9373-6c4c9ff5ec2b
