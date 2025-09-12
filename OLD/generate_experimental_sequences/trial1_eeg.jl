### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ d1ffbd6e-41f6-11f0-2884-51de482dd6fe
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

# ╔═╡ 0d99d4cc-18f9-4f5e-ab23-6bff9bdce753
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

# ╔═╡ 52303bea-d08a-4b0d-bb30-97633d428811
# General parameters
begin
	sessions = 1:2
end

# ╔═╡ 4bbb7c6d-71d3-4447-8359-348c86d05bdc
categories = vec([string(uppercase, lowercase) for lowercase in 'a':'z', uppercase in 'A':'Z'])

# ╔═╡ 83266903-2dd2-43e1-93fb-b5c5271bb02d
md"""# PILT"""

# ╔═╡ 57ca9c3d-6d91-43cf-8fc4-a8ad8dfce0dd
# PILT parameters
begin
	# PILT Parameters
	PILT_trials_per_block = 10
		
end

# ╔═╡ 149549b9-5ff3-451c-9f91-41a0c240e4e8
md"""## Post-PILT test"""

# ╔═╡ ad996e31-abdb-46c9-8930-6cfa7f833cb0
	# Post-PILT test parameters
	PILT_test_n_blocks = 3


# ╔═╡ cc409bce-df5e-4b77-906e-20b8342d5b5b
md"""## Reversal task"""

# ╔═╡ 531a2596-52c2-43d6-94d3-de273705a79c
# Reversal task parameters
begin
	rev_n_blocks = 30
	rev_n_trials = 80
	rev_prop_confusing = vcat([0, 0.1, 0.1, 0.2, 0.2], fill(0.3, rev_n_blocks - 5))
	rev_criterion = vcat(
		[8, 7, 6, 6, 5], 
		shuffled_fill(
			3:8, 
			rev_n_blocks - 5; 
			rng = Xoshiro(0)
		)
	)
end

# ╔═╡ 1ac016d1-7de7-405f-987e-446efca5ff90
function find_lcm_denominators(props::Vector{Float64})
	# Convert to rational numbers
	rational_props = rationalize.(props)
	
	# Extract denominators
	denominators = [denominator(r) for r in rational_props]
	
	# Compute the LCM of all denominators
	smallest_int = foldl(lcm, denominators)
end

# ╔═╡ af283fc5-50b0-47f6-97c6-33d5b189e76d
# Reversal task structure
reversal = let random_seed = 1

	# Compute minimal mini block length to accomodate proportions
	mini_block_length = find_lcm_denominators(rev_prop_confusing)
	@info "Randomizing in miniblocks of $mini_block_length trials"

	# Function to create high magnitude values for miniblock
	mini_block_high_mag(p, rng) = shuffle(rng, vcat(
		fill(1., round(Int64, mini_block_length * (1-p))),
		fill(0.01, round(Int64, mini_block_length * p))
	))

	# Function to create high magntidue values for block
	block_high_mag(p, rng) = 
		vcat(
			[mini_block_high_mag(p, rng) 
				for _ in 1:(div(rev_n_trials, mini_block_length))]...)

	# Set random seed
	rng = Xoshiro(random_seed)

	# Initialize
	feedback_optimal = Vector{Vector{Float64}}()

	# Make sure first sixs blocks don't start with confusing feedback on first trial
	dist_diff = 11
	while isempty(feedback_optimal) || 
		!all([bl[1] != 0.01 for bl in feedback_optimal[1:6]]) || dist_diff > 2

		# Assign blocks
		feedback_optimal = [block_high_mag(p, rng) for p in rev_prop_confusing]

		# Check distribution of confusing feedback
		dist = (x -> permutedims(reshape(x, div(length(x), 10), 10), (2,1))).(feedback_optimal)

		dist = vcat(dist...)

		dist = vec(sum(dist .== 1., dims = 1))
		dist_diff = maximum(abs.(diff(dist)))
	end

	# Function to compute feedback_suboptimal from feedback_optimal
	inverter(x) = 1 ./ (100 * x)

	# Draw whether right is optimal on even or odd blocks
	evenodd = Dict(s => shuffle(rng, [isodd, iseven]) for s in Symbol.(sessions))

	# Create timeline variables
	timeline = Dict(s => [[Dict(
		:block => bl,
		:trial => t,
		:feedback_left => evenodd[s][1](bl) ? feedback_optimal[bl][t] : 
			inverter(feedback_optimal[bl][t]),
		:feedback_right => evenodd[s][2](bl) ? feedback_optimal[bl][t] : 
			inverter(feedback_optimal[bl][t]),
		:optimal_right => evenodd[s][2](bl),
		:criterion => rev_criterion[bl]
	) for t in 1:rev_n_trials] for bl in 1:rev_n_blocks] for s in Symbol.(sessions))

	# Convert to DataFrame since eeg experiment code wants csv
	rows = []
	
	for (session, blocks) in timeline
	    for block_trials in blocks
	        for trial_dict in block_trials
	            push!(rows, merge(Dict(:session => parse(Int, string(session))), trial_dict))
	        end
	    end
	end
	
	df = DataFrame(rows)
		
end

# ╔═╡ cfb06155-e1b9-4e19-ae61-72d762f24529
let
	# Save to file
	CSV.write("results/trial1_eeg_reversal.csv", reversal)
end

# ╔═╡ 6d835516-0be7-42d0-bc46-d4e14dfd077d
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
		:feedback => StatsBase.mode => :magnitude,
		:feedback => mean => :EV
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

	# Create a priority queue for balanced selection based on pair counts
	pair_counts = Dict{Vector{Float64}, Int}()

	# Function to retrieve attribute of stimulus
	stim_attr(s, attr) = stimuli[stimuli.stimulus .== s, :][!, attr][1]

	for b in 1:test_n_blocks

		used_stimuli = Set{String}()

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
			if mag1 == mag2 # Exclude same magnitude
				continue
			end
		    if !haskey(magnitude_pairs, key)
		        magnitude_pairs[key] = []
		    end
		    push!(magnitude_pairs[key], pair)
		end
		
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
		feedback_right = [stimuli[stimuli.stimulus .== p[2], :].magnitude[1] for p in final_pairs],
		feedback_left = [stimuli[stimuli.stimulus .== p[1], :].magnitude[1] for p in final_pairs],
		EV_right = [stimuli[stimuli.stimulus .== p[2], :].EV[1] for p in final_pairs],
		EV_left = [stimuli[stimuli.stimulus .== p[1], :].EV[1] for p in final_pairs],
		original_block_right = [stimuli[stimuli.stimulus .== p[2], :].block[1] for p in final_pairs],
		original_block_left = [stimuli[stimuli.stimulus .== p[1], :].block[1] for p in final_pairs]
	)

	# Same / different block variable
	pairs_df.same_block = pairs_df.original_block_right .== pairs_df.original_block_left

	# Valence variables
	pairs_df.valence_left = sign.(pairs_df.EV_left)
	pairs_df.valence_right = sign.(pairs_df.EV_right)
	pairs_df.same_valence = pairs_df.valence_left .== pairs_df.valence_right

	# Compute sequence stats
	prop_same_block = (mean(pairs_df.same_block)) 
	prop_same_valence = (mean(pairs_df.same_valence))
	n_same_magnitude = sum(pairs_df.feedback_right .== pairs_df.feedback_left)
	
	pairs_df, prop_same_block, prop_same_valence, n_same_magnitude
end

# ╔═╡ b04341bc-82e0-4660-8809-42d852b5a95d
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
	dev_valence = abs.(prop_block .- 0.5)

	# Choose best sequence
	chosen = pass_magnitude[argmin(dev_valence)]

	# Return sequence and stats
	return create_test_sequence(task, random_seed = chosen, same_weight = same_weight)
end

# ╔═╡ 15527827-4835-42a8-88c5-cba1d6e9719a
function discretize_to_quantiles(values::Vector{T}, reference_values::Vector{T}, num_quantiles::Int) where T
    quantile_edges = quantile(reference_values, range(0, 1, length=num_quantiles+1))
    
    labels = ["[$(round(quantile_edges[i], sigdigits=3)), $(round(quantile_edges[i+1], sigdigits=3)))" for i in 1:num_quantiles]
    
    function assign_quantile(v)
        for (i, edge) in enumerate(quantile_edges[2:end])
            if v <= edge
                return labels[i]
            end
        end
        return labels[end]
    end
    
    categorical_values = CategoricalArray([assign_quantile(v) for v in values], ordered=true, levels=labels)
    return categorical_values
end

# ╔═╡ 71e38508-9a0d-4ab3-9e39-75f529d5672a
PILT_blocks = let rng = Xoshiro(0)

	coins = [-1., -0.5, -0.01, 0.01, 0.5, 1.]

	# Create all magnitdue combinations
	all_outcome_combinations = collect(product(coins, coins))

	outcome_combinations = vcat(
		filter(x -> x[1] < x[2], vec(all_outcome_combinations)), # First iteration of all relevant combinations
		[all_outcome_combinations[i, j] for (i,j) in zip(1:5, 2:6)] # Adding 5 more adjacent to the diagonal
	)

	outcome_combinations_extra = [all_outcome_combinations[i, j] for (i,j) in zip(1:4, 3:6)]

	# Convert to vector of vectors
	outcome_combinations = collect.(outcome_combinations)
	outcome_combinations_extra = collect.(outcome_combinations_extra)

	# Shuffle order within pair, keeping B stim outcome distribution flat
	criteria = false 
	while !criteria

		# Shuffle
		shuffle!.(rng, outcome_combinations)

		# Collect B stimuli values
		B_stim = [x[2] for x in outcome_combinations]

		B_stim_counts = values(countmap(B_stim))

		criteria = maximum(B_stim_counts) - minimum(B_stim_counts) == 1

	end

	# Shuffle order, under constraints
	criteria = false 
	
	while !criteria
		shuffle!(rng, outcome_combinations)

		# Summarize stim B outcome transitions--------
		outcome_B = [x[2] for x in outcome_combinations]

		transitions = [(outcome_B[i], outcome_B[i+1]) for i in 1:(length(outcome_B) - 1)]

		trans_counts = values(countmap(transitions))


        # Constraints:
        # 1. First block rewards only
        # 2. Blocks 2 and 3 with mixed valence
        # 3. Block 4 with punishments only
        # 4. Maximize exposure to most transition types
		criteria = all(outcome_combinations[1] .> 0) &&
			all((x -> (x[1] * x[2]) < 0).(outcome_combinations[2:3])) &&
			all(outcome_combinations[4] .< 0) &&
			maximum(trans_counts) <= 1 &&
            [-1.0, 1.] ∉ sort.(outcome_combinations[1:4]) &&
            all((x -> x[2] - x[1]).(outcome_combinations[1:4]) .!= 0.)

	end

	# Shuffle order of extra blocks
	shuffle!.(rng, outcome_combinations_extra)
	shuffle!(rng, outcome_combinations_extra)

	# Concatenate
	outcome_combinations = vcat(outcome_combinations, outcome_combinations_extra)
	
	# Arrange into DataFrame
	PILT_blocks_long = DataFrame(
		block = repeat(1:length(outcome_combinations), outer = 2),
		n_confusing = repeat(vcat(
			fill(0, 4), 
			fill(1, 4), 
			fill(2, length(outcome_combinations) - 8)
		), outer = 2),
		stimulus = repeat(["A", "B"], inner = length(outcome_combinations)),
		primary_outcome = vcat(
			[x[1] for x in outcome_combinations], 
			[x[2] for x in outcome_combinations]
		)
	)

	sort!(PILT_blocks_long, [:block, :stimulus])

	# Function to assign secondary outcomes
	function shuffle_secondary_outcome(n::Int64, primary::Float64)

		secondary = repeat(
			filter(x -> x != primary, coins), 
			n ÷ (length(coins) - 1)
		)

		distances = abs.(coins .- primary)

		furthest = coins[partialsortperm(distances, 1:(rem(n, length(coins) - 1)), rev=true)]

		secondary = vcat(secondary, furthest)

		return shuffle(rng, secondary)

	end

	# Tell apart deterministic from probabilistic blocks
	PILT_blocks_long.deterministic = PILT_blocks_long.n_confusing .== 0

	# Add secondary outcome, shuffle under contraints such that the first two blocks primary outcome matches secondary in sign, and that stimulus B (repeating) has uniform distribution
	criteria2 = false
	n_attempts = 0
	while !criteria2 && n_attempts < 100000
		DataFrames.transform!(
			groupby(PILT_blocks_long, [:deterministic, :primary_outcome]),
			[:deterministic, :primary_outcome] => 
				((d, o) -> ifelse.(
					d,
					fill(missing, length(o)),
					shuffle_secondary_outcome(
						length(o), unique(o)[1]))) => :secondary_outcome
				
		)

		# EV
		PILT_blocks_long.EV = ifelse.(
			ismissing.(PILT_blocks_long.secondary_outcome),
			PILT_blocks_long.primary_outcome,
			PILT_blocks_long.primary_outcome .* 0.8 + PILT_blocks_long.secondary_outcome .* 0.2
		)

        # Long to wide to compute EV difference
        PILT_blocks = unstack(
            PILT_blocks_long,
            [:block],
            :stimulus,
            :EV
        )

        # # Compute EV difference
        PILT_blocks.EV_abs_diff = abs.(PILT_blocks.A - PILT_blocks.B)

        # # Discretize to five bins from zero to two
        PILT_blocks.EV_abs_diff_bin = ceil.((PILT_blocks.EV_abs_diff .+ eps()) ./ 0.4) * 0.4

        EV_diff_bin_counts = countmap(PILT_blocks.EV_abs_diff_bin)

        EV_diff_counts_sd = std(values(EV_diff_bin_counts))

        # println(EV_diff_bin_counts)
        # println(maximum(EV_diff_bin_counts) - minimum(EV_diff_bin_counts))

        EV_diff_uniform = EV_diff_counts_sd < 2.85

        # Check if the first two blocks primary outcome matches secondary in sign
		first_mathces = all((r -> sign(r.primary_outcome) == 
			sign(r.secondary_outcome)).(eachrow(filter(x -> x.block in [5,6], PILT_blocks_long))))


        # Check if stimulus B has uniform distribution
		EV_B_bin = discretize_to_quantiles(
			PILT_blocks_long.EV,
			PILT_blocks_long.EV,
			5
		)[PILT_blocks_long.stimulus .== "B"]

		EV_B_bin_counts = values(countmap(EV_B_bin))

		B_uniform = maximum(EV_B_bin_counts) - minimum(EV_B_bin_counts) <= 2

		criteria2 = first_mathces && B_uniform && EV_diff_uniform

        if criteria2
            println(EV_diff_counts_sd)
            println(EV_diff_bin_counts)
        end
        
        n_attempts += 1

	end
	
	# Long to wide
	PILT_blocks = leftjoin(
		unstack(
			PILT_blocks_long,
			[:block, :n_confusing],
			:stimulus,
			:primary_outcome,
			renamecols = x -> "primary_outcome_$(x)"
		),
		unstack(
			PILT_blocks_long,
			[:block, :n_confusing],
			:stimulus,
			:secondary_outcome,
			renamecols = x -> "secondary_outcome_$(x)"
		),
		on = [:block, :n_confusing]
	)

	# Compute EV
	PILT_blocks.EV_A = ifelse.(
		ismissing.(PILT_blocks.secondary_outcome_A),
		PILT_blocks.primary_outcome_A,
		PILT_blocks.primary_outcome_A .* 0.8 + PILT_blocks.secondary_outcome_A .* 0.2
	)

	PILT_blocks.EV_B = ifelse.(
		ismissing.(PILT_blocks.secondary_outcome_B),
		PILT_blocks.primary_outcome_B,
		PILT_blocks.primary_outcome_B .* 0.8 + PILT_blocks.secondary_outcome_B .* 0.2
	)

    PILT_blocks.optimal_A = PILT_blocks.EV_A .> PILT_blocks.EV_B

    @assert all(PILT_blocks.EV_A .!= PILT_blocks.EV_B) "EVs are not different"

    @info "Proportion of blocks with repeating stimulus B optimal: $(mean(PILT_blocks.EV_B .> PILT_blocks.EV_A))"
	@info "Proportion of blocks withing the first 20 with repeating stimulus B optimal: $(mean(PILT_blocks.EV_B[1:20] .> PILT_blocks.EV_A[1:20]))"

	PILT_blocks

end

# ╔═╡ 4c093483-9436-44b5-bcdd-299538d4d05b
# Assign stimulus images
PILT_stimuli = let random_seed = 0

	@info length(categories)

	# Assign stimulus pairs
	stimuli = vcat(
		[
			insertcols(assign_stimuli_and_optimality(;
				n_phases = 1,
				n_pairs = fill(1, maximum(PILT_blocks.block)),
				categories = categories,
				rng = Xoshiro(random_seed),
				ext = "png"
		), 1, :session => s) for s in sessions
		]...
	)

    @info "Categories left after PILT assignment: $(length(categories))"

	@info "Proportion of blocks on which the novel category is optimal : $(mean(stimuli.optimal_A))"

    # Check that optimal_A is the same for each session
    @assert allequal([gdf.optimal_A for gdf in groupby(stimuli, :session)]) "optimal_A is not the same for each session"

    # Check that the set of stimuli used is different for each session
    @assert !any(intersect(gdf1.stimulus_A, gdf2.stimulus_A) != [] for (gdf1, gdf2) in combinations(groupby(stimuli, :session), 2)) "Stimuli used in different sessions are not disjoint"

	stimuli
end


# ╔═╡ 2a83846a-cc71-44b0-b75c-1b2b487d256c
# Create PILT sequence
PILT = let rng = Xoshiro(2)

    position_criterion = false
    PILT_sequence = DataFrame()
    while !position_criterion 

		# All positions of common feedback
		common_feedback_positions = Dict(
			n_confusing => shuffle(rng, collect(combinations(1:PILT_trials_per_block, n_confusing)))
			for n_confusing in unique(PILT_blocks.n_confusing)
		)


        # Helper function to assign common feedback
        get_sequence = function(indices::Vector{Int})
            v = fill(true, PILT_trials_per_block)
            v[indices] .= false
            return v
        end

        # Assign common feedback
        common_feedback = vcat(
            [
                n_confusing == 0 ?  fill(true, PILT_trials_per_block) : get_sequence(pop!(common_feedback_positions[n_confusing])) for n_confusing in PILT_blocks.n_confusing 
            ]...
        )

        # Make into DataFrame
        PILT_sequence = DataFrame(
            block = repeat(PILT_blocks.block, inner = PILT_trials_per_block),
            trial = repeat(1:PILT_trials_per_block, outer = length(PILT_blocks.block)),
            feedback_common = common_feedback,
            n_confusing = repeat(PILT_blocks.n_confusing, inner = PILT_trials_per_block)
        )

        # Compute proportion of common feedback per position
        position_sd = std(combine(
            groupby(PILT_sequence, :trial),
            :feedback_common => mean => :feedback_common
        ).feedback_common)

        position_criterion = position_sd < 0.0282

        if position_criterion
            @info "SD of position proportions: $(position_sd)"
        end

    end

    # Merge with blocks
    PILT_sequence = leftjoin(
        PILT_sequence,
        select(PILT_blocks, [:block, :n_confusing, :optimal_A, :primary_outcome_A, :primary_outcome_B, :secondary_outcome_A, :secondary_outcome_B]),
        on = [:block, :n_confusing]
    )

    # Create feedback_A and feedback_B
    PILT_sequence.feedback_A = ifelse.(
        PILT_sequence.feedback_common,
        PILT_sequence.primary_outcome_A,
        PILT_sequence.secondary_outcome_A
    )
    PILT_sequence.feedback_B = ifelse.(
        PILT_sequence.feedback_common,
        PILT_sequence.primary_outcome_B,
        PILT_sequence.secondary_outcome_B
    )

    # Merge with stimuli
    PILT_sequence = leftjoin(
        PILT_sequence,
        select(PILT_stimuli, [:block, :session, :stimulus_A, :stimulus_B]),
        on = [:block]
    )

    sort!(PILT_sequence, [:session, :block, :trial])

    # Randomize appearance on left / right
    DataFrames.transform!(
        groupby(PILT_sequence, [:session, :block]),
        :n_confusing => (x -> shuffled_fill([true, false], length(x), rng = rng)) => :A_on_right
    )

    # Create stimulus on left / right
    PILT_sequence.stimulus_right = ifelse.(
		PILT_sequence.A_on_right,
		PILT_sequence.stimulus_A,
		PILT_sequence.stimulus_B
	)

    PILT_sequence.stimulus_left = ifelse.(
		.!PILT_sequence.A_on_right,
		PILT_sequence.stimulus_A,
		PILT_sequence.stimulus_B
	)

    # Create optimal_right
    PILT_sequence.optimal_right = (PILT_sequence.A_on_right .& PILT_sequence.optimal_A) .| (.!PILT_sequence.A_on_right .& .!PILT_sequence.optimal_A)

    # Create feedback_right and feedback_left
    PILT_sequence.feedback_right = ifelse.(
        PILT_sequence.A_on_right,
        PILT_sequence.feedback_A,
        PILT_sequence.feedback_B
    )
    PILT_sequence.feedback_left = ifelse.(
        .!PILT_sequence.A_on_right,
        PILT_sequence.feedback_A,
        PILT_sequence.feedback_B
    )

    # Add variables needed for experiment script
	insertcols!(
		PILT_sequence,
		:n_stimuli => 2,
		:n_groups => 1,
		:stimulus_group => 1,
		:stimulus_group_id => PILT_sequence.block,
		:stimulus_middle => "",
		:feedback_middle => "",
		:optimal_side => "",
		:present_pavlovian => true,
		:early_stop => true,
        :valence => 0 # Meaning mixed
	)

end


# ╔═╡ b7034df7-868c-49b9-b883-e050738e0c33
# Validate task DataFrame
let task = PILT
	@assert maximum(task.block) == length(unique(task.block)) "Error in block numbering"

	@assert all(combine(groupby(task, :session), 
		:block => issorted => :sorted).sorted) "Task structure not sorted by block"

	@assert all(combine(groupby(task, [:session, :block]), 
		:trial => issorted => :sorted).sorted) "Task structure not sorted by trial number"

    @assert all(combine(
        groupby(task, [:session, :block, :n_confusing]),
        :feedback_common => (x -> sum(.!x)) => :feedback_rare
    ) |> df -> df.feedback_rare .== df.n_confusing) "Number of rare feedbacks does not match n_confusing"

    ev_right = select(
        groupby(task, [:session, :block]),
        :feedback_A => mean => :EV_A,
        :feedback_B => mean => :EV_B,
        :A_on_right,
        :optimal_right
    )

    ev_right.ev_right_bigger = ifelse.(
        ev_right.A_on_right,
        ev_right.EV_A .> ev_right.EV_B,
        ev_right.EV_B .> ev_right.EV_A
    )

    @assert all(ev_right.ev_right_bigger .== ev_right.optimal_right) "Optimality does not match EVs"

	@info "Overall proportion of common feedback: $(round(mean(task.feedback_common), digits = 2))"

	# Count losses to allocate coins in to safe for beginning of task
	worst_loss = countmap(
        ifelse.(
            task.feedback_right .< task.feedback_left,
            ifelse.(
                task.feedback_right .< 0,
                task.feedback_right,
                0
            ),
            ifelse.(
                task.feedback_left .< 0,
                task.feedback_left,
                0
            )
        )
    )

	@info "Worst possible loss in this task is of these coin numbers: $worst_loss"

end


# ╔═╡ 9d3736ba-004f-4d1a-9595-e138ba6cb2b9
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
	


	save("results/trial1_eeg_pilt_trial_plan.png", f, pt_per_unit = 1)

	f

end


# ╔═╡ 48620cba-b84f-4c14-aafa-1f39ffa99787
let
	CSV.write("results/trial1_eeg_PILT.csv", PILT)
end

# ╔═╡ 6e7d02b8-9889-49fc-b1b0-e82e7826f9ff
PILT_test_template = let s = 1,
	task = filter(x -> x.session == s && x.block < 21, PILT)
	
	# Find test sequence for each session
	test, pb, pv, nm = find_best_test_sequence(
		task,
		n_seeds = 100, # Number of random seeds to try
		same_weight = 1. # Weight reducing the number of same magntiude pairs
	) 

	# Add session variable
	insertcols!(test, 1, :session => s)

	@info "Session $s: proportion of same block pairs: $pb"
	@info "Session $s: proportion of same valence pairs: $pv"

	# Create magnitude_pair variable
	test.EV_pair = [sort([r.EV_left, r.EV_right]) for r in eachrow(test)]
	test.primary_outcome_pair = [sort([r.feedback_left, r.feedback_right]) for r in eachrow(test)]

	# Create optimal_right
	test.optimal_right = test.EV_right .> test.EV_left

	test
end


# ╔═╡ dae14142-7ad7-4846-9923-4643e7a2fa36
PILT_test = let PILT_test_template = copy(PILT_test_template)

	# Create stimulus dict to replace equivalent stimuli
	stimuli_dict = unique(
		select(
			filter(x -> x.block < 21, PILT),
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
		for s in sessions]...
	)

end

# ╔═╡ d0665196-1862-4bbd-b766-47052f72f07e
# Test test sequence
let
	@assert all(PILT_test_template.stimulus_left .== filter(x -> x.session == 1, PILT_test).stimulus_left) "Final sequence for wk0 does not match template"

	@assert all(PILT_test_template.stimulus_right .== filter(x -> x.session == 1, PILT_test).stimulus_right) "Final sequence for wk0 does not match template"

	@assert all(PILT_test.EV_left .!= PILT_test.EV_right) "EVs are not different"

	@assert all(PILT_test.feedback_left .!= PILT_test.feedback_right)

	@assert all(PILT_test.stimulus_right .!= PILT_test.stimulus_left)

	trials_per_session = combine(
		groupby(PILT_test, :session),
		:trial => length => :n
	).n

	@info "Test trials per session: $(only(unique(trials_per_session)))"

	@assert length(values(countmap(PILT_test_template.primary_outcome_pair))) == 15 "Didn't cover all primary outcome combinations"
end

# ╔═╡ 01440120-ec95-4c62-92cf-cc50bedf99ec
let
	# Save to file
	CSV.write("results/trial1_eeg_PILT_test.csv", PILT_test)
end

# ╔═╡ Cell order:
# ╠═d1ffbd6e-41f6-11f0-2884-51de482dd6fe
# ╠═0d99d4cc-18f9-4f5e-ab23-6bff9bdce753
# ╠═52303bea-d08a-4b0d-bb30-97633d428811
# ╠═4bbb7c6d-71d3-4447-8359-348c86d05bdc
# ╠═83266903-2dd2-43e1-93fb-b5c5271bb02d
# ╠═57ca9c3d-6d91-43cf-8fc4-a8ad8dfce0dd
# ╠═71e38508-9a0d-4ab3-9e39-75f529d5672a
# ╠═4c093483-9436-44b5-bcdd-299538d4d05b
# ╠═2a83846a-cc71-44b0-b75c-1b2b487d256c
# ╠═b7034df7-868c-49b9-b883-e050738e0c33
# ╠═9d3736ba-004f-4d1a-9595-e138ba6cb2b9
# ╠═48620cba-b84f-4c14-aafa-1f39ffa99787
# ╠═149549b9-5ff3-451c-9f91-41a0c240e4e8
# ╠═ad996e31-abdb-46c9-8930-6cfa7f833cb0
# ╠═6e7d02b8-9889-49fc-b1b0-e82e7826f9ff
# ╠═dae14142-7ad7-4846-9923-4643e7a2fa36
# ╠═d0665196-1862-4bbd-b766-47052f72f07e
# ╠═01440120-ec95-4c62-92cf-cc50bedf99ec
# ╠═cc409bce-df5e-4b77-906e-20b8342d5b5b
# ╠═531a2596-52c2-43d6-94d3-de273705a79c
# ╠═af283fc5-50b0-47f6-97c6-33d5b189e76d
# ╠═cfb06155-e1b9-4e19-ae61-72d762f24529
# ╠═1ac016d1-7de7-405f-987e-446efca5ff90
# ╠═b04341bc-82e0-4660-8809-42d852b5a95d
# ╠═6d835516-0be7-42d0-bc46-d4e14dfd077d
# ╠═15527827-4835-42a8-88c5-cba1d6e9719a
