### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ 2d7211b4-b31e-11ef-3c0b-e979f01c47ae
begin
	cd("/home/jovyan")
	import Pkg
	
	# activate the shared project environment
    Pkg.activate("$(pwd())/environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	# using CairoMakie, Random, DataFrames, Distributions, StatsBase,
	# 	ForwardDiff, LinearAlgebra, JLD2, FileIO, CSV, Dates, JSON, RCall, Turing, Printf, Combinatorics, JuMP, HiGHS, AlgebraOfGraphics, CategoricalArrays
	# using LogExpFunctions: logistic, logit
	# using IterTools: product

	using CairoMakie, CSV, DataFrames, Combinatorics, StatsBase, Random, CategoricalArrays, AlgebraOfGraphics
	using IterTools: product

	# Turing.setprogress!(false)

	script_dir = dirname(@__FILE__)
	stim_dir = "$(script_dir)/trial1_stimuli"
	result_dir = "$(script_dir)/results"

	# include("$(pwd())/PILT_models.jl")
	# include("$(pwd())/sample_utils.jl")
	# include("$(pwd())/model_utils.jl")
	# include("$(pwd())/plotting_utils.jl")
	# include("$(pwd())/fetch_preprocess_data.jl")
	include("$(script_dir)/sequence_utils.jl")
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

	for s in sessions
		open("$(result_dir)/trial1_$(s)_sequences.js", "w") do file
		end
	end

end

# ╔═╡ de74293f-a452-4292-b5e5-b4419fb70feb
categories = let

	categories = DataFrame(CSV.File("$(stim_dir)/stimuli.csv")).concept |> unique

	@info "Found $(length(categories)) categories"

	categories
end

# ╔═╡ 56abf8a4-acad-4408-a86c-aad2d5aa3cd7
md"""# PILT"""

# ╔═╡ 1f791569-cfad-4bc8-9aef-ea324ea7be23
# PILT parameters
begin
	# PILT Parameters
	PILT_trials_per_block = 10
			
	# Post-PILT test parameters
	PILT_test_n_blocks = 4

end

# ╔═╡ c34ecc54-289c-4f6f-91eb-e2d643a9c647
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


# ╔═╡ e34ea9b5-50e6-463f-afba-f3e845186019
PILT_blocks = let rng = Xoshiro(0)

	coins = [-1., -0.5, -0.01, 0.01, 0.5, 1.]

	# Create all magnitdue combinations
	outcome_combinations = collect(product(coins, coins))

	outcome_combinations = filter(x -> x[1] < x[2], vec(outcome_combinations))

	# Convert to vector of vectors
	outcome_combinations = collect.(outcome_combinations)

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

        EV_diff_uniform = EV_diff_counts_sd < 2.5

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

		B_uniform = maximum(EV_B_bin_counts) - minimum(EV_B_bin_counts) == 1

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

	PILT_blocks

end

# ╔═╡ 30d04b83-46b4-4e13-84b3-17a404c2d8be
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
				rng = Xoshiro(random_seed)
		), 1, :session => s) for s in sessions[2:end]
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


# ╔═╡ 54d7e5e4-ea89-4efc-8ae6-078d60c7bcb1
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


# ╔═╡ 15cf4ca3-3cad-41f3-8ed0-23af0a5b6d61
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


# ╔═╡ af5c2af1-2a59-42ef-99c7-96958df12d93
# Visualize PILT sequence
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
	


	save("$(result_dir)/trial1_pilt_trial_plan.png", f, pt_per_unit = 1)

	f

end


# ╔═╡ 5db56189-97ae-4ab0-969c-3c6a9bf787a7
md"""## Screening session"""

# ╔═╡ 2005851c-6cb4-4849-a587-3562b2a14dd7
# Parameters
begin
	scr_PILT_n_trials = 50
	scr_PILT_prop_common::Rational = 9//10
	scr_PILT_first_confusing_trial = 4
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

	@assert all(PILT_test.EV_left .!= PILT_test.EV_right) "EVs are not different"

	@assert all(PILT_test.feedback_left .!= PILT_test.feedback_right)

	@assert all(PILT_test.stimulus_right .!= PILT_test.stimulus_left)

	trials_per_session = combine(
		groupby(PILT_test, :session),
		:trial => length => :n
	).n

	@info "Test trials per session: $(only(unique(trials_per_session)))"

	@assert length(values(countmap(PILT_test_template.primary_outcome_pair))) == 15 "Didn't cover all primary outcome combinations"

	@assert allequal(values(countmap(PILT_test_template.primary_outcome_pair))) "Distribution over primary outcome combinations is not uniform"
end

# ╔═╡ 9f7b362c-5a60-4af1-a7e1-64b9665eee1e
md"""# RLX"""

# ╔═╡ 54f6f217-3ae5-49c7-9456-a5abcbbdc62f
# RLX Parameters
begin
	RLX_prop_fifty = 0.2
	RLX_shaping_n = 20
	RLX_test_n_blocks = 1
end

# ╔═╡ 9f300301-b018-4bea-8fc4-4bc889b11afd
triplet_order = let
	triplet_order = DataFrame(CSV.File(
		"$(script_dir)/data/wm_stimulus_sequence.csv"))

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

# ╔═╡ d11a7fc9-6e2c-48a1-bce8-6b5f39df3eda
md"""## Reversal task"""

# ╔═╡ 3c167a3c-5577-40ce-af0d-317417c8e934
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

# ╔═╡ d9085924-a630-4897-8a87-bd5943098c49
function save_json_to_js(object, var_name::String, filename::String)
    # Convert to JSON String
    json_string = JSON.json(object)

    # Add JS variable assignment
    json_string = "const $(var_name) = '$json_string';"

    # Append the JSON string to the file
	@info "writing $var_name to $filename"
    open(filename, "a") do file
        write(file, json_string)
    end
end

# ╔═╡ 7f254beb-3d4d-437d-a491-e45512b578ce
function save_to_JSON(
	df::DataFrame, 
	filename::Function,
	var_name
)
	# Initialize an empty dictionary to store the grouped data
	sess_dict = Dict()
	
	# Iterate through unique blocks and their respective rows
	for s in unique(df.session)

		session_df = filter(x -> x.session == s, df)
		session_groups = []
		for b in unique(session_df.block)
		    # Filter the rows corresponding to the current block
		    block_group = filter(x -> x.block == b, session_df)
		    
		    # Convert each row in the block group to a dictionary and collect them into a list
		    push!(session_groups, [Dict(pairs(row)) for row in eachrow(block_group)])
		end
		
		# Store session data using session name as key
		sess_dict[string(s)] = session_groups
	end
	
	for (k, v) in sess_dict
		save_json_to_js(v, var_name, filename(k))
	end

end

# ╔═╡ 47bfbee6-eaf4-4290-90f4-7b40a11bf27b
let
	# Save to file
	save_to_JSON(PILT_test, s -> "$(result_dir)/trial1_$(s)_sequences.js", "PILT_test_json")
	CSV.write("$(result_dir)/trial1_PILT_test.csv", PILT_test)
end

# ╔═╡ 4899facf-6759-49c3-9905-8a418c9ebe7c
function find_lcm_denominators(props::Vector{Float64})
	# Convert to rational numbers
	rational_props = rationalize.(props)
	
	# Extract denominators
	denominators = [denominator(r) for r in rational_props]
	
	# Compute the LCM of all denominators
	smallest_int = foldl(lcm, denominators)
end

# ╔═╡ 64a0768b-dee5-4a5a-b9b2-cfc3a1c6a6e0
# Reversal task structure
rev_feedback_optimal, rev_timeline = let random_seed = 1

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
	
	for (k, v) in timeline
		save_json_to_js(v, "reversal_json", "$(result_dir)/trial1_$(string(k))_sequences.js")
	end

	feedback_optimal, timeline
end

# ╔═╡ f4b4556f-28bd-4745-8709-1955abd891df
let

	f = Figure(size = (700, 600))

	mp1 = data(
		DataFrame(
			block = repeat(1:rev_n_blocks, 2),
			prop = vcat(rev_prop_confusing, 1. .- rev_prop_confusing),
			feedback_type = repeat(["Confusing", "Common"], inner = rev_n_blocks)
		)
	) * mapping(
		:block => "Block", 
		:prop => "Proportion of trials", 
		color = :feedback_type => "", 
		stack = :feedback_type) * visual(BarPlot)

	plt1 = draw!(f[1,1], mp1, axis = (; yticks = [0., 0.5, 0.7, 0.8, 0.9, 1.]))

	legend!(f[1,1], plt1, 
		valign = 1.18,
		tellheight = false, 
		framevisible = false,
		orientation = :horizontal,
		labelsize = 14
	)

	rowgap!(f[1,1].layout, 0)

	rev_confusing = DataFrame(
		block = repeat(1:rev_n_blocks, inner = rev_n_trials),
		trial = repeat(1:rev_n_trials, outer = rev_n_blocks),
		feedback_common = vcat(rev_feedback_optimal...) .== 1.
	)

	mp2 = data(rev_confusing) * 
		mapping(:trial => "Trial", :block => "Block", :feedback_common) *
		visual(Heatmap)

	draw!(f[1,2], mp2, 
		axis = (; 
			yreversed = true, 
			yticks = [1, 10, 20, 30],
			subtitle = "Confusing Feedback"
		)
	)

	insertcols!(
		rev_confusing,
		:rel_trial => rev_confusing.trial .- div.(rev_confusing.trial, 10) .* 10 .+ 1
	)
	
	mp3 = data(
		combine(
			groupby(rev_confusing, :rel_trial), 
			:feedback_common => (x -> mean(.!x)) => :feedback_confusing
		)
	) * mapping(
		:rel_trial => "Trial", 
		:feedback_confusing => "Prop. confusing feedback"
	) * visual(ScatterLines)

	draw!(f[2,1], mp3)

	mp4 = mapping(1:rev_n_blocks => "Block", rev_criterion => "# optimal choices)") * visual(ScatterLines)

	draw!(f[2,2], mp4, axis = (; 
		yticks = 3:8, 
		xticks = [1, 10, 20, 30], 
		subtitle = "Reversal criterion")
	)

	save("$(result_dir)/trial1_reversal_sequence.png", f, pt_per_unit = 1)

	f

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
"""
    prop_fill_shuffle(values, props, n; rng=Xoshiro(1))

Generate a shuffled vector containing `n` elements, where each unique value in `values`  
is allocated according to the proportions specified in `props`. 

# Arguments
- `values::AbstractVector`: A vector of unique values to be sampled.  
- `props::Vector{Float64}`: A vector of proportions (summing to 1) corresponding to `values`.  
- `n::Int64`: The total number of elements in the output vector.  
- `rng::AbstractRNG`: (Optional) Random number generator for shuffling (default: `Xoshiro(1)`).  

# Returns
- A shuffled vector of length `n` containing the values allocated based on `props`.  
"""
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

# ╔═╡ 263807b6-8212-41a5-936a-40c962c02150
scr_PILT_seq = let rng = Xoshiro(0)

	denom_prop = denominator(scr_PILT_prop_common)

	@assert rem(scr_PILT_n_trials, denom_prop) == 0 "Proportion cannot be represented in scr_PILT_n_trials"

	# Shuffled order of feedback common
	feedback_common = vcat(
		[prop_fill_shuffle(
			[true, false], 
			float.([scr_PILT_prop_common, 1-scr_PILT_prop_common]), denom_prop; 
			rng = rng) for _ in 1:(scr_PILT_n_trials ÷ denom_prop - 1)]...
	)

	# Add first sequence with first confusing trial
	feedback_common = vcat(
		vcat(
			fill(true, scr_PILT_first_confusing_trial - 1),
			[false],
			fill(true, denom_prop - scr_PILT_first_confusing_trial)
		),
		feedback_common
	)

	@assert length(feedback_common) == scr_PILT_n_trials "Problem with allocation of sequence"

	feedback_common

end

# ╔═╡ 8e7ac8e9-3de3-469a-b8b3-2ec88bb0b356
scr_PILT = let rng = Xoshiro(3)

	# Initiate DataFrame with sequence and constant variables
	scr_PILT = DataFrame(
		session = "screening",
		block = 1,
		valence = 0,
		trial = 1:length(scr_PILT_seq),
		feedback_common = scr_PILT_seq,
		present_pavlovian = false,
		n_stimuli = 2,
		early_stop = true,
		n_groups = 1,
		stimulus_group = 1,
		stimulus_group_id = 1,
		stimulus_middle = "",
		feedback_middle = "",
		optimal_side = "",
	)

	# Create feedback_optimal variable
	denom_prop = denominator(scr_PILT_prop_common)

	scr_PILT.feedback_optimal = ifelse.(
		scr_PILT.feedback_common,
		1.,
		0.01
	)

	scr_PILT.feedback_suboptimal = ifelse.(
		scr_PILT.feedback_common,
		0.01,
		0.5
	)

	# Assign right / left
	scr_PILT.optimal_right = vcat(
		[prop_fill_shuffle(
			[true, false], 
			[0.5, 0.5], 
			denom_prop; 
			rng = rng) for _ in 1:(scr_PILT_n_trials ÷ denom_prop)]...
	)

	# Assign feedback_right and feedback_left
	scr_PILT.feedback_right = ifelse.(
		scr_PILT.optimal_right,
		scr_PILT.feedback_optimal,
		scr_PILT.feedback_suboptimal
	)

	scr_PILT.feedback_left = ifelse.(
		.!scr_PILT.optimal_right,
		scr_PILT.feedback_optimal,
		scr_PILT.feedback_suboptimal
	)

	# Assign stimuli
	stimuli = [popat!(categories, rand(rng, 1:length(categories))) * "_1.jpg" for _ in 1:2]

	scr_PILT.stimulus_right = ifelse.(
		scr_PILT.optimal_right,
		stimuli[1],
		stimuli[2]
	)

	scr_PILT.stimulus_left = ifelse.(
		.!scr_PILT.optimal_right,
		stimuli[1],
		stimuli[2]
	)

	scr_PILT

end

# ╔═╡ cc7b17ab-0d77-4be4-bb53-53325ca85145
let
	# Bind screening to rest of sessions
	all_PILT = vcat(scr_PILT, PILT, cols = :union)
	
	# Save to file
	save_to_JSON(all_PILT, s -> "$(result_dir)/trial1_$(s)_sequences.js", "PILT_json")
	CSV.write("$(result_dir)/trial1_PILT.csv", all_PILT)

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
	# Save to file
	save_to_JSON(RLWM, s -> "$(result_dir)/trial1_$(s)_sequences.js", "WM_json")
	CSV.write("$(result_dir)/trial1_WM.csv", RLWM)
end

# ╔═╡ 1491f0f9-0c40-41ca-b7a9-055259f66eb3
RLWM_test = let

	RLWM_test = DataFrame()

	for gdf in groupby(RLWM, :session)

		# Reset random seed so that randomization is the same across sessions
		rng = Xoshiro(0)

		# Get unique stimuli
		RLWM_stimuli = combine(groupby(gdf, :stimulus_left), :trial => length)

		filter!(x -> x.trial_length > 2, RLWM_stimuli)

		RLWM_stimuli = RLWM_stimuli.stimulus_left
	
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
			:EV_left => 1.,
			:EV_right => 1.,
			:same_valence => true,
			:same_block => true,
			:original_block_left => 1,
			:original_block_right => 1,
			:optimal_right => :true
		)

		RLWM_test = vcat(RLWM_test, RLWM_test_session)

	end

	RLWM_test
	
end

# ╔═╡ b28f57a2-8aab-45e9-9d16-4c3b9fcf3828
let
	# Save to file
	save_to_JSON(RLWM_test, s -> "$(result_dir)/trial1_$(s)_sequences.js", "WM_test_json")
	CSV.write("$(result_dir)/trial1_WM_test.csv", RLWM_test)
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

	@info combine(groupby(RLWM_test, :session), :trial => length => :n)

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

# ╔═╡ Cell order:
# ╠═2d7211b4-b31e-11ef-3c0b-e979f01c47ae
# ╠═114f2671-1888-4b11-aab1-9ad718ababe6
# ╠═3cb53c43-6b38-4c95-b26f-36697095f463
# ╠═de74293f-a452-4292-b5e5-b4419fb70feb
# ╠═56abf8a4-acad-4408-a86c-aad2d5aa3cd7
# ╠═1f791569-cfad-4bc8-9aef-ea324ea7be23
# ╠═e34ea9b5-50e6-463f-afba-f3e845186019
# ╠═30d04b83-46b4-4e13-84b3-17a404c2d8be
# ╠═54d7e5e4-ea89-4efc-8ae6-078d60c7bcb1
# ╠═15cf4ca3-3cad-41f3-8ed0-23af0a5b6d61
# ╠═af5c2af1-2a59-42ef-99c7-96958df12d93
# ╠═7f254beb-3d4d-437d-a491-e45512b578ce
# ╠═c34ecc54-289c-4f6f-91eb-e2d643a9c647
# ╟─5db56189-97ae-4ab0-969c-3c6a9bf787a7
# ╠═2005851c-6cb4-4849-a587-3562b2a14dd7
# ╠═263807b6-8212-41a5-936a-40c962c02150
# ╠═8e7ac8e9-3de3-469a-b8b3-2ec88bb0b356
# ╠═cc7b17ab-0d77-4be4-bb53-53325ca85145
# ╠═b13e8f5f-7522-497e-92d1-51d782fca33b
# ╠═e51b4cec-6fd2-49d3-a9ec-ebbe960a7f49
# ╠═8700a65a-3117-4d62-98f9-26f2839ba6a2
# ╠═dd7112c9-35ac-4d02-a9c4-1e19efad0f31
# ╠═47bfbee6-eaf4-4290-90f4-7b40a11bf27b
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
# ╠═d11a7fc9-6e2c-48a1-bce8-6b5f39df3eda
# ╠═3c167a3c-5577-40ce-af0d-317417c8e934
# ╠═64a0768b-dee5-4a5a-b9b2-cfc3a1c6a6e0
# ╠═d9085924-a630-4897-8a87-bd5943098c49
# ╠═f4b4556f-28bd-4745-8709-1955abd891df
# ╠═4899facf-6759-49c3-9905-8a418c9ebe7c
# ╠═68873d3e-054d-4ab4-9d89-73586bb0370e
# ╠═f89e88c9-ebfc-404f-964d-acff5c7f8985
# ╠═2c75cffc-1adc-44b6-bed3-12ed0c7025b7
