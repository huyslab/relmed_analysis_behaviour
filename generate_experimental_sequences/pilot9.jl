# Load pacakges
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

categories = let

	categories = DataFrame(CSV.File("generate_experimental_sequences/trial1_stimuli/stimuli.csv")).concept |> unique

	@info "Found $(length(categories)) categories"

	categories
end


# PILT parameters
begin
	# PILT Parameters
	PILT_trials_per_block = 10
			
	# Post-PILT test parameters
	PILT_test_n_blocks = 5
end

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

PILT_blocks = let rng = Xoshiro(0)

	coins = [-1., -0.5, -0.01, 0.01, 0.5, 1.]

	# Create all magnitdue combinations
	outcome_combinations = collect(product(coins, coins))

	# Select only upper triangular
	outcome_combinations = 
		outcome_combinations[triu(trues(size(outcome_combinations)))]

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


	PILT_blocks

end


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
		), 1, :session => s) for s in 1:2
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

# Create PILT sequence
PILT_sequence = let rng = Xoshiro(0)

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
        common_feedback = common_feedback,
        n_confusing = repeat(PILT_blocks.n_confusing, inner = PILT_trials_per_block)
    )

    # Merge with blocks
    PILT_sequence = leftjoin(
        PILT_sequence,
        select(PILT_blocks, [:block, :n_confusing, :primary_outcome_A, :primary_outcome_B, :secondary_outcome_A, :secondary_outcome_B]),
        on = [:block, :n_confusing]
    )

    # Create feedback_A and feedback_B
    PILT_sequence.feedback_A = ifelse.(
        PILT_sequence.common_feedback,
        PILT_sequence.primary_outcome_A,
        PILT_sequence.secondary_outcome_A
    )
    PILT_sequence.feedback_B = ifelse.(
        PILT_sequence.common_feedback,
        PILT_sequence.primary_outcome_B,
        PILT_sequence.secondary_outcome_B
    )

    # Merge with stimuli
    PILT_sequence = leftjoin(
        PILT_sequence,
        select(PILT_stimuli, [:block, :session, :optimal_A, :stimulus_A, :stimulus_B]),
        on = [:block]
    )

    sort!(PILT_sequence, [:session, :block, :trial])

    # Randomize appearance on left / right
    DataFrames.transform!(
        groupby(PILT_sequence, [:session, :block]),
        :trial => (x -> shuffled_fill([true, false], length(x), rng = rng)) => :A_on_right
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


end

