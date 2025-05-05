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
			maximum(trans_counts) <= 1

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
	B_score = Inf
	while !criteria2
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


		first_mathces = all((r -> sign(r.primary_outcome) == 
			sign(r.secondary_outcome)).(eachrow(filter(x -> x.block in [5,6], PILT_blocks_long))))

		EV_B_bin = discretize_to_quantiles(
			PILT_blocks_long.EV,
			PILT_blocks_long.EV,
			5
		)[PILT_blocks_long.stimulus .== "B"]

		EV_B_bin_counts = values(countmap(EV_B_bin))

		B_uniform = maximum(EV_B_bin_counts) - minimum(EV_B_bin_counts) == 1

		criteria2 = first_mathces && B_uniform

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

let
	
	autocor(PILT_blocks.EV_B), autocor(PILT_blocks.EV_A)
	
end

let
	@info quantile(vcat(PILT_blocks.EV_A, PILT_blocks.EV_B), range(0., 1., 6))

	@info minimum(abs.(vcat(PILT_blocks.EV_A, PILT_blocks.EV_B)))
	
	hist(vcat(PILT_blocks.EV_A, PILT_blocks.EV_B), bins = 40)

end

function assign_stimuli_and_optimality(;
	n_phases::Int64,
	n_pairs::Vector{Int64}, # Number of pairs in each block. Assume same for all phases
	categories::AbstractVector = [('A':'Z')[div(i - 1, 26) + 1] * ('a':'z')[rem(i - 1, 26)+1] 
		for i in 1:(sum(n_pairs) * 2 * n_phases + n_phases)],
	random_seed::Int64 = 1,
	ext::String = "jpg"
	)

	total_n_pairs = sum(n_pairs) # Number of pairs needed
	
	# @assert rem(length(n_pairs), 2) == 0 "Code only works for even number of blocks per sesion"

	@assert length(categories) >= sum(total_n_pairs) * n_phases + n_phases "Not enought categories supplied"

	# Compute how many repeating categories we will have
	n_repeating = sum(min.(n_pairs[2:end], n_pairs[1:end-1]))

	# Assign whether repeating is optimal and shuffle
	repeating_optimal = shuffle(
		Xoshiro(random_seed),
		vcat(
			fill(true, div(n_repeating, 2)),
			fill(false, div(n_repeating, 2) + rem(n_repeating, 2))
		)
	)

	# Assign whether categories that cannot repeat are optimal
	rest_optimal = shuffle(
		vcat(
			fill(true, div(total_n_pairs - n_repeating, 2) + 
				rem(total_n_pairs - n_repeating, 2)),
			fill(false, div(total_n_pairs - n_repeating, 2))
		)
	)

	# Initialize vectors for stimuli. A is always novel, B may be repeating
	stimulus_A = []
	stimulus_B = []
	optimal_B = []
	
	for j in 1:n_phases
		for (i, p) in enumerate(n_pairs)
	
			# Choose repeating categories for this block
			n_repeating = ((i > 1) && minimum([p, n_pairs[i - 1]])) * 1
			append!(
				stimulus_B,
				stimulus_A[(end - n_repeating + 1):end]
			)
	
			# Fill up stimulus_repeating with novel categories if not enough to repeat
			for _ in 1:(p - n_repeating)
				push!(
					stimulus_B,
					popfirst!(categories)
				)
			end
			
			# Choose novel categories for this block
			for _ in 1:p
				push!(
					stimulus_A,
					popfirst!(categories)
				)
			end

			# Populate who is optimal vector
			for _ in 1:(n_repeating)
				push!(
					optimal_B,
					popfirst!(repeating_optimal)
				)
			end

			for _ in 1:(p - n_repeating)
				push!(
					optimal_B,
					popfirst!(rest_optimal)
				)
			end
		end
	end

	stimulus_A = (x -> x * "_1.$ext").(stimulus_A)
	stimulus_B = (x -> x * "_2.$ext").(stimulus_B)

	return DataFrame(
		phase = repeat(1:n_phases, inner = total_n_pairs),
		block = repeat(
			vcat([fill(i, p) for (i, p) in enumerate(n_pairs)]...), n_phases),
		pair = repeat(
			vcat([1:p for p in n_pairs]...), n_phases),
		stimulus_A = stimulus_A,
		stimulus_B = stimulus_B,
		optimal_A = .!(optimal_B)
	)

end

# Assign stimulus images
PILT_stimuli = let random_seed = 0

	# Shuffle categories
	shuffle!(Xoshiro(random_seed), categories)

	@info length(categories)

	# Assign stimulus pairs
	stimuli = assign_stimuli_and_optimality(;
		n_phases = 1,
		n_pairs = fill(1, nrow(PILT_blocks)),
		categories = categories,
		random_seed = random_seed
	)

	@info "Proportion of blocks on which the novel category is optimal : $(mean(stimuli.optimal_A))"

	rename!(stimuli, :phase => :session) # For compatibility with multi-phase sessions

	stimuli
end

let

	PILT_blocks_stims = leftjoin(
		PILT_blocks,
		PILT_stimuli,
		on = :block
	)

	PILT_blocks_stims.EV_B_bin = discretize_to_quantiles(
		PILT_blocks_stims.EV_B,
		vcat(PILT_blocks_stims.EV_A, PILT_blocks_stims.EV_B),
		3
	)

	PILT_blocks_stims.previous_EV_B_bin = vcat(
		[missing], (PILT_blocks_stims.EV_B_bin)[1:(end-1)]
	)

	PILT_blocks_stims.repeating_optimal = vcat(
		[missing], (.! PILT_blocks_stims.optimal_A)[2:end]
	)

	PILT_blocks_stims.repeating_previous_optimal = vcat(
		[missing], (.! PILT_blocks_stims.optimal_A)[1:(end-1)]
	)

	combine(
		groupby(
			PILT_blocks_stims, 
			[:EV_B_bin, :previous_EV_B_bin]
		),
		:EV_B_bin => length => :n
	) 

end

