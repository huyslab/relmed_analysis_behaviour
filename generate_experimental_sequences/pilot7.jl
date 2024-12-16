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
	categories = (s -> replace(s, ".jpg" => "")[1:(end-1)]).(readdir("generate_experimental_sequences/pilot7_stims"))

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

	# # Full deterministic task
	# WM_n_confusing = fill(0, WM_n_total_blocks) # Per block

	# # For uniform shuffling of triplets in block
	# WM_triplet_mini_block_size = 2

end

# ╔═╡ 5b37feb9-30c2-4e72-bba9-08f3b4e1c499
function assign_triplet_stimuli_and_optimality(;
	n_phases::Int64,
	n_groups::Vector{Int64}, # Number of groups in each block. Assume same for all phases
	categories::Vector{String} = [('A':'Z')[div(i - 1, 26) + 1] * ('a':'z')[rem(i - 1, 26)+1] 
		for i in 1:(sum(n_groups) * 3 * n_phases + n_phases)],
	random_seed::Int64 = 1
)

	# Copy categories so that it is not changed
	this_cats = copy(categories)

	total_n_groups = sum(n_groups) # Number of pairs needed
	
	@assert rem(length(n_groups), 2) == 0 "Code only works for even number of blocks per sesion"

	# Compute how many repeating categories we will have
	n_repeating = sum(min.(n_groups[2:end], n_groups[1:end-1]))

	rng = Xoshiro(random_seed)

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

	stimulus_A = (x -> x * "1.jpg").(stimulus_A)
	stimulus_B = (x -> x * "1.jpg").(stimulus_B)
	stimulus_C = (x -> x * "2.jpg").(stimulus_C)

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
		triplet = repeat(
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
task, common_per_pos, EV_per_pos =
let random_seed = 1
	
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

# ╔═╡ 8e9ffd82-89ec-4a63-83a8-54dfde7192a0
# Checks
let
	delays = combine(
		groupby(task, [:block, :set_size]),
		:stimulus_group => (x -> length(unique(count_delays(x)))) => :n_unique_delays,
		[:stimulus_group, :set_size] => ((t, s) -> Ref(counts(count_delays(t), 1:(2*s[1]-1)))) => :distribution
	)

	@assert all(delays.n_unique_delays .== 2 .* delays.set_size .- 1) "Number of unique delay values between triplets should be exactly 2*ns-1"

	@assert all(all.((x -> 3 .≤ x .≤ 6).(delays.distribution))) "Distribution of delays far from uniform"

	@info "Each delay value appears $(minimum(minimum.(delays.distribution))) - $(maximum(maximum.(delays.distribution))) times"
	

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


	# save("results/pilot6_pilt_trial_plan.png", f, pt_per_unit = 1)

	f

end

# ╔═╡ ed848098-c8cf-4c8b-adda-850c58e828e6
# Assign stimulus images
stimuli, stim_score = let

	shuffle!(Xoshiro(0), categories)

	best_stimuli = DataFrame()
	best_score = 0

	for random_seed in 1:10

		# Assign stimulus pairs
		stimuli = assign_triplet_stimuli_and_optimality(;
			n_phases = 2,
			n_groups = disallowmissing(block_order.set_size),
			# categories = categories,
			random_seed = random_seed
		)
	
		rename!(stimuli, :phase => :session) # For compatibility with multi-phase sessions
	
		stimuli.repeating_optimal = ifelse.(
			stimuli.repeating_C,
			stimuli.optimal_stimulus .== "C",
			missing
		)
	
		# Compute repeating_prev_optimal
		repeating_prev_optimal::Vector{Union{Missing, Bool}} = fill(missing, nrow(stimuli))
	
		for (i, r) in enumerate(eachrow(stimuli))
			if r.repeating_C
				repeating_prev_optimal[i] = only(stimuli.optimal_stimulus[stimuli.stimulus_A .== replace(r.stimulus_C, "2" => "1")]) == "A"
			end
		end
	
		stimuli.repeating_prev_optimal = repeating_prev_optimal
	
		# Add valence and prev_valence
		stimuli = leftjoin(
			stimuli,
			insertcols(
				block_order[!, [:block, :valence]],
				:prev_valence => vcat([missing], block_order.valence[1:(end-1)])
			),
			on = :block,
			order = :left
		)
	
		# Score on uniformity of distribution across variables important for generalization 
		generaliation_ns = combine(
			groupby(
				dropmissing(stimuli, [:repeating_optimal, :repeating_prev_optimal]), [:repeating_optimal, :repeating_prev_optimal, :valence, :prev_valence]),
			:repeating_optimal => length => :n
		)
		
		score = minimum(generaliation_ns.n) - 1000 * (nrow(generaliation_ns) != 16)

		if score > best_score
			best_stimuli = stimuli
			best_score = score
		end

	end

	
	best_stimuli, best_score
end

# ╔═╡ 34f390db-aeff-41e3-934e-05406b76a3dd
let
	generaliation_ns = combine(
		groupby(
			dropmissing(stimuli, [:repeating_optimal, :repeating_prev_optimal]), [:repeating_optimal, :repeating_prev_optimal, :valence, :prev_valence]),
		:repeating_optimal => length => :n
	)
end

# ╔═╡ Cell order:
# ╠═2d7211b4-b31e-11ef-3c0b-e979f01c47ae
# ╠═114f2671-1888-4b11-aab1-9ad718ababe6
# ╠═de74293f-a452-4292-b5e5-b4419fb70feb
# ╠═c05d90b6-61a7-4f9e-a03e-3e11791da6d0
# ╠═699245d7-1493-4f94-bcfc-83184ca521eb
# ╠═7e078cb5-c615-4dc8-9060-3b69c86648b6
# ╠═ed848098-c8cf-4c8b-adda-850c58e828e6
# ╠═34f390db-aeff-41e3-934e-05406b76a3dd
# ╠═8e9ffd82-89ec-4a63-83a8-54dfde7192a0
# ╠═87035e3e-e7ce-4320-a440-c150c4547c02
# ╠═5b37feb9-30c2-4e72-bba9-08f3b4e1c499
# ╠═b4a3c42d-ebc6-4d7f-a451-271fc3a5132d
# ╠═69afd881-0c45-48c9-8db9-699f9ae23bec
# ╠═fdbe5c4e-29cd-4d24-bbe5-40d24d5f98f4
# ╠═3952d2ef-a5c6-40a1-9373-6c4c9ff5ec2b
