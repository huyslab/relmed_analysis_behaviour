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

	categories = filter(x -> x in keys(keeps), unique(categories))

	@info "Found $(length(categories)) categories"

	categories
end

# ╔═╡ ffe06202-d829-4145-ae26-4a95449d64e6
md"""# RLWM"""

# ╔═╡ 05f25eb8-3a48-4d16-9837-84d1fdf5c806
triplet_order = let
	triplet_order = DataFrame(CSV.File(
		"generate_experimental_sequences/pilot7_wm_stimulus_sequence_longer.csv"))

	select!(
		triplet_order, 
		:stimset => :stimulus_group,
		:delay
	)
end

# ╔═╡ c05d90b6-61a7-4f9e-a03e-3e11791da6d0
# RLWM Parameters
begin
	RLWM_prop_fifty = 0.2
	RLWM_shaping_n = 20
end

# ╔═╡ 25afe21f-71d2-45ca-b2db-5edf830a2d62
"ABC" .== 'A'

# ╔═╡ 3fa8c293-ac47-4acd-bdb7-9313286ee464
function assign_triplet_stimuli_RLWM(
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

# ╔═╡ f5916a9f-ddcc-4c03-9328-7dd76c4c74b2
# Create deterministic block
RLWM_det_block = let rng = Xoshiro(0)
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
			[1 - RLWM_prop_fifty, RLWM_prop_fifty],
			length(x),
			rng = rng
			)
		) => :feedback_optimal
	) 

	@info "Proportion fifty pence: $(mean(det_block.feedback_optimal .== 0.5))"

	# Assign stimuli categories
	stimuli = assign_triplet_stimuli_RLWM((categories),
		maximum(det_block.stimulus_group);
		rng = rng
	)

	# Merge with trial structure
	det_block = innerjoin(
		det_block,
		stimuli,
		on = :stimulus_group,
		order = :left
	)

	# Assign stimuli locations
	orderings = [join(p) for p in permutations(["A", "B", "C"])]
	DataFrames.transform!(
		groupby(det_block, :stimulus_group),
		:trial => (
			x -> shuffled_fill(orderings, length(x))
		) => :stimulus_ordering
	)

	@info "Proportion optimal in each location: $([round(mean((x -> x[i] == 'A').(det_block.stimulus_ordering)), digits = 3) for i in 1:3])"

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

# ╔═╡ 6eec5cf5-5416-4158-a53f-5ea2b9c9e4a4
# Create probabilistic block
RLWM_prob_block = let rng = Xoshiro(1)
	n_trials = nrow(triplet_order)

	# Basic variables
	prob_block = DataFrame(
		block = fill(2, n_trials),
		trial = 1:n_trials,
		stimulus_group = triplet_order.stimulus_group,
		delay = triplet_order.delay
	)

	# Draw high feedback
	DataFrames.transform!(
		groupby(prob_block, :stimulus_group),
		:trial => (x -> prop_fill_shuffle(
			[1., 0.5],
			[1 - RLWM_prop_fifty, RLWM_prop_fifty],
			length(x),
			rng = rng
			)
		) => :feedback_high
	) 

	@info "Proportion fifty pence: $(mean(prob_block.feedback_high .== 0.5))"

	# Draw feedback_common
	prob_block[!, :feedback_common] .= true
	prob_block[[5, 17], :feedback_common] = [false, false]

	prob_block[(RLWM_shaping_n + 1):end, :feedback_common] = DataFrames.transform(
		groupby(prob_block[(RLWM_shaping_n + 1):end, :], :stimulus_group),
		:trial => (x -> prop_fill_shuffle(
			[true, false],
			[0.8, 0.2],
			length(x),
			rng = rng
		)) => :feedback_common
	).feedback_common

	# Create feedback optimal and suboptimal
	prob_block.feedback_optimal = ifelse.(
		prob_block.feedback_common,
		prob_block.feedback_high,
		fill(0.01, nrow(prob_block))
	)

	prob_block.feedback_suboptimal = ifelse.(
		.!prob_block.feedback_common,
		prob_block.feedback_high,
		fill(0.01, nrow(prob_block))
	)


	# # Assign stimuli categories
	stimuli = assign_triplet_stimuli_RLWM((categories),
		maximum(prob_block.stimulus_group);
		rng = rng
	)

	# Merge with trial structure
	prob_block = innerjoin(
		prob_block,
		stimuli,
		on = :stimulus_group,
		order = :left
	)

	# Assign stimuli locations
	orderings = [join(p) for p in permutations(["A", "B", "C"])]
	DataFrames.transform!(
		groupby(prob_block, :stimulus_group),
		:trial => (
			x -> shuffled_fill(orderings, length(x))
		) => :stimulus_ordering
	)

	@info "Proportion optimal in each location: $([round(mean((x -> x[i] == 'A').(prob_block.stimulus_ordering)), digits = 3) for i in 1:3])"

	# Assign stimulus identity to location
	DataFrames.transform!(
		prob_block,
		[:stimulus_ordering, :stimulus_A, :stimulus_B, :stimulus_C] =>
		((o, a, b, c) -> [[a b c][i, (Int(o[i][1]) - Int('A') + 1)] for i in 1:length(o)]) => :stimulus_left,
		[:stimulus_ordering, :stimulus_A, :stimulus_B, :stimulus_C] =>
		((o, a, b, c) -> [[a b c][i, (Int(o[i][2]) - Int('A') + 1)] for i in 1:length(o)]) => :stimulus_middle,
		[:stimulus_ordering, :stimulus_A, :stimulus_B, :stimulus_C] =>
		((o, a, b, c) -> [[a b c][i, (Int(o[i][3]) - Int('A') + 1)] for i in 1:length(o)]) => :stimulus_right		
	)

	# Assign feedback to location, by covention stimulus_A is optimal
	prob_block.feedback_left = ifelse.(
		(x -> x[1] == 'A').(prob_block.stimulus_ordering),
		prob_block.feedback_optimal,
		prob_block.feedback_suboptimal
	)
	
	prob_block.feedback_middle = ifelse.(
		(x -> x[2] == 'A').(prob_block.stimulus_ordering),
		prob_block.feedback_optimal,
		prob_block.feedback_suboptimal
	)

	prob_block.feedback_right = ifelse.(
		(x -> x[3] == 'A').(prob_block.stimulus_ordering),
		prob_block.feedback_optimal,
		prob_block.feedback_suboptimal
	)

	prob_block

end

# ╔═╡ e3bff0b9-306a-4bf9-8cbd-fe0e580bd118
RLWM = let
	RLWM = vcat(RLWM_det_block, RLWM_prob_block, cols = :union)

	# Session variable
	RLWM.session .= 1

	# Valence variable
	RLWM.valence .= 1

	# Apperance variable
	DataFrames.transform!(
		groupby(RLWM, [:block, :stimulus_group]),
		:trial => (x -> 1:length(x)) => :appearance
	)

	# Cumulative triplet index
	RLWM.stimulus_group_id = RLWM.stimulus_group .+ (maximum(RLWM.stimulus_group) .* (RLWM.block .- 1))

	# Create optimal_side variable
	RLWM.optimal_side = [["left", "middle", "right"][findfirst('A', o)] for o in RLWM.stimulus_ordering]


	# Add variables needed for experiment code
	insertcols!(
		RLWM,
		:n_stimuli => 3,
		:optimal_right => "",
		:present_pavlovian => false,
		:n_groups => maximum(RLWM.stimulus_group),
		:early_stop => false
	)

	

	RLWM
end

# ╔═╡ f9be1490-8e03-445f-b36e-d8ceff894751
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

# ╔═╡ eecaac0c-e051-4543-988c-e969de3a8567
let
	save_to_JSON(RLWM, "results/pilot7_WM.json")
	CSV.write("results/pilot7_WM.csv", RLWM)
end

# ╔═╡ 2d860d01-17af-46df-909b-54faca2d2fe4
# Visualize seuqnce
let task = RLWM

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

	# Plot confusing appearnce by triplet
	mp = data(fp) * mapping(
		:appearance => "Appearance",
		:stimulus_group_id => "Triplet",
		:color
	) * visual(Heatmap)

	draw!(f[1,2], mp, axis = (; yreversed = true))

	f

end

# ╔═╡ 9e4e639f-c078-4000-9f01-63bded0dbd82
md"""## PILT"""

# ╔═╡ 85deb936-2204-4fe8-a0dd-a23f527f813d
md"""## Post-PILT test"""

# ╔═╡ 2b7204c6-4fc6-41d2-b446-1c6bf75750b7


# ╔═╡ 0089db22-38ad-4d9c-88a2-12b82361384f
# ╠═╡ disabled = true
#=╠═╡
let
	save_to_JSON(test, "results/pilot7_PILT_test.json")
	CSV.write("results/pilot7_PILT_test.csv", test)
end
  ╠═╡ =#

# ╔═╡ 6dff1b52-b0f0-4895-89d3-f732791e11c5
# ╠═╡ disabled = true
#=╠═╡
# Choose test sequence with best stats
function find_best_test_sequence(
	task::DataFrame; # PILT task structure
	n_trials::Int64, # Number of test trials
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
			n_trials = n_trials,
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
  ╠═╡ =#

# ╔═╡ db7b8c41-7160-4a77-a058-26086d09b7a4
# ╠═╡ disabled = true
#=╠═╡
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
  ╠═╡ =#

# ╔═╡ 62ca4f41-0b0d-4125-a85b-0a9752714d64
# ╠═╡ disabled = true
#=╠═╡
function create_test_sequence(;
	stimuli::DataFrame,
	n_trials::Int64,
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
	used_stimuli = Dict(s => 0 for s in unique_stimuli)

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
					min_count = minimum(values(used_stimuli))
		            
					if (used_stimuli[pair[1]] == min_count) && (used_stimuli[pair[2]] == min_count)  && 
						stim_attr(pair[1], "block") == stim_attr(pair[2], "block")
					
		                push!(block_pairs, pair)
						used_stimuli[pair[1]] += 1
						used_stimuli[pair[2]] += 1
		                pair_counts[key] += 1
		                found_pair = true
		                break  # Stop going over pairs
		            end
		        end
	
				# Then try different block pair
		        for pair in pairs
					min_count = minimum(values(used_stimuli))
		            if !found_pair &&
						(used_stimuli[pair[1]] == min_count) && (used_stimuli[pair[2]] == min_count)
					
		                push!(block_pairs, pair)
						used_stimuli[pair[1]] += 1
						used_stimuli[pair[2]] += 1
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

		    if (length(block_pairs) * 2) == length(unique_stimuli)
		        break  # Exit if all stimuli are used or no valid pairs remain
		    end
		end

		# Add more from under-represented magnitdues
		while true

			found_pair = false

			magn_keys = sort(collect(keys(magnitude_pairs)), by = x -> pair_counts[x] + same_weight * (x[1] == x[2]))

			filter!(x -> pair_counts[x] < n_trials, magn_keys)

			println(magn_keys)

			if isempty(magn_keys)
				break
			end
			
			for key in magn_keys

				pairs = magnitude_pairs[key]

				least_used = filter(x -> x[1] in unique(vcat(pairs...)), used_stimuli)

				least_used = minimum(values(least_used))

	
				# First try to find a same block pair				
		        for pair in pairs
		            
					if (used_stimuli[pair[1]] == least_used) && (used_stimuli[pair[2]] == least_used)  && 
						stim_attr(pair[1], "block") == stim_attr(pair[2], "block")
					
		                push!(block_pairs, pair)
						used_stimuli[pair[1]] += 1
						used_stimuli[pair[2]] += 1
		                pair_counts[key] += 1
		                found_pair = true
		                break  # Stop going over pairs
		            end
		        end
	
				# Then try different block pair
		        for pair in pairs

					if !found_pair &&
						(used_stimuli[pair[1]] == least_used) && (used_stimuli[pair[2]] == least_used)
					
		                push!(block_pairs, pair)
						used_stimuli[pair[1]] += 1
						used_stimuli[pair[2]] += 1
		                pair_counts[key] += 1
		                found_pair = true
		                break  # Stop going over pairs
		            end
		        end
		        
		        if found_pair
		            break  # Restart the outer loop if a pair was found
		        end

			end

		end	

		# Step 7 - Shuffle pair order
		# shuffle!(rng, block_pairs)

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
		block = repeat(1:test_n_blocks, inner = length(final_pairs) ÷ test_n_blocks),
		trial = 1:(length(final_pairs)),
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
  ╠═╡ =#

# ╔═╡ dbb4369f-4928-4c18-8bc8-c3b27ac93462
# ╠═╡ disabled = true
#=╠═╡
test = let
	
	# Find test sequence
	test, pb, pv, pm, sss = find_best_test_sequence(
		task;
		n_trials = 4,
		n_seeds = 1, # Number of random seeds to try
		same_weight = 20. # Weight reducing the number of same magntiude pairs
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
  ╠═╡ =#

# ╔═╡ 5b37feb9-30c2-4e72-bba9-08f3b4e1c499
# ╠═╡ disabled = true
#=╠═╡
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

  ╠═╡ =#

# ╔═╡ b4a3c42d-ebc6-4d7f-a451-271fc3a5132d
# ╠═╡ disabled = true
#=╠═╡
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
  ╠═╡ =#

# ╔═╡ 69afd881-0c45-48c9-8db9-699f9ae23bec
# ╠═╡ disabled = true
#=╠═╡
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
  ╠═╡ =#

# ╔═╡ fdbe5c4e-29cd-4d24-bbe5-40d24d5f98f4
# ╠═╡ disabled = true
#=╠═╡
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
  ╠═╡ =#

# ╔═╡ 3952d2ef-a5c6-40a1-9373-6c4c9ff5ec2b
# ╠═╡ disabled = true
#=╠═╡
function count_valence_transitions(valence::AbstractVector)
	# Preallocate Dict
	transitions = Dict((i, j) => 0 for i in [1, -1] for j in [1, -1])

	# Loop and count
	for i in eachindex(valence)[2:end]
		transitions[(valence[i-1], valence[i])] += 1
	end

	return transitions
end
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═2d7211b4-b31e-11ef-3c0b-e979f01c47ae
# ╠═114f2671-1888-4b11-aab1-9ad718ababe6
# ╠═de74293f-a452-4292-b5e5-b4419fb70feb
# ╟─ffe06202-d829-4145-ae26-4a95449d64e6
# ╠═05f25eb8-3a48-4d16-9837-84d1fdf5c806
# ╠═c05d90b6-61a7-4f9e-a03e-3e11791da6d0
# ╠═f5916a9f-ddcc-4c03-9328-7dd76c4c74b2
# ╠═6eec5cf5-5416-4158-a53f-5ea2b9c9e4a4
# ╠═e3bff0b9-306a-4bf9-8cbd-fe0e580bd118
# ╠═25afe21f-71d2-45ca-b2db-5edf830a2d62
# ╠═f9be1490-8e03-445f-b36e-d8ceff894751
# ╠═eecaac0c-e051-4543-988c-e969de3a8567
# ╠═2d860d01-17af-46df-909b-54faca2d2fe4
# ╠═3fa8c293-ac47-4acd-bdb7-9313286ee464
# ╠═68873d3e-054d-4ab4-9d89-73586bb0370e
# ╠═f89e88c9-ebfc-404f-964d-acff5c7f8985
# ╠═9e4e639f-c078-4000-9f01-63bded0dbd82
# ╠═85deb936-2204-4fe8-a0dd-a23f527f813d
# ╠═2b7204c6-4fc6-41d2-b446-1c6bf75750b7
# ╠═0089db22-38ad-4d9c-88a2-12b82361384f
# ╠═6dff1b52-b0f0-4895-89d3-f732791e11c5
# ╠═db7b8c41-7160-4a77-a058-26086d09b7a4
# ╠═62ca4f41-0b0d-4125-a85b-0a9752714d64
# ╠═dbb4369f-4928-4c18-8bc8-c3b27ac93462
# ╠═5b37feb9-30c2-4e72-bba9-08f3b4e1c499
# ╠═b4a3c42d-ebc6-4d7f-a451-271fc3a5132d
# ╠═69afd881-0c45-48c9-8db9-699f9ae23bec
# ╠═fdbe5c4e-29cd-4d24-bbe5-40d24d5f98f4
# ╠═3952d2ef-a5c6-40a1-9373-6c4c9ff5ec2b
