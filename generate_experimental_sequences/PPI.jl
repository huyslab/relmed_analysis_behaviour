### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ 32998df2-96a4-11ef-2120-437ed324f530
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
	include("$(pwd())/stats_utils.jl")
	include("$(pwd())/plotting_utils.jl")
	include("$(pwd())/single_p_QL.jl")
	include("$(pwd())/fetch_preprocess_data.jl")
	include("$(pwd())/generate_experimental_sequences/sequence_utils.jl")
	nothing
end

# ╔═╡ bd84e28a-e3b1-45b5-87bc-5a11154a8228
# Reversal task parameters
begin
	rev_n_blocks = 25
	rev_n_trials = 100
	rev_prop_confusing = vcat([0, 0.1, 0.1, 0.2, 0.2], fill(0.3, rev_n_blocks - 5))
	rev_criterion = vcat(
		[8, 7, 6, 6, 5], 
		shuffled_fill(
			3:8, 
			rev_n_blocks - 5; 
			random_seed = 0
		)
	)
end

# ╔═╡ 3b87bcbf-31ac-430d-9b1f-970d94129c10
# PILT parameters
begin
	# PILT Parameters
	set_sizes = [1,3,7]
	block_per_set = 4 # Including reward and punishment
	trials_per_pair = 7

	# Total number of blocks
	n_total_blocks = length(set_sizes) * block_per_set

	# Total number of pairs
	n_total_pairs = sum(set_sizes .* block_per_set)

	pair_mini_block_size = 2

	# Full deterministic task
	n_confusing = fill(0, n_total_blocks) # Per block

	# Initial Q value for simulation - average outcome
	aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])
	
	# Post-PILT test parameters
	test_n_blocks = 2

end

# ╔═╡ d6787cc4-7638-410a-92a9-600f05758390
md"""# PILT"""

# ╔═╡ d8b48d35-96bb-44b7-b138-2dfbb1c6a33a
# Assign valence and set size per block
valence_set_size = let random_seed = 0
	
	# # All combinations of set sizes and valence
	@assert iseven(block_per_set) # Requisite for code below

	n_repetitions = div(block_per_set, 2) # How many repetitions of each cell
	valence_set_size = DataFrame(
		n_pairs = repeat(set_sizes, inner = block_per_set),
		valence = repeat([1, -1], inner = n_repetitions, outer = div(n_total_blocks, block_per_set)),
		super_block = repeat(1:n_repetitions, outer = div(n_total_blocks, n_repetitions))
	)

	# Shuffle set size and valence, making sure set size rises gradually and valence is varied in first three blocks, and positive in the first
	rng = Xoshiro(random_seed)
	
	while (valence_set_size[1:3, :n_pairs] != set_sizes) | 
		allequal(valence_set_size[1:3, :valence]) | 
		(valence_set_size.valence[1] == -1)

		DataFrames.transform!(
			groupby(valence_set_size, :super_block),
			:super_block => (x -> shuffle(rng, 1:length(x))) => :order
		)
		
		sort!(valence_set_size, [:super_block, :order])
	end

	# Add n_confusing
	valence_set_size.n_confusing = n_confusing

	# Add block variable
	valence_set_size.block = 1:nrow(valence_set_size)

	# Return
	valence_set_size
end

# ╔═╡ fbace823-d259-452e-bce1-585a71e591e2
# # Create trial sequence
feedback_sequence = let random_seed = 0

	# Create pair list
	task = combine(
		groupby(valence_set_size, 
			[:block, :n_pairs, :valence, :n_confusing]),
		:n_pairs => (x -> repeat(vcat([1:xi for xi in x]...), inner = trials_per_pair)) => :pair
	)

	# Add cumulative pair variable
	pairs = sort(unique(task[!, [:block, :pair, :n_pairs, :valence]]), [:block, :pair])
	pairs.cpair = 1:nrow(pairs)
	
	# Assign fifty_high
	equal_distribution = false

	rng = Xoshiro(random_seed)
	while !equal_distribution
		pairs.fifty_high = shuffled_fill([true, false], 
			nrow(pairs), rng = rng)

		# Compute sum of fifty_high by valence
		fifty_high = combine(
			groupby(pairs, [:valence]),
			:fifty_high => sum => :fifty_high
		).fifty_high 

		# Check if distributed equally
		equal_valence = all((x -> x in [sum(fifty_high) ÷ (length(fifty_high)), sum(fifty_high) ÷ (length(fifty_high)) + 1]).(fifty_high))

		# Compute sum of fifty_high by n_pairs
		fifty_high = combine(
			groupby(pairs, [:n_pairs]),
			:fifty_high => sum => :fifty_high,
			:fifty_high => length => :n
		)

		# Check if distributed equally
		equal_pairs = all((r -> r.fifty_high == r.n / 2).(eachrow(fifty_high)))
	

		equal_distribution = equal_valence && equal_pairs
	end

	# Join into task
	task = innerjoin(
		task,
		pairs[!, Not(:valence, :n_pairs)],
		on = [:block, :pair],
		order = :left
	)


	# Add apperance count variable
	DataFrames.transform!(
		groupby(task, [:block, :pair]),
		:block => (x -> 1:length(x)) => :appearance
	)

	sort!(task, [:block, :appearance, :pair])

	task.mini_block = (task.appearance .÷ pair_mini_block_size) .+ 1

	# Shuffle pair appearance	
	DataFrames.transform!(
		groupby(task, [:block, :mini_block]),
		:pair => (x -> shuffle(rng, 1:length(x))) => :order_var
	)

	sort!(task, [:block, :mini_block, :order_var])

	# Reorder apperance
	DataFrames.transform!(
		groupby(task, [:block, :pair]),
		:block => (x -> 1:length(x)) => :appearance
	)

	# Trial counter
	DataFrames.transform!(
		groupby(task, :block),
		:block => (x -> 1:length(x)) => :trial
	)

	# Create deterministic sequence
	task[!, :feedback_common] .= true

	# Randomly assign magnitude
	DataFrames.transform!(
		groupby(task, :cpair),
		:fifty_high => 
			(x -> shuffled_fill([0.5, x[1] ? 1. : 0.01], length(x), rng = rng)) => :variable_magnitude
	)

	# Compute optimal and suboptimal feedback
	task.feedback_optimal = ifelse.(
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

	task.feedback_suboptimal = ifelse.(
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

	task

end

# ╔═╡ 2f6b150c-5056-4862-944d-8a9fa9e7b309
categories = shuffle(Xoshiro(0), unique([replace(s, ".png" => "")[1:(end-1)] for s in 
		readlines("generate_experimental_sequences/pilot4_stim_list.txt")])) 

# ╔═╡ b7376c48-758e-42d4-83ce-b1241c5330ff
# Assign stimulus images
stimuli = let random_seed = 0
	# Load stimulus names
	categories = shuffle(Xoshiro(random_seed), unique([replace(s, ".png" => "")[1:(end-1)] for s in 
		readlines("generate_experimental_sequences/pilot4_stim_list.txt")]))

	# Assign stimulus pairs
	stimuli = assign_stimuli_and_optimality(;
		n_phases = 1,
		n_pairs = valence_set_size.n_pairs,
		categories = categories,
		random_seed = random_seed
	)

	rename!(stimuli, :phase => :session) # For compatibility with multi-phase sessions
end

# ╔═╡ 6727e3c1-b715-4f61-8c6b-8e40592a1f1e
# Add stimulus assignments to sequences DataFrame, and assign right / left
task = let random_seed = 0

	# Join stimuli and sequences
	task = innerjoin(
		feedback_sequence,
		stimuli,
		on = [:block, :pair],
		order = :left
	)

	@assert nrow(task) == nrow(feedback_sequence) "Problem in join operation"

	# Assign right / left, equal proportions within each pair
	rng = Xoshiro(random_seed)

	DataFrames.transform!(
		groupby(task, [:block, :cpair]),
		:block => 
			(x -> shuffled_fill([true, false], length(x); random_seed = random_seed)) =>
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

	task
end

# ╔═╡ aae008c0-7420-478c-8a9f-4b1b33c80c82
# Validate task DataFrame
let
	@assert maximum(task.block) == length(unique(task.block)) "Error in block numbering"

	@assert issorted(task.block) "Task structure not sorted by block"

	@assert all(combine(groupby(task, [:session, :block]), 
		:trial => issorted => :sorted).sorted) "Task structure not sorted by trial number"

	@assert all(sign.(task.valence) == sign.(task.feedback_right)) "Valence doesn't match feedback sign"

	@assert all(sign.(task.valence) == sign.(task.feedback_left)) "Valence doesn't match feedback sign"

	@assert sum(unique(task[!, [:session, :block, :valence]]).valence) == 0 "Number of reward and punishment blocks not equal"

	@info "Overall proportion of common feedback: $(round(mean(task.feedback_common), digits = 2))"

	@assert all(combine(groupby(task, :cpair),
		:appearance => maximum => :max_appear
	).max_appear .== trials_per_pair) "Didn't find exactly $trials_per_pair apperances per pair"

	@assert all((task.variable_magnitude .== abs.(task.feedback_right)) .| 
		(task.variable_magnitude .== abs.(task.feedback_left))) ":variable_magnitude, which is used for sequnece optimization, doesn't match end result column :feedback_right no :feedback_left"

	@assert all(combine(
		groupby(task, [:cpair]),
		[:feedback_optimal, :feedback_suboptimal] =>
			((o, s) -> ((0.5 ∈ abs.(o)) && (0.5 ∉ abs.(s))) || ((0.5 ∉ abs.(o)) && (0.5 ∈ abs.(s)))) => :test
	).test) "50 pence not allocated correctly"

	# Count losses to allocate coins in to safe for beginning of task
	worst_loss = filter(x -> x.valence == -1, task) |> 
		df -> ifelse.(
			df.feedback_right .< df.feedback_left, 
			df.feedback_right, 
			df.feedback_left) |> 
		countmap

	@info "Worst possible loss in this task is of these coin numbers: $worst_loss"

end

# ╔═╡ dba92a2d-1837-45b5-8da6-75420a5e860b
let
	save_to_JSON(task, "results/pilot_PPI_pilt.json")
	CSV.write("results/pilot_PPI_pilt.csv", task)
end

# ╔═╡ 1dbed335-ee1a-4bfa-8f06-b133297d7059
md"""## Post-PILT test"""

# ╔═╡ dbfdb959-bbd9-4d4f-99a2-c3e7f2a0b11b
function create_test_sequence(
	pilt_task::DataFrame;
	random_seed::Int64, 
	same_weight::Float64 = 6.5
) 
	
	rng = Xoshiro(random_seed)

	# Extract stimuli and their common feedback from task structure
	stimuli = vcat([rename(
		pilt_task[pilt_task.feedback_common, [:session, :n_pairs, :block, :cpair, Symbol("stimulus_$s"), Symbol("feedback_$s")]],
		Symbol("stimulus_$s") => :stimulus,
		Symbol("feedback_$s") => :feedback
	) for s in ["right", "left"]]...)

	# Summarize magnitude per stimulus
	stimuli = combine(
		groupby(stimuli, [:session, :n_pairs, :block, :cpair, :stimulus]),
		:feedback => (x -> mean(unique(x))) => :magnitude
	)

	# Step 1: Identify unique stimuli
	unique_stimuli = unique(stimuli.stimulus)

	# Step 2: Define existing pairs
	create_pair_list(d) = [filter(x -> x.cpair == p, d).stimulus 
		for p in unique(stimuli.cpair)]

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

# ╔═╡ b6f76392-f2cc-4e0f-8992-bbeef464ec0b
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
	chosen = pass_magnitude[argmin(dev_block .+ dev_valence)]

	# Return sequence and stats
	return create_test_sequence(task, random_seed = chosen, same_weight = same_weight)
end

# ╔═╡ 86873aa8-2836-4683-83ec-bddcc64b1c90
test = let
	# Find test sequence for main part
	test, pb, pv, nm = find_best_test_sequence(
		task,
		n_seeds = 50, # Number of random seeds to try
		same_weight = 4. # Weight reducing the number of same magntiude pairs
	) 

	@info "Proportion of same block pairs: $pb"
	@info "Proportion of same valence pairs: $pv"
	@info "Number of same magnitude pairs: $nm"

	# Create magnitude_pair variable
	test.magnitude_pair = [sort([r.magnitude_left, r.magnitude_right]) for r in eachrow(test)]

	# Create column session
	test[!, :session] .= 1

	test
end

# ╔═╡ df7de94c-cc4a-4360-afaa-1b27c7697807
let
	save_to_JSON(test, "results/pilot_PPI_pilt_test.json")
	CSV.write("results/pilot_PPI_pilt_test.csv", test)
end

# ╔═╡ 6464a2af-8770-4a92-988d-5123adf96957
md"""# Reversal task"""

# ╔═╡ c7506813-df22-4c00-90fe-37591de5bc64
function find_lcm_denominators(props::Vector{Float64})
	# Convert to rational numbers
	rational_props = rationalize.(props)
	
	# Extract denominators
	denominators = [denominator(r) for r in rational_props]
	
	# Compute the LCM of all denominators
	smallest_int = foldl(lcm, denominators)
end

# ╔═╡ 199e2206-5e55-45bd-88b1-e9718df6bf63
# Reversal task structure
rev_feedback_optimal = let random_seed = 0

	# Compute minimal mini block length to accomodate proportions
	mini_block_length = find_lcm_denominators(rev_prop_confusing)

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
	while isempty(feedback_optimal) || 
		!all([bl[1] != 0.01 for bl in feedback_optimal[1:6]])
		feedback_optimal = [block_high_mag(p, rng) for p in rev_prop_confusing]
	end

	# Function to compute feedback_suboptimal from feedback_optimal
	inverter(x) = 1 ./ (100 * x)

	# Create timeline variables
	timeline = [[Dict(
		:block => bl,
		:trial => t,
		:feedback_left => isodd(bl) ? feedback_optimal[bl][t] : 
			inverter(feedback_optimal[bl][t]),
		:feedback_right => iseven(bl) ? feedback_optimal[bl][t] : 
			inverter(feedback_optimal[bl][t]),
		:optimal_right => iseven(bl),
		:criterion => rev_criterion[bl]
	) for t in 1:rev_n_trials] for bl in 1:rev_n_blocks]
	
	# Convert to JSON String
	json_string = JSON.json(timeline)

	# Add JS variable assignment
	json_string = "const reversal_json = '$json_string';"
		
	# Write the JSON string to the file
	open("results/pilot_PPI_reversal_sequence.js", "w") do file
	    write(file, json_string)
	end

	feedback_optimal
end

# ╔═╡ 7a3b1891-4287-410f-b5fa-41734b0c7613
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

	mp3 = data(
		combine(
			groupby(rev_confusing, :trial),
			:feedback_common => (x -> mean(.!x)) => :feedback_confusing
		)
	) * mapping(:trial => "Trial position", :feedback_confusing => "Prop. confusing trials") * (visual(Scatter) + smooth())

	draw!(f[2,1], mp3)

	mp4 = mapping(1:rev_n_blocks => "Block", rev_criterion => "# optimal choices)") * visual(ScatterLines)

	draw!(f[2,2], mp4, axis = (; 
		yticks = 3:8, 
		xticks = [1, 10, 20, 30], 
		subtitle = "Reversal criterion")
	)

	save("results/pilot_PPI_reversal_sequence.png", f, pt_per_unit = 1)

	f

end

# ╔═╡ Cell order:
# ╠═32998df2-96a4-11ef-2120-437ed324f530
# ╠═bd84e28a-e3b1-45b5-87bc-5a11154a8228
# ╠═3b87bcbf-31ac-430d-9b1f-970d94129c10
# ╟─d6787cc4-7638-410a-92a9-600f05758390
# ╠═d8b48d35-96bb-44b7-b138-2dfbb1c6a33a
# ╠═fbace823-d259-452e-bce1-585a71e591e2
# ╠═2f6b150c-5056-4862-944d-8a9fa9e7b309
# ╠═b7376c48-758e-42d4-83ce-b1241c5330ff
# ╠═6727e3c1-b715-4f61-8c6b-8e40592a1f1e
# ╠═aae008c0-7420-478c-8a9f-4b1b33c80c82
# ╠═dba92a2d-1837-45b5-8da6-75420a5e860b
# ╟─1dbed335-ee1a-4bfa-8f06-b133297d7059
# ╠═86873aa8-2836-4683-83ec-bddcc64b1c90
# ╠═df7de94c-cc4a-4360-afaa-1b27c7697807
# ╠═dbfdb959-bbd9-4d4f-99a2-c3e7f2a0b11b
# ╠═b6f76392-f2cc-4e0f-8992-bbeef464ec0b
# ╟─6464a2af-8770-4a92-988d-5123adf96957
# ╠═c7506813-df22-4c00-90fe-37591de5bc64
# ╠═199e2206-5e55-45bd-88b1-e9718df6bf63
# ╠═7a3b1891-4287-410f-b5fa-41734b0c7613
