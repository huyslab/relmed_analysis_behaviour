### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ 784d74ba-21c7-454e-916e-2c54ed0e6911
begin
	cd("/home/jovyan/")
    import Pkg
    # activate the shared project environment
    Pkg.activate("relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate
	using Random, DataFrames, JSON, CSV, StatsBase, CairoMakie
end

# ╔═╡ 87b8c113-cc45-4fb7-b6b6-056afbdb246b
"""
Assigns stimulus filenames and determines the optimal stimulus in each pair.

# Arguments
- `n_phases::Int64`: Number of phases (or blocks) in the session.
- `n_pairs::Vector{Int64}`: Vector containing the number of pairs in each block. Assumes the same number of pairs for all phases.
- `categories::Vector{String}`: Vector of category labels to generate stimulus filenames. Default is the combination of letters 'A' to 'Z' and 'a' to 'z', repeated as necessary to cover the number of stimuli required.

# Returns
- `stimulus_A::Vector{String}`: Vector of filenames for the "A" stimuli in each pair.
- `stimulus_B::Vector{String}`: Vector of filenames for the "B" stimuli in each pair.
- `optimal_A::Vector{Int64}`: Vector indicating which stimulus in each pair is the optimal one (1 if stimulus A is optimal, 0 if stimulus B is optimal).

# Description
1. The function first validates that the number of blocks is even and that there are enough categories to cover all stimuli.
2. It generates filenames for two stimuli per pair: `stimulus_A` and `stimulus_B`. The filenames are based on the provided `categories` vector, with "2.png" for `stimulus_A` and "1.png" for `stimulus_B`.
3. The function then randomly assigns which stimulus in each pair is the optimal one (`optimal_A`), ensuring that exactly half of the stimuli are marked as optimal in a balanced way.
4. A loop ensures that the repeating category in each block and the optimal stimulus are relatively independent.

# Constraints
- Assumes an even number of blocks per session.
- Ensures that there are enough category labels to generate filenames for all stimuli in all phases.
"""
function assign_stimuli_and_optimality(;
	n_phases::Int64,
	n_pairs::Vector{Int64}, # Number of pairs in each block. Assume same for all phases
	categories::Vector{String} = [('A':'Z')[div(i - 1, 26) + 1] * ('a':'z')[rem(i - 1, 26)+1] 
		for i in 1:(sum(n_pairs) * 2 * n_phases + n_phases)],
	random_seed::Int64 = 1
	)

	total_n_pairs = sum(n_pairs) # Number of pairs needed
	
	@assert rem(length(n_pairs), 2) == 0 "Code only works for even number of blocks per sesion"

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

	stimulus_A = (x -> x * "1.png").(stimulus_A)
	stimulus_B = (x -> x * "2.png").(stimulus_B)

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

# ╔═╡ 1b3aca46-c259-43f7-8b06-9ffc63e36228
"""
    save_to_JSON(df::DataFrame, file_path::String)

Saves a given task sequence `DataFrame` to a JSON file, organizing the data by session and block.

# Arguments
- `df::DataFrame`: The DataFrame containing task data to be saved. The DataFrame must have at least `session` and `block` columns to structure the data.
- `file_path::String`: The path (including file name) where the JSON file will be saved.

# Procedure
1. The function groups the DataFrame rows by `session` and then by `block`.
2. Each row within a block is converted into a dictionary.
3. The grouped data is converted to a JSON string.
4. The JSON string is written to the specified file path.

# Notes
- This function assumes that the DataFrame includes `session` and `block` columns for proper grouping.
- The resulting JSON file will contain a nested list structure, where each session contains its respective blocks, and each block contains rows of data represented as dictionaries.
"""
function save_to_JSON(
	df::DataFrame, 
	file_path::String
)
	# Initialize an empty dictionary to store the grouped data
	json_groups = []
	
	# Iterate through unique blocks and their respective rows
	for s in unique(df.session)
		session_groups = []
		for b in unique(df.block)
		    # Filter the rows corresponding to the current block
		    block_group = df[(df.block .== b) .&& (df.session .== s), :]
		    
		    # Convert each row in the block group to a dictionary and collect them into a list
		    push!(session_groups, [Dict(pairs(row)) for row in eachrow(block_group)])
		end
		push!(json_groups, session_groups)
	end
	
	# Convert to JSON String
	json_string = JSON.json(json_groups)
		
	# Write the JSON string to the file
	open(file_path, "w") do file
	    write(file, json_string)
	end

end

# ╔═╡ 94a4ac24-2d30-4410-ae5f-6432f9e2973e
"""
    generate_multiple_n_confusing_sequences(; 
        n_trials::Vector{Int64}, n_confusing::Vector{Int64}, n_pairs::Vector{Int64}
    ) -> DataFrame

Generates sequences that determine which trials in a task will have confusing feedback, based on the number of confusing trials per block. The assignment of confusing trials is random, with the restriction that the last trial in each block cannot be confusing.

# Arguments
- `n_trials::Vector{Int64}`: Number of trials per block.
- `n_confusing::Vector{Int64}`: Number of confusing feedback trials per block.
- `n_pairs::Vector{Int64}`: Number of stimulus pairs per block.

# Returns
- `DataFrame`: A DataFrame where each row represents a trial in a block, and the `feedback_common` column specifies whether the trial will have common feedback (`true`) or confusing feedback (`false`). The structure includes:
  - `cblock`: The cumulative block number.
  - `pair`: The stimulus pair for the trial.
  - `appearance`: The appearance number for the pair in the block.
  - `feedback_common`: A boolean value indicating whether the trial has common feedback (`true`) or confusing feedback (`false`).

# Procedure
1. For each block, trials are grouped by stimulus pairs, ensuring that the total number of trials is divisible by the number of pairs.
2. For each pair, a sequence of trials is generated where a specified number of trials receive confusing feedback (`false`), while the rest receive common feedback (`true`).
3. The feedback sequence for each pair is shuffled randomly, with the constraint that the last trial in the sequence must always have common feedback (`true`).
4. The resulting sequences for all blocks are concatenated into a single DataFrame.

# Notes
- An assertion ensures that the total number of trials per block is divisible by the number of stimulus pairs (`n_trials[i] % n_pairs[i] == 0`).
- This function provides a random but constrained assignment of confusing trials for mixed task designs with varying numbers of blocks, pairs, and confusing trials.
"""
function generate_multiple_n_confusing_sequences(;
		n_trials::Vector{Int64}, # Per block
		n_confusing::Vector{Int64},
		n_pairs::Vector{Int64} # Per block
)

	@assert all(rem.(n_trials, n_pairs) .== 0) "Not all n_trials divisible by n_pairs"

	seqs = []

	for (i, (t, c, p)) in enumerate(zip(n_trials, n_confusing, n_pairs))

		nttp = div(t, p) # Number of trials per pair

		seq = DataFrame(
			cblock = fill(i, t),
			pair = repeat(1:p, inner = nttp),
			appearance = repeat(1:nttp, outer = p),
			feedback_common = vcat(
				[
					vcat(shuffle(
						vcat(
							fill(false, c),
							fill(true, nttp - c - 1)
						)
					), [true]) # Shuffle, making sure last is not confusing
					for _ in 1:p
				]...
			)
		)
		push!(seqs, seq)
	end

	return vcat(seqs...)

end

# ╔═╡ dea0a1fd-7ec3-4004-af9e-3f3155f19ec0
"""
    prepare_task_structure(; 
        n_sessions::Int64, n_blocks::Int64, n_trials::Vector{Int64}, 
        n_pairs::Vector{Int64}, n_confusing::Vector{Int64}, valence::Vector{Int64}, 
        categories::Vector{String}, stop_after::Union{Int64, Missing}, output_file::String, 
        high_reward_magnitudes::Vector{Vector{Float64}}, low_reward_magnitudes::Vector{Vector{Float64}}
    ) -> DataFrame

Generates a task structure for a Probabilistic Inference Learning Task (PILT) by organizing trial sequences, reward feedback magnitudes, stimulus pairing, and saving the resulting structure. 

# Arguments
- `n_sessions::Int64`: Number of sessions.
- `n_blocks::Int64`: Number of blocks per session.
- `n_trials::Vector{Int64}`: Number of trials per block.
- `n_pairs::Vector{Int64}`: Number of stimulus pairs per block.
- `n_confusing::Vector{Int64}`: Number of trials with confusing feedback per block.
- `valence::Vector{Int64}`: Specifies the valence (positive/negative) of rewards for each block.
- `categories::Vector{String}`: List of stimulus categories.
- `stop_after::Union{Int64, Missing}`: Number of trials after which the task should stop, or `missing` if no early stopping is required.
- `output_file::String`: Name of the output file to save the task data.
- `high_reward_magnitudes::Vector{Vector{Float64}}`: High reward magnitudes for each stimulus pair, sorted by blocks.
- `low_reward_magnitudes::Vector{Vector{Float64}}`: Low reward magnitudes for each stimulus pair, sorted by blocks.

# Returns
- `task::DataFrame`: A DataFrame representing the complete task structure, including stimulus file on right / left, feedback, and optimality.

# Procedure
1. Validates input dimensions to ensure consistency across sessions, blocks, and trials.
2. Assigns stimuli and determines their optimality for each trial.
3. Creates feedback sequences (confusing/common) for each block and trial.
4. Organizes trials into a DataFrame, setting attributes such as session, block, and trial number.
5. Randomizes the presentation of stimuli and shuffles their appearance order.
6. Joins feedback sequences and stimulus information into the task structure.
7. Randomly assigns the reward magnitudes for each trial and computes feedback for stimuli.
8. Saves the generated task structure to JSON and CSV formats.
"""
function prepare_task_strucutre(;
	n_sessions::Int64,
	n_blocks::Int64,
	n_trials::Vector{Int64}, # Per block
	n_pairs::Vector{Int64}, # Per block
	n_confusing::Vector{Int64}, # Per block
	valence::Vector{Int64}, # Per block
	categories::Vector{String},
	stop_after::Union{Int64, Missing},
	output_file::String,
	high_reward_magnitudes::Vector{Vector{Float64}}, # Per pair
	low_reward_magnitudes::Vector{Vector{Float64}} # Per pair
) 

	# Checks
	n_blocks_total = n_sessions * n_blocks
	n_pairs_total = sum(n_pairs)
	
	@assert length(n_confusing) == n_blocks_total "Length of n_confusing does not match total number of blocks specified"

	@assert length(n_pairs) == n_blocks_total "Length of n_pairs does not match total number of blocks specified"

	@assert length(n_trials) == n_blocks_total "Length of n_trials does not match total number of blocks specified"

	@assert length(valence) == n_blocks_total "Length of valence does not match total number of blocks specified"

	@assert length(high_reward_magnitudes) == n_pairs_total "Length of high_reward_magnitudes does not match total number of stimulus pairs specified"

	@assert length(low_reward_magnitudes) == n_pairs_total "Length of low_reward_magnitudes does not match total number of stimulus pairs specified"

	# Assign stimulus pairs
	stimuli = assign_stimuli_and_optimality(;
		n_phases = n_sessions,
		n_pairs = n_pairs,
		categories = categories
	)

	rename!(stimuli, :phase => :session) # For compatibility with multi-phase sessions

	# Create sequences of confusing / common feedback
	feedback_sequences = generate_multiple_n_confusing_sequences(
		n_trials = n_trials,
		n_confusing = n_confusing,
		n_pairs = n_pairs
	)

	# Prepare task DataFrame
	task = DataFrame(
		session = repeat(1:n_sessions, inner = sum(n_trials)),
		block = vcat([fill(b, t) for (b, t) in enumerate(n_trials)]...),
		n_pairs = vcat([fill(n_pairs[b], t) for (b, t) in enumerate(n_trials)]...),
		valence = vcat([fill(valence[b], t) for (b, t) in enumerate(n_trials)]...),
		pair = vcat([repeat(1:p, inner = div(n_trials[i], p)) for (i, p) in enumerate(n_pairs)]...)
	)

	# Cumulative block number across sessions
	task.cblock = task.block .+ (task.session .- 1) .* maximum(task.block)

	# Shuffle pair in trial
	transform!(
		groupby(task, [:session, :block, :cblock]),
		:pair => shuffle => :pair
	)

	# Number apperanece of each pair
	transform!(
		groupby(task, [:session, :block, :pair]),
		:pair => (x -> 1:length(x)) => :appearance
	)
	
	# Number trials
	transform!(
		groupby(task, [:session, :block]),
		:pair => (x -> 1:length(x)) => :trial
	)
	
	# Join with sequences	
	task = innerjoin(
		task,
		feedback_sequences,
		on = [:cblock, :pair, :appearance],
		order = :left
	)

	# Join with stimuli
	task = innerjoin(
		task,
		stimuli,
		on = [:session, :block, :pair],
		order = :left
	)

	# Assign whether optimal stimulus is on right
	transform!(
		groupby(task, :cblock),
		:trial => (x -> shuffle(
			collect(Iterators.take(Iterators.cycle(shuffle([true, false])), length(x)))
		)) => :optimal_right
	)

	# Assign stimulus on right
	task.stimulus_right = ifelse.(
		task.optimal_right,
		ifelse.(
			task.optimal_A,
			task.stimulus_A,
			task.stimulus_B
		),
		ifelse.(
			task.optimal_A,
			task.stimulus_B,
			task.stimulus_A
		)
	)

	# Assign stimulus on left
	task.stimulus_left = ifelse.(
		task.optimal_right,
		ifelse.(
			task.optimal_A,
			task.stimulus_B,
			task.stimulus_A
		),
		ifelse.(
			task.optimal_A,
			task.stimulus_A,
			task.stimulus_B
		)
	)

	# Compute cumulative pair number
	pairs = unique(task[!, [:cblock, :pair]])
	pairs.cpair = 1:nrow(pairs)

	task = innerjoin(
		task,
		pairs,
		on = [:cblock, :pair],
		order = :left
	)
	
	
	# Shuffle reward magnitudes
	transform!(
		groupby(task, [:session, :block, :pair]),
		:cpair => (
			x -> shuffle(collect(Iterators.take(
				Iterators.cycle(shuffle(high_reward_magnitudes[x[1]])),
				length(x)
			)))
		) => :high_magnitude,
		:cpair => (
			x -> shuffle(collect(Iterators.take(
				Iterators.cycle(shuffle(low_reward_magnitudes[x[1]])),
				length(x)
			)))
		) => :low_magnitude
	)

	# Compute better feedback value
	task.better_feedback = ifelse.(
		task.valence .== 1,
		task.high_magnitude,
		.- task.low_magnitude
	)

	task.worse_feedback = ifelse.(
		task.valence .== 1,
		task.low_magnitude,
		.- task.high_magnitude
	)

	# Assign feedback to stimulus
	task.feedback_right = ifelse.(
		task.feedback_common,
		ifelse.(
			task.optimal_right,
			task.better_feedback,
			task.worse_feedback
		),
		ifelse.(
			.!task.optimal_right,
			task.better_feedback,
			task.worse_feedback
		)
	)

	task.feedback_left = ifelse.(
		task.feedback_common,
		ifelse.(
			.!task.optimal_right,
			task.better_feedback,
			task.worse_feedback
		),
		ifelse.(
			task.optimal_right,
			task.better_feedback,
			task.worse_feedback
		)
	)

	# Save to file
	save_to_JSON(task, "results/$output_file.json")
	CSV.write("results/$output_file.csv", task)
	
	return task
end

# ╔═╡ 2c31faf8-8b32-4709-ba3a-43ee9376a3c4
# Create PILT sequence for pilot 2
task = let set_sizes = 1:3,
	block_per_set = 6,
	trials_per_pair = 10,
	random_seed = 10

	# Set random seed
	Random.seed!(random_seed)

	# Load stimulus names
	categories = shuffle(unique([replace(s, ".png" => "")[1:(end-1)] for s in 
		readlines("generate_experimental_sequences/allimages.txt")]))

	# Total number of blocks
	n_total_blocks = length(set_sizes) * block_per_set

	# All combinations of set sizes and valence
	@assert iseven(block_per_set) # Requisite for code below
	
	valence_set_size = DataFrame(
		n_pairs = repeat(set_sizes, inner = block_per_set),
		valence = repeat([1, -1], inner = 3, outer = div(n_total_blocks, 6))
	)

	# Shuffle, making sure set size rises gradually
	while valence_set_size[1:3, :n_pairs] != [1, 2, 3]
		valence_set_size.block = shuffle(1:n_total_blocks)
		sort!(valence_set_size, :block)
	end

	# Total number of pairs
	n_total_pairs = sum(valence_set_size.n_pairs)

	# Shuffle high reward magnitudes for each pair
	high_reward_magnitudes = Iterators.take(
		Iterators.cycle([[0.5, 1.], [1.]]), n_total_pairs
	) |> collect |> shuffle


	# Assign low reward magnitudes for each pair based on high
	low_reward_magnitudes = ifelse.(
		(x -> x == [1.]).(high_reward_magnitudes),
		fill([0.5, 0.01], n_total_pairs),
		fill([0.01], n_total_pairs)
	)

	# Prepare and save task sequence
	prepare_task_strucutre(;
		n_sessions = 1,
		n_blocks = n_total_blocks,
		n_trials = valence_set_size.n_pairs .* trials_per_pair, # Per block
		categories = categories,
		valence = valence_set_size.valence,
		stop_after = 5,
		output_file = "pilot2",
		n_confusing = vcat([0, 1, 1], fill(2, n_total_blocks - 3)), # Per block
		n_pairs = valence_set_size.n_pairs, # Per block
		high_reward_magnitudes = high_reward_magnitudes, # Per pair
		low_reward_magnitudes = low_reward_magnitudes # Per pair
	) 

end

# ╔═╡ 08366330-e5da-4f12-85c5-fc780c4a98a2
# Visualize PILT seuqnce
let

	# Proportion of confusing by trial number
	confusing_location = combine(
		groupby(task, :trial),
		:feedback_common => (x -> mean(.!x)) => :feedback_confusing
	)

	# Plot
	f = Figure()

	ax_prob = Axis(
		f[1,1],
		xlabel = "Trial #",
		ylabel = "Prop. confusing feedback"
	)

	scatter!(
		ax_prob,
		confusing_location.trial,
		confusing_location.feedback_confusing
	)

	# Plot confusing trials by block
	ax_heatmap = Axis(
		f[1, 2],
		xlabel = "Trial #",
		ylabel = "Block",
		yreversed = true
	)

	heatmap!(
		task.trial,
		task.block,
		.!task.feedback_common
	)

	f

end

# ╔═╡ b176448a-74a5-4304-b2a2-95bd9298afb5
# Count losses to allocate coins in to safe for beginning of task
filter(x -> x.valence == -1, task).worse_feedback |> countmap

# ╔═╡ 723476b8-8df9-417c-b941-a6af097656c9
# Create sequence of post-PILT test
test_pairs = let n_blocks = 2,
	random_seed = 3

	# Set random seed
	rng = Xoshiro(random_seed)

	# Intialize DataFrame and summary stats for checking
	test_pairs_wide = DataFrame()
	prop_same_original = 0.
	prop_same_valence = 1.

	# Make sure with have exactly 1/3 pairs that were previously in same block
	# and 1/2 that were of the same valence
	while !(prop_same_original == 1/3) || !(prop_same_valence == 0.5)
		
		# Extract list of stimuli from PILT task sequence
		stimuli = vcat([rename(
			task[!, [:session, :n_pairs, :block, :cpair, Symbol("stimulus_$s"), Symbol("feedback_$s")]],
			Symbol("stimulus_$s") => :stimulus,
			Symbol("feedback_$s") => :feedback
		) for s in ["right", "left"]]...)

		# Summarize EV per stimulus
		stimuli = combine(
			groupby(stimuli, [:session, :n_pairs, :block, :cpair, :stimulus]),
			:feedback => mean => :EV
		)

		# Function that creates a list of pairs from DataFrame
		create_pair_list(d) = [filter(x -> x.cpair == p, d).stimulus 
			for p in unique(stimuli.cpair)]

		# Pairs used in PILT - will add post-PILT test blocks into this as we go
		used_pairs = create_pair_list(stimuli)

		# Function to check whether pair is novel
		check_novel(p) = !(p in used_pairs) && !(reverse(p) in used_pairs)
	
		# Initialize long format DataFrame
		test_pairs = DataFrame()

		# Run over neede block number
		for bl in 1:n_blocks

			# Variable to record whether suggested pairs are novel
			all_within_novel = false

			# Intialize DataFrame for pairs that were in the same block
			block_within_pairs = DataFrame()

			# Create within-block pairs
			while !all_within_novel
				
				# Select blocks with n_pairs > 1
				multi_pair_blocks = groupby(
					filter(x -> x.n_pairs > 1, stimuli), 
					[:session, :block]
				)

				# For each block, pick one stimulus from each of two pairs
				for (i, gdf) in enumerate(multi_pair_blocks)
			
					if gdf.n_pairs[1] == 3
						chosen_pairs = sample(rng, unique(gdf.cpair), 2, replace = false)
						tdf = filter(x -> x.cpair in chosen_pairs, gdf)
					else
						tdf = copy(gdf)
					end
			
					stim = vcat([sample(rng, filter(x -> x.cpair == c, tdf).stimulus, 1) 
						for c in unique(tdf.cpair)]...)
	
					append!(block_within_pairs, DataFrame(
						block = fill(bl, 2),
						cpair = fill(i, 2),
						stimulus = stim
					))
				end

				# Check that picked combinations are novel
				pair_list = create_pair_list(block_within_pairs)
	
				all_within_novel = all(check_novel.(pair_list))
	
			end

			# Add to block list
			block_pairs = block_within_pairs

			# Stimuli not included previously
			remaining_stimuli = 
				filter(x -> !(x in block_pairs.stimulus), stimuli.stimulus)

			# Variable for checking whether remaining pairs are novel
			all_between_novel = false
	
			block_between_pairs = DataFrame()

			# Add across-block pairs
			while !all_between_novel

				# Assign pairs
				block_between_pairs = DataFrame(
					block = fill(bl, length(remaining_stimuli)),
					cpair = repeat(
						(maximum(block_pairs.cpair) + 1):maximum(stimuli.cpair), 
						inner = 2
					),
					stimulus = shuffle(rng, remaining_stimuli)
				)

				# Check novelty
				pair_list = create_pair_list(block_between_pairs)
		
				all_between_novel = all(check_novel.(pair_list))
			end

			# Add to lists of pairs for checking future block against
			append!(block_pairs, block_between_pairs)
	
			append!(test_pairs, block_pairs)
			
		end

		# Add EV from PILT
		test_pairs = innerjoin(test_pairs, 
			rename(stimuli, :block => :original_block)[!, Not(:cpair)],
			on = :stimulus
		)

		# Compute EV difference and match in block and valence
		test_pairs_wide = combine(
			groupby(test_pairs, [:block, :cpair]),
			:stimulus => maximum => :stimulus_A,
			:stimulus => minimum => :stimulus_B,
			:EV => diff => :EV_diff,
			:EV => (x -> sign(x[1]) == sign(x[2])) => :same_valence,
			:original_block => (x -> x[1] == x[2]) => :same_block
		)
		
		prop_same_original = mean(test_pairs_wide.same_block)
	
		prop_same_valence = mean(test_pairs_wide.same_valence)

end

	@info "Proportion of pairs that were in the same original block: $prop_same_original"
	@info "Proprotion of pairs with the same valence: $prop_same_valence"

	# Assign right / left stimulus randomly
	A_on_right = sample(rng, [true, false], nrow(test_pairs_wide))

	test_pairs_wide.stimulus_right = ifelse.(
		A_on_right, 
		test_pairs_wide.stimulus_A,
		test_pairs_wide.stimulus_B
	)

	test_pairs_wide.stimulus_left = ifelse.(
		.!A_on_right, 
		test_pairs_wide.stimulus_A,
		test_pairs_wide.stimulus_B
	)

	# Save to file
	save_to_JSON(task, "results/pilot2_test.json")
	CSV.write("results/pilot2_test.csv", task)
	
	test_pairs_wide
end

# ╔═╡ 95143c27-80b7-42c1-a065-723d405c3c4d
# Plot post-PILT test
let
	# Histogram of all Abs. Δ EV
	f = Figure()

	ax = Axis(
		f[1,1],
		xlabel = "Abs. Δ EV"
	)
	
	hist!(ax, abs.(test_pairs.EV_diff))

	# Abs. Δ EV by same / different block
	block_group = ifelse.(
		test_pairs.same_block, 
		fill(1, nrow(test_pairs)), 
		ifelse.(
			test_pairs.same_valence,
			fill(2, nrow(test_pairs)),
			fill(3, nrow(test_pairs))
		)
	)

	ax_block = Axis(
		f[1,2],
	)

	ax_block.xticks = (1:3,
		["Same block", "Different block\nsame valence", "Different\nvalence"]
	)

	scatter!(
		block_group + rand(nrow(test_pairs)) * 0.25,
		abs.(test_pairs.EV_diff)
	)


	f
	
end

# ╔═╡ Cell order:
# ╠═784d74ba-21c7-454e-916e-2c54ed0e6911
# ╠═2c31faf8-8b32-4709-ba3a-43ee9376a3c4
# ╠═08366330-e5da-4f12-85c5-fc780c4a98a2
# ╠═b176448a-74a5-4304-b2a2-95bd9298afb5
# ╠═95143c27-80b7-42c1-a065-723d405c3c4d
# ╠═723476b8-8df9-417c-b941-a6af097656c9
# ╠═87b8c113-cc45-4fb7-b6b6-056afbdb246b
# ╠═dea0a1fd-7ec3-4004-af9e-3f3155f19ec0
# ╠═1b3aca46-c259-43f7-8b06-9ffc63e36228
# ╠═94a4ac24-2d30-4410-ae5f-6432f9e2973e
