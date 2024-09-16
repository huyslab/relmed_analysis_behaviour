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

# ╔═╡ 45394a5a-6e96-11ef-27e9-5dddb818f955
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

# ╔═╡ 04caadb8-6095-46e2-84b3-67eb535f4725
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

	save_to_JSON(task, "results/$output_file.json")
	CSV.write("results/$output_file.csv", task)
	
	return task
end

# ╔═╡ 2c31faf8-8b32-4709-ba3a-43ee9376a3c4
task = let set_sizes = 1:3,
	block_per_set = 6,
	trials_per_pair = 10

	Random.seed!(0)


	categories = shuffle(unique([replace(s, ".png" => "")[1:(end-1)] for s in 
		readlines("generate_experimental_sequences/allimages.txt")]))

	n_total_blocks = length(set_sizes) * block_per_set

	# All combinations of set sizes and valence
	@assert iseven(block_per_set)
	
	valence_set_size = DataFrame(
		n_pairs = repeat(set_sizes, inner = block_per_set),
		valence = repeat([1, -1], inner = 3, outer = div(n_total_blocks, 6))
	)

	# Shuffle, making sure set size rises gradually
	while valence_set_size[1:3, :n_pairs] != [1, 2, 3]
		valence_set_size.block = shuffle(1:n_total_blocks)
		sort!(valence_set_size, :block)
	end

	n_total_pairs = sum(valence_set_size.n_pairs)
	
	high_reward_magnitudes = Iterators.take(
		Iterators.cycle([[0.5, 1.], [1.]]), n_total_pairs
	) |> collect |> shuffle

	

	low_reward_magnitudes = ifelse.(
		(x -> x == [1.]).(high_reward_magnitudes),
		fill([0.5, 0.01], n_total_pairs),
		fill([0.01], n_total_pairs)
	)

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

# ╔═╡ b176448a-74a5-4304-b2a2-95bd9298afb5
filter(x -> x.valence == -1, task).worse_feedback |> countmap

# ╔═╡ 8b0207af-84d9-4f65-97c7-451cb8012497
let

	confusing_location = combine(
		groupby(task, :trial),
		:feedback_common => (x -> mean(.!x)) => :feedback_confusing
	)

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

	ax_heatmap = Axis(
		f[1, 2],
		xlabel = "Trial #",
		ylabel = "Block"
	)

	heatmap!(
		task.trial,
		task.block,
		.!task.feedback_common
	)

	f

end

# ╔═╡ Cell order:
# ╠═784d74ba-21c7-454e-916e-2c54ed0e6911
# ╠═45394a5a-6e96-11ef-27e9-5dddb818f955
# ╠═04caadb8-6095-46e2-84b3-67eb535f4725
# ╠═2c31faf8-8b32-4709-ba3a-43ee9376a3c4
# ╠═b176448a-74a5-4304-b2a2-95bd9298afb5
# ╠═8b0207af-84d9-4f65-97c7-451cb8012497
# ╠═1b3aca46-c259-43f7-8b06-9ffc63e36228
# ╠═94a4ac24-2d30-4410-ae5f-6432f9e2973e
