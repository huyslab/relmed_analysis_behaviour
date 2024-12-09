# This file contains functions for creating trial sequences, and optimizing FI under contraints.
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

"""
    FI_for_feedback_sequence(; task::AbstractDataFrame, ρ::Float64, a::Float64, initV::Float64, n_blocks::Int64 = 500, summary_method::Function = tr)

Compute the Fisher Information (FI) for a given sequence of feedback in a probabilistic instrumental learning task using a simple Q-learning model.

# Arguments
- `task::AbstractDataFrame`: The DataFrame containing the feedback sequences. It should include columns for `block`, `trial`, `feedback_optimal`, and `feedback_suboptimal`.
- `ρ::Float64`: Reward sensitivity to be used for simulating data.
- `a::Float64`: Learning rate to be used for simulatind data (unconstrained scale).
- `initV::Float64`: The initial value of the Q-values for both choices (optimal and suboptimal).
- `n_blocks::Int64`: The number of simulated blocks over which the Fisher Information is summarized. Defaults to 500.
- `summary_method::Function`: A function used to summarize the Fisher Information matrix. Defaults to `tr` (trace).

# Returns
- The computed Fisher Information (FI) for the feedback sequence.

# Details
The function first simulates feedback sequences using a Q-learning model with the provided parameters (`ρ`, `a`, and `initV`).

After simulation, the Fisher Information is computed using the provided model `single_p_QL`, which maps the task data to the model parameters and calculates FI using the specified summary method.
"""
function FI_for_feedback_sequence(;
	task::AbstractDataFrame,
	ρ::Float64,
	a::Float64,
	initV::Union{Float64, Nothing} = nothing,
	n_blocks::Int64 = 500,
	summary_method::Function = tr
)

	# Sample from prior
	prior_sample = simulate_single_p_QL(
			n_blocks;
			block = task.block,
			outcomes = hcat(task.feedback_suboptimal, task.feedback_optimal),
			initV = fill(initV, 1, 2),
			random_seed = 0,
			prior_ρ = Dirac(ρ),
			prior_a = Dirac(a)
		)

	prior_sample = innerjoin(prior_sample, 
			task[!, [:block, :trial, :feedback_optimal, :feedback_suboptimal]],
			on = [:block, :trial]
	)

	prior_sample.block = prior_sample.PID

	prior_sample[!, :PID] .= 1

	# Compute FI
	return FI(
		data = prior_sample,
		model = single_p_QL,
		map_data_to_model = map_data_to_single_p_QL,
		param_names = [:a, :ρ],
		initV = initV,
		summary_method = summary_method
	)
end

"""
    sum_FI_for_feedback_sequence(; task::AbstractDataFrame, ρ_vals::AbstractVector, a_vals::AbstractVector, initV::Float64, n_blocks::Int64 = 200, within_summary_method::Function = det, across_summary_method::Function = median)

Compute and summarize the Fisher Information (FI) across multiple combinations of Q-learning parameters.

# Arguments
- `task::AbstractDataFrame`: The task data containing feedback sequence.
- `ρ_vals::AbstractVector`: Vector of reward sensitivity values to compute FI over.
- `a_vals::AbstractVector`: Vector of learning rate values for action selection.
- `initV::Float64`: Initial value for Q-values.
- `n_blocks::Int64`: Number of blocks for simulation. Defaults to 200.
- `within_summary_method::Function`: Method to summarize the FI matrix (default: determinant `det`).
- `across_summary_method::Function`: Method to summarize FI across different parameter combinations (default: `median`).

# Returns
- A scalar summarizing the FI over the parameter space using the specified across-summary method.
"""
function sum_FI_for_feedback_sequence(;
	task::AbstractDataFrame,
	ρ_vals::AbstractVector,
	a_vals::AbstractVector,
	initV::Union{Float64, Nothing} = nothing,
	n_blocks::Int64 = 200,
	within_summary_method::Function = det,
	across_summary_method::Function = median
)
	
	FIs = Matrix{Float64}(undef, length(ρ_vals), length(a_vals))
	
	for (i, ρ) in enumerate(ρ_vals)
		for (j, a) in enumerate(a_vals)
			FIs[i, j] = FI_for_feedback_sequence(;
				task = task,
				ρ = ρ,
				a = a,
				initV = initV,
				summary_method = within_summary_method,
				n_blocks = n_blocks
			) / n_blocks
		end
	end

	return across_summary_method(FIs)
end

"""
    compute_save_FIs_for_all_seqs(; n_trials::Int64, n_confusing::Int64, fifty_high::Bool, FI_res::Int64 = 6)

Compute and save Fisher Information (FI) for all possible feedback sequences with a specified number of confusing trials and reward magnitude distributions.

# Arguments
- `n_trials::Int64`: The total number of trials in each sequence.
- `n_confusing::Int64`: The number of confusing feedback trials in each sequence.
- `fifty_high::Bool`: Whether to include high-magnitude (1.0) and low-magnitude (0.01) rewards, or uniform magnitudes (0.5).
- `FI_res::Int64`: The resolution of Fisher Information computation, i.e., the number of grid points for parameter values (default: 6).
- `initV::Float64`: The initial value of the Q-values for both choices.

# Details
This function computes FI for all possible combinations of:
1. Sequences of common (true) and confusing (false) feedback, constrained by the number of confusing trials (`n_confusing`).
2. Sequences of feedback magnitudes, either alternating between high (1.0) and low (0.01) or fixed at 0.5 depending on the value of `fifty_high`.

The Fisher Information is computed over a grid of values for the parameters `ρ` (learning rate) and `a` (exploration-exploitation balance) using the `sum_FI_for_feedback_sequence` function. The results are saved to a file for future use.

# Returns
- `FIs`: A 3D matrix where each entry corresponds to the FI for a specific combination of feedback sequence and magnitude sequence.
- `common_seqs`: All possible sequences of common/confusing feedback.
- `magn_seq`: All possible sequences of feedback magnitudes.
"""
function compute_save_FIs_for_all_seqs(;
	n_trials::Int64,
	n_confusing::Int64,
	fifty_high::Bool,
	FI_res::Int64 = 6,
    initV::Union{Float64, Nothing} = nothing
)

	filename = "saved_models/FI/FIs_$(n_trials)_$(n_confusing)_$(fifty_high).jld2"

	if !isfile(filename)
		# All possible sequences of confusing feedback
		common_seqs = collect(
			multiset_permutations(
				vcat(
					fill(false, n_confusing), 
					fill(true, n_trials - n_confusing)
				),
				n_trials
			)
		)

		# All possible sequences of magnitude
		magn_seq = collect(
			multiset_permutations(
				vcat(
					fill(.5, div(n_trials, 2)), 
					fill(fifty_high ? 1. : 0.01, div(n_trials, 2))
				),
				n_trials
			)
		)

		# Compute FIs ---------------

		# Preallocate
		lk = ReentrantLock()
		FIs = fill(fill(-99., FI_res, FI_res), length(common_seqs), length(magn_seq))

		# Compute in parallel
		Threads.@threads for i in eachindex(common_seqs)
			for (j, magn) in enumerate(magn_seq)
				thisFI = sum_FI_for_feedback_sequence(;
						task = sequence_to_task_df(;
							feedback_common = common_seqs[i],
							feedback_magnitude_high = fifty_high ? magn : fill(1., n_trials),
							feedback_magnitude_low = fifty_high ? fill(0.01, n_trials) : magn
						),
						ρ_vals = range(1., 10., length = FI_res),
						a_vals = range(-1.5, 1.5, length = FI_res),
						initV = initV,
						across_summary_method = identity,
						n_blocks = 200
					) 

				lock(lk) do
					FIs[i,j] = thisFI
				end
			end
		end

		FIs = zscore_avg_matrices(FIs)

		# Save
		JLD2.@save filename FIs common_seqs magn_seq

	else
		JLD2.@load filename FIs common_seqs magn_seq
	end

	return FIs, common_seqs, magn_seq
end

"""
    optimize_FI_distribution(; n_wanted::Vector{Int64}, FIs::Vector{Matrix{Float64}}, common_seqs::Vector{Vector{Vector{Bool}}}, magn_seqs::Vector{Vector{Vector{Float64}}}, ω_FI::Float64, filename::String)

Optimize the selection of sequences to maximize Fisher Information (FI) while maintaining uniform distributions of confusing feedback trials and feedback magnitude across trial positions.

# Arguments
- `n_wanted::Vector{Int64}`: The number of sequences to select from each category.
- `FIs::Vector{Matrix{Float64}}`: Fisher Information matrices for each sequence in each category.
- `common_seqs::Vector{Vector{Vector{Bool}}}`: Sequences of common (vs. confusing) feedback positions for each category.
- `magn_seqs::Vector{Vector{Vector{Float64}}}`: Sequences of feedback magnitudes for each category.
- `ω_FI::Float64`: The weight assigned to maximizing FI relative to uniformity of distributions.
- `filename::String`: The file name to save or load the results of the optimization.

# Details
This function formulates and solves an optimization problem that selects a set of feedback sequences to maximize Fisher Information, subject to the constraint that the distribution of confusing feedback and feedback magnitude remains uniform across trial positions. The function ensures that each common feedback sequence is chosen only once across related categories and enforces the selection of a desired number of sequences for each category.

The objective function balances maximizing Fisher Information and minimizing deviations in the proportion of common feedback and the mean feedback magnitude across trial positions, controlled by the weight parameter `ω_FI`. A higher `ω_FI` prioritizes FI maximization, while a lower value emphasizes distribution uniformity.

The selected sequence indices are either saved to or loaded from the specified file, depending on whether the file already exists.

# Returns
- `selected_idx`: Indices of the selected sequences for each category.
"""
function optimize_FI_distribution(;
	n_wanted::Vector{Int64}, # How many sequences wanted of each category
	FIs::Vector{Matrix{Float64}}, # Fisher information for all the sequences in each category
	common_seqs::Vector{Vector{Vector{Bool}}}, # Sequences of common feedback position in each category
	magn_seqs::Vector{Vector{Vector{Float64}}}, # Sequences of feedback magnitude in each category
	ω_FI::Float64, # Weight of FI vs uniform distributions.
	filename::String # Filename to save results
)

	@assert all([size(FIs[s]) == size(FIs[s+1]) for s in 2:2:length(FIs)]) "Assuming inputs are arranged in pairs matching in sizes, except first stimulus. Common sequences will be constrained to be only chosen once across pairs"

	@assert all([size(FIs[s], 1) == length(common_seqs[s]) for s in eachindex(FIs)]) "FIs and common_seqs not matching in size"

	@assert all([size(FIs[s], 2) == length(magn_seqs[s]) for s in eachindex(FIs)]) "FIs and magn_seqs not matching in size"

	if !isfile(filename)
	
		# Number of available sequences per dimension, category
		n_common_seqs = [length(cmn) for cmn in common_seqs]
		n_magn_seqs = [length(magn) for magn in magn_seqs]
		n_cats = length(FIs)
	
		# Number of trials in block
		n_trials = length(common_seqs[1][1])
	
		# Proportion of common feedback trials
		common_prop = mean(vcat(common_seqs...))
	
		# Maximum magnitude for normalizing
		magn_max = maximum(vcat(vcat(magn_seqs...)...))
	
		# Average magnitude
		magn_avg = mean(vcat(magn_seqs...)) ./ magn_max
	
		# Maximum FIs for normalization
		FIs_max = maximum(vcat([vec(fi) for fi in FIs]...))
	
		# # # Create the optimization model
		model = Model(HiGHS.Optimizer)
	
		set_time_limit_sec(model, 720.)
	
		# # Decision variables: x[v] is 1 if vector v is selected, 0 otherwise
		xs = [@variable(model, [1:c, 1:m], Bin) 
			for (c,m) in zip(n_common_seqs, n_magn_seqs)]
	
		# Mean vector variables: mu_common[i] is the proportion of common feedback of selected vectors at position i
		@variable(model, mu_common[i = 1:n_trials])
	
		# Mean vector variables: mu_magn[i] is the mean magnitude of selected vectors at position i
		@variable(model, mu_magn[i = 1:n_trials])
	
	
		# Constraint: Exactly n_wanted vectors should be selected
		for s in eachindex(xs)
			@constraint(model, sum(xs[s]) == n_wanted[s])
	
			if iseven(s) # Make sure no common sequence is chosen twice across variations of magnitude
				# Each row (sequence) is selected exactly once across all columns
				for i in 1:n_common_seqs[s]
				    @constraint(model, sum(xs[s][i,j] for j in 1:n_magn_seqs[s]) + sum(xs[s+1][i,j] for j in 1:n_magn_seqs[s+1]) <= 1)  # Each row selected at most once
				end
			end
				
			# Each column (magnitude) is selected exactly once across all rows
			for j in 1:n_magn_seqs[s]
				@constraint(model, sum(xs[s][i,j] for i in 1:n_common_seqs[s]) <= 1)  # Each column selected at most once
			end
			
		end
	
		# # Constraints to calculate the mean vector
		for i in 1:n_trials
	
			# Compute average common feedback
			@constraint(
				model, 
				mu_common[i] == sum([common_seqs[s][v][i] * sum(xs[s][v,j] for j in 1:n_magn_seqs[s]) for s in eachindex(xs) for v in 1:n_common_seqs[s]]) / sum(n_wanted)
			)
	
			# Compute average magnitude
			@constraint(
				model, 
				mu_magn[i] == sum([magn_seqs[s][v][i] * sum(xs[s][j,v] for j in 1:n_common_seqs[s]) for s in eachindex(xs) for v in 1:n_magn_seqs[s]]) / (sum(n_wanted) * magn_max)
			)
		end
	
		# Auxiliary variables for absolute deviations
		@variable(model, common_abs_dev[1:n_trials])
		@variable(model, magn_abs_dev[1:n_trials])
	
		# Constraints for absolute deviations
		for i in 1:n_trials
	
			# Proportion of common feedback
			@constraint(model, common_abs_dev[i] >= mu_common[i] - common_prop[i])
			@constraint(model, common_abs_dev[i] >= common_prop[i] - mu_common[i])
	
			# Average magnitude
			@constraint(model, magn_abs_dev[i] >= (mu_magn[i] - magn_avg[i]))
			@constraint(model, magn_abs_dev[i] >= (magn_avg[i] - mu_magn[i]))
	
		end
	
	
		# Objective: Maximize the total score and minimize the mean vector deviation
		@objective(
			model, 
			Max, 
			ω_FI * sum(sum(FIs[s][i,j] * xs[s][i,j] for i in 1:n_common_seqs[s] 
				for j in 1:n_magn_seqs[s]) for s in 1:n_cats) / (sum(n_wanted) * FIs_max)  -
			((1 - ω_FI) * (mean(common_abs_dev[i] for i in 1:n_trials) 
			+ mean(magn_abs_dev[i] for i in 1:n_trials)
			) / 2
			)
		)
	
		# Solve the optimization problem
		set_silent(model)
		optimize!(model)
	
		# Check the status of the solution
		status = termination_status(model)
		if status == MOI.OPTIMAL
			@info "Optimal solution found"
		elseif status == MOI.INFEASIBLE_OR_UNBOUNDED
			@info "Problem infeasible or unbounded"
		else
			@info "Solver terminated with status: $status"
		end
	
		# Retrieve the solution
		selected_idx = [[(i,j) for i in 1:n_common_seqs[s] 
				for j in 1:n_magn_seqs[s] if value(xs[s][i,j]) > 0.5] for s in 1:n_cats]

		JLD2.@save filename selected_idx
	else
		JLD2.@load filename selected_idx
	end
		
	
	# Print FI stats
	avg_FI = [mean(FIs[s][idx...]) for s in eachindex(FIs) for idx in selected_idx[s]]

	quantile_F = [round(mean(vec(FIs[s]) .< avg_FI[s]) .* 100, digits = 2) for s in eachindex(FIs)]

	@info "FI quantile for each category: $quantile_F"
	
	# Compute and print distributions stats
	chosen_common = hcat([common_seqs[s][idx[1]] for s in eachindex(common_seqs) for idx in selected_idx[s]]...)

	common_per_pos = vec(sum(chosen_common, dims = 2))

	@info "Number of common feedback trials per trial position $(common_per_pos)"

	chosen_magn = hcat([magn_seqs[s][idx[2]] for s in eachindex(magn_seqs) for idx in selected_idx[s]]...)

	EV_per_pos = vec(mean(chosen_magn, dims = 2))

	@info "EV per trial position $(round.(EV_per_pos, digits = 2))"
		
	return selected_idx, common_per_pos, EV_per_pos
end

"""
    sequence_to_task_df(; feedback_common::Vector{Bool}, feedback_magnitude_high::Vector{Float64}, feedback_magnitude_low::Vector{Float64})

Create a task DataFrame for the PILT based on a given feedback sequence.

# Arguments
- `feedback_common::Vector{Bool}`: A sequence indicating whether feedback is common (true) or confusing (false).
- `feedback_magnitude_high::Vector{Float64}`: A sequence of high-magnitude feedback values.
- `feedback_magnitude_low::Vector{Float64}`: A sequence of low-magnitude feedback values.

# Returns
- A DataFrame with columns for `block`, `feedback_optimal`, `feedback_suboptimal`, and `trial`, representing the task structure.
"""
function sequence_to_task_df(;
	feedback_common::Vector{Bool}, # Sequence of common (true) / confusing (false) feedback
	feedback_magnitude_high::Vector{Float64}, # Sequence of high magnitude feedback,
	feedback_magnitude_low::Vector{Float64}, # Sequence of low magnitude feedback
)
	# Check inputs
	@assert length(feedback_common) == length(feedback_magnitude_high)
	@assert length(feedback_magnitude_low) == length(feedback_magnitude_high)
	@assert sum(feedback_magnitude_high) > sum(feedback_magnitude_low)

	n_trials = length(feedback_common)

	# Build data frame
	task = DataFrame(
		block = fill(1, n_trials),
		feedback_optimal = ifelse.(
			feedback_common, 
			feedback_magnitude_high, 
			feedback_magnitude_low
		), # Swap feedback magnitude on confusing trials
		feedback_suboptimal = ifelse.(
			.!feedback_common, 
			feedback_magnitude_high, 
			feedback_magnitude_low
		),
		trial = 1:n_trials
	)

	return task

end

"""
    shuffled_fill(values::AbstractVector, n::Int64; random_seed::Int64 = 0)

Generate a shuffled vector by filling `n` trials with the values from `values`, distributing them as evenly as possible.

# Arguments
- `values::AbstractVector`: The set of values to distribute across trials.
- `n::Int64`: The total number of trials.
- `random_seed::Int64`: Seed for random number generation (default: 0).

# Returns
- A shuffled vector of length `n` with the values from `values` distributed as evenly as possible.
"""
function shuffled_fill(
	values::AbstractVector, # Values to fill vector
	n::Int64; # How many trials overall
	random_seed::Int64 = 0
)	
	# Create vector with as equal number of appearance for each value as possible
	shuffled_values = shuffle(Xoshiro(random_seed), values)
	unshuffled_vector = collect(Iterators.take(Iterators.cycle(shuffled_values), n))

	return shuffle(Xoshiro(random_seed + 1), unshuffled_vector)
end

"""
    zscore_avg_matrices(matrices::Array{Matrix{Float64}})

Standardize each element in a set of matrices by converting them to z-scores element-wise, and then compute the average matrix.

# Arguments
- `matrices::Array{Matrix{Float64}}`: An array of matrices, all of the same size.

# Returns
- `avg_mat`: The average matrix after z-scoring across all input matrices.
"""
function zscore_avg_matrices(matrices::Array{Matrix{Float64}})
	
    # Ensure all matrices are of the same size
    m, n = size(matrices[1])
    for mat in matrices
        @assert size(mat) == (m, n) "All matrices must be of the same dimensions"
    end

    # Initialize arrays for means and standard deviations
    mean_vals = zeros(m, n)
    std_vals = zeros(m, n)
	
    # Calculate means and standard deviations across matrices by position
    for i in 1:m, j in 1:n
        values = [mat[i, j] for mat in matrices]
        mean_vals[i, j] = mean(values)
        std_vals[i, j] = std(values, corrected=true)
    end

    # Z-score each element in each matrix
    zscored_matrices = [ (mat .- mean_vals) ./ std_vals for mat in matrices ]
	
	# # Average each matrix
	avg_mat = [mean(m) for m in zscored_matrices]

end

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