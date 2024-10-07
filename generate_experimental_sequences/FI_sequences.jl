### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ 62fb2c44-70f7-11ef-2499-9d1ed7c02f46
begin
	cd("/home/jovyan")
	import Pkg
	
	# activate the shared project environment
    Pkg.activate("$(pwd())/relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	using CairoMakie, Random, DataFrames, Distributions, StatsBase,
		ForwardDiff, LinearAlgebra, JLD2, FileIO, CSV, Dates, JSON, RCall, Turing, ParetoSmooth, MCMCDiagnosticTools, Printf, Combinatorics, JuMP, HiGHS
	using LogExpFunctions: logistic, logit

	Turing.setprogress!(false)

	include("$(pwd())/PILT_models.jl")
	include("$(pwd())/sample_utils.jl")
	include("$(pwd())/stats_utils.jl")
	include("$(pwd())/plotting_utils.jl")
	include("$(pwd())/single_p_QL.jl")
	include("$(pwd())/fetch_preprocess_data.jl")
	nothing
end

# ╔═╡ a4db20c5-a30b-454f-a04b-ffbe25459e07
aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])

# ╔═╡ c1b8f566-a260-4e9d-9712-61b9b53fb74c
# Sample datasets from prior
begin	
	prior_sample = let
		# Load sequence from file
		task = task_vars_for_condition("00")
	
		prior_sample = simulate_single_p_QL(
			2;
			block = task.block,
			outcomes = task.outcomes,
			initV = fill(aao, 1, 2),
			random_seed = 0,
			prior_ρ = truncated(Normal(0., 2.), lower = 0.),
			prior_a = Normal()
		)
	
	
		leftjoin(prior_sample, 
			task.task[!, [:block, :trial, :feedback_optimal, :feedback_suboptimal]],
			on = [:block, :trial]
		)
	
	end

end

# ╔═╡ 2b7e29ee-742c-4afd-95b9-602b3e42a6f2
function map_data_to_single_p_QL(
	data::AbstractDataFrame
)

	tdata = dropmissing(data)

	return (;
		block = collect(tdata.block),
		choice = tdata.choice,
		outcomes = hcat(tdata.feedback_suboptimal, tdata.feedback_optimal)
	)

end

# ╔═╡ a16f068f-a86d-4d91-990b-bbba4a0da511
function FI_for_feedback_sequence(;
	task::AbstractDataFrame,
	ρ::Float64,
	a::Float64,
	initV::Float64,
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
		initV = fill(aao, 1, 2),
		summary_method = summary_method
	)
end

	

# ╔═╡ bb3917a6-50a3-4b07-ba23-c64e0efe4097
function sum_FI_for_feedback_sequence(;
	task::AbstractDataFrame,
	ρ_vals::AbstractVector,
	a_vals::AbstractVector,
	initV::Float64,
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

# ╔═╡ e5395ac6-7525-445d-8e7e-d161c9f74f93
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

# ╔═╡ f5b0c228-d73a-4001-b272-b00e6fc2446c
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

# ╔═╡ cb26d70f-1d42-4ff5-9803-46af4e48127a
# compute_save_FIs_for_all_seqs(;
# 	n_trials = 10,
# 	n_confusing = 2,
# 	fifty_high = false
# )

# ╔═╡ e21e131f-75fd-4d4a-9d84-e06a124783c9
# compute_save_FIs_for_all_seqs(;
# 	n_trials = 10,
# 	n_confusing = 2,
# 	fifty_high = true
# )

# ╔═╡ 15ff66ec-ade9-4e94-be2f-643e0c50cdeb
# compute_save_FIs_for_all_seqs(;
# 	n_trials = 10,
# 	n_confusing = 1,
# 	fifty_high = false
# )

# ╔═╡ 753d4d21-1346-42ed-860a-b04019fa0e74
function optimize_FI_distribution(;
	n_wanted::Vector{Int64}, # How many sequences wanted of each category
	FIs::Vector{Matrix{Float64}}, # Fisher information for all the sequences in each category
	common_seqs::Vector{Vector{Vector{Bool}}}, # Sequences of common feedback position in each category
	magn_seqs::Vector{Vector{Vector{Float64}}}, # Sequences of feedback magnitude in each category
	ω_FI::Float64 # Weight of FI vs uniform distributions.
)
	
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

	# # # Create the optimization model
	model = Model(HiGHS.Optimizer)

	# # Decision variables: x[v] is 1 if vector v is selected, 0 otherwise
	xs = [@variable(model, [1:c, 1:m], Bin) 
		for c in n_common_seqs for m in n_magn_seqs]

	# Mean vector variables: mu_common[i] is the proportion of common feedback of selected vectors at position i
	@variable(model, mu_common[i = 1:n_trials])

	# Mean vector variables: mu_magn[i] is the mean magnitude of selected vectors at position i
	@variable(model, mu_magn[i = 1:n_trials])


	# Constraint: Exactly n_wanted vectors should be selected
	for s in eachindex(xs)
		@constraint(model, sum(xs[s]) == n_wanted[s])

	# Each row (sequence) is selected exactly once across all columns
	for i in 1:n_common_seqs[s]
	    @constraint(model, sum(xs[s][i,j] for j in 1:n_magn_seqs[s]) <= 1)  # Each row selected at most once
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
			mu_common[i] == sum([common_seqs[s][v][i] * xs[s][v] for s in eachindex(xs) for v in 1:n_common_seqs[s]]) / sum(n_wanted)
		)

		# Compute average magnitude
		@constraint(
			model, 
			mu_magn[i] == sum([magn_seqs[s][v][i] * xs[s][v] for s in eachindex(xs) for v in 1:n_magn_seqs[s]]) / (sum(n_wanted) + magn_max)
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
			for j in 1:n_magn_seqs[s]) for s in 1:n_cats)  -
		((1 - ω_FI) * (mean(common_abs_dev[i] for i in 1:n_trials) + mean(magn_abs_dev[i] for i in 1:n_trials)) / 2)
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

	return selected_idx
end

# ╔═╡ 2fc7e7bc-8940-4f98-9225-9535b878bd76
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

# ╔═╡ 269bf7ae-5a00-40fd-afc0-b45c92c43597
function compute_save_FIs_for_all_seqs(;
	n_trials::Int64,
	n_confusing::Int64,
	fifty_high::Bool,
	FI_res::Int64 = 6
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
						initV = aao,
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

# ╔═╡ b76a5feb-e86d-46fb-8c92-8dc662248f6b
let
	FIs, common_seqs, magn_seqs = compute_save_FIs_for_all_seqs(;
		n_trials = 10,
		n_confusing = 1,
		fifty_high = true
	)

	size(FIs), size(common_seqs), size(magn_seqs)

	optimize_FI_distribution(
		n_wanted = [5],
		FIs = [FIs],
		common_seqs = [common_seqs],
		magn_seqs = [magn_seqs],
		ω_FI = 0.35
	)
end

# ╔═╡ 1e9e4f9c-6a61-4f0d-a17b-08ba5875616f
let n_trials = 10,
	n_confusing = 2,
	FI_res = 6

	# All possible sequences of confusing feedback
	sequences = collect(
		multiset_permutations(
			vcat(
				fill(false, n_confusing), 
				fill(true, n_trials - n_confusing)
			),
			n_trials
		)
	)

	# Compute FIs ---------------
	lk = ReentrantLock()
	FIs = fill(fill(-99., FI_res, FI_res), length(sequences))
	
	# Preallocate
	Threads.@threads for i in eachindex(sequences)
		thisFI = sum_FI_for_feedback_sequence(;
				task = sequence_to_task_df(;
					feedback_common = sequences[i],
					feedback_magnitude_high = fill(1., n_trials),
					feedback_magnitude_low = fill(0.01, n_trials)
				),
				ρ_vals = range(1., 10., length = FI_res),
				a_vals = range(-1.5, 1.5, length = FI_res),
				initV = aao,
				across_summary_method = identity,
				n_blocks = 200
			)

		lock(lk) do
			FIs[i] = thisFI
		end
	end

	FIs = zscore_avg_matrices(FIs)

	# Sort
	sorted_FIs = sort(FIs)
	sorted_seq = hcat(sequences[sortperm(FIs)]...) 

	# Plot heatmap
	f = Figure()

	ax = Axis(
		f[1,1],
		xlabel = "Sequence",
		ylabel = "FI",
		yticks = 1:2:length(sequences),
		ytickformat = 
			values -> ["$(round(sorted_FIs[round(Int64, v)], digits = 2))" 
				for v in values]
	)

	hidexdecorations!(ax, label = false)

	heatmap!(ax, sorted_seq)
	
	f
		
end

# ╔═╡ 6d9999c8-5ca5-4bee-986f-09d6b96fd1f6
binomial(10, 5) * 10 / 60

# ╔═╡ 5dbe7bc7-f348-4efa-91f3-17a1261a4e78
# Look at FI across parameter range for one sequence
let n_trials = 10,
	FI_res = 10

	# Create task dataframe
	task = sequence_to_task_df(;
		feedback_common = shuffled_fill([true, false], n_trials),
		feedback_magnitude_high = fill(1., n_trials),
		feedback_magnitude_low = shuffled_fill([0.01, 0.5], n_trials)
	)

	# Compute FI for each parameter combination
	FIs = sum_FI_for_feedback_sequence(;
		task = task,
		ρ_vals = range(0., 10., length = FI_res),
		a_vals = range(-2., 2., length = FI_res),
		initV = aao,
		across_summary_method = identity
	)

	println(mean(FIs))

	# Plot
	f = Figure()

	ax = Axis(
		f[1,1],
		aspect = 1
	)

	heatmap!(
		ax,
		range(0., 10., length = 15),
		range(-2., 2., length = 15),
		FIs
	)

	ax_d = Axis(
		f[1,2]
	)

	hist!(
		ax_d,
		vec(FIs)
	)

	f
end

# ╔═╡ Cell order:
# ╠═62fb2c44-70f7-11ef-2499-9d1ed7c02f46
# ╠═a4db20c5-a30b-454f-a04b-ffbe25459e07
# ╠═c1b8f566-a260-4e9d-9712-61b9b53fb74c
# ╠═2b7e29ee-742c-4afd-95b9-602b3e42a6f2
# ╠═a16f068f-a86d-4d91-990b-bbba4a0da511
# ╠═bb3917a6-50a3-4b07-ba23-c64e0efe4097
# ╠═e5395ac6-7525-445d-8e7e-d161c9f74f93
# ╠═f5b0c228-d73a-4001-b272-b00e6fc2446c
# ╠═cb26d70f-1d42-4ff5-9803-46af4e48127a
# ╠═e21e131f-75fd-4d4a-9d84-e06a124783c9
# ╠═15ff66ec-ade9-4e94-be2f-643e0c50cdeb
# ╠═b76a5feb-e86d-46fb-8c92-8dc662248f6b
# ╠═753d4d21-1346-42ed-860a-b04019fa0e74
# ╠═269bf7ae-5a00-40fd-afc0-b45c92c43597
# ╠═2fc7e7bc-8940-4f98-9225-9535b878bd76
# ╠═1e9e4f9c-6a61-4f0d-a17b-08ba5875616f
# ╠═6d9999c8-5ca5-4bee-986f-09d6b96fd1f6
# ╠═5dbe7bc7-f348-4efa-91f3-17a1261a4e78
