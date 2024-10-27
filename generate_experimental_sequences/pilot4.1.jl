### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ 3e76cffe-94a1-11ef-39a4-a5be586dbb76
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

# ╔═╡ 0b0e5ab3-781d-4a19-a00f-2fdbdffd60e5
# PILT parameters
begin
	# PILT Parameters
	set_sizes = [1,3,7]
	block_per_set = 6 # Including reward and punishment
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

# ╔═╡ 8de26077-086b-478f-80ff-776ff7688bdf
begin
	# Set theme
	inter_bold = assetpath(pwd() * "/fonts/Inter/Inter-Bold.ttf")
	
	th = Theme(
		font = "Helvetica",
		fontsize = 16,
		Axis = (
			xgridvisible = false,
			ygridvisible = false,
			rightspinevisible = false,
			topspinevisible = false,
			xticklabelsize = 14,
			yticklabelsize = 14,
			spinewidth = 1.5,
			xtickwidth = 1.5,
			ytickwidth = 1.5
		)
	)
	
	set_theme!(th)
end

# ╔═╡ a57e46e9-7a8f-4190-90d6-62272c8fb036
# Assign valence and set size per block
valence_set_size = let random_seed = 1
	
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

# ╔═╡ f9ee951c-5a34-451f-ad6f-081e6135f170
# # Create trial sequence
feedback_sequence = let random_seed = 1

	rng = Xoshiro(random_seed)

	# Create pair list
	task = combine(
		groupby(valence_set_size, 
			[:block, :n_pairs, :valence, :n_confusing]),
		:n_pairs => (x -> repeat(vcat([1:xi for xi in x]...), inner = trials_per_pair)) => :pair
	)

	# Add cumulative pair variable
	pairs = sort(unique(task[!, [:block, :pair, :n_pairs, :valence]]), [:block, :pair])
	pairs.cpair = 1:nrow(pairs)
	

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

	# Compute optimal and suboptimal feedback
	task.feedback_optimal = ifelse.(
		task.valence .> 0,
		fill(1., nrow(task)),
		fill(-0.01, nrow(task))
	)

	task.feedback_suboptimal = ifelse.(
		task.valence .> 0,
		fill(0.01, nrow(task)),
		fill(-1, nrow(task)),
	)

	task

end

# ╔═╡ af64279f-6b19-4129-9bb4-c109452de07d
# Assign stimulus images
stimuli = let random_seed = 0
	# Load stimulus names
	categories = shuffle(Xoshiro(random_seed), unique([replace(s, ".png" => "")[1:(end-1)] for s in 
		readlines("generate_experimental_sequences/pilot4.1_stim_list.txt")]))

	# Assign stimulus pairs
	stimuli = assign_stimuli_and_optimality(;
		n_phases = 1,
		n_pairs = valence_set_size.n_pairs,
		categories = categories,
		random_seed = random_seed
	)

	rename!(stimuli, :phase => :session) # For compatibility with multi-phase sessions
end

# ╔═╡ 8dfd1290-7988-4c61-8342-f4accf876bb6
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

# ╔═╡ 08639317-881c-4af9-bad5-9fdfc98519ed
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

	# Count losses to allocate coins in to safe for beginning of task
	worst_loss = filter(x -> x.valence < 0, task) |> 
		df -> ifelse.(
			df.feedback_right .< df.feedback_left, 
			df.feedback_right, 
			df.feedback_left) |> 
		countmap

	@info "Worst possible loss in this task is of these coin numbers: $worst_loss"

end

# ╔═╡ ccfb8336-4ea0-4474-99b9-cdbf02d8fd42
let
	save_to_JSON(task, "results/pilot4.1_pilt.json")
	CSV.write("results/pilot4_pilt.1.csv", task)
end

# ╔═╡ Cell order:
# ╠═3e76cffe-94a1-11ef-39a4-a5be586dbb76
# ╠═0b0e5ab3-781d-4a19-a00f-2fdbdffd60e5
# ╠═8de26077-086b-478f-80ff-776ff7688bdf
# ╠═a57e46e9-7a8f-4190-90d6-62272c8fb036
# ╠═f9ee951c-5a34-451f-ad6f-081e6135f170
# ╠═af64279f-6b19-4129-9bb4-c109452de07d
# ╠═8dfd1290-7988-4c61-8342-f4accf876bb6
# ╠═08639317-881c-4af9-bad5-9fdfc98519ed
# ╠═ccfb8336-4ea0-4474-99b9-cdbf02d8fd42
