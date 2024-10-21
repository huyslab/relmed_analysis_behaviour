### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ d5f0abd6-8cc2-11ef-0c92-7168bbb88d55
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

# ╔═╡ ca091875-8948-43ef-a27b-8e57948a17f1
# Reversal task parameters
begin
	rev_n_blocks = 30
	rev_n_trials = 50
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

# ╔═╡ 0d181fd0-4bfc-4c55-9723-da22fcffecb1
# PILT parameters
begin
	# PILT Parameters
	set_sizes = [1,3,5]
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

# ╔═╡ 2daa3d69-0e75-4234-bad4-50a6861eb54f
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

# ╔═╡ 87b95069-f729-40e8-8fa3-73741fe1c74f
md"""# PILT"""

# ╔═╡ fdcf523b-c623-4ecc-82f3-a2aa9fedc1d8
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

# ╔═╡ 60b14a3a-ba7a-467a-80f0-3e1bb1125bf9
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

# ╔═╡ 64b6dbec-653c-4741-b451-eb0b725e444f
categories = shuffle(unique([replace(s, ".png" => "")[1:(end-1)] for s in 
		readlines("generate_experimental_sequences/pilot4_stim_list.txt")])) 

# ╔═╡ 7f79e8e2-7a19-44dd-a785-992d09bb7aa2
"tent" in categories

# ╔═╡ bfd8026b-5abb-4be3-86d4-b68eb92a5e93
# Assign stimulus images
stimuli = let random_seed = 0
	# Load stimulus names
	categories = shuffle(unique([replace(s, ".png" => "")[1:(end-1)] for s in 
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

# ╔═╡ 6ef0d562-2b41-4867-ad21-00097ed65a62
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

# ╔═╡ 70869f63-ec33-4b09-9bd5-9a66360d72a3
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
			df.feedback_right < df.feedback_left, 
			df.feedback_right, 
			df.feedback_left) |> 
		countmap

	@info "Worst possible loss in this task is of these coin numbers: $worst_loss"

end

# ╔═╡ 436b3d1a-5a46-4f9c-ad52-fa08b9a488a3
let
	save_to_JSON(task, "results/pilot4_pilt.json")
	CSV.write("results/pilot4_pilt.csv", task)
end

# ╔═╡ d27b7e4e-4156-481e-b5b9-58a84920c640
md"""# Reversal task"""

# ╔═╡ b9db2e21-068b-4148-80b4-8c48edf8c4ec
function find_lcm_denominators(props::Vector{Float64})
	# Convert to rational numbers
	rational_props = rationalize.(props)
	
	# Extract denominators
	denominators = [denominator(r) for r in rational_props]
	
	# Compute the LCM of all denominators
	smallest_int = foldl(lcm, denominators)
end

# ╔═╡ e1745880-0a58-4cca-ab6a-0c98de5430a1
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
	open("results/pilot4_reversal_sequence.js", "w") do file
	    write(file, json_string)
	end

	feedback_optimal
end

# ╔═╡ efab3964-0ec5-4df1-870c-b20ce2882337
let

	f = Figure(size = (700, 300))

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

	mp3 = mapping(1:rev_n_blocks => "Block", rev_criterion => "# optimal choices)") * visual(ScatterLines)

	draw!(f[1,3], mp3, axis = (; 
		yticks = 3:8, 
		xticks = [1, 10, 20, 30], 
		subtitle = "Reversal criterion")
	)

	save("results/pilot4_reversal_sequence.png", f, pt_per_unit = 1)

	f

end

# ╔═╡ Cell order:
# ╠═d5f0abd6-8cc2-11ef-0c92-7168bbb88d55
# ╠═ca091875-8948-43ef-a27b-8e57948a17f1
# ╠═0d181fd0-4bfc-4c55-9723-da22fcffecb1
# ╠═2daa3d69-0e75-4234-bad4-50a6861eb54f
# ╟─87b95069-f729-40e8-8fa3-73741fe1c74f
# ╠═fdcf523b-c623-4ecc-82f3-a2aa9fedc1d8
# ╠═60b14a3a-ba7a-467a-80f0-3e1bb1125bf9
# ╠═64b6dbec-653c-4741-b451-eb0b725e444f
# ╠═7f79e8e2-7a19-44dd-a785-992d09bb7aa2
# ╠═bfd8026b-5abb-4be3-86d4-b68eb92a5e93
# ╠═6ef0d562-2b41-4867-ad21-00097ed65a62
# ╠═70869f63-ec33-4b09-9bd5-9a66360d72a3
# ╠═436b3d1a-5a46-4f9c-ad52-fa08b9a488a3
# ╟─d27b7e4e-4156-481e-b5b9-58a84920c640
# ╠═b9db2e21-068b-4148-80b4-8c48edf8c4ec
# ╠═e1745880-0a58-4cca-ab6a-0c98de5430a1
# ╠═efab3964-0ec5-4df1-870c-b20ce2882337
