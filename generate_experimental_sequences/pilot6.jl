### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ 99c994e4-9c36-11ef-2c8f-d5829be639eb
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

# ╔═╡ 65ec8b8f-9eba-467b-bb19-9f0c72b8933e
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

# ╔═╡ ea917db6-ec27-454f-8b4e-9df65d65064b
md"""## PILT"""

# ╔═╡ 381e61e2-7d51-4070-8ad1-ce9e63015eb6
# PILT parameters
begin
	# PILT Parameters
	PILT_blocks_per_valence = 8
	PILT_trials_per_block = 10
	
	PILT_total_blocks = PILT_blocks_per_valence * 2
	PILT_n_confusing = vcat([0, 1, 1], fill(2, PILT_total_blocks - 3))
		
	# Post-PILT test parameters
	test_n_blocks = 2
end

# ╔═╡ 687f5ae6-86c6-449f-86f5-5ed359e6d580
function has_consecutive_repeats(vec::Vector, n::Int = 3)
    count = 1
    for i in 2:length(vec)
        if vec[i] == vec[i - 1]
            count += 1
            if count > n
                return true
            end
        else
            count = 1
        end
    end
    return false
end

# ╔═╡ 31128edd-5d2d-49e9-8f65-842bb42639f9
# Assign valence and set size per block
PILT_block_attr = let random_seed = 3
	
	# # All combinations of set sizes and valence
	block_attr = DataFrame(
		session = repeat(1:2, inner = PILT_blocks_per_valence * 2),
		block = repeat(1:PILT_total_blocks, outer = 2),
		valence = repeat([1, -1], inner = PILT_blocks_per_valence, outer = 2),
		fifty_high = repeat([true, false], outer = PILT_blocks_per_valence * 2)
	)

	# Shuffle set size and valence, making sure valence is varied in first three blocks, and positive in the first
	rng = Xoshiro(random_seed)

	first_three_same = true
	first_block_punishement = true
	too_many_repeats = true
	while first_three_same || first_block_punishement || too_many_repeats

		DataFrames.transform!(
			groupby(block_attr, :session),
			:block => (x -> shuffle(rng, x)) => :block
		)
		
		sort!(block_attr, [:session, :block])

		# Compute criterion variables
		first_three_same = any(
			allequal.(
				[filter(x -> x.session == s, block_attr)[1:3, :valence] for s in 1:2]
			)
		)
		
		first_block_punishement = any(
			[filter(x -> x.session == s, block_attr).valence[1] == -1 for s in 1:2]
		)

		too_many_repeats = any([has_consecutive_repeats(filter(x -> x.session == s, block_attr).valence) for s in 1:2])
	end

	# Add n_confusing
	block_attr.n_confusing = repeat(PILT_n_confusing, 2)

	# Return
	block_attr
end

# ╔═╡ c7c5b78e-ad76-4877-8c80-af9151105544
# Create feedback sequences per pair
PILT_sequences, common_per_pos, EV_per_pos = 
	let random_seed = 0
	
	# Compute how much we need of each sequence category
	n_confusing_wanted = combine(
		groupby(PILT_block_attr, [:n_confusing, :fifty_high]),
		:block => length => :n
	)
	
	# Generate all sequences and compute FI
	FI_seqs = [compute_save_FIs_for_all_seqs(;
		n_trials = 10,
		n_confusing = r.n_confusing,
		fifty_high = r.fifty_high,
		initV = nothing
	) for r in eachrow(n_confusing_wanted)]

	# Unpack results
	common_seqs = [x[2] for x in FI_seqs]
	magn_seqs = [x[3] for x in FI_seqs]

	# Choose sequences optimizing FI under contraints
	chosen_idx, common_per_pos, EV_per_pos = optimize_FI_distribution(
		n_wanted = n_confusing_wanted.n,
		FIs = [x[1] for x in FI_seqs],
		common_seqs = common_seqs,
		magn_seqs = magn_seqs,
		ω_FI = 0.1,
		filename = "results/exp_sequences/pilot6_opt.jld2"
	)

	@assert length(vcat(chosen_idx...)) == nrow(PILT_block_attr) "Number of saved optimize sequences does not match number of sequences needed. Delete file and rerun."

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
		idx = repeat(1:n_sequences, inner = PILT_trials_per_block),
		sequence = repeat(vcat([1:length(x) for x in chosen_common]...), 
			inner = PILT_trials_per_block),
		trial = repeat(1:PILT_trials_per_block, n_sequences),
		feedback_common = vcat(vcat(chosen_common...)...),
		variable_magnitude = vcat(vcat(chosen_magn...)...)
	)

	# Create n_confusing and fifty_high varaibles
	DataFrames.transform!(
		groupby(task, :idx),
		:feedback_common => (x -> PILT_trials_per_block - sum(x)) => :n_confusing,
		:variable_magnitude => (x -> 1. in x) => :fifty_high
	)

	# Add sequnces variable to PILT_block_attr
	DataFrames.transform!(
		groupby(PILT_block_attr, [:n_confusing, :fifty_high]),
		:block => (x -> shuffle(rng, 1:length(x))) => :sequence
	)

	# Combine with block attributes
	task = innerjoin(
		task,
		PILT_block_attr,
		on = [:n_confusing, :fifty_high, :sequence],
		order = :left
	)

	@assert nrow(task) == length(vcat(vcat(chosen_common...)...)) "Problem with join operation"
	@assert nrow(unique(task[!, [:session, :block]])) == PILT_total_blocks * 2 "Problem with join operation"
		
	@assert mean(task.fifty_high) == 0.5 "Proportion of blocks with 50 pence in high magnitude option expected to be 0.5"

	# Sort by block
	sort!(task, [:session, :block, :trial])

	# Remove auxillary variables
	select!(task, Not([:sequence, :idx]))

	# Compute low and high feedback
	task.feedback_high = ifelse.(
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

	task.feedback_low = ifelse.(
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

# ╔═╡ e1b66454-7700-4d79-9df2-59e13bd031ee
# Assign stimulus images
PILT_stimuli = let random_seed = 0
	# Load stimulus names
	categories = shuffle(Xoshiro(random_seed), unique([replace(s, ".png" => "")[1:(end-1)] for s in 
		readlines("generate_experimental_sequences/pilot6_stim_list.txt")]))

	# Assign stimulus pairs
	stimuli = assign_stimuli_and_optimality(;
		n_phases = 2,
		n_pairs = fill(1, PILT_total_blocks),
		categories = categories,
		random_seed = random_seed
	)

	@info "Proportion of blocks on which the novel category is optimal : $(mean(stimuli.optimal_A))"

	rename!(stimuli, :phase => :session) # For compatibility with multi-phase sessions

	stimuli
end

# ╔═╡ 1cedb080-d46b-45a2-b781-aadb1d9a48d0
# Add stimulus assignments to sequences DataFrame, and assign right / left
PILT_task = let random_seed = 1

	# Join stimuli and sequences
	task = innerjoin(
		PILT_sequences,
		PILT_stimuli,
		on = [:session, :block],
		order = :left
	)

	@assert nrow(task) == nrow(PILT_sequences) "Problem in join operation"

	# Assign right / left, equal proportions within each pair
	rng = Xoshiro(random_seed)

	DataFrames.transform!(
		groupby(task, [:session, :block]),
		:block => 
			(x -> shuffled_fill([true, false], length(x); rng = rng)) =>
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

# ╔═╡ 6507d118-6977-4023-b43c-0d483a720f96
# Validate task DataFrame
let task = PILT_task
	@assert maximum(task.block) == length(unique(task.block)) "Error in block numbering"

	@assert all(combine(groupby(task, :session), 
		:block => issorted => :sorted).sorted) "Task structure not sorted by block"

	@assert all(combine(groupby(task, [:session, :block]), 
		:trial => issorted => :sorted).sorted) "Task structure not sorted by trial number"

	@assert all(sign.(task.valence) == sign.(task.feedback_right)) "Valence doesn't match feedback sign"

	@assert all(sign.(task.valence) == sign.(task.feedback_left)) "Valence doesn't match feedback sign"

	@assert sum(unique(task[!, [:session, :block, :valence]]).valence) == 0 "Number of reward and punishment blocks not equal"

	@info "Overall proportion of common feedback: $(round(mean(task.feedback_common), digits = 2))"

	@assert all((task.variable_magnitude .== abs.(task.feedback_right)) .| 
		(task.variable_magnitude .== abs.(task.feedback_left))) ":variable_magnitude, which is used for sequnece optimization, doesn't match end result column :feedback_right no :feedback_left"

	# Count losses to allocate coins in to safe for beginning of task
	worst_loss = filter(x -> x.valence == -1, task) |> 
		df -> ifelse.(
			df.feedback_right .< df.feedback_left, 
			df.feedback_right, 
			df.feedback_left) |> 
		countmap

	@info "Worst possible loss in this task is of these coin numbers: $worst_loss"

end

# ╔═╡ a64f05e8-9eb6-435c-850e-b69c04c6721b
let
	save_to_JSON(PILT_task, "results/pilot6_PILT.json")
	CSV.write("results/pilot6_PILT.csv", PILT_task)
end

# ╔═╡ b05f81e5-837d-4a7d-8b6a-73628568e106
# Visualize PILT seuqnce
let task = PILT_task

	f = Figure(size = (700, 300))

	# Proportion of confusing by trial number
	confusing_location = combine(
		groupby(task, [:session, :trial]),
		:feedback_common => (x -> mean(.!x)) => :feedback_confusing
	)

	mp1 = data(confusing_location) * mapping(
		:trial => "Trial", 
		:feedback_confusing => "Prop. confusing feedback",
		color = :session => nonnumeric => "Session",
		group = :session => nonnumeric
	) * visual(ScatterLines)

	plt1 = draw!(f[1,1], mp1)

	legend!(
		f[1,1], 
		plt1,
		tellwidth = false,
		tellheight = false,
		valign = 1.2,
		halign = 0.,
		framevisible = false
	)

	# Plot confusing trials by block
	fp = insertcols(
		task,
		:color => ifelse.(
			task.feedback_common,
			(task.valence .+ 1) .÷ 2,
			fill(3, nrow(task))
		)
	)

	for (i, s) in enumerate(unique(fp.session))
		mp = data(filter(x -> x.session == s, fp)) * mapping(
			:trial => "Trial",
			:block => "Block",
			:color
		) * visual(Heatmap)

		draw!(f[1,i+1], mp, axis = (; yreversed = true, subtitle = "Session $i"))
	end
	


	# save("results/eeg_trial_plan.png", f, pt_per_unit = 1)

	f

end

# ╔═╡ Cell order:
# ╠═99c994e4-9c36-11ef-2c8f-d5829be639eb
# ╠═65ec8b8f-9eba-467b-bb19-9f0c72b8933e
# ╟─ea917db6-ec27-454f-8b4e-9df65d65064b
# ╠═381e61e2-7d51-4070-8ad1-ce9e63015eb6
# ╠═687f5ae6-86c6-449f-86f5-5ed359e6d580
# ╠═31128edd-5d2d-49e9-8f65-842bb42639f9
# ╠═c7c5b78e-ad76-4877-8c80-af9151105544
# ╠═e1b66454-7700-4d79-9df2-59e13bd031ee
# ╠═1cedb080-d46b-45a2-b781-aadb1d9a48d0
# ╠═6507d118-6977-4023-b43c-0d483a720f96
# ╠═a64f05e8-9eb6-435c-850e-b69c04c6721b
# ╠═b05f81e5-837d-4a7d-8b6a-73628568e106
