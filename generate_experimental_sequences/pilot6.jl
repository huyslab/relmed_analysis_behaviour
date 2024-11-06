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
	PILT_test_n_blocks = 2
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
	


	save("results/pilot6_pilt_trial_plan.png", f, pt_per_unit = 1)

	f

end

# ╔═╡ f3391ced-7620-40b9-9c05-341e2c7106d2
md"""## Post-PILT test"""

# ╔═╡ 3268d6e0-9dd6-46b6-8234-c04bd63a48ef
function create_test_sequence(
	pilt_task::DataFrame;
	random_seed::Int64, 
	same_weight::Float64 = 6.5,
	test_n_blocks::Int64 = PILT_test_n_blocks
) 
	
	rng = Xoshiro(random_seed)

	# Extract stimuli and their common feedback from task structure
	stimuli = vcat([rename(
		pilt_task[pilt_task.feedback_common, [:session, :block, Symbol("stimulus_$s"), Symbol("feedback_$s")]],
		Symbol("stimulus_$s") => :stimulus,
		Symbol("feedback_$s") => :feedback
	) for s in ["right", "left"]]...)

	# Summarize magnitude per stimulus
	stimuli = combine(
		groupby(stimuli, [:session, :block, :stimulus]),
		:feedback => (x -> mean(unique(x))) => :magnitude
	)

	# Step 1: Identify unique stimuli
	unique_stimuli = unique(stimuli.stimulus)

	# Step 2: Define existing pairs
	create_pair_list(d) = [filter(x -> x.block == p, d).stimulus 
		for p in unique(stimuli.block)]

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

# ╔═╡ 160bb7c6-cdd4-4c42-87e6-6ccf987d3f7f
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

# ╔═╡ 2dd93d65-9b8f-4d6f-8a96-d9dd0c2ea345
PILT_test = let task = PILT_task
	
	# Find test sequence for each session
	PILT_test = []
	for s in 1:2
		test, pb, pv, nm = find_best_test_sequence(
			filter(x -> x.session == s, task),
			n_seeds = 10, # Number of random seeds to try
			same_weight = 0.01 # Weight reducing the number of same magntiude pairs
		) 

		# Add session variable
		insertcols!(test, 1, :session => s)

		push!(PILT_test, test)
	
		@info "Session $s: proportion of same block pairs: $pb"
		@info "Session $s: proportion of same valence pairs: $pv"
		@info "Session $s: number of same magnitude pairs: $nm"
	end

	# Concatenate together
	PILT_test = vcat(PILT_test...)

	# Create magnitude_pair variable
	PILT_test.magnitude_pair = [sort([r.magnitude_left, r.magnitude_right]) for r in eachrow(PILT_test)]

	@assert all([diff(collect(extrema(values(countmap(filter(x -> x.session == s, PILT_test).magnitude_pair)))))[1] == 1 for s in 1:2]) "Distribution across magnitude pairs could be more equal"

	@assert all([length(unique(filter(x -> x.session == 2, PILT_test).magnitude_pair)) for s in 1:2] .== binomial(length(unique(PILT_test.magnitude_left)), 2)) "Not all magnitude pairs present"

	PILT_test
end

# ╔═╡ edca0f2a-dba2-4882-944c-3b28e6c14a90
let
	save_to_JSON(PILT_test, "results/pilot6_PILT_test.json")
	CSV.write("results/pilot6_PILT_test.csv", PILT_test)
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
# ╟─f3391ced-7620-40b9-9c05-341e2c7106d2
# ╠═2dd93d65-9b8f-4d6f-8a96-d9dd0c2ea345
# ╠═edca0f2a-dba2-4882-944c-3b28e6c14a90
# ╠═3268d6e0-9dd6-46b6-8234-c04bd63a48ef
# ╠═160bb7c6-cdd4-4c42-87e6-6ccf987d3f7f
