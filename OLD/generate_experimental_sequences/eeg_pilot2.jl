### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ fee6c3cd-243b-46f8-aae1-c19b3effa88d
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

# ╔═╡ 2f4eb9a3-4a49-46eb-867c-61583a84aea0
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

# ╔═╡ 9425106c-bbef-11ef-3059-19f67d4354e0
# Reversal task parameters
begin
	rev_n_blocks = 40
	rev_n_trials = 80
	rev_prop_confusing = vcat([0, 0.1, 0.1, 0.2, 0.2], fill(0.3, rev_n_blocks - 5))
	rev_criterion = vcat(
		[8, 7, 6, 6, 5], 
		shuffled_fill(
			3:8, 
			rev_n_blocks - 5; 
			rng = Xoshiro(2)
		)
	)
end

# ╔═╡ 10343528-6896-4521-84ba-e0fbdc483b28
function find_lcm_denominators(props::Vector{Float64})
	# Convert to rational numbers
	rational_props = rationalize.(props)
	
	# Extract denominators
	denominators = [denominator(r) for r in rational_props]
	
	# Compute the LCM of all denominators
	smallest_int = foldl(lcm, denominators)
end

# ╔═╡ f21e290b-1a1b-4120-b00d-c07f03c248a4
# Reversal task structure
rev_feedback_optimal = let random_seed = 1

	# Compute minimal mini block length to accomodate proportions
	mini_block_length = find_lcm_denominators(rev_prop_confusing)
	@info "Randomizing in miniblocks of $mini_block_length trials"

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
	dist_diff = 11
	while isempty(feedback_optimal) || 
		!all([bl[1] != 0.01 for bl in feedback_optimal[1:6]]) || dist_diff > 2

		# Assign blocks
		feedback_optimal = [block_high_mag(p, rng) for p in rev_prop_confusing]

		# Check distribution of confusing feedback
		dist = (x -> permutedims(reshape(x, div(length(x), 10), 10), (2,1))).(feedback_optimal)

		dist = vcat(dist...)

		dist = vec(sum(dist .== 1., dims = 1))
		dist_diff = maximum(abs.(diff(dist)))
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
		
	# Write the JSON string to the file
	open("results/eeg_pilot2_reversal_sequence.js", "w") do file
	    write(file, json_string)
	end

	feedback_optimal
end

# ╔═╡ 67285fe5-9381-45dd-a31e-e9f2649c0498
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
			yticks = [1, 10, 20, 30, 40],
			subtitle = "Confusing Feedback"
		)
	)

	insertcols!(
		rev_confusing,
		:rel_trial => rev_confusing.trial .- div.(rev_confusing.trial, 10) .* 10 .+ 1
	)
	
	mp3 = data(
		combine(
			groupby(rev_confusing, :rel_trial), 
			:feedback_common => (x -> mean(.!x)) => :feedback_confusing
		)
	) * mapping(
		:rel_trial => "Trial", 
		:feedback_confusing => "Prop. confusing feedback"
	) * visual(ScatterLines)

	draw!(f[2,1], mp3)

	mp4 = mapping(1:rev_n_blocks => "Block", rev_criterion => "# optimal choices)") * visual(ScatterLines)

	draw!(f[2,2], mp4, axis = (; 
		yticks = 3:8, 
		xticks = [1, 10, 20, 30, 40], 
		subtitle = "Reversal criterion")
	)

	save("results/eeg_pilot2_reversal_sequence.png", f, pt_per_unit = 1)

	f

end

# ╔═╡ Cell order:
# ╠═fee6c3cd-243b-46f8-aae1-c19b3effa88d
# ╠═2f4eb9a3-4a49-46eb-867c-61583a84aea0
# ╠═9425106c-bbef-11ef-3059-19f67d4354e0
# ╠═10343528-6896-4521-84ba-e0fbdc483b28
# ╠═f21e290b-1a1b-4120-b00d-c07f03c248a4
# ╠═67285fe5-9381-45dd-a31e-e9f2649c0498
