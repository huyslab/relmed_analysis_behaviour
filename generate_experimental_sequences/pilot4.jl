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
# ╠═2daa3d69-0e75-4234-bad4-50a6861eb54f
# ╠═b9db2e21-068b-4148-80b4-8c48edf8c4ec
# ╠═e1745880-0a58-4cca-ab6a-0c98de5430a1
# ╠═efab3964-0ec5-4df1-870c-b20ce2882337
