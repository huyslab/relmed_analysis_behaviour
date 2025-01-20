### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ ef1a09ce-d1b5-11ef-2994-91cdad7ede58
begin
	cd("/home/jovyan")
	import Pkg
	
	# activate the shared project environment
    Pkg.activate("$(pwd())/relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
		using CairoMakie, Random, DataFrames, Distributions, StatsBase,
		ForwardDiff, LinearAlgebra, JSON, CSV, JLD2, AlgebraOfGraphics, Dates, Turing, HypothesisTests
		using LogExpFunctions: logistic, logit
	
	Turing.setprogress!(false)
	
	include("$(pwd())/working_memory/simulate.jl")
	include("$(pwd())/working_memory/collins_RLWM.jl")
	include("$(pwd())/working_memory/plotting_utils.jl")
	include("$(pwd())/working_memory/model_utils.jl")
	#include("$(pwd())/fetch_preprocess_data.jl")

	# Load data
	#_, _, _, _, _, WM_data, _, _ = load_pilot6_data()

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

# ╔═╡ 5e07b79a-d16f-4279-b1ad-771ab599a832
include("$(pwd())/working_memory/simulate.jl")

# ╔═╡ b25136ed-5791-45cd-bb6a-45fdd8aeba9c
begin
	include("$(pwd())/working_memory/simulate.jl")
	include("$(pwd())/working_memory/collins_RLWM.jl")
end

# ╔═╡ 82326a87-0cba-4027-97f7-1e0fa60e1549
begin
	function uniform_chi_squared(df::DataFrame, mss::Int)
	    # Simple chi-squared helper
	    chisq(o, e) = sum((o .- e).^2 .// e)
		pds = 1:2*mss-1
		cto = countmap(df.delay)
		observed = [get(cto, d, 0) for d in pds]
		expected = fill(nrow(df)÷length(pds), length(pds))
	    return chisq(observed, expected)
	end

	function bimodal_chi_squared(df::DataFrame, mss::Int; peak_ratio::Float64=0.4)
	    # Setup
	    pds = 1:2*mss-1
	    total = nrow(df)
	    n_bins = length(pds)
	    
	    # Create bimodal expected distribution
	    μ1 = n_bins * 0.4  # First peak at 25%
	    μ2 = n_bins * 0.9  # Second peak at 75%
	    σ = n_bins * 0.15   # Width of peaks
	    
	    # Generate expected counts
	    weights = @. peak_ratio * pdf(Normal(μ1, σ), 1:n_bins) + 
	                (1-peak_ratio) * pdf(Normal(μ2, σ), 1:n_bins)
	    expected = round.(Int, total * weights / sum(weights))
	    
	    # Adjust to match total
	    diff = total - sum(expected)
	    expected[end] += diff
	   
	    # Calculate observed
	    cto = countmap(df.delay)
	    observed = [get(cto, d, 0) for d in pds]
	    
	    # Chi-squared
	    return sum((observed .- expected).^2 ./ expected)
	end
	
	function best_seed_for_delays(;
		ns::Int, mc::Int, d::Float64, ntries::Int, ntol::Int
	)
	    best_seed  = -1
	    best_score = Inf
		best_tol = -1
	
	    for s in 1:ntries
			for tol in 1:ntol
			    df = generate_delay_sequence(;
					no_sets=ns,
				    max_count=mc,
				    difficulty=d,
				    seed=s,
				    tolerance=tol,
					return_struct=false
				)
		        # Example "score": mean absolute deviation from target "delay_g"
		        score = bimodal_chi_squared(df, maximum(df.set_size))
		        if score < best_score
		            best_score = score
		            best_seed  = s
					best_tol = tol
		        end
			end
	    end
	
	    return best_seed, best_score, best_tol
	end
end

# ╔═╡ 3461ed6b-fa89-4bd9-b94a-6ef55f3bdf2c
begin
    se, nc, dd = 6, 20, .5
	bse, bsc, bst = best_seed_for_delays(ns=se, mc=nc, d=dd, ntries=10000, ntol=5)
end

# ╔═╡ 878db40c-4b7a-45bb-bddd-a24d08202359
begin
	task_struct = generate_delay_sequence(
		no_sets=se,
		max_count=nc,
		difficulty=dd,
		seed=bse,
		tolerance=bst,
		return_struct=true,
		n_options = 3, # number of options to choose from
	    coins = [0.01, 0.5, 1.0],
	    punish = true
	)
end

# ╔═╡ 2f9398ce-6520-454d-b131-2d32082a9850
let
	set_aog_theme!()
	update_theme!(fontsize=30)
	delay_seq = deepcopy(task_struct)

	axis = (width = 300, height = 300)
	delay_seq.ss = string.(delay_seq.set_size)
	delay_pt = data(delay_seq) * mapping(:trial_ovl, :delay, color=:ss)
		
	draw(delay_pt; axis = axis)
end

# ╔═╡ 67809b95-8a52-4359-b31a-651e4df21dcd
let
	set_aog_theme!()
	update_theme!(fontsize=30)
	delay_seq = deepcopy(task_struct)

	axis = (width = 400, height = 300)
	delay_freq = data(delay_seq) * frequency() * mapping(:delay)

	#delay_seq.ss = string.(delay_seq.set_size)
	#delay_by_ss = delay_freq * mapping(layout = :ss)
		
	draw(delay_freq; axis = axis)
end

# ╔═╡ 24b87114-c3d2-4e69-8cd4-a08a82ea0c5f
begin
	set_theme!(th)
	prior_sample_hlwm_broad = simulate_from_prior(
	    100;
		model = HLWM_collins_continuous,
		priors = Dict(
			:β => 25., # fixed inverse temperature
	        :a => Normal(0., 2.), # RL reward learning rate
			:F_wm => Normal(0., 2.), # working memory forgetting rate
	        :w0 => Beta(2, 2), # prop. of WM to RL weight (i.e., 0.5 ===)
		),
		parameters = [:a_pos, :F_wm, :w0],
		fixed_struct = task_struct,
		gq = true,
		random_seed = 1
	)
	f_hlwm_broad = optimization_calibration(
		prior_sample_hlwm_broad,
		optimize_multiple,
		estimate = "MAP",
		model = HLWM_collins_continuous,
		priors = Dict(
			:β => 25., # fixed inverse temperature
	        :a => Normal(0., 4.), # RL reward learning rate
			:F_wm => Normal(0., 4.), # working memory forgetting rate
	        :w0 => Beta(1.1, 1.1), # prop. of WM to RL weight (i.e., 0.5 ===)
		),
		parameters = [:a_pos, :F_wm, :w0],
		n_starts = 5
	)

	f_hlwm_broad
end

# ╔═╡ f153aa3b-985d-494f-bb8b-e1446d0633e9
let
	df = leftjoin(
		prior_sample_hlwm_broad, task_struct;
		on=[:block, :trial, :valence, :stimset, :set_size, :feedback_optimal, :feedback_suboptimal1, :feedback_suboptimal2],
		order=:left
	)
	#df.delay_grp = ifelse.(df.delay .>= median(df.delay), "high delay", "low delay")

	# f = Figure(size=(1000, 300))
	# plot_sim_q_value_acc!(
	# 	f,
	# 	df;
	# 	colA = [:Q_optimal, :W_optimal],
	# 	colB = [[:Q_suboptimal1, :Q_suboptimal2], [:W_suboptimal1, :W_suboptimal2]],
	# 	choice_val = 3.0,
	# 	ylab = ("Q-value", "W-value"),
	# 	group = :delay_grp,
	# 	norm = nothing,
	# 	legend = true,
	# 	colors = Makie.colorschemes[:seaborn_pastel]
	# )
	# f
	f = plot_prior_predictive_by_valence(
		df,
		[:Q_optimal, :Q_suboptimal1, :Q_suboptimal2];
		W_cols = [:W_optimal, :W_suboptimal1, :W_suboptimal2],
		choice_val = 3.0,
		ylab = ("Q-value", "W-value"),
		fig_size = (1000, 1000),
		group = nothing,
		norm = nothing,
		legend = false,
		colors = Makie.colorschemes[:seaborn_pastel6]
	)
	f
end

# ╔═╡ Cell order:
# ╠═ef1a09ce-d1b5-11ef-2994-91cdad7ede58
# ╠═82326a87-0cba-4027-97f7-1e0fa60e1549
# ╠═5e07b79a-d16f-4279-b1ad-771ab599a832
# ╠═3461ed6b-fa89-4bd9-b94a-6ef55f3bdf2c
# ╠═878db40c-4b7a-45bb-bddd-a24d08202359
# ╠═2f9398ce-6520-454d-b131-2d32082a9850
# ╠═67809b95-8a52-4359-b31a-651e4df21dcd
# ╠═b25136ed-5791-45cd-bb6a-45fdd8aeba9c
# ╠═24b87114-c3d2-4e69-8cd4-a08a82ea0c5f
# ╠═f153aa3b-985d-494f-bb8b-e1446d0633e9
