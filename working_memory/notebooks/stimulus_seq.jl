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
	# include("$(pwd())/working_memory/collins_RLWM.jl")
	# include("$(pwd())/working_memory/plotting_utils.jl")
	# include("$(pwd())/working_memory/model_utils.jl")
	# include("$(pwd())/fetch_preprocess_data.jl")

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

# ╔═╡ 8883dea5-bedc-49ba-916d-a4592603fdd2
include("$(pwd())/working_memory/simulate.jl")

# ╔═╡ 82326a87-0cba-4027-97f7-1e0fa60e1549
begin
	function uniform_chi_squared(df::DataFrame, ns::Int)
	    # Simple chi-squared helper
	    chisq(o, e) = sum((o .- e).^2 .// e)
	    total_stat = 0.0
	    for s in 3:ns
	        sub = df[df.set_size .== s .&& df.delay .!= 0, :]
	        possible_delays = 1:floor(s+s÷2)
	        cto, cte = countmap(df.delay), countmap(df.delay_g)
			observed = [get(cto, d, 0) for d in possible_delays]
			expected = [get(cte, d, 0) for d in possible_delays]
	        total_stat += chisq(observed, expected)*s # weight by set-size
	    end
	    return total_stat
	end
	
	function best_seed_for_delays(ns::Int; ntries::Int=100, start::Int=10, ntol::Int=5)
	    best_seed  = -1
	    best_score = Inf
		best_tol = -1
	
	    for s in 1:ntries
			for tol in 1:ntol
			    df = generate_delay_sequence(ns, start; seed=s, tolerance=tol)
		        # Example "score": mean absolute deviation from target "delay_g"
		        score = uniform_chi_squared(df, ns)
		        if score < best_score
		            best_score = score
		            best_seed  = s
					best_tol = tol
		        end
			end
	    end
	
	    return best_seed, best_score, best_tol
	end

	bse, bsc, bst = best_seed_for_delays(7; ntries=10000, start=0, ntol=5)
	test1 = generate_delay_sequence(7, 10; seed=bse, tolerance=bst)
end

# ╔═╡ 3b58eb9d-5abc-46da-9857-0c54e823b649
bse

# ╔═╡ 2f9398ce-6520-454d-b131-2d32082a9850
let
	set_aog_theme!()
	update_theme!(fontsize=30)

	axis = (width = 300, height = 300)
	#test1.ss = string.(test1.set_size)
	delay_pt = data(test1) * mapping(:trial, :delay, color=:ss)
	#delay_by_ss = delay_freq * mapping(layout = :ss)
		
	draw(delay_pt; axis = axis)
end

# ╔═╡ 67809b95-8a52-4359-b31a-651e4df21dcd
let
	set_aog_theme!()
	update_theme!(fontsize=30)

	axis = (width = 400, height = 300)
	test1.ss = string.(test1.set_size)
	delay_freq = data(test1) * frequency() * mapping(:delay)
	delay_by_ss = delay_freq * mapping(layout = :ss)
		
	draw(delay_by_ss; axis = axis)
end

# ╔═╡ 2ae530c1-004c-40d5-baf6-4a06bc6ddc06
mean(abs.(sort(test1.delay) .- sort(test1.delay_g)))

# ╔═╡ Cell order:
# ╠═ef1a09ce-d1b5-11ef-2994-91cdad7ede58
# ╠═8883dea5-bedc-49ba-916d-a4592603fdd2
# ╠═82326a87-0cba-4027-97f7-1e0fa60e1549
# ╠═3b58eb9d-5abc-46da-9857-0c54e823b649
# ╠═2f9398ce-6520-454d-b131-2d32082a9850
# ╠═67809b95-8a52-4359-b31a-651e4df21dcd
# ╠═2ae530c1-004c-40d5-baf6-4a06bc6ddc06
