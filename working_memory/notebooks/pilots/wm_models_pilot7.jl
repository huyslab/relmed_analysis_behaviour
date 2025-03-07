### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ ae3bc190-fb4a-11ef-1af4-354fd7e21bfe
begin
	cd("/home/jovyan")
	import Pkg
	
	# activate the shared project environment
    Pkg.activate("$(pwd())/relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
		using CairoMakie, Random, DataFrames, Distributions, StatsBase,
		ForwardDiff, LinearAlgebra, JSON, CSV, JLD2, AlgebraOfGraphics, Dates, Turing
		using LogExpFunctions: logistic, logit
	
	Turing.setprogress!(false)
	
	include("$(pwd())/working_memory/simulate.jl")
	include("$(pwd())/working_memory/continuous_HLWM.jl")
	include("$(pwd())/working_memory/plotting_utils.jl")
	include("$(pwd())/working_memory/model_utils.jl")
	include("$(pwd())/fetch_preprocess_data.jl")

	# # Load data
	_, _, _, _, _, WM_data, _, _ = load_pilot7_data(; return_version="7.0")

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

# ╔═╡ 60471a68-1673-4c49-b5e1-2bc547973027
begin
	# Load data
	df = prepare_WM_data(WM_data, continuous=true)
	
	# structure
	pilot7_wm = DataFrame(CSV.File("data/pilot7_WM.csv")) # task structure for ppcs
	nothing
end

# ╔═╡ 6e36058b-4afe-4e8d-b843-2106f50ebe3d
begin
	function true_vs_preds(
		df1::DataFrame;
		strct1::DataFrame
	)
		df1 = leftjoin(df1, strct1; on = [:block, :valence, :trial, :stimset])
		
		df1 = stack(df1, [:true_choice, :predicted_choice])
		filter!(x -> !(x.variable == "true_choice" && ismissing(x.value)), df1)
		# df1.chce_ss = df1.variable .* string.(df1.set_size)
		rename!(df1, (:value => :choice))

		return df1
	end
	
	function plot_true_vs_preds(choice_df1::DataFrame, choice_df2::DataFrame)
		f = Figure(size = (1000, 300))
		plot_prior_accuracy!(
			f[1,1], choice_df1;
			choice_val = 3.0, group = :variable, 
			pid_col = :PID, error_band = "se", legend = true,
			legend_pos = :bottom, legend_rows = 2, 
			title = "", colors = Makie.colorschemes[:Reds_3]
		)
		plot_prior_accuracy!(
			f[1,2], choice_df2;
			choice_val = 3.0, group = :variable,
		    pid_col = :PID, error_band = "se",legend = true,
			legend_pos = :bottom, legend_rows = 2,
			title = "", colors = Makie.colorschemes[:Reds_3]
		)
		return f
	end
end

# ╔═╡ a91c74dd-4e6c-4969-abf1-358b49c3276c
md"
### Raw data
"

# ╔═╡ 8263814d-f633-4dd1-9623-053af24afcc1
let
	f = Figure(size = (600, 400))
	plot_prior_accuracy!(
		f[1,1],
		df;
		choice_val = 3.,
		group = nothing,
		pid_col = :PID,
		legend = true,
		legend_rows = 1,
		legend_pos = :bottom,
		title = "Overall",
		error_band = "se"
	)
	f
end

# ╔═╡ 5720a30f-a87d-4c85-a949-e7511ac65d00
md"
### Model fits
"

# ╔═╡ d7810152-b252-44f0-b606-ddd08e21002e
begin
	# single update model
	hlwm_ests, hlwm_choices = optimize_multiple(
		df;
		model = HLWM_collins_continuous,
		estimate = "MAP",
		include_true = false,
		priors = Dict(
			:β => 50., # fixed inverse temperature
	        :a => Dirac(99), #Normal(0., 2.), # RL reward learning rate
			:F_wm => Dirac(-99), # working memory forgetting rate
	        :w0 => Beta(2, 2), # prop. of WM to RL weight (i.e., 0.5 ===)
		),
		parameters = [:a_pos, :F_wm, :w0],
		gq_struct = pilot7_wm,
		n_starts = 5
	)
end

# ╔═╡ 597326cc-12f4-4bd8-9b05-f83e5bcbd882
begin
	summarystats(a2α.(hlwm_ests.F_wm))
end

# ╔═╡ 26c3005d-2ee4-4387-a853-626527aa562f
md"
#### Fitted parameters & likelihood
"

# ╔═╡ 51d2f15a-aaa9-4295-a176-aa0117109e97
let
	f = Figure(size = (1200, 300))
	
	# Define parameters
	s1a, s1ll = hlwm_ests.a_pos, hlwm_ests.loglike
	s1f, s1w = hlwm_ests.F_wm, hlwm_ests.w0

	# Set labels
	labs1 = (xlabel = "a_pos", ylabel = "F_wm", zlabel = "log-likelihood", title = "Learning + forgetting")
	labs2 = (xlabel = "F_wm", ylabel = "w0", zlabel = "log-likelihood", title = "WM forgetting + weight")
	labs3 = (xlabel = "a_pos", ylabel = "F_wm", zlabel = "w0", title = "All parameters")

	# Plot
	ax1, ax2, ax3 = Axis3(f[1,1]; labs1...), Axis3(f[1,2]; labs2...), Axis3(f[1,3]; labs3...)
	scatter!(ax1, s1a, s1f, s1ll)
	scatter!(ax2, s1f, s1w, s1ll)
	scatter!(ax3, s1a, s1f, s1w)
	f
end

# ╔═╡ d43a96f1-576d-4e2b-9682-c9f0547b47af
md"
#### Simulated data from fitted parameters
"

# ╔═╡ 68fd75db-aa7b-4b47-81a9-b66175ede3c3
begin
	np = maximum(hlwm_ests.PID)
	prior_sample_hlwm = simulate_from_prior(
	    100;
		model = HLWM_collins_continuous,
		priors = Dict(
			:β => 50., # fixed inverse temperature
	        :a => Dirac(99.), #DiscreteNonParametric(hlwm_ests.a_pos, fill(1/np, np)),
			:F_wm => Dirac(-99), #DiscreteNonParametric(hlwm_ests.F_wm, fill(1/np, np)),
	        :w0 => DiscreteNonParametric(hlwm_ests.w0, fill(1/np, np))
		),
		parameters = [:a_pos, :F_wm, :w0],
		fixed_struct = pilot7_wm,
		continuous = true,
		gq = true,
		random_seed = 1
	)
	nothing
end

# ╔═╡ 1431c579-d65f-41a2-8d8f-a6febee12492
let
	f = Figure(size=(1000, 400))
	plot_sim_q_value_acc!(
		f,
		prior_sample_hlwm;
		colA = [:Q_optimal, :W_optimal],
		colB = [[:Q_suboptimal1, :Q_suboptimal2], [:W_suboptimal1, :W_suboptimal2]],
		choice_val = 3.0,
		ylab = ("Q-value", "W-value"),
		group = :delay,
		norm = nothing,
		legend = true,
		legend_rows=3,
		colors = Makie.to_colormap(:nuuk50)[20:2:50]
	)
	f
end

# ╔═╡ 3978b38d-bdfb-408e-b713-4fa14f993593
begin
	df1 = leftjoin(hlwm_choices, pilot7_wm; on = [:block, :valence, :trial, :stimset])
	df1 = stack(df1, [:true_choice, :predicted_choice])
	filter!(x -> !(x.variable == "true_choice" && ismissing(x.value)), df1)
	#df1.chce_ss = df1.variable .* string.(df1.set_size)
	rename!(df1, (:value => :choice))
	
	f = Figure(size = (600, 400))
	plot_prior_accuracy!(
		f[1,1], df1;
		choice_val = 3.0, group = :variable, 
		pid_col = :PID, error_band = "se", legend = true,
		legend_pos = :bottom, legend_rows = 2, 
		title = "", colors = Makie.colorschemes[:Reds_3]
	)
	f
end

# ╔═╡ Cell order:
# ╠═ae3bc190-fb4a-11ef-1af4-354fd7e21bfe
# ╠═60471a68-1673-4c49-b5e1-2bc547973027
# ╠═6e36058b-4afe-4e8d-b843-2106f50ebe3d
# ╟─a91c74dd-4e6c-4969-abf1-358b49c3276c
# ╠═8263814d-f633-4dd1-9623-053af24afcc1
# ╟─5720a30f-a87d-4c85-a949-e7511ac65d00
# ╠═d7810152-b252-44f0-b606-ddd08e21002e
# ╠═597326cc-12f4-4bd8-9b05-f83e5bcbd882
# ╟─26c3005d-2ee4-4387-a853-626527aa562f
# ╠═51d2f15a-aaa9-4295-a176-aa0117109e97
# ╟─d43a96f1-576d-4e2b-9682-c9f0547b47af
# ╠═68fd75db-aa7b-4b47-81a9-b66175ede3c3
# ╠═1431c579-d65f-41a2-8d8f-a6febee12492
# ╠═3978b38d-bdfb-408e-b713-4fa14f993593
