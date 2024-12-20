### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ c3e866b8-b623-11ef-215c-c36622ba89dd
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
	include("$(pwd())/working_memory/collins_RLWM.jl")
	include("$(pwd())/working_memory/plotting_utils.jl")
	include("$(pwd())/working_memory/model_utils.jl")
	include("$(pwd())/fetch_preprocess_data.jl")

	# Load data
	_, _, _, _, _, WM_data, _, _ = load_pilot6_data()

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

# ╔═╡ 5ff26fc6-657b-4a7b-94d9-5b5599ec5d89
md"
## Fitting A.G.C. model to pilot datasets
"

# ╔═╡ 255d8c52-5a41-449c-b2af-3ae640661918
begin
	# Load data
	df = prepare_WM_data(WM_data)
	
	# structure
	pilot6_wm = load_wm_structure_csv("pilot6_WM") # task structure for ppcs
	sess1_str = filter(x -> x.block <= 10, pilot6_wm)
	sess2_str = filter(x -> x.block > 10, pilot6_wm)
	nothing
end

# ╔═╡ edc8f2e1-39ee-467e-91a7-b3f8f3a7802f
begin
	function true_vs_preds(
		df1::DataFrame,
		df2::DataFrame;
		strct1::DataFrame,
		strct2::DataFrame
	)
		df1 = leftjoin(df1, strct1; on = [:block, :valence, :trial, :stimset])
		df2 = leftjoin(df2, strct2; on = [:block, :valence, :trial, :stimset])
		
		df1 = stack(df1, [:true_choice, :predicted_choice])
		filter!(x -> !(x.variable == "true_choice" && ismissing(x.value)), df1)
		df1.chce_ss = df1.variable .* string.(df1.set_size)
		rename!(df1, (:value => :choice))
		
		df2 = stack(df2, [:true_choice, :predicted_choice])
		filter!(x -> !(x.variable == "true_choice" && ismissing(x.value)), df2)
		df2.chce_ss = df2.variable .* string.(df2.set_size)
		rename!(df2, (:value => :choice))

		return df1, df2
	end
	
	function plot_true_vs_preds(choice_df1::DataFrame, choice_df2::DataFrame)
		f = Figure(size = (1000, 600))
		plot_prior_accuracy!(
			f[1,1], filter(x -> x.set_size == 1, choice_df1);
			choice_val = 3.0, group = :variable, 
			pid_col = :PID, error_band = "se", legend = true,
			legend_pos = :bottom, legend_rows = 2, 
			title = "Session 1 (set size = 1)", colors = Makie.colorschemes[:Reds_3]
		)
		plot_prior_accuracy!(
			f[1,2], filter(x -> x.set_size == 1, choice_df2);
			choice_val = 3.0, group = :variable,
		    pid_col = :PID, error_band = "se",legend = true,
			legend_pos = :bottom, legend_rows = 2,
			title = "Session 2 (set size = 1)", colors = Makie.colorschemes[:Reds_3]
		)
		plot_prior_accuracy!(
			f[2,1], filter(x -> x.set_size == 7, choice_df1);
			choice_val = 3.0, group = :variable, 
			pid_col = :PID, error_band = "se", legend = true, legend_pos = :bottom, legend_rows = 2, 
			title = "Session 1 (set size = 7)", colors = Makie.colorschemes[:Blues_3]
		)
		plot_prior_accuracy!(
			f[2,2], filter(x -> x.set_size == 7, choice_df2);
			choice_val = 3.0, group = :variable, 
		    pid_col = :PID, error_band = "se",legend = true,
			legend_pos = :bottom, legend_rows = 2,
			title = "Session 2 (set size = 7)", colors = Makie.colorschemes[:Blues_3]
		)
		return f
	end
end

# ╔═╡ 61b26843-b400-4606-86eb-e73251962a1b
md"
### Choice accuracy by set size
"

# ╔═╡ 565c5e45-6ce0-4863-8508-d03c71d55492
let
	f = Figure(size = (800, 1000))
	plot_prior_accuracy!(
		f[1,1:2],
		df;
		choice_val = 3.,
		group = :set_size,
		pid_col = :PID,
		legend = true,
		legend_rows = 1,
		legend_pos = :bottom,
		title = "Overall",
		error_band = "se"
	)
	plot_prior_accuracy!(
		f[2,1],
		filter(x -> x.valence > 0, df);
		choice_val = 3.,
		group = :set_size,
		pid_col = :PID,
		legend = true,
		legend_rows = 1,
		legend_pos = :bottom,
		title = "Reward blocks",
		error_band = "se"
	)
	plot_prior_accuracy!(
		f[2, 2],
		filter(x -> x.valence < 0, df);
		choice_val = 3.,
		group = :set_size,
		pid_col = :PID,
		legend = true,
		legend_rows = 1,
		legend_pos = :bottom,
		title = "Punishment blocks",
		error_band = "se"
	)
	plot_prior_accuracy!(
		f[3,1],
		filter(x -> x.session == "1", df);
		choice_val = 3.,
		group = :set_size,
		pid_col = :PID,
		legend = true,
		legend_rows = 1,
		legend_pos = :bottom,
		title = "Session 1",
		error_band = "se"
	)
	plot_prior_accuracy!(
		f[3, 2],
		filter(x -> x.session == "2", df);
		choice_val = 3.,
		group = :set_size,
		pid_col = :PID,
		legend = true,
		legend_rows = 1,
		legend_pos = :bottom,
		title = "Session 2",
		error_band = "se"
	)
	f
end

# ╔═╡ d749dca3-b3a7-4573-a374-05acec1afeeb
md"
### Model fits
"

# ╔═╡ 116fe57d-1a80-4f33-af7b-aeccdd98a54d
md"
#### Standard RLWM model
"

# ╔═╡ 20afec7e-bbd3-4cb2-a682-84974becaf46
begin
	# single update model
	rlwm_ests_s1, rlwm_choices_s1 = optimize_multiple(
		filter(x -> x.session == "1", df);
		model = RLWM_collins24,
		estimate = "MAP",
		include_true = false,
		priors = Dict(
	        :β => 25., # fixed inverse temperature
	        :a => Normal(-2., 1.), # RL reward learning rate
	        :bRL => Normal(1., 1.), # LR difference between RL reward / punishment
	        :bWM => Normal(1., 1.), # punishment learning rate for working memory
	        :E => Normal(-2., 1.), # undirected noise
	        :F_wm => Normal(-2., 4.), # working memory forgetting rate
	        :w0 => Normal(2., 1.), # initial working memory weight
        	:C => truncated(Normal(4., 2.), lower=1, upper=7) # WM capacity
		),
		parameters = [:a_pos, :bRL, :bWM, :E, :F_wm, :w0, :C],
		gq_struct = sess1_str,
		n_starts = 5
	)
	rlwm_ests_s2, rlwm_choices_s2 = optimize_multiple(
		filter(x -> x.session == "2", df);
		model = RLWM_collins24,
		estimate = "MAP",
		include_true = false,
		priors = Dict(
	        :β => 25., # fixed inverse temperature
	        :a => Normal(-2., 1.), # RL reward learning rate
	        :bRL => Normal(1., 1.), # LR difference between RL reward / punishment
	        :bWM => Normal(1., 1.), # punishment learning rate for working memory
	        :E => Normal(-2., 1.), # undirected noise
	        :F_wm => Normal(-2., 4.), # working memory forgetting rate
	        :w0 => Normal(2., 1.), # initial working memory weight
        	:C => truncated(Normal(4., 2.), lower=1, upper=7) # WM capacity
		),
		parameters = [:a_pos, :bRL, :bWM, :E, :F_wm, :w0, :C],
		gq_struct = sess2_str,
		n_starts = 5
	)
end

# ╔═╡ 519ffac2-9179-49db-a732-d50bf5f6fe38
let
	f = Figure(size = (1200, 300))
	
	# Define parameters
	s1a1, s1a2, s1ll = rlwm_ests_s1.a_pos, rlwm_ests_s1.bRL, rlwm_ests_s1.loglike
	s2a1, s2a2, s2ll = rlwm_ests_s2.a_pos, rlwm_ests_s2.bRL, rlwm_ests_s2.loglike
	s1f, s1a3 = rlwm_ests_s1.F_wm, rlwm_ests_s1.bWM
	s2f, s2a3 = rlwm_ests_s2.F_wm, rlwm_ests_s2.bWM
	s1w, s1c = rlwm_ests_s1.w0, rlwm_ests_s1.C
	s2w, s2c = rlwm_ests_s2.w0, rlwm_ests_s2.C

	# Set labels
	labs1 = (xlabel = "a_pos", ylabel = "bRL", zlabel = "log-likelihood", title = "RL parameters")
	labs2 = (xlabel = "F_wm", ylabel = "bWM", zlabel = "log-likelihood", title = "WM forgetting + bias")
	labs3 = (xlabel = "w0", ylabel = "C", zlabel = "log-likelihood", title = "WM weight + capacity")

	# Plot
	ax1, ax2, ax3 = Axis3(f[1,1]; labs1...), Axis3(f[1,2]; labs2...), Axis3(f[1,3]; labs3...)
	scatter!(ax1, s1a1, s1a2, s1ll)
	scatter!(ax1, s2a1, s2a2, s2ll)
	scatter!(ax2, s1f, s1a3, s1ll)
	scatter!(ax2, s2f, s2a3, s2ll)
	scatter!(ax3, s1w, s1c, s1ll)
	scatter!(ax3, s2w, s2c, s2ll)
	f
end

# ╔═╡ dc5534aa-b00a-42d8-a34b-e1fc7794baef
let
	choice_df1, choice_df2 = true_vs_preds(
		rlwm_choices_s1,
		rlwm_choices_s2;
		strct1 = sess1_str,
		strct2 = sess2_str
	)
	plot_true_vs_preds(choice_df1, choice_df2)
end

# ╔═╡ e5cc4228-ab7b-4a5a-a4c7-b798d4ec18dd
function reliability_scatter!(
	f::GridPosition;
	df::AbstractDataFrame,
	xlabel::AbstractString,
	ylabel::AbstractString,
	xcol::Symbol = :x,
	ycol::Symbol = :y,
	subtitle::AbstractString = "",
	tickformat::Union{Function, Makie.Automatic} = Makie.automatic,
	correct_r::Bool = false # Whether to apply Spearman Brown
)	

	# Compute correlation
	r = cor(df[!, xcol], df[!, ycol])

	# Text
	r_text = "n = $(nrow(df)), r = $(round(r; digits = 2))"

	# Plot
	mp = data(df) *
			mapping(xcol, ycol) *
			(visual(Scatter) + linear()) +
		mapping([0], [1]) *
			visual(ABLines, linestyle = :dash, color = :gray70)
	
	draw!(f, mp; axis=(;
		xlabel = xlabel, 
		ylabel = ylabel, 
		xtickformat = tickformat,
		ytickformat = tickformat,
		subtitle = subtitle
	))

	if r > 0
		Label(
			f,
			r_text,
			fontsize = 16,
			font = :bold,
			halign = 0.975,
			valign = 0.1,
			tellheight = false,
			tellwidth = false
		)
	end

end

# ╔═╡ bf3d9424-491d-4ea7-adce-8867bfb535a6
let
	retest_df = leftjoin(rlwm_ests_s1, rlwm_ests_s2, on = :PID, makeunique=true)
	dropmissing!(retest_df)

	fig=Figure(;size=(800, 1000))
	reliability_scatter!(
		fig[1,1]; df=retest_df, xlabel="Session 1", ylabel="Session 2", 
		xcol=:a_pos, ycol=:a_pos_1, subtitle="RL reward learning rate"
	)
	reliability_scatter!(
		fig[1,2]; df=retest_df, xlabel="Session 1", ylabel="Session 2", 
		xcol=:bRL, ycol=:bRL_1, subtitle="RL punishment learning rate bias"
	)
	reliability_scatter!(
		fig[2,1]; df=retest_df, xlabel="Session 1", ylabel="Session 2", 
		xcol=:bWM, ycol=:bWM_1, subtitle="WM punishment learning rate bias"
	)
	reliability_scatter!(
		fig[2,2]; df=retest_df, xlabel="Session 1", ylabel="Session 2", 
		xcol=:F_wm, ycol=:F_wm_1, subtitle="WM forgetting"
	)
	reliability_scatter!(
		fig[3,1]; df=retest_df, xlabel="Session 1", ylabel="Session 2", 
		xcol=:w0, ycol=:w0_1, subtitle="WM initial weighting"
	)
	reliability_scatter!(
		fig[3,2]; df=retest_df, xlabel="Session 1", ylabel="Session 2", 
		xcol=:E, ycol=:E_1, subtitle="Undirected policy noise"
	)
	reliability_scatter!(
		fig[4,1:2]; df=retest_df, xlabel="Session 1", ylabel="Session 2", 
		xcol=:C, ycol=:C_1, subtitle="WM capacity"
	)
	fig
end

# ╔═╡ 8b06d38e-2b28-4a19-80de-03cb012f35be
md"
#### HLWM model
"

# ╔═╡ 930f843a-e3d7-42ad-b7c3-1da5939f8d6b
begin
	# single update model
	hlwm_ests_s1, hlwm_choices_s1 = optimize_multiple(
		filter(x -> x.session == "1", df);
		model = HLWM_collins24,
		estimate = "MAP",
		include_true = false,
		priors = Dict(
	        :β => 25., # fixed inverse temperature
	        :a => Normal(-2., 1.), # RL reward learning rate
	        :bWM => Normal(1., 1.), # punishment learning rate for working memory
	        :E => Normal(-2., 1.), # undirected noise
	        :F_wm => Normal(-2., 4.), # working memory forgetting rate
	        :w0 => Normal(2., 1.), # initial working memory weight
        	:C => truncated(Normal(4., 2.), lower=1, upper=7) # WM capacity
		),
		parameters = [:a_pos, :bWM, :E, :F_wm, :w0, :C],
		gq_struct = sess1_str,
		n_starts = 5
	)
	hlwm_ests_s2, hlwm_choices_s2 = optimize_multiple(
		filter(x -> x.session == "2", df);
		model = HLWM_collins24,
		estimate = "MAP",
		include_true = false,
		priors = Dict(
	        :β => 25., # fixed inverse temperature
	        :a => Normal(-2., 1.), # RL reward learning rate
	        :bWM => Normal(1., 1.), # punishment learning rate for working memory
	        :E => Normal(-2., 1.), # undirected noise
	        :F_wm => Normal(-2., 4.), # working memory forgetting rate
	        :w0 => Normal(2., 1.), # initial working memory weight
        	:C => truncated(Normal(4., 2.), lower=1, upper=7) # WM capacity
		),
		parameters = [:a_pos, :bWM, :E, :F_wm, :w0, :C],
		gq_struct = sess2_str,
		n_starts = 5
	)
end

# ╔═╡ e51d9a96-bc02-4e16-8d30-459bb9f7a734
let
	f = Figure(size = (1200, 300))
	
	# Define parameters
	s1a1, s1a2, s1ll = hlwm_ests_s1.a_pos, hlwm_ests_s1.E, hlwm_ests_s1.loglike
	s2a1, s2a2, s2ll = hlwm_ests_s2.a_pos, hlwm_ests_s2.E, hlwm_ests_s2.loglike
	s1f, s1a3 = hlwm_ests_s1.F_wm, hlwm_ests_s1.bWM
	s2f, s2a3 = hlwm_ests_s2.F_wm, hlwm_ests_s2.bWM
	s1w, s1c = hlwm_ests_s1.w0, hlwm_ests_s1.C
	s2w, s2c = hlwm_ests_s2.w0, hlwm_ests_s2.C

	# Set labels
	labs1 = (xlabel = "a_pos", ylabel = "E", zlabel = "log-likelihood", title = "HL parameters")
	labs2 = (xlabel = "F_wm", ylabel = "bWM", zlabel = "log-likelihood", title = "WM forgetting + bias")
	labs3 = (xlabel = "w0", ylabel = "C", zlabel = "log-likelihood", title = "WM weight + capacity")

	# Plot
	ax1, ax2, ax3 = Axis3(f[1,1]; labs1...), Axis3(f[1,2]; labs2...), Axis3(f[1,3]; labs3...)
	scatter!(ax1, s1a1, s1a2, s1ll)
	scatter!(ax1, s2a1, s2a2, s2ll)
	scatter!(ax2, s1f, s1a3, s1ll)
	scatter!(ax2, s2f, s2a3, s2ll)
	scatter!(ax3, s1w, s1c, s1ll)
	scatter!(ax3, s2w, s2c, s2ll)
	f
end

# ╔═╡ 7cad2528-caf4-4318-b355-b7c2bc8295b0
let
	choice_df1, choice_df2 = true_vs_preds(
		hlwm_choices_s1,
		hlwm_choices_s2;
		strct1 = sess1_str,
		strct2 = sess2_str
	)
	plot_true_vs_preds(choice_df1, choice_df2)
end

# ╔═╡ d7e98acb-bfd7-4f91-8470-0a92c39a8fc2
let
	retest_df = leftjoin(hlwm_ests_s1, hlwm_ests_s2, on = :PID, makeunique=true)
	dropmissing!(retest_df)

	fig=Figure(;size=(800, 1000))
	reliability_scatter!(
		fig[1,1]; df=retest_df, xlabel="Session 1", ylabel="Session 2", 
		xcol=:a_pos, ycol=:a_pos_1, subtitle="RL reward learning rate"
	)
	reliability_scatter!(
		fig[1,2]; df=retest_df, xlabel="Session 1", ylabel="Session 2", 
		xcol=:bWM, ycol=:bWM_1, subtitle="WM punishment learning rate bias"
	)
	reliability_scatter!(
		fig[2,1]; df=retest_df, xlabel="Session 1", ylabel="Session 2", 
		xcol=:F_wm, ycol=:F_wm_1, subtitle="WM forgetting"
	)
	reliability_scatter!(
		fig[2,2]; df=retest_df, xlabel="Session 1", ylabel="Session 2", 
		xcol=:w0, ycol=:w0_1, subtitle="WM initial weighting"
	)
	reliability_scatter!(
		fig[3,1]; df=retest_df, xlabel="Session 1", ylabel="Session 2", 
		xcol=:E, ycol=:E_1, subtitle="Undirected policy noise"
	)
	reliability_scatter!(
		fig[2,2]; df=retest_df, xlabel="Session 1", ylabel="Session 2", 
		xcol=:C, ycol=:C_1, subtitle="WM capacity"
	)
	fig
end

# ╔═╡ 629ff7e2-a279-4b55-906a-224f466c398b
md"
#### Reparameterised HLWM model
"

# ╔═╡ c88dc070-3819-409b-abba-56a76c9ea8b1
begin
	# single update model
	hlwm_el_ests_s1, hlwm_el_choices_s1 = optimize_multiple(
		filter(x -> x.session == "1", df);
		model = RLWM_elig,
		estimate = "MAP",
		include_true = false,
		priors = Dict(
			:β => 25., # fixed inverse temperature
	        :a => Normal(-2., 1.), # RL reward learning rate
			:bRL => Normal(1., 1.), # punishment learning rate for RL
	        :bWM => Normal(1., 1.), # punishment learning rate for working memory
			:w0 => Normal(3., 2.), # WM weight
			:λ_wm => Beta(4., 1.) # memory eligibility trace decay
		),
		parameters = [:a_pos, :bRL, :bWM, :w0, :λ_wm],
		gq_struct = sess1_str,
		n_starts = 5
	)
	hlwm_el_ests_s2, hlwm_el_choices_s2 = optimize_multiple(
		filter(x -> x.session == "2", df);
		model = RLWM_elig,
		estimate = "MAP",
		include_true = false,
		priors = Dict(
			:β => 25., # fixed inverse temperature
	        :a => Normal(-2., 1.), # RL reward learning rate
			:bRL => Normal(1., 1.), # punishment learning rate for RL
	        :bWM => Normal(1., 1.), # punishment learning rate for working memory
			:w0 => Normal(3., 2.), # WM weight
			:λ_wm => Beta(4., 1.) # memory eligibility trace decay
		),
		parameters = [:a_pos, :bRL, :bWM, :w0, :λ_wm],
		gq_struct = sess2_str,
		n_starts = 5
	)
end

# ╔═╡ 8cdb4251-d220-4f1a-ad67-54db7992fa3f
let
	f = Figure(size = (800, 300))
	
	# Define parameters
	s1a, s1e, s1ll = hlwm_el_ests_s1.a_pos, hlwm_el_ests_s1.E, hlwm_el_ests_s1.loglike
	s2a, s2e, s2ll = hlwm_el_ests_s2.a_pos, hlwm_el_ests_s2.E, hlwm_el_ests_s2.loglike
	s1w, s1d = hlwm_el_ests_s1.bWM, hlwm_el_ests_s1.λ_wm
	s2w, s2d = hlwm_el_ests_s2.bWM, hlwm_el_ests_s2.λ_wm
	# Set labels
	labs1 = (xlabel = "a_pos", ylabel = "E", zlabel = "log-likelihood", title = "Habit learning rate + noise")
	labs2 = (xlabel = "bWM", ylabel = "λ_wm", zlabel = "log-likelihood", title = "WM bias + eligibility trace decay")

	# Plot
	ax1, ax2 = Axis3(f[1,1]; labs1...), Axis3(f[1,2]; labs2...)
	scatter!(ax1, s1a, s1e, s1ll)
	scatter!(ax1, s2a, s2e, s2ll)
	scatter!(ax2, s1w, s1d, s1ll)
	scatter!(ax2, s2w, s2d, s2ll)
	f
end

# ╔═╡ 5df900d4-b408-43e1-ba68-26b964ffa777
let
	choice_df1, choice_df2 = true_vs_preds(
		hlwm_el_choices_s1,
		hlwm_el_choices_s2;
		strct1 = sess1_str,
		strct2 = sess2_str
	)
	plot_true_vs_preds(choice_df1, choice_df2)
end

# ╔═╡ 6b84ec76-850b-471d-987e-947999f8402a
let
	retest_df = leftjoin(
		hlwm_el_ests_s1, hlwm_el_ests_s2, on = :PID, makeunique=true
	)
	dropmissing!(retest_df)

	fig=Figure(;size=(800, 500))
	reliability_scatter!(
		fig[1,1]; df=retest_df, xlabel="Session 1", ylabel="Session 2", 
		xcol=:a_pos, ycol=:a_pos_1, subtitle="RL reward learning rate"
	)
	reliability_scatter!(
		fig[1,2]; df=retest_df, xlabel="Session 1", ylabel="Session 2", 
		xcol=:bWM, ycol=:bWM_1, subtitle="WM punishment learning rate bias"
	)
	reliability_scatter!(
		fig[2,1]; df=retest_df, xlabel="Session 1", ylabel="Session 2", 
		xcol=:E, ycol=:E_1, subtitle="Undirected policy noise"
	)
	reliability_scatter!(
		fig[2,2]; df=retest_df, xlabel="Session 1", ylabel="Session 2", 
		xcol=:λ_wm, ycol=:λ_wm_1, subtitle="Elegibility trace decay"
	)
	fig
end

# ╔═╡ Cell order:
# ╟─5ff26fc6-657b-4a7b-94d9-5b5599ec5d89
# ╠═c3e866b8-b623-11ef-215c-c36622ba89dd
# ╠═255d8c52-5a41-449c-b2af-3ae640661918
# ╟─edc8f2e1-39ee-467e-91a7-b3f8f3a7802f
# ╟─61b26843-b400-4606-86eb-e73251962a1b
# ╟─565c5e45-6ce0-4863-8508-d03c71d55492
# ╟─d749dca3-b3a7-4573-a374-05acec1afeeb
# ╟─116fe57d-1a80-4f33-af7b-aeccdd98a54d
# ╠═20afec7e-bbd3-4cb2-a682-84974becaf46
# ╠═519ffac2-9179-49db-a732-d50bf5f6fe38
# ╠═dc5534aa-b00a-42d8-a34b-e1fc7794baef
# ╟─e5cc4228-ab7b-4a5a-a4c7-b798d4ec18dd
# ╠═bf3d9424-491d-4ea7-adce-8867bfb535a6
# ╟─8b06d38e-2b28-4a19-80de-03cb012f35be
# ╠═930f843a-e3d7-42ad-b7c3-1da5939f8d6b
# ╠═e51d9a96-bc02-4e16-8d30-459bb9f7a734
# ╠═7cad2528-caf4-4318-b355-b7c2bc8295b0
# ╠═d7e98acb-bfd7-4f91-8470-0a92c39a8fc2
# ╟─629ff7e2-a279-4b55-906a-224f466c398b
# ╠═c88dc070-3819-409b-abba-56a76c9ea8b1
# ╠═8cdb4251-d220-4f1a-ad67-54db7992fa3f
# ╠═5df900d4-b408-43e1-ba68-26b964ffa777
# ╠═6b84ec76-850b-471d-987e-947999f8402a
