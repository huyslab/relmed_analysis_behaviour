### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ 3647aaf8-becd-11ef-17b7-1348cec601a2
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

# ╔═╡ 8e96fc42-2eda-4aca-a766-fe6793ebe2ed
md"
### Forgetting x habit models
"

# ╔═╡ a0300a7d-600b-4661-aadd-6afbaebe0486
begin
	# Load data
	df = prepare_WM_data(WM_data)
	
	# structure
	pilot6_wm = load_wm_structure_csv("pilot6_WM") # task structure for ppcs
	sess1_str = filter(x -> x.block <= 10, pilot6_wm)
	sess2_str = filter(x -> x.block > 10, pilot6_wm)
	nothing
end

# ╔═╡ d0d375c2-b3ff-4d24-bcf9-3a333f15c5fa
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

# ╔═╡ 3b6d2229-7d15-448b-8062-fdf7f5a25d66
md"
### Raw data
"

# ╔═╡ 5f5ce9fe-c168-431e-bd48-19e06f81eabd
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

# ╔═╡ c4559cb4-98c2-4dae-b1d1-a163783855d1
md"
### Model fits
"

# ╔═╡ 84fd6ba3-f471-4550-a9da-8e09ccd927a8
begin
	# single update model
	hlwm_el_ests_s1, hlwm_el_choices_s1 = optimize_multiple(
		filter(x -> x.session == "1", df);
		model = HLWM_forget,
		estimate = "MAP",
		include_true = false,
		priors = Dict(
			:β => 25., # fixed inverse temperature
	        :a => Normal(0., 4.), # RL reward learning rate
	        :bWM => Normal(0., 4.), # punishment learning rate for working memory
			:F_wm => Normal(0., 4.), # working memory forgetting rate
	        :w0 => Beta(1.1, 1.1), # prop. of WM to RL weight (i.e., 0.5 ===)
			# :λ_wm => Beta(4., 2.) # memory eligibility trace decay
		),
		parameters = [:a_pos, :bWM, :F_wm, :w0],
		gq_struct = sess1_str,
		n_starts = 5
	)
	hlwm_el_ests_s2, hlwm_el_choices_s2 = optimize_multiple(
		filter(x -> x.session == "2", df);
		model = HLWM_forget,
		estimate = "MAP",
		include_true = false,
		priors = Dict(
			:β => 25., # fixed inverse temperature
	        :a => Normal(0., 4.), # RL reward learning rate
	        :bWM => Normal(0., 4.), # punishment learning rate for working memory
			:F_wm => Normal(0., 4.), # working memory forgetting rate
	        :w0 => Beta(1.1, 1.1), # prop. of WM to RL weight (i.e., 0.5 ===)
			# :λ_wm => Beta(4., 2.) # memory eligibility trace decay
		),
		parameters = [:a_pos, :bWM, :F_wm, :w0],
		gq_struct = sess2_str,
		n_starts = 5
	)
end

# ╔═╡ 8c4f9aeb-84e2-4942-814f-00284432bf2c
let
	f = Figure(size = (800, 300))
	
	# Define parameters
	s1a, s1e, s1ll = hlwm_el_ests_s1.a_pos, hlwm_el_ests_s1.bWM, hlwm_el_ests_s1.loglike
	s2a, s2e, s2ll = hlwm_el_ests_s2.a_pos, hlwm_el_ests_s2.bWM, hlwm_el_ests_s2.loglike
	s1w, s1d = hlwm_el_ests_s1.w0, hlwm_el_ests_s1.F_wm
	s2w, s2d = hlwm_el_ests_s2.w0, hlwm_el_ests_s2.F_wm
	# Set labels
	labs1 = (xlabel = "a_pos", ylabel = "bWM", zlabel = "log-likelihood", title = "Habit learning rate + noise")
	labs2 = (xlabel = "w0", ylabel = "F_wm", zlabel = "log-likelihood", title = "WM bias + forgetting")

	# Plot
	ax1, ax2 = Axis3(f[1,1]; labs1...), Axis3(f[1,2]; labs2...)
	scatter!(ax1, s1a, s1e, s1ll)
	scatter!(ax1, s2a, s2e, s2ll)
	scatter!(ax2, s1w, s1d, s1ll)
	scatter!(ax2, s2w, s2d, s2ll)
	f
end

# ╔═╡ 88b7c2e9-7fc2-4885-8f0e-ca349e61e349
let
	choice_df1, choice_df2 = true_vs_preds(
		hlwm_el_choices_s1,
		hlwm_el_choices_s2;
		strct1 = sess1_str,
		strct2 = sess2_str
	)
	plot_true_vs_preds(choice_df1, choice_df2)
end

# ╔═╡ e26b9dc7-7135-421f-8885-d7d8bbd7ce66
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

# ╔═╡ b338dfe6-96a6-4967-9932-35f64beabf64
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
		xcol=:w0, ycol=:w0_1, subtitle="WM weight bias"
	)
	reliability_scatter!(
		fig[2,2]; df=retest_df, xlabel="Session 1", ylabel="Session 2", 
		xcol=:F_wm, ycol=:F_wm_1, subtitle="Forgetting factor"
	)
	fig
end

# ╔═╡ 45b2382b-fd20-4ad1-ae81-adef02767fd1
md"
### Parameter recovery
"

# ╔═╡ 119bec4a-7950-48bf-bc37-7456d1e17248
begin
	fit_df = leftjoin(filter(x -> x.session == "1", df), hlwm_el_ests_s1; on=:PID)
	f_hlwm_forget = optimization_calibration(
		fit_df,
		optimize_multiple,
		estimate = "MAP",
		model = HLWM_forget,
		priors = Dict(
			:β => 25., # fixed inverse temperature
	        :a => Normal(0., 4.), # RL reward learning rate
	        :bWM => Normal(0., 4.), # punishment learning rate for working memory
			:F_wm => Normal(0., 4.), # working memory forgetting rate
	        :w0 => Beta(1.1, 1.1), # prop. of WM to RL weight (i.e., 0.5 ===)
		),
		parameters = [:a_pos, :bWM, :F_wm, :w0],
		n_starts = 5
	)

	f_hlwm_forget
end

# ╔═╡ 0d1aa1e3-abb1-42b9-8108-7acb156737f0
begin
	fit_df2 = leftjoin(filter(x -> x.session == "2", df), hlwm_el_ests_s2; on=:PID)
	f_hlwm_forget_s2 = optimization_calibration(
		fit_df2,
		optimize_multiple,
		estimate = "MAP",
		model = HLWM_forget,
		priors = Dict(
			:β => 25., # fixed inverse temperature
	        :a => Normal(0., 4.), # RL reward learning rate
	        :bWM => Normal(0., 4.), # punishment learning rate for working memory
			:F_wm => Normal(0., 4.), # working memory forgetting rate
	        :w0 => Beta(1.1, 1.1), # prop. of WM to RL weight (i.e., 0.5 ===)
		),
		parameters = [:a_pos, :bWM, :F_wm, :w0],
		n_starts = 5
	)

	f_hlwm_forget_s2
end

# ╔═╡ Cell order:
# ╟─8e96fc42-2eda-4aca-a766-fe6793ebe2ed
# ╠═3647aaf8-becd-11ef-17b7-1348cec601a2
# ╠═a0300a7d-600b-4661-aadd-6afbaebe0486
# ╟─d0d375c2-b3ff-4d24-bcf9-3a333f15c5fa
# ╟─3b6d2229-7d15-448b-8062-fdf7f5a25d66
# ╠═5f5ce9fe-c168-431e-bd48-19e06f81eabd
# ╟─c4559cb4-98c2-4dae-b1d1-a163783855d1
# ╠═84fd6ba3-f471-4550-a9da-8e09ccd927a8
# ╠═8c4f9aeb-84e2-4942-814f-00284432bf2c
# ╠═88b7c2e9-7fc2-4885-8f0e-ca349e61e349
# ╟─e26b9dc7-7135-421f-8885-d7d8bbd7ce66
# ╠═b338dfe6-96a6-4967-9932-35f64beabf64
# ╟─45b2382b-fd20-4ad1-ae81-adef02767fd1
# ╠═119bec4a-7950-48bf-bc37-7456d1e17248
# ╠═0d1aa1e3-abb1-42b9-8108-7acb156737f0