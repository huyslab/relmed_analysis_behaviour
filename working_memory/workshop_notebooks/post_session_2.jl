### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ cd74d0cc-a0d4-4b93-ae70-36f2e961744d
begin
	cd("/home/jovyan")
	import Pkg
	
	# activate the shared project environment
    Pkg.activate("$(pwd())/relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
		using CairoMakie, Random, DataFrames, Distributions, StatsBase,
		ForwardDiff, LinearAlgebra, JLD2, FileIO, CSV, Dates, JSON, RCall, Turing, ParetoSmooth, MCMCDiagnosticTools, Printf, AlgebraOfGraphics
		using LogExpFunctions: logistic, logit
	
	Turing.setprogress!(false)
	
	include("$(pwd())/fetch_preprocess_data.jl")
	include("$(pwd())/working_memory/simulate.jl")
	include("$(pwd())/working_memory/RL+WM_multiaction_models.jl")	
	include("$(pwd())/working_memory/plotting_utils.jl")
	include("$(pwd())/working_memory/model_utils.jl")

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

# ╔═╡ 252143a6-a665-11ef-342a-611b7078e403
md"
## Plots for working memory in workshop
"

# ╔═╡ d3fa9ccc-373d-4b6c-a957-c44a970c4664
begin
	# Load data
	_, _, _, _, _, WM_data, _, _ = load_pilot6_data()
	df = prepare_WM_data(WM_data)
	
	# structure
	pilot6_wm = load_wm_structure_csv("pilot6_WM") # task structure for ppcs
	sess1_str = filter(x -> x.block <= 10, pilot6_wm)
	sess2_str = filter(x -> x.block > 10, pilot6_wm)
	nothing
end

# ╔═╡ c34ed6ad-4188-4fe8-bd54-8dded373728f
begin
	function true_vs_preds(df1::DataFrame, df2::DataFrame; strct::DataFrame)
		df1 = leftjoin(df1, strct; on = [:block, :valence, :trial, :pair])
		df2 = leftjoin(df2, strct; on = [:block, :valence, :trial, :pair])
		
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
			f[1,1], filter(x -> x.set_size == 3, choice_df1);
			choice_val = 3.0, group = :variable, 
			pid_col = :PID, error_band = "se", legend = true,
			legend_pos = :bottom, legend_rows = 2, 
			title = "Single (set size = 3)", colors = Makie.colorschemes[:Reds_3]
		)
		plot_prior_accuracy!(
			f[1,2], filter(x -> x.set_size == 3, choice_df2);
			choice_val = 3.0, group = :variable,
		    pid_col = :PID, error_band = "se",legend = true,
			legend_pos = :bottom, legend_rows = 2,
			title = "Reciprocal (set size = 3)", colors = Makie.colorschemes[:Reds_3]
		)
		plot_prior_accuracy!(
			f[2,1], filter(x -> x.set_size == 21, choice_df1);
			choice_val = 3.0, group = :variable, 
			pid_col = :PID, error_band = "se", legend = true, legend_pos = :bottom, legend_rows = 2, 
			title = "Single (set size = 21)", colors = Makie.colorschemes[:Blues_3]
		)
		plot_prior_accuracy!(
			f[2,2], filter(x -> x.set_size == 21, choice_df2);
			choice_val = 3.0, group = :variable, 
		    pid_col = :PID, error_band = "se",legend = true,
			legend_pos = :bottom, legend_rows = 2,
			title = "Reciprocal (set size = 21)", colors = Makie.colorschemes[:Blues_3]
		)
		return f
	end
end

# ╔═╡ 815a8229-1420-479c-a6b7-e835afaa58da
function workshop_reliability_scatter!(
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
	
	# Spearman-Brown correction
	if correct_r
		r = spearman_brown(r)
	end

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
			valign = 0.025,
			tellheight = false,
			tellwidth = false
		)
	end

end

# ╔═╡ 8937e80e-81a1-4314-9391-213994933679
md"
## Choice accuracy by set size
"

# ╔═╡ 9a951062-ab2e-4fc6-9826-b45718caf366
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

# ╔═╡ 73f18d96-fe5c-48ad-87bb-968c0df3bdd5
md"
## Model fits
"

# ╔═╡ 9e36111d-6b22-49ec-8c3e-ef3701ae4e9c
md"
#### Q-learning model with two learning rates
"

# ╔═╡ 87479aa6-d8a2-4801-860f-df408243dc41
begin
	# single update model
	qs_ests_s1, qs_choices_s1 = optimize_multiple(
		filter(x -> x.session == "1", df);
		model = RL_multi_2set,
		estimate = "MAP",
		include_true = false,
		priors = Dict(
			:ρ => truncated(Normal(0., 5.), lower = 0.),
			:a1 => Normal(0., 2.),
        	:a2 => Normal(0., 2.)
		),
		parameters = [:ρ, :a1, :a2],
		transformed = Dict(:a1 => :α1, :a2 => :α2),
		gq_struct = sess1_str,
		n_starts = 5
	)
	qs_ests_s2, qs_choices_s2 = optimize_multiple(
		filter(x -> x.session == "2", df);
		model = RL_multi_2set,
		estimate = "MAP",
		include_true = false,
		priors = Dict(
			:ρ => truncated(Normal(0., 5.), lower = 0.),
			:a1 => Normal(0., 2.),
        	:a2 => Normal(0., 2.)
		),
		parameters = [:ρ, :a1, :a2],
		transformed = Dict(:a1 => :α1, :a2 => :α2),
		gq_struct = sess2_str,
		n_starts = 5
	)

	# reciprocal update model
	qr_ests_s1, qr_choices_s1 = optimize_multiple(
		filter(x -> x.session == "1", df);
		model = RL_multi_2set_recip,
		estimate = "MAP",
		include_true = false,
		priors = Dict(
			:ρ => truncated(Normal(0., 5.), lower = 0.),
			:a1 => Normal(0., 2.),
        	:a2 => Normal(0., 2.)
		),
		parameters = [:ρ, :a1, :a2],
		transformed = Dict(:a1 => :α1, :a2 => :α2),
		gq_struct = sess1_str,
		n_starts = 5
	)
	qr_ests_s2, qr_choices_s2 = optimize_multiple(
		filter(x -> x.session == "2", df);
		model = RL_multi_2set_recip,
		estimate = "MAP",
		include_true = false,
		priors = Dict(
			:ρ => truncated(Normal(0., 5.), lower = 0.),
			:a1 => Normal(0., 2.),
        	:a2 => Normal(0., 2.)
		),
		parameters = [:ρ, :a1, :a2],
		transformed = Dict(:a1 => :α1, :a2 => :α2),
		gq_struct = sess2_str,
		n_starts = 5
	)
	
	nothing
end

# ╔═╡ f0c0fb08-bbb8-4876-ad6d-eab5f817d4e5
let
	f = Figure(size = (800, 300))
	
	# Define parameters
	s1ρ1, s1α1s, s1ll1 = qs_ests_s1.ρ, qs_ests_s1.α1, qs_ests_s1.loglike
	s2ρ1, s2α1s, s2ll1 = qs_ests_s2.ρ, qs_ests_s2.α1, qs_ests_s2.loglike
	s1ρ2, s1α1r, s1ll2 = qr_ests_s1.ρ, qr_ests_s1.α1, qr_ests_s1.loglike
	s2ρ2, s2α1r, s2ll2 = qr_ests_s2.ρ, qr_ests_s2.α1, qr_ests_s2.loglike

	# Set labels
	labs1 = (xlabel = "ρ", ylabel = "α1", zlabel = "log-likelihood", title = "Single")
	labs2 = (xlabel = "ρ", ylabel = "α1", zlabel = "log-likelihood", title = "Reciprocal")

	# Plot
	ax1, ax2 = Axis3(f[1,1]; labs1...), Axis3(f[1,2]; labs2...)
	scatter!(ax1, s1ρ1, s1α1s, s1ll1)
	scatter!(ax1, s2ρ1, s2α1s, s2ll1)
	scatter!(ax2, s1ρ2, s1α1r, s1ll2)
	scatter!(ax2, s2ρ2, s2α1r, s2ll2)
	f
end

# ╔═╡ 11e19837-c3c6-40c8-bd3b-e1ff9fca00b6
md"
#### Palimpsest working memory model
"

# ╔═╡ 50323172-e475-4cdd-99cc-a3419ac22887
md"
A) **Overall** capacity + no averaging
"

# ╔═╡ 53d8fa96-127a-42d4-9928-d3f36d616f45
begin
	# single update model
	wm_ests_s1, wm_choices_s1 = optimize_multiple(
		filter(x -> x.session == "1", df);
		model = WM_multi_all_outc_pmst_sgd_sum,
		estimate = "MAP",
		include_true = false,
		priors = Dict(
			:ρ => truncated(Normal(0., 3.), lower = 0.),
			:C => truncated(Normal(5., 3.), lower = 1.)
		),
		parameters = [:ρ, :C],
		gq_struct = sess1_str,
		n_starts = 5
	)
	wm_ests_s2, wm_choices_s2 = optimize_multiple(
		filter(x -> x.session == "2", df);
		model = WM_multi_all_outc_pmst_sgd_sum,
		estimate = "MAP",
		include_true = false,
		priors = Dict(
			:ρ => truncated(Normal(0., 3.), lower = 0.),
			:C => truncated(Normal(5., 3.), lower = 1.)
		),
		parameters = [:ρ, :C],
		gq_struct = sess2_str,
		n_starts = 5
	)
	
	# reciprocal update model
	wmr_ests_s1, wmr_choices_s1 = optimize_multiple(
		filter(x -> x.session == "1", df);
		model = WM_multi_all_outc_pmst_sgd_sum_recip,
		estimate = "MAP",
		include_true = false,
		priors = Dict(
			:ρ => truncated(Normal(0., 3.), lower = 0.),
			:C => truncated(Normal(5., 3.), lower = 1.)
		),
		parameters = [:ρ, :C],
		gq_struct = sess1_str,
		n_starts = 5
	)
	wmr_ests_s2, wmr_choices_s2 = optimize_multiple(
		filter(x -> x.session == "2", df);
		model = WM_multi_all_outc_pmst_sgd_sum_recip,
		estimate = "MAP",
		include_true = false,
		priors = Dict(
			:ρ => truncated(Normal(0., 3.), lower = 0.),
			:C => truncated(Normal(5., 3.), lower = 1.)
		),
		parameters = [:ρ, :C],
		gq_struct = sess2_str,
		n_starts = 5
	)
	nothing
end

# ╔═╡ 09c121c4-dd34-42c6-a506-d05634f53f4b
let
	f = Figure(size = (800, 300))
	
	# Define parameters
	s1ρ1, s1C1, s1ll1 = wm_ests_s1.ρ, wm_ests_s1.C, wm_ests_s1.loglike
	s2ρ1, s2C1, s2ll1 = wm_ests_s2.ρ, wm_ests_s2.C, wm_ests_s2.loglike
	s1ρ2, s1C2, s1ll2 = wmr_ests_s1.ρ, wmr_ests_s1.C, wmr_ests_s1.loglike
	s2ρ2, s2C2, s2ll2 = wmr_ests_s2.ρ, wmr_ests_s2.C, wmr_ests_s2.loglike

	# Set labels
	labs1 = (xlabel = "ρ", ylabel = "C", zlabel = "log-likelihood", title = "Single")
	labs2 = (xlabel = "ρ", ylabel = "C", zlabel = "log-likelihood", title = "Reciprocal")

	# Plot
	ax1, ax2 = Axis3(f[1,1]; labs1...), Axis3(f[1,2]; labs2...)
	scatter!(ax1, s1ρ1, s1C1, s1ll1)
	scatter!(ax1, s2ρ1, s2C1, s2ll1)
	scatter!(ax2, s1ρ2, s1C2, s1ll2)
	scatter!(ax2, s2ρ2, s2C2, s2ll2)
	f
end

# ╔═╡ 95e141ba-7011-4e9c-84a8-7605f3516db1
md"
B) **Stimulus-specific** capacity + no averaging
"

# ╔═╡ a235c7c1-f5a6-4bf3-8ddc-9ec44f56117e
begin
	# single update model
	wmss_ests_s1, wmss_choices_s1 = optimize_multiple(
		filter(x -> x.session == "1", df);
		model = WM_multi_pmst_sgd_sum,
		estimate = "MAP",
		include_true = false,
		priors = Dict(
			:ρ => truncated(Normal(0., 3.), lower = 0.),
			:C => truncated(Normal(3., 3.), lower = 1.)
		),
		parameters = [:ρ, :C],
		gq_struct = sess1_str,
		n_starts = 5
	)
	wmss_ests_s2, wmss_choices_s2 = optimize_multiple(
		filter(x -> x.session == "2", df);
		model = WM_multi_pmst_sgd_sum,
		estimate = "MAP",
		include_true = false,
		priors = Dict(
			:ρ => truncated(Normal(0., 3.), lower = 0.),
			:C => truncated(Normal(3., 3.), lower = 1.)
		),
		parameters = [:ρ, :C],
		gq_struct = sess2_str,
		n_starts = 5
	)
	
	# reciprocal update model
	wmssr_ests_s1, wmssr_choices_s1 = optimize_multiple(
		filter(x -> x.session == "1", df);
		model = WM_multi_pmst_sgd_sum_recip,
		estimate = "MAP",
		include_true = false,
		priors = Dict(
			:ρ => truncated(Normal(0., 3.), lower = 0.),
			:C => truncated(Normal(3., 3.), lower = 1.)
		),
		parameters = [:ρ, :C],
		gq_struct = sess1_str,
		n_starts = 5
	)
	wmssr_ests_s2, wmssr_choices_s2 = optimize_multiple(
		filter(x -> x.session == "2", df);
		model = WM_multi_pmst_sgd_sum_recip,
		estimate = "MAP",
		include_true = false,
		priors = Dict(
			:ρ => truncated(Normal(0., 3.), lower = 0.),
			:C => truncated(Normal(3., 3.), lower = 1.)
		),
		parameters = [:ρ, :C],
		gq_struct = sess2_str,
		n_starts = 5
	)
	nothing
end

# ╔═╡ af0e6ebe-249b-46ab-9f0f-e44caf8055df
let
	f = Figure(size = (800, 300))
	
	# Define parameters
	s1ρ1, s1C1, s1ll1 = wmss_ests_s1.ρ, wmss_ests_s1.C, wmss_ests_s1.loglike
	s2ρ1, s2C1, s2ll1 = wmss_ests_s2.ρ, wmss_ests_s2.C, wmss_ests_s2.loglike
	s1ρ2, s1C2, s1ll2 = wmssr_ests_s1.ρ, wmssr_ests_s1.C, wmssr_ests_s1.loglike
	s2ρ2, s2C2, s2ll2 = wmssr_ests_s2.ρ, wmssr_ests_s2.C, wmssr_ests_s2.loglike

	# Set labels
	labs1 = (xlabel = "ρ", ylabel = "C", zlabel = "log-likelihood", title = "Single")
	labs2 = (xlabel = "ρ", ylabel = "C", zlabel = "log-likelihood", title = "Reciprocal")

	# Plot
	ax1, ax2 = Axis3(f[1,1]; labs1...), Axis3(f[1,2]; labs2...)
	scatter!(ax1, s1ρ1, s1C1, s1ll1)
	scatter!(ax1, s2ρ1, s2C1, s2ll1)
	scatter!(ax2, s1ρ2, s1C2, s1ll2)
	scatter!(ax2, s2ρ2, s2C2, s2ll2)
	f
end

# ╔═╡ 3a2f75c0-62ed-4a9b-8e09-a771ee5c3830
md"
### Posterior predictions
"

# ╔═╡ 76299691-ba71-4c6b-8226-aa0036a0c074
md"
#### Q-learning model
"

# ╔═╡ 8df1a8b6-3de5-40a2-b549-702402895c7f
md"
**Session 1**
"

# ╔═╡ 13b6ccfa-a377-4158-8809-17f16e9511b2
let
	choice_df1, choice_df2 = true_vs_preds(qs_choices_s1, qr_choices_s1; strct = sess1_str)
	plot_true_vs_preds(choice_df1, choice_df2)
end

# ╔═╡ dcbdbe12-2c0c-4205-98b0-fe04d2d8812a
md"
**Session 2**
"

# ╔═╡ e45c5d7f-70f9-4caa-a16f-8ed6dd89e8dc
let
	choice_df1, choice_df2 = true_vs_preds(qs_choices_s2, qr_choices_s2; strct = sess2_str)
	plot_true_vs_preds(choice_df1, choice_df2)
end

# ╔═╡ e1d2f0ee-a6b4-4ccd-93dd-f35271b3096e
md"
#### Working memory models
"

# ╔═╡ cc653756-8962-4f27-a8ec-95f848f5016c
md"
A) **Overall** capacity + no averaging
"

# ╔═╡ 8fc367ad-0a43-4df1-94fd-6b9c5f6a3973
md"
**Session 1**
"

# ╔═╡ 3c330108-a4fc-41f5-a6ae-04077f03345c
let
	choice_df1, choice_df2 = true_vs_preds(wm_choices_s1, wmr_choices_s1; strct = sess1_str)
	plot_true_vs_preds(choice_df1, choice_df2)
end

# ╔═╡ e880e378-616d-4020-a4c1-e3b918e3bc81
md"
**Session 2**
"

# ╔═╡ eecbad77-245b-4637-bc1a-f5930afbdc24
let
	choice_df1, choice_df2 = true_vs_preds(wm_choices_s2, wmr_choices_s2; strct = sess2_str)
	plot_true_vs_preds(choice_df1, choice_df2)
end

# ╔═╡ 43eddd3a-7518-47e8-a988-343dd57fd8fb
md"
B) **Stimulus-specific** capacity + no averaging
"

# ╔═╡ 64e0d111-94db-43f9-afb6-b1f538e7d4ad
md"
**Session 1**
"

# ╔═╡ 36f6b691-997b-490a-8f86-a71d565b2ce9
let
	choice_df1, choice_df2 = true_vs_preds(wmss_choices_s1, wmssr_choices_s1; strct = sess1_str)
	plot_true_vs_preds(choice_df1, choice_df2)
end

# ╔═╡ 90af826c-9ba4-4496-b37b-a1716dcc2f5a
md"
**Session 2**
"

# ╔═╡ 2edec29a-c4c3-4cd8-baf3-03f71ada674b
let
	choice_df1, choice_df2 = true_vs_preds(wmss_choices_s2, wmssr_choices_s2; strct = sess2_str)
	plot_true_vs_preds(choice_df1, choice_df2)
end

# ╔═╡ 5950c9d6-c581-493d-b726-beba0315a8eb
md"
## Test-retest
"

# ╔═╡ f1b6aaf2-0226-4d2a-bad0-71e4471857e3
md"
### Extended Q-learning model
"

# ╔═╡ 2938d136-24c7-4162-bf27-88a9c4a35702
begin
	qs_splithalf = optimize_multiple_by_factor(
		df;
		model = RL_multi_2set,
		factor = :session,
		priors = Dict(
			:ρ => truncated(Normal(0., 5.), lower = 0.),
			:a1 => Normal(0., 2.),
        	:a2 => Normal(0., 2.)
		),
		estimate = "MAP",
		include_true = false,
		parameters = [:ρ, :a1, :a2],
		transformed = Dict(:a1 => :α1, :a2 => :α2),
		n_starts = 5
	)
	# reciprocal update model
	qr_splithalf = optimize_multiple_by_factor(
		df;
		model = RL_multi_2set_recip,
		factor = :session,
		priors = Dict(
			:ρ => truncated(Normal(0., 5.), lower = 0.),
			:a1 => Normal(0., 2.),
        	:a2 => Normal(0., 2.)
		),
		estimate = "MAP",
		include_true = false,
		parameters = [:ρ, :a1, :a2],
		transformed = Dict(:a1 => :α1, :a2 => :α2),
		n_starts = 5
	)
	nothing
end

# ╔═╡ 1764420b-3cb9-4516-9b28-33ea6e7cf985
md"
**Single update**
"

# ╔═╡ 8398a4d3-2161-49ac-b6eb-374228aef9f2
let
	retest_df = leftjoin(qs_splithalf..., on = :PID, makeunique=true)
	dropmissing!(retest_df)
	fig=Figure(;size=(800, 700))
	workshop_reliability_scatter!(
		fig[1,1:2];
		df=retest_df,
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=:ρ,
		ycol=:ρ_1,
		subtitle="Test-retest: reward sensitivity",
		correct_r=false
	)
	workshop_reliability_scatter!(
		fig[2,1];
		df=retest_df,
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=:α1,
		ycol=:α1_1,
		subtitle="Test-retest: learning rate (set-size = 3)",
		correct_r=false
	)
	workshop_reliability_scatter!(
		fig[2,2];
		df=retest_df,
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=:α2,
		ycol=:α2_1,
		subtitle="Test-retest: learning rate (set-size = 21)",
		correct_r=false
	)
	fig
end

# ╔═╡ c5590646-9909-46a9-95bb-79f79f377008
md"
**Reciprocal update**
"

# ╔═╡ b2971e33-797c-4c1c-be55-c18830067982
let
	retest_df = leftjoin(qr_splithalf..., on = :PID, makeunique=true)
	dropmissing!(retest_df)
	fig=Figure(;size=(800, 700))
	workshop_reliability_scatter!(
		fig[1,1:2];
		df=retest_df,
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=:ρ,
		ycol=:ρ_1,
		subtitle="Test-retest: reward sensitivity",
		correct_r=false
	)
	workshop_reliability_scatter!(
		fig[2,1];
		df=retest_df,
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=:α1,
		ycol=:α1_1,
		subtitle="Test-retest: learning rate (set-size = 3)",
		correct_r=false
	)
	workshop_reliability_scatter!(
		fig[2,2];
		df=retest_df,
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=:α2,
		ycol=:α2_1,
		subtitle="Test-retest: learning rate (set-size = 21)",
		correct_r=false
	)
	fig
end

# ╔═╡ 147ffa2a-1524-4daa-b228-05a5d4e0c292
md"
### Palimpsest working memory models
"

# ╔═╡ 59643cec-9163-45ea-b0cc-9a1402aa9b52
md"
A) **Overall** capacity
"

# ╔═╡ c902b164-6d41-4d7a-a8c5-c58dbdcd9853
begin
	wm_splithalf = optimize_multiple_by_factor(
		df;
		model = WM_multi_all_outc_pmst_sgd_sum,
		factor = :session,
		priors = Dict(
			:ρ => truncated(Normal(0., 3.), lower = 0.),
			:C => truncated(Normal(5., 3.), lower = 1.)
		),
		estimate = "MAP",
		include_true = false,
		parameters = [:ρ, :C],
		n_starts = 5
	)
	# reciprocal update model
	wmr_splithalf = optimize_multiple_by_factor(
		df;
		model = WM_multi_all_outc_pmst_sgd_sum_recip,
		factor = :session,
		priors = Dict(
			:ρ => truncated(Normal(0., 3.), lower = 0.),
			:C => truncated(Normal(5., 3.), lower = 1.)
		),
		estimate = "MAP",
		include_true = false,
		parameters = [:ρ, :C],
		n_starts = 5
	)
	nothing
end

# ╔═╡ a4e26c75-7c77-4c09-9149-f8f236f0c9d7
md"
**Single update**
"

# ╔═╡ 609f38ad-a107-4dcb-a109-03a81470eb84
let
	retest_df = leftjoin(wm_splithalf..., on = :PID, makeunique=true)
	dropmissing!(retest_df)
	fig=Figure(;size=(800, 700))
	workshop_reliability_scatter!(
		fig[1,1];
		df=retest_df,
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=:ρ,
		ycol=:ρ_1,
		subtitle="Test-retest: reward sensitivity",
		correct_r=false
	)
	workshop_reliability_scatter!(
		fig[2,1];
		df=retest_df,
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=:C,
		ycol=:C_1,
		subtitle="Test-retest: overall capacity",
		correct_r=false
	)
	fig
end

# ╔═╡ 06081dd1-78d8-4bad-a64b-703619e7f12e
md"
**Reciprocal update**
"

# ╔═╡ f80c84e3-41c9-429e-b39b-0050f057bd9d
let
	retest_df = leftjoin(wmr_splithalf..., on = :PID, makeunique=true)
	dropmissing!(retest_df)
	fig=Figure(;size=(800, 700))
	workshop_reliability_scatter!(
		fig[1,1];
		df=retest_df,
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=:ρ,
		ycol=:ρ_1,
		subtitle="Test-retest: reward sensitivity",
		correct_r=false
	)
	workshop_reliability_scatter!(
		fig[2,1];
		df=retest_df,
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=:C,
		ycol=:C_1,
		subtitle="Test-retest: overall capacity",
		correct_r=false
	)
	fig
end

# ╔═╡ f5a78d8e-ab11-4204-8337-2db5176c6718
md"
B) **Stimulus-specific** capacity
"

# ╔═╡ b0242d24-2463-492c-81d7-61f0ef6f7583
begin
	wmss_splithalf = optimize_multiple_by_factor(
		df;
		model = WM_multi_pmst_sgd_sum,
		factor = :session,
		priors = Dict(
			:ρ => truncated(Normal(0., 3.), lower = 0.),
			:C => truncated(Normal(3., 3.), lower = 1.)
		),
		estimate = "MAP",
		include_true = false,
		parameters = [:ρ, :C],
		n_starts = 5
	)
	# reciprocal update model
	wmssr_splithalf = optimize_multiple_by_factor(
		df;
		model = WM_multi_pmst_sgd_sum_recip,
		factor = :session,
		priors = Dict(
			:ρ => truncated(Normal(0., 3.), lower = 0.),
			:C => truncated(Normal(3., 3.), lower = 1.)
		),
		estimate = "MAP",
		include_true = false,
		parameters = [:ρ, :C],
		n_starts = 5
	)
	nothing
end

# ╔═╡ 395efdab-12c6-43d0-8bb7-8714c0af8f54
md"
**Single update**
"

# ╔═╡ 4756e64e-0a1b-4a4c-8408-e3fe6e7c7ecb
let
	retest_df = leftjoin(wmss_splithalf..., on = :PID, makeunique=true)
	dropmissing!(retest_df)
	fig=Figure(;size=(800, 700))
	workshop_reliability_scatter!(
		fig[1,1];
		df=retest_df,
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=:ρ,
		ycol=:ρ_1,
		subtitle="Test-retest: reward sensitivity",
		correct_r=false
	)
	workshop_reliability_scatter!(
		fig[2,1];
		df=retest_df,
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=:C,
		ycol=:C_1,
		subtitle="Test-retest: stimulus-specific capacity",
		correct_r=false
	)
	fig
end

# ╔═╡ 140b5d49-c254-4325-b942-e78d933f3a37
md"
**Reciprocal update**
"

# ╔═╡ 795450a9-1f4c-41eb-8af4-06bb3bcfeae5
let
	retest_df = leftjoin(wmssr_splithalf..., on = :PID, makeunique=true)
	dropmissing!(retest_df)
	fig=Figure(;size=(800, 700))
	workshop_reliability_scatter!(
		fig[1,1];
		df=retest_df,
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=:ρ,
		ycol=:ρ_1,
		subtitle="Test-retest: reward sensitivity",
		correct_r=false
	)
	workshop_reliability_scatter!(
		fig[2,1];
		df=retest_df,
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=:C,
		ycol=:C_1,
		subtitle="Test-retest: stimulus-specific capacity",
		correct_r=false
	)
	fig
end

# ╔═╡ Cell order:
# ╟─252143a6-a665-11ef-342a-611b7078e403
# ╠═cd74d0cc-a0d4-4b93-ae70-36f2e961744d
# ╠═d3fa9ccc-373d-4b6c-a957-c44a970c4664
# ╟─c34ed6ad-4188-4fe8-bd54-8dded373728f
# ╟─815a8229-1420-479c-a6b7-e835afaa58da
# ╟─8937e80e-81a1-4314-9391-213994933679
# ╠═9a951062-ab2e-4fc6-9826-b45718caf366
# ╟─73f18d96-fe5c-48ad-87bb-968c0df3bdd5
# ╟─9e36111d-6b22-49ec-8c3e-ef3701ae4e9c
# ╠═87479aa6-d8a2-4801-860f-df408243dc41
# ╠═f0c0fb08-bbb8-4876-ad6d-eab5f817d4e5
# ╟─11e19837-c3c6-40c8-bd3b-e1ff9fca00b6
# ╟─50323172-e475-4cdd-99cc-a3419ac22887
# ╠═53d8fa96-127a-42d4-9928-d3f36d616f45
# ╠═09c121c4-dd34-42c6-a506-d05634f53f4b
# ╟─95e141ba-7011-4e9c-84a8-7605f3516db1
# ╠═a235c7c1-f5a6-4bf3-8ddc-9ec44f56117e
# ╠═af0e6ebe-249b-46ab-9f0f-e44caf8055df
# ╠═3a2f75c0-62ed-4a9b-8e09-a771ee5c3830
# ╟─76299691-ba71-4c6b-8226-aa0036a0c074
# ╟─8df1a8b6-3de5-40a2-b549-702402895c7f
# ╠═13b6ccfa-a377-4158-8809-17f16e9511b2
# ╟─dcbdbe12-2c0c-4205-98b0-fe04d2d8812a
# ╠═e45c5d7f-70f9-4caa-a16f-8ed6dd89e8dc
# ╟─e1d2f0ee-a6b4-4ccd-93dd-f35271b3096e
# ╟─cc653756-8962-4f27-a8ec-95f848f5016c
# ╟─8fc367ad-0a43-4df1-94fd-6b9c5f6a3973
# ╠═3c330108-a4fc-41f5-a6ae-04077f03345c
# ╟─e880e378-616d-4020-a4c1-e3b918e3bc81
# ╠═eecbad77-245b-4637-bc1a-f5930afbdc24
# ╟─43eddd3a-7518-47e8-a988-343dd57fd8fb
# ╟─64e0d111-94db-43f9-afb6-b1f538e7d4ad
# ╠═36f6b691-997b-490a-8f86-a71d565b2ce9
# ╟─90af826c-9ba4-4496-b37b-a1716dcc2f5a
# ╠═2edec29a-c4c3-4cd8-baf3-03f71ada674b
# ╟─5950c9d6-c581-493d-b726-beba0315a8eb
# ╟─f1b6aaf2-0226-4d2a-bad0-71e4471857e3
# ╠═2938d136-24c7-4162-bf27-88a9c4a35702
# ╟─1764420b-3cb9-4516-9b28-33ea6e7cf985
# ╠═8398a4d3-2161-49ac-b6eb-374228aef9f2
# ╟─c5590646-9909-46a9-95bb-79f79f377008
# ╠═b2971e33-797c-4c1c-be55-c18830067982
# ╟─147ffa2a-1524-4daa-b228-05a5d4e0c292
# ╟─59643cec-9163-45ea-b0cc-9a1402aa9b52
# ╠═c902b164-6d41-4d7a-a8c5-c58dbdcd9853
# ╟─a4e26c75-7c77-4c09-9149-f8f236f0c9d7
# ╠═609f38ad-a107-4dcb-a109-03a81470eb84
# ╟─06081dd1-78d8-4bad-a64b-703619e7f12e
# ╠═f80c84e3-41c9-429e-b39b-0050f057bd9d
# ╟─f5a78d8e-ab11-4204-8337-2db5176c6718
# ╠═b0242d24-2463-492c-81d7-61f0ef6f7583
# ╟─395efdab-12c6-43d0-8bb7-8714c0af8f54
# ╠═4756e64e-0a1b-4a4c-8408-e3fe6e7c7ecb
# ╟─140b5d49-c254-4325-b942-e78d933f3a37
# ╠═795450a9-1f4c-41eb-8af4-06bb3bcfeae5
