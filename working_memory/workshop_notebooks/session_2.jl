### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ 54a03c1e-a2b8-11ef-2c40-695bce51ab75
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

# ╔═╡ 11ebe9db-0809-48e7-ab4b-d5bee1fae9b1
md"
# Pilot session 2 for workshop
"

# ╔═╡ 2a0c2f77-4af2-44f0-adbe-3d9e52d24a59
begin
	# Load data
	_, _, _, _, _, WM_data, _, _ = load_pilot6_data()
	df = prepare_WM_data(WM_data)
	filter!(x -> x.session == "2", df)
	pilot6_wm = load_wm_structure_csv("pilot6_WM") # task structure for ppcs
	sess2 = filter(x -> x.block <= 10, pilot6_wm)
	nothing
end

# ╔═╡ 892fc799-cc39-4a86-b899-cdcd1816c01b
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

# ╔═╡ dea7d258-486f-4ca3-8bcf-a2240fdef345
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

# ╔═╡ 0eac1304-e5c7-4022-a648-2a3ad0293d36
md"
## Choice accuracy by set size
"

# ╔═╡ 419ae40b-f067-43ab-aff7-af7817ab93db
let
	f = Figure(size = (800, 600))
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
	f
end

# ╔═╡ fa15d0b0-705c-4f62-8d94-4d7dec28b487
md"
#### Splits
"

# ╔═╡ 242afc6b-5881-47c4-b86b-f117d6c213d1
begin
	df.early .= Float64.(df.block) .<= maximum(df.block)/2
	DataFrames.transform!(
		groupby(df, [:prolific_pid]), :trial_index => (x -> ordinalrank(x, rev = false)) => :trial_number
	)
	df.even .= Int64.(df.trial_number) .% 2 .=== 0
	nothing
end

# ╔═╡ c1bca834-a7bd-470e-831f-3ddf3d3c0a9a
let
	f = Figure(size = (800, 600))
	plot_prior_accuracy!(
		f[1,1],
		filter(x -> x.early, df);
		choice_val = 3.,
		group = :set_size,
		pid_col = :PID,
		legend = true,
		legend_rows = 1,
		legend_pos = :bottom,
		title = "First half",
		error_band = "se"
	)
	plot_prior_accuracy!(
		f[1,2],
		filter(x -> !(x.early), df);
		choice_val = 3.,
		group = :set_size,
		pid_col = :PID,
		legend = true,
		legend_rows = 1,
		legend_pos = :bottom,
		title = "Second half",
		error_band = "se"
	)
	plot_prior_accuracy!(
		f[2, 1],
		filter(x -> x.even, df);
		choice_val = 3.,
		group = :set_size,
		pid_col = :PID,
		legend = true,
		legend_rows = 1,
		legend_pos = :bottom,
		title = "Even trials",
		error_band = "se"
	)
	plot_prior_accuracy!(
		f[2, 2],
		filter(x -> !(x.even), df);
		choice_val = 3.,
		group = :set_size,
		pid_col = :PID,
		legend = true,
		legend_rows = 1,
		legend_pos = :bottom,
		title = "Odd trials",
		error_band = "se"
	)
	f
end

# ╔═╡ 946360aa-3a1a-4ea0-b4ef-c32453e65b19
md"
### Model fits
"

# ╔═╡ 6fe27779-90e2-459e-adb9-2afc166d37f8
md"
##### Q-learning model with two learning rates
"

# ╔═╡ 3108a5a5-37b9-4b76-a529-dab5e3eecff2
begin
	# single update model
	qs_ests, qs_choices = optimize_multiple(
		df;
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
		gq_struct = sess2,
		n_starts = 10
	)
	# reciprocal update model
	qr_ests, qr_choices = optimize_multiple(
		df;
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
		gq_struct = sess2,
		n_starts = 10
	)
	nothing
end

# ╔═╡ 19d06113-a41f-4b5f-a353-225cec794db0
let
	f = Figure(size = (800, 300))
	
	# Define parameters
	ρ1, α1s, ll1 = qs_ests.ρ, qs_ests.α1, qs_ests.loglike
	ρ2, α1r, ll2 = qr_ests.ρ, qr_ests.α1, qr_ests.loglike

	# Set labels
	labs1 = (xlabel = "ρ", ylabel = "α1", zlabel = "log-likelihood", title = "Single")
	labs2 = (xlabel = "ρ", ylabel = "α1", zlabel = "log-likelihood", title = "Reciprocal")

	# Plot
	ax1, ax2 = Axis3(f[1,1]; labs1...), Axis3(f[1,2]; labs2...)
	scatter!(ax1, ρ1, α1s, ll1)
	scatter!(ax2, ρ2, α1r, ll2)
	f
end

# ╔═╡ 34db522e-26ce-4954-8660-5d4f4c5c53a5
md"
##### Palimpsest working memory model
"

# ╔═╡ aa77e9de-514b-43dd-b0c9-f542c6ab7c2f
md"
A) **Overall** capacity + averaging
"

# ╔═╡ 1c46ae0f-658c-40df-82fa-385c0b5365ee
begin
	# single update model
	wm_ests, wm_choices = optimize_multiple(
		df;
		model = WM_multi_all_outc_pmst_sgd_sum,
		estimate = "MAP",
		include_true = false,
		priors = Dict(
			:ρ => truncated(Normal(0., 3.), lower = 0.),
			:C => truncated(Normal(5., 3.), lower = 1.)
		),
		parameters = [:ρ, :C],
		gq_struct = sess2,
		n_starts = 5
	)
	# reciprocal update model
	wmr_ests, wmr_choices = optimize_multiple(
		df;
		model = WM_multi_all_outc_pmst_sgd_sum_recip,
		estimate = "MAP",
		include_true = false,
		priors = Dict(
			:ρ => truncated(Normal(0., 3.), lower = 0.),
			:C => truncated(Normal(5., 3.), lower = 1.)
		),
		parameters = [:ρ, :C],
		gq_struct = sess2,
		n_starts = 5
	)
	nothing
end

# ╔═╡ 03918685-bda8-46b2-a1ac-be20e8781e9f
let
	f = Figure(size = (800, 300))
	
	# Define parameters
	ρ1, C1, ll1 = wm_ests.ρ, wm_ests.C, wm_ests.loglike
	ρ2, C2, ll2 = wmr_ests.ρ, wmr_ests.C, wmr_ests.loglike

	# Set labels
	labs1 = (xlabel = "ρ", ylabel = "C", zlabel = "log-likelihood", title = "Single")
	labs2 = (xlabel = "ρ", ylabel = "C", zlabel = "log-likelihood", title = "Reciprocal")

	# Plot
	ax1, ax2 = Axis3(f[1,1]; labs1...), Axis3(f[1,2]; labs2...)
	scatter!(ax1, ρ1, C1, ll1)
	scatter!(ax2, ρ2, C2, ll2)
	f
end

# ╔═╡ 157d7977-0c94-47bf-9df0-47abf0887eeb
# ╠═╡ disabled = true
#=╠═╡
md"
B) **Stimulus-specific** capacity + no averaging
"
  ╠═╡ =#

# ╔═╡ 6429d353-3892-4d62-9a9e-cf167639e4fe
# ╠═╡ disabled = true
#=╠═╡
begin
	# single update model
	wmss_ests, wmss_choices = optimize_multiple(
		df;
		model = WM_multi_pmst_sgd_sum,
		estimate = "MAP",
		include_true = false,
		priors = Dict(
			:ρ => truncated(Normal(0., 3.), lower = 0.),
			:C => truncated(Normal(3., 3.), lower = 1.)
		),
		parameters = [:ρ, :C],
		gq_struct = sess2,
		n_starts = 5
	)
	# reciprocal update model
	wmssr_ests, wmssr_choices = optimize_multiple(
		df;
		model = WM_multi_pmst_sgd_sum_recip,
		estimate = "MAP",
		include_true = false,
		priors = Dict(
			:ρ => truncated(Normal(0., 3.), lower = 0.),
			:C => truncated(Normal(3., 3.), lower = 1.)
		),
		parameters = [:ρ, :C],
		gq_struct = sess2,
		n_starts = 5
	)
	nothing
end
  ╠═╡ =#

# ╔═╡ 4fe85f44-2dbe-450b-859a-85cfbf5a8024
# ╠═╡ disabled = true
#=╠═╡
let
	f = Figure(size = (800, 300))
	
	# Define parameters
	ρ1, C1, ll1 = wmss_ests.ρ, wmss_ests.C, wmss_ests.loglike
	ρ2, C2, ll2 = wmssr_ests.ρ, wmssr_ests.C, wmssr_ests.loglike

	# Set labels
	labs1 = (xlabel = "ρ", ylabel = "C", zlabel = "log-likelihood", title = "Single")
	labs2 = (xlabel = "ρ", ylabel = "C", zlabel = "log-likelihood", title = "Reciprocal")

	# Plot
	ax1, ax2 = Axis3(f[1,1]; labs1...), Axis3(f[1,2]; labs2...)
	scatter!(ax1, ρ1, C1, ll1)
	scatter!(ax2, ρ2, C2, ll2)
	f
end
  ╠═╡ =#

# ╔═╡ ee33a3e7-2d4c-40a1-9255-b57f3a73df80
md"
#### Posterior predictions
"

# ╔═╡ 0faebc26-016a-4683-b26a-a248fa9a0dee
md"
##### Q-learning model
"

# ╔═╡ 56e84e90-f53c-474e-a0aa-d5d97cb8f255
let
	choice_df1, choice_df2 = true_vs_preds(qs_choices, qr_choices; strct = sess2)
	plot_true_vs_preds(choice_df1, choice_df2)
end

# ╔═╡ fa7fcb02-e12d-4bc5-8886-e50460a04186
md"
##### Working memory model
"

# ╔═╡ f266fcdb-5000-4458-9328-684d70f896f0
md"
A) **Overall** capacity
"

# ╔═╡ 9c9c4f38-3631-4a02-8a0e-f4583702e377
let
	choice_df1, choice_df2 = true_vs_preds(wm_choices, wmr_choices; strct = sess2)
	plot_true_vs_preds(choice_df1, choice_df2)
end

# ╔═╡ b34d0881-4480-480a-9673-e3cf5ba8a909
md"
B) **Stimulus-specific capacity**
"

# ╔═╡ 11e8bd77-e192-4e00-9e15-d8aa2edae2e8
#=╠═╡
let
	choice_df1, choice_df2 = true_vs_preds(wmss_choices, wmssr_choices; strct = sess2)
	plot_true_vs_preds(choice_df1, choice_df2)
end
  ╠═╡ =#

# ╔═╡ 75bf8e63-f786-4d1a-9dc2-d2ad3b2e586e
md"
## Test-retest
"

# ╔═╡ cc95996c-c5fe-4c6a-8785-ae819721bd0a
md"
#### Model-free internal consistency
"

# ╔═╡ d6eb32d4-df29-4427-9cd4-b7a5ed5a4224
let
	mean_acc = combine(
		groupby(df, [:prolific_pid, :even]), :isOptimal => mean => :mean_accuracy
	)
	mean_acc_ssz = combine(
		groupby(df, [:prolific_pid, :even, :set_size]), :isOptimal => mean => :mean_accuracy
	)
	mean_acc = unstack(mean_acc, :even, :mean_accuracy) |> dropmissing!

	mean_acc_diff = unstack(mean_acc_ssz, :set_size, :mean_accuracy)
	mean_acc_diff.ss_acc_diff .= mean_acc_diff[!, Symbol("3")] .- mean_acc_diff[!, Symbol("21")]
	mean_acc_diff = mean_acc_diff[:, [:prolific_pid, :even, :ss_acc_diff]]
	mean_acc_diff = unstack(mean_acc_diff, :even, :ss_acc_diff) |> dropmissing!
	
	mean_acc_ssz = unstack(mean_acc_ssz, :even, :mean_accuracy) |> dropmissing!

	# figure
	fig=Figure(;size=(800, 1100))
	workshop_reliability_scatter!(
		fig[1,1:2];
		df=mean_acc,
		xlabel="Even trials",
		ylabel="Odd trials",
		xcol=Symbol("true"),
		ycol=Symbol("false"),
		subtitle="Session 2 split-half: overall mean accuracy",
		correct_r=false
	)
	workshop_reliability_scatter!(
		fig[2,1];
		df=filter(x -> x.set_size == 3, mean_acc_ssz),
		xlabel="Even trials",
		ylabel="Odd trials",
		xcol=Symbol("true"),
		ycol=Symbol("false"),
		subtitle="Session 2 split-half: mean accuracy (set size = 3)",
		correct_r=false
	)
	workshop_reliability_scatter!(
		fig[2,2];
		df=filter(x -> x.set_size == 21, mean_acc_ssz),
		xlabel="Even trials",
		ylabel="Odd trials",
		xcol=Symbol("true"),
		ycol=Symbol("false"),
		subtitle="Session 2 split-half: mean accuracy (set size = 21)",
		correct_r=false
	)
	workshop_reliability_scatter!(
		fig[3,1:2];
		df=mean_acc_diff,
		xlabel="Even trials",
		ylabel="Odd trials",
		xcol=Symbol("true"),
		ycol=Symbol("false"),
		subtitle="Session 2 split-half: mean difference in accuracy",
		correct_r=false
	)
	fig
end

# ╔═╡ 26eb6122-96a2-45d2-b479-240a0da3eef8
md"
#### Extended Q-learning model
"

# ╔═╡ 519b1a8b-e600-4fcb-84cc-cc3afc5d5b06
begin
	qs_splithalf = optimize_multiple_by_factor(
		df;
		model = RL_multi_2set,
		factor = :even,
		priors = Dict(
			:ρ => truncated(Normal(0., 5.), lower = 0.),
			:a1 => Normal(0., 2.),
        	:a2 => Normal(0., 2.)
		),
		estimate = "MAP",
		include_true = false,
		parameters = [:ρ, :a1, :a2],
		# transformed = Dict(:a1 => :α1, :a2 => :α2),
		n_starts = 5
	)
	# reciprocal update model
	qr_splithalf = optimize_multiple_by_factor(
		df;
		model = RL_multi_2set_recip,
		factor = :even,
		priors = Dict(
			:ρ => truncated(Normal(0., 5.), lower = 0.),
			:a1 => Normal(0., 2.),
        	:a2 => Normal(0., 2.)
		),
		estimate = "MAP",
		include_true = false,
		parameters = [:ρ, :a1, :a2],
		# transformed = Dict(:a1 => :α1, :a2 => :α2),
		n_starts = 5
	)
	nothing
end

# ╔═╡ 1a8b8235-e31e-4d1e-9829-5cb44daeb017
md"
**Single update**
"

# ╔═╡ 372bc57f-3bde-4275-9440-a28aae76bba5
let
	retest_df = leftjoin(qs_splithalf..., on = :PID, makeunique=true)
	retest_df[!, :Δa_s1] .= retest_df.a2 .- retest_df.a1
	retest_df[!, :Δa_s2] .= retest_df.a2_1 .- retest_df.a1_1
	
	fig=Figure(;size=(800, 1000))
	workshop_reliability_scatter!(
		fig[1,1:2];
		df=retest_df,
		xlabel="Even trials",
		ylabel="Odd trials",
		xcol=:ρ,
		ycol=:ρ_1,
		subtitle="Session 2 split-half: reward sensitivity",
		correct_r=false
	)
	workshop_reliability_scatter!(
		fig[2,1];
		df=retest_df,
		xlabel="Even trials",
		ylabel="Odd trials",
		xcol=:a1,
		ycol=:a1_1,
		subtitle="Session 2 split-half: learning rate (set-size = 3)",
		correct_r=false
	)
	workshop_reliability_scatter!(
		fig[2,2];
		df=retest_df,
		xlabel="Even trials",
		ylabel="Odd trials",
		xcol=:a2,
		ycol=:a2_1,
		subtitle="Session 2 split-half: learning rate (set-size = 21)",
		correct_r=false
	)
	workshop_reliability_scatter!(
		fig[3,1:2];
		df=retest_df,
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=:Δa_s1,
		ycol=:Δa_s2,
		subtitle="Session 2 split-half: learning rate difference",
		correct_r=false
	)
	fig
end

# ╔═╡ 9a42da85-8531-4d91-944e-77a9378d6b0b
md"
**Reciprocal update**
"

# ╔═╡ 38ea51ca-f146-4f45-a32d-1aca601c5c17
let
	retest_df = leftjoin(qr_splithalf..., on = :PID, makeunique=true)
	retest_df[!, :Δa_s1] .= retest_df.a2 .- retest_df.a1
	retest_df[!, :Δa_s2] .= retest_df.a2_1 .- retest_df.a1_1
	
	fig=Figure(;size=(800, 1000))
	workshop_reliability_scatter!(
		fig[1,1:2];
		df=retest_df,
		xlabel="Even trials",
		ylabel="Odd trials",
		xcol=:ρ,
		ycol=:ρ_1,
		subtitle="Session 2 split-half: reward sensitivity",
		correct_r=false
	)
	workshop_reliability_scatter!(
		fig[2,1];
		df=retest_df,
		xlabel="Even trials",
		ylabel="Odd trials",
		xcol=:a1,
		ycol=:a1_1,
		subtitle="Session 2 split-half: learning rate (set-size = 3)",
		correct_r=false
	)
	workshop_reliability_scatter!(
		fig[2,2];
		df=retest_df,
		xlabel="Even trials",
		ylabel="Odd trials",
		xcol=:a2,
		ycol=:a2_1,
		subtitle="Session 2 split-half: learning rate (set-size = 21)",
		correct_r=false
	)
	workshop_reliability_scatter!(
		fig[3,1:2];
		df=retest_df,
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=:Δa_s1,
		ycol=:Δa_s2,
		subtitle="Session 2 split-half: learning rate difference",
		correct_r=false
	)
	fig
end

# ╔═╡ 4b2074cd-9e36-49cd-898e-8c096388e175
md"
#### Palimpsest working memory models
"

# ╔═╡ 20616d82-c334-4e8b-9e67-0bb9270d5c4b
md"
A) **Overall** capacity
"

# ╔═╡ da71edc6-5867-4cba-8c71-fbe225011e20
begin
	wm_splithalf = optimize_multiple_by_factor(
		df;
		model = WM_multi_all_outc_pmst_sgd_sum,
		factor = :even,
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
		factor = :even,
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

# ╔═╡ 5a770983-740c-404c-b5d5-ea37e2c77cc4
md"
**Single update**
"

# ╔═╡ 3dfc8850-b62c-401d-a4a4-c2ddb135fe8d
let
	retest_df = leftjoin(wm_splithalf..., on = :PID, makeunique=true)
	fig=Figure(;size=(800, 700))
	workshop_reliability_scatter!(
		fig[1,1];
		df=retest_df,
		xlabel="Even trials",
		ylabel="Odd trials",
		xcol=:ρ,
		ycol=:ρ_1,
		subtitle="Session 1 even-odd: reward sensitivity",
		correct_r=false
	)
	workshop_reliability_scatter!(
		fig[2,1];
		df=retest_df,
		xlabel="Even trials",
		ylabel="Odd trials",
		xcol=:C,
		ycol=:C_1,
		subtitle="Session 1 even-odd: overall capacity",
		correct_r=false
	)
	fig
end

# ╔═╡ ef6a0662-f688-4941-bebe-694b02375638
md"
**Reciprocal update**
"

# ╔═╡ 3a29b6ee-06a6-4ce5-a0ab-7bc020ad6d13
let
	retest_df = leftjoin(wmr_splithalf..., on = :PID, makeunique=true)
	fig=Figure(;size=(800, 700))
	workshop_reliability_scatter!(
		fig[1,1];
		df=retest_df,
		xlabel="Even trials",
		ylabel="Odd trials",
		xcol=:ρ,
		ycol=:ρ_1,
		subtitle="Session 1 even-odd: reward sensitivity",
		correct_r=false
	)
	workshop_reliability_scatter!(
		fig[2,1];
		df=retest_df,
		xlabel="Even trials",
		ylabel="Odd trials",
		xcol=:C,
		ycol=:C_1,
		subtitle="Session 1 even-odd: overall capacity",
		correct_r=false
	)
	fig
end

# ╔═╡ 361e6a8d-aa7c-45cb-8034-67f1b77d2d2c
# ╠═╡ disabled = true
#=╠═╡
md"
B) **Stimulus-specific** capacity
"
  ╠═╡ =#

# ╔═╡ 84360976-a52e-4b13-8a0a-a50f52860eb9
# ╠═╡ disabled = true
#=╠═╡
begin
	wmss_splithalf = optimize_multiple_by_factor(
		df;
		model = WM_multi_pmst_sgd_sum,
		factor = :even,
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
		factor = :even,
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
  ╠═╡ =#

# ╔═╡ a2e68387-a919-4a84-ad41-7b857ca34c00
# ╠═╡ disabled = true
#=╠═╡
md"
**Single update**
"
  ╠═╡ =#

# ╔═╡ 0b110e8d-d325-4aaf-b00f-b6be55332743
# ╠═╡ disabled = true
#=╠═╡
let
	retest_df = leftjoin(wmss_splithalf..., on = :PID, makeunique=true)
	fig=Figure(;size=(800, 700))
	workshop_reliability_scatter!(
		fig[1,1];
		df=retest_df,
		xlabel="Even trials",
		ylabel="Odd trials",
		xcol=:ρ,
		ycol=:ρ_1,
		subtitle="Session 1 even-odd: reward sensitivity",
		correct_r=false
	)
	workshop_reliability_scatter!(
		fig[2,1];
		df=retest_df,
		xlabel="Even trials",
		ylabel="Odd trials",
		xcol=:C,
		ycol=:C_1,
		subtitle="Session 1 even-odd: stimulus-specific capacity",
		correct_r=false
	)
	fig
end
  ╠═╡ =#

# ╔═╡ 4d5490da-dfa3-41fb-8812-f9da163dbbdf
# ╠═╡ disabled = true
#=╠═╡
md"
**Reciprocal update**
"
  ╠═╡ =#

# ╔═╡ 444e2e69-6aa5-4baa-99da-78eb01896715
# ╠═╡ disabled = true
#=╠═╡
let
	retest_df = leftjoin(wmssr_splithalf..., on = :PID, makeunique=true)
	fig=Figure(;size=(800, 700))
	workshop_reliability_scatter!(
		fig[1,1];
		df=retest_df,
		xlabel="Even trials",
		ylabel="Odd trials",
		xcol=:ρ,
		ycol=:ρ_1,
		subtitle="Session 1 even-odd: reward sensitivity",
		correct_r=false
	)
	workshop_reliability_scatter!(
		fig[2,1];
		df=retest_df,
		xlabel="Even trials",
		ylabel="Odd trials",
		xcol=:C,
		ycol=:C_1,
		subtitle="Session 1 even-odd: stimulus-specific capacity",
		correct_r=false
	)
	fig
end
  ╠═╡ =#

# ╔═╡ Cell order:
# ╟─11ebe9db-0809-48e7-ab4b-d5bee1fae9b1
# ╠═54a03c1e-a2b8-11ef-2c40-695bce51ab75
# ╠═2a0c2f77-4af2-44f0-adbe-3d9e52d24a59
# ╟─892fc799-cc39-4a86-b899-cdcd1816c01b
# ╟─dea7d258-486f-4ca3-8bcf-a2240fdef345
# ╟─0eac1304-e5c7-4022-a648-2a3ad0293d36
# ╠═419ae40b-f067-43ab-aff7-af7817ab93db
# ╟─fa15d0b0-705c-4f62-8d94-4d7dec28b487
# ╠═242afc6b-5881-47c4-b86b-f117d6c213d1
# ╠═c1bca834-a7bd-470e-831f-3ddf3d3c0a9a
# ╟─946360aa-3a1a-4ea0-b4ef-c32453e65b19
# ╟─6fe27779-90e2-459e-adb9-2afc166d37f8
# ╠═3108a5a5-37b9-4b76-a529-dab5e3eecff2
# ╠═19d06113-a41f-4b5f-a353-225cec794db0
# ╟─34db522e-26ce-4954-8660-5d4f4c5c53a5
# ╟─aa77e9de-514b-43dd-b0c9-f542c6ab7c2f
# ╠═1c46ae0f-658c-40df-82fa-385c0b5365ee
# ╠═03918685-bda8-46b2-a1ac-be20e8781e9f
# ╟─157d7977-0c94-47bf-9df0-47abf0887eeb
# ╠═6429d353-3892-4d62-9a9e-cf167639e4fe
# ╠═4fe85f44-2dbe-450b-859a-85cfbf5a8024
# ╟─ee33a3e7-2d4c-40a1-9255-b57f3a73df80
# ╟─0faebc26-016a-4683-b26a-a248fa9a0dee
# ╠═56e84e90-f53c-474e-a0aa-d5d97cb8f255
# ╟─fa7fcb02-e12d-4bc5-8886-e50460a04186
# ╟─f266fcdb-5000-4458-9328-684d70f896f0
# ╠═9c9c4f38-3631-4a02-8a0e-f4583702e377
# ╟─b34d0881-4480-480a-9673-e3cf5ba8a909
# ╠═11e8bd77-e192-4e00-9e15-d8aa2edae2e8
# ╟─75bf8e63-f786-4d1a-9dc2-d2ad3b2e586e
# ╟─cc95996c-c5fe-4c6a-8785-ae819721bd0a
# ╠═d6eb32d4-df29-4427-9cd4-b7a5ed5a4224
# ╟─26eb6122-96a2-45d2-b479-240a0da3eef8
# ╠═519b1a8b-e600-4fcb-84cc-cc3afc5d5b06
# ╟─1a8b8235-e31e-4d1e-9829-5cb44daeb017
# ╠═372bc57f-3bde-4275-9440-a28aae76bba5
# ╠═9a42da85-8531-4d91-944e-77a9378d6b0b
# ╠═38ea51ca-f146-4f45-a32d-1aca601c5c17
# ╟─4b2074cd-9e36-49cd-898e-8c096388e175
# ╟─20616d82-c334-4e8b-9e67-0bb9270d5c4b
# ╠═da71edc6-5867-4cba-8c71-fbe225011e20
# ╟─5a770983-740c-404c-b5d5-ea37e2c77cc4
# ╠═3dfc8850-b62c-401d-a4a4-c2ddb135fe8d
# ╟─ef6a0662-f688-4941-bebe-694b02375638
# ╠═3a29b6ee-06a6-4ce5-a0ab-7bc020ad6d13
# ╟─361e6a8d-aa7c-45cb-8034-67f1b77d2d2c
# ╠═84360976-a52e-4b13-8a0a-a50f52860eb9
# ╟─a2e68387-a919-4a84-ad41-7b857ca34c00
# ╠═0b110e8d-d325-4aaf-b00f-b6be55332743
# ╟─4d5490da-dfa3-41fb-8812-f9da163dbbdf
# ╠═444e2e69-6aa5-4baa-99da-78eb01896715
