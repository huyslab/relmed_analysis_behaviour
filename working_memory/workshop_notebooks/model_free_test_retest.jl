### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ 1fbb7054-9dd9-4b8f-a849-593d2dc21adb
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

# ╔═╡ 15052f56-a68c-11ef-0907-5d4e2070ad61
md"
## Model-free accuracy plots and choice test-retest
"

# ╔═╡ be015260-3dd9-4729-b6d1-05cee8e20bb8
begin
	# Load data
	_, _, _, _, _, WM_data, _, _ = load_pilot6_data()
	df = prepare_WM_data(WM_data)
	nothing
end

# ╔═╡ fc5f297b-a507-4bcf-9a05-8dbd6d95319c
begin
	PILT_data, _, _, _ = load_pilot4x_data()
	filter!(x -> x.version == "4.1", PILT_data)
	PILT_data = exclude_PLT_sessions(PILT_data, required_n_blocks = 18)
	PILT_data_clean = filter(x -> x.choice != "noresp", PILT_data)
	df_pilot4 = prepare_for_fit(PILT_data_clean; pilot4=true)[1]
	nothing
end

# ╔═╡ 927da0dc-e94e-42fa-a286-67f128af37b0
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

# ╔═╡ b845e8d8-8090-4173-a780-79b2051ce5b6
md"
## Choice accuracy by set size
"

# ╔═╡ 4a495817-049f-4118-bcfb-d85c9d4df837
let
	f = Figure(size = (1000, 800))
	plot_prior_accuracy!(
		f[1,1:2],
		df;
		choice_val = 3.,
		group = :set_size,
		pid_col = :PID,
		legend = true,
		legend_rows = 1,
		legend_pos = :bottom,
		title = "Overall choice accuracy by set size",
		error_band = "se"
	)
	
	df[!, :valence_type] .= ifelse.(df.valence .== 1, "reward", "punishment")
	df[!, :ssgrp] .= string.(df.set_size) .* " (" .* df.valence_type .* ")"
	plot_prior_accuracy!(
		f[2,1],
		df;
		choice_val = 3.,
		group = :ssgrp,
		pid_col = :PID,
		legend = true,
		legend_rows = 2,
		legend_pos = :bottom,
		title = "Reward vs. punishment",
		error_band = "se"
	)

	df[!, :sess_ss] .=
		string.(df.set_size) .* " (session " .* string.(df.session) .* ")"
	plot_prior_accuracy!(
		f[2, 2],
		df;
		choice_val = 3.,
		group = :sess_ss,
		pid_col = :PID,
		legend = true,
		legend_rows = 2,
		legend_pos = :bottom,
		title = "Session 1 vs. Session 2",
		error_band = "se"
	)
	f
end

# ╔═╡ e8ecdef8-4974-4068-aed5-dced1273ccb9
md"
#### Previous pilot with 2 options and 3 different set sizes
"

# ╔═╡ 20e6ebfd-c2ed-469d-9848-c281b8ebf55c
let
	f = Figure(size = (800, 500))
	plot_prior_accuracy!(
		f[1,1],
		df_pilot4;
		choice_val = 1.,
		group = :set_size,
		pid_col = :PID,
		legend = true,
		legend_rows = 1,
		legend_pos = :bottom,
		title = "Overall choice accuracy by set size for prev. pilot",
		error_band = "se"
	)
	f
end

# ╔═╡ 8c731a4e-b530-43a4-9979-31ca87621a62
md"
#### Session 1 alone
"

# ╔═╡ 6b1c6ec2-f159-4805-8abf-d5b14b20b986
let
	f = Figure(size = (1000, 400))
	plot_prior_accuracy!(
		f[1,1],
		filter(x -> x.session == "1", df);
		choice_val = 3.,
		group = :set_size,
		pid_col = :PID,
		legend = true,
		legend_rows = 1,
		legend_pos = :bottom,
		title = "Overall choice accuracy by set size",
		error_band = "se"
	)
	
	df[!, :valence_type] .= ifelse.(df.valence .== 1, "reward", "punishment")
	df[!, :ssgrp] .= string.(df.set_size) .* " (" .* df.valence_type .* ")"
	plot_prior_accuracy!(
		f[1,2],
		filter(x -> x.session == "1", df);
		choice_val = 3.,
		group = :ssgrp,
		pid_col = :PID,
		legend = true,
		legend_rows = 2,
		legend_pos = :bottom,
		title = "Reward vs. punishment",
		error_band = "se"
	)
	f
end

# ╔═╡ 15325f92-cf2e-4712-8e70-d84298f1fef3
md"
#### Session 2 alone
"

# ╔═╡ 3f20d6f9-48d4-4d95-99c6-4543f4db9fdc
let
	f = Figure(size = (1000, 400))
	plot_prior_accuracy!(
		f[1,1],
		filter(x -> x.session == "2", df);
		choice_val = 3.,
		group = :set_size,
		pid_col = :PID,
		legend = true,
		legend_rows = 1,
		legend_pos = :bottom,
		title = "Overall choice accuracy by set size",
		error_band = "se"
	)
	
	df[!, :valence_type] .= ifelse.(df.valence .== 1, "reward", "punishment")
	df[!, :ssgrp] .= string.(df.set_size) .* " (" .* df.valence_type .* ")"
	plot_prior_accuracy!(
		f[1,2],
		filter(x -> x.session == "2", df);
		choice_val = 3.,
		group = :ssgrp,
		pid_col = :PID,
		legend = true,
		legend_rows = 2,
		legend_pos = :bottom,
		title = "Reward vs. punishment",
		error_band = "se"
	)
	f
end

# ╔═╡ 965f3dda-b1c2-477c-9cd2-d925d7009116
md"
## Test-retest for participant behaviour
"

# ╔═╡ 599e3277-2ca6-4b99-a669-411150df3654
let
	mean_acc = combine(
		groupby(df, [:prolific_pid, :session]), :isOptimal => mean => :mean_accuracy
	)
	mean_acc_ssz = combine(
		groupby(df, [:prolific_pid, :session, :set_size]), :isOptimal => mean => :mean_accuracy
	)
	mean_acc = unstack(mean_acc, :session, :mean_accuracy) |> dropmissing!

	mean_acc_diff = unstack(mean_acc_ssz, :set_size, :mean_accuracy)
	mean_acc_diff.ss_acc_diff .= mean_acc_diff[!, Symbol("3")] .- mean_acc_diff[!, Symbol("21")]
	mean_acc_diff = mean_acc_diff[:, [:prolific_pid, :session, :ss_acc_diff]]
	mean_acc_diff = unstack(mean_acc_diff, :session, :ss_acc_diff) |> dropmissing!
	
	mean_acc_ssz = unstack(mean_acc_ssz, :session, :mean_accuracy) |> dropmissing!
	
	fig=Figure(;size=(800, 1100))
	workshop_reliability_scatter!(
		fig[1,1:2];
		df=mean_acc,
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=Symbol("1"),
		ycol=Symbol("2"),
		subtitle="Test-retest: overall mean accuracy",
		correct_r=false
	)
	workshop_reliability_scatter!(
		fig[2,1];
		df=filter(x -> x.set_size == 3, mean_acc_ssz),
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=Symbol("1"),
		ycol=Symbol("2"),
		subtitle="Test-retest: mean accuracy (set size = 3)",
		correct_r=false
	)
	workshop_reliability_scatter!(
		fig[2,2];
		df=filter(x -> x.set_size == 21, mean_acc_ssz),
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=Symbol("1"),
		ycol=Symbol("2"),
		subtitle="Test-retest: mean accuracy (set size = 21)",
		correct_r=false
	)
	workshop_reliability_scatter!(
		fig[3,1:2];
		df=mean_acc_diff,
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=Symbol("1"),
		ycol=Symbol("2"),
		subtitle="Test-retest: mean difference in accuracy",
		correct_r=false
	)
	fig
end

# ╔═╡ Cell order:
# ╟─15052f56-a68c-11ef-0907-5d4e2070ad61
# ╠═1fbb7054-9dd9-4b8f-a849-593d2dc21adb
# ╠═be015260-3dd9-4729-b6d1-05cee8e20bb8
# ╠═fc5f297b-a507-4bcf-9a05-8dbd6d95319c
# ╟─927da0dc-e94e-42fa-a286-67f128af37b0
# ╟─b845e8d8-8090-4173-a780-79b2051ce5b6
# ╠═4a495817-049f-4118-bcfb-d85c9d4df837
# ╟─e8ecdef8-4974-4068-aed5-dced1273ccb9
# ╠═20e6ebfd-c2ed-469d-9848-c281b8ebf55c
# ╟─8c731a4e-b530-43a4-9979-31ca87621a62
# ╠═6b1c6ec2-f159-4805-8abf-d5b14b20b986
# ╟─15325f92-cf2e-4712-8e70-d84298f1fef3
# ╠═3f20d6f9-48d4-4d95-99c6-4543f4db9fdc
# ╟─965f3dda-b1c2-477c-9cd2-d925d7009116
# ╠═599e3277-2ca6-4b99-a669-411150df3654
