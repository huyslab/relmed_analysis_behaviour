### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ e953ab1e-2f64-11f0-287e-afd1a5effa1f
begin
	cd("/home/jovyan/")
    import Pkg
    # activate the shared project environment
    Pkg.activate("relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	using Random, DataFrames, JSON, CSV, StatsBase, JLD2, HTTP, CairoMakie, Printf, Distributions, CategoricalArrays, AlgebraOfGraphics, Dates, SHA
	using Tidier, GLM, MixedModels, PlutoUI, LaTeXStrings, ColorSchemes
	using LogExpFunctions: logistic, logit
	import OpenScienceFramework as OSF
	include("fetch_preprocess_data.jl")
	include("sample_utils.jl")
	include("plotting_utils.jl")
	include("osf_utils.jl")
	include("vigour_utils.jl")
	nothing
end

# ╔═╡ da326055-094a-4b3a-8a0c-dbb87366b3ae
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

# ╔═╡ ce32edbe-2883-4b02-ad66-49228b6c2bac
function prepare_test_data(df::DataFrame; task::String = "pilt")

	# Select rows
	test_data = filter(x -> (x.trial_type == "PILT") && (x.trialphase == "$(task)_test"), df)

	# Select columns
	test_data = test_data[:, Not(map(col -> all(ismissing, col), eachcol(test_data)))]

	# Change all block names to same type
	test_data.block = string.(test_data.block)

	# Sort
	sort!(test_data, [:participant_id, :session, :block, :trial])

	return test_data

end


# ╔═╡ 9eb71529-0dc7-4951-9e2e-96ec95f4df89
function load_pilot9_data(; force_download = false)
	datafile = "data/pilot9.jld2"

	# Load data or download from REDCap
	if !isfile(datafile) || force_download
		jspsych_json, records = get_REDCap_data("pilot9"; file_field = "data", record_id_field = "record_id")
	
		jspsych_data = REDCap_data_to_df(jspsych_json, records; participant_id_field = "participant_id", start_time_field = "sitting_start_time")

		remove_testing!(jspsych_data)

		JLD2.@save datafile jspsych_data
	else
		JLD2.@load datafile jspsych_data
	end

	# Exctract PILT
	PILT_data = prepare_PLT_data(jspsych_data; trial_type = "PILT")

	# Seperate out PILT
	filter!(x -> x.trialphase == "pilt", PILT_data)
	
	# Extract post-PILT test
	test_data = prepare_test_data(jspsych_data)

	# Exctract vigour
	vigour_data = prepare_vigour_data(jspsych_data) 

	# Extract PIT
	PIT_data = prepare_PIT_data(jspsych_data)

	# Extract max press rate data
	max_press_data = prepare_max_press_data(jspsych_data)

	# Extract control data
	control_task_data, control_report_data = prepare_control_data(jspsych_data) 

	return PILT_data, test_data, vigour_data, PIT_data, max_press_data, control_task_data, control_report_data, jspsych_data
end


# ╔═╡ b00006c6-ef32-45d3-b5a2-d6614f39bc31
begin
	PILT_data, test_data, vigour_data, PIT_data, max_press_data, control_task_data, control_report_data, jspsych_data = load_pilot9_data(;force_download = false)
	nothing
end

# ╔═╡ c2896ab1-cf21-4198-bfe8-f672e372d03b
PIT_doubt_PID = ["677afec8954cc9493035dc74", "67f7f3a41fcb35f0eaac48ef", "680d70786b6a8ca0af164c89", "66226d86cdae406d48c9e4c6", "681106ac93d01f1615c6f003"]

# ╔═╡ 781c45d2-d4b6-463f-87bf-a01db1fc0372
md"""# PILT"""

# ╔═╡ 34e147de-0af9-45a6-aa56-785d7d071153
let
	@assert all((x -> x in ["right", "left", "noresp"]).(unique(PILT_data.response))) "Unexpected values in response"
	
	@assert all(PILT_data.chosen_feedback .== ifelse.(
		PILT_data.response .== "right",
		PILT_data.feedback_right,
		ifelse.(
			PILT_data.response .== "left",
			PILT_data.feedback_left,
			minimum.(hcat.(PILT_data.feedback_left, PILT_data.feedback_right))
		)
	)) "`chosen_feedback` doens't match feedback and choice"


end

# ╔═╡ aaf73642-aaec-40a9-95f8-f7f0f25095c7
let
	# Clean data
	PILT_data_clean = exclude_PLT_sessions(PILT_data, required_n_blocks = 21)
	PILT_data_clean = filter(x -> x.response != "noresp", PILT_data_clean)

	# Sumarrize by participant and trial
	acc_curve = combine(
		groupby(PILT_data_clean, [:prolific_pid, :trial]),
		:response_optimal => mean => :acc
	)

	sort!(acc_curve, [:prolific_pid, :trial])

	# Summarize by trial
	acc_curve_sum = combine(
		groupby(acc_curve, :trial),
		:acc => mean => :acc,
		:acc => sem => :se
	)

	# Plot
	mp = (data(acc_curve) * mapping(
		:trial => "Trial #",
		:acc => "Prop. optimal choice",
		group = :prolific_pid,
		color = :prolific_pid
	) * visual(Lines, linewidth = 1, alpha = 0.7)) +
	(data(acc_curve_sum) * 
	mapping(
		:trial => "Trial #",
		:acc => "Prop. optimal choice"
	) * visual(Lines, linewidth = 4))
	
	
	draw(mp; legend = (; show = false))
end

# ╔═╡ 5373906f-51fb-43aa-91f7-20656785a386
let
	test_data_clean = filter(x -> x.response != "no_resp", test_data)
	
	test_data_clean.EV_diff = test_data_clean.EV_right .- test_data_clean.EV_left

	test_sum = combine(
		groupby(test_data_clean, :EV_diff),
		:response => (x -> mean(x .== "right")) => :right_chosen
	)

	mp = data(test_sum) * mapping(:EV_diff, :right_chosen) * visual(Scatter)

	draw(mp)
end

# ╔═╡ 0b1ead8e-ae59-4add-a1ea-e5420cadae5f
md"""
# Vigour
"""

# ╔═╡ e0becc69-04ab-4df0-b56b-df4acc8aca9c
begin
	filter!(x -> !(x.prolific_pid in []), vigour_data);
	transform!(vigour_data, [:trial_presses, :trial_duration] => ((x, y) -> x .* 1000 ./ y) => :press_per_sec);
	
	nothing;
	@chain vigour_data begin
		@filter(press_per_sec > 11)
		@count(prolific_pid)
	end
end

# ╔═╡ a7eb19ed-d7e8-4bb4-a2f3-17426035217d
# ╠═╡ disabled = true
#=╠═╡
let
	n_miss_df = @chain vigour_data begin
		@mutate(pig = "Mag " * string(magnitude) * ", FR " * string(ratio))
		@group_by(prolific_pid, pig)
		@summarize(n_miss = sum(trial_presses == 0))
		@ungroup()
		@mutate(short_id = last.(prolific_pid, 5))
	end
	# Create the plot
	plot = data(n_miss_df) * 
		   mapping(:n_miss, layout=:pig) * 
		   histogram(bins=5)
	
	# Set up the axis
	axis = (
		xlabel = "# No-response trials",
		ylabel = "Count (# participants)"
	)
	
	# Draw the plot
	draw(plot; axis, figure=(;title="No-response trial distribution in Vigour task"))
end
  ╠═╡ =#

# ╔═╡ 8c4fca45-504b-4be8-bbcc-e48e497b3f0e
let
	plot_presses_vs_var(@filter(vigour_data, trial_number > 0); x_var=:reward_per_press, y_var=:press_per_sec, xlab="Reward/press", ylab = "Press/sec", combine=false)
end

# ╔═╡ 56a77ffc-be7d-474a-a0f8-5951fc87b424
let
	avg_df = @chain vigour_data begin
		@group_by(trial_number)
		@summarize(trial_presses = mean(trial_presses), se = mean(trial_presses)/sqrt(length(prolific_pid)))
		@ungroup
	end
	p = data(vigour_data) * mapping(:trial_number, :trial_presses) * AlgebraOfGraphics.linear() + data(avg_df) * mapping(:trial_number, :trial_presses) * visual(ScatterLines)
    draw(p)
end

# ╔═╡ c0f66cdc-4d68-4f2d-b4c6-65de4821091f
md"""
# PIT
"""

# ╔═╡ 67b741f1-eb82-4e14-9ba6-79d903e0b2d0
begin
	filter!(x -> !(x.prolific_pid in []), PIT_data);
	transform!(PIT_data, [:trial_presses, :trial_duration] => ((x, y) -> x .* 1000 ./ y) => :press_per_sec);
	nothing;
end

# ╔═╡ 6dee7572-0add-4e83-970d-2e629fa7c632
# ╠═╡ disabled = true
#=╠═╡
let
	n_miss_df =  @chain PIT_data begin
		# @filter(coin != 0)
		@arrange(prolific_pid, magnitude, ratio)
		@mutate(pig = "Mag " * string(magnitude) * ", FR " * string(ratio))
		@group_by(prolific_pid, pig)
		@summarize(n_miss = sum(trial_presses == 0))
		@ungroup()
		@mutate(short_id = last.(prolific_pid, 5))
	end
	# Create the plot
	plot = data(n_miss_df) * 
		   mapping(:n_miss, layout=:pig) * 
		   histogram(bins=5)
	
	# Set up the axis
	axis = (
		xlabel = "# No-response trials",
		ylabel = "Count (# participants)"
	)
	
	# Draw the plot
	draw(plot; axis, figure=(;title="No-response trial distribution in PIT task"))
end
  ╠═╡ =#

# ╔═╡ 67f673b9-58a3-4b1a-a0a2-690758d621c1
let
	common_rpp = unique(PIT_data.reward_per_press)
	instrumental_data = @chain PIT_data begin
		@filter(coin==0)
		@bind_rows(vigour_data)
		@mutate(trialphase=categorical(trialphase, levels=["vigour_trial", "pit_trial"], ordered=true))
		@mutate(trialphase=~recode(trialphase, "vigour_trial" => "Vigour", "pit_trial" => "PIT w/o coin"))
		@filter(reward_per_press in !!common_rpp)
	end
	plot_presses_vs_var(instrumental_data; x_var=:reward_per_press, y_var=:press_per_sec, grp_var = :trialphase, xlab="Reward/press", ylab = "Press/sec", combine=false)
end

# ╔═╡ 187e681c-1207-46f3-9cb3-9e6dcc40f625
let
	df = @chain PIT_data begin
		@mutate(session=if_else(trial_number <= 36, "Trial: 1-36", "Trial: 37-72"))
		@arrange(prolific_pid, session, magnitude, ratio)
		@mutate(pig = "Mag " * string(magnitude) * ", FR " * string(ratio))
		@mutate(pig=categorical(pig,levels=["Mag 2, FR 16","Mag 2, FR 8","Mag 5, FR 8","Mag 1, FR 1"]))
	end
	grouped_data, avg_w_data = avg_presses_w_fn(df, [:session, :coin, :pig], :press_per_sec)
	p = data(avg_w_data) *
	mapping(:coin=>nonnumeric, :avg_y, col=:pig=>nonnumeric, row=:session=>nonnumeric) *
	(
	visual(Lines, linewidth=1, color=:gray) +
	visual(Errorbars, whiskerwidth=4) *
	mapping(:se_y, color=:coin => nonnumeric) +
	visual(Scatter) *
	mapping(color=:coin => nonnumeric)
	)
	draw(p, scales(Color = (; palette=:PRGn_7)); axis=(;xlabel="Pavlovian stimuli (coin)", ylabel="Press/sec", width=150, height=150, xticklabelrotation=pi/4))
end

# ╔═╡ ae08f0a0-d7f3-475a-ac6e-7c6a305737cc
begin
	# If the test_data is not corrected, then the accuracy for negative is wrong (and should be reversed).
	PIT_acc_df = @chain test_data begin
		@filter(block == "pavlovian")
		@mutate(
			EV_left = case_when(
				contains(stimulus_left, "PIT4") => -0.01,
				contains(stimulus_left, "PIT6") => -1.0,
				true => EV_left),
			EV_right = case_when(
				contains(stimulus_right, "PIT4") => -0.01,
				contains(stimulus_right, "PIT6") => -1.0,
				true => EV_right)
		)
		@mutate(valence = ifelse(EV_left * EV_right < 0, "Different", ifelse(EV_left > 0, "Positive", "Negative")))
		@group_by(valence)
		@summarize(acc = mean(skipmissing((EV_right > EV_left) == (key == "arrowright"))))
		@ungroup
	end
	@info "PIT acc for NOT in the same valence: $(round(PIT_acc_df.acc[PIT_acc_df.valence.=="Different"]...; digits=2))"
	@info "PIT acc for in the positive valence: $(round(PIT_acc_df.acc[PIT_acc_df.valence.=="Positive"]...; digits=2))"
	@info "PIT acc for in the negative valence: $(round(PIT_acc_df.acc[PIT_acc_df.valence.=="Negative"]...; digits=2))"

	p = @chain test_data begin
		@filter(block == "pavlovian")
		@mutate(
			EV_left = case_when(
				contains(stimulus_left, "PIT4") => -0.01,
				contains(stimulus_left, "PIT6") => -1.0,
				true => EV_left),
			EV_right = case_when(
				contains(stimulus_right, "PIT4") => -0.01,
				contains(stimulus_right, "PIT6") => -1.0,
				true => EV_right)
		)
		@mutate(valence = ifelse(EV_left * EV_right < 0, "Different", ifelse(EV_left > 0, "Positive", "Negative")))
		@mutate(correct = (EV_right > EV_left) == (key == "arrowright"))
		@filter(!ismissing(correct))
		@group_by(prolific_pid, session, valence)
		@summarize(acc = mean(correct))
		@ungroup
		data(_) * mapping(:valence, :acc, color=:valence) * visual(RainClouds; clouds=hist, hist_bins = 20, plot_boxplots = false)
	end
	fig = Figure(;size=(8, 6) .* 144 ./ 2.54)
	p = draw!(fig[1,1], p, scales(Color = (; palette=[colorant"gray", ColorSchemes.Set3_5[[4,5]]...])); axis=(;xlabel="Pavlovian valence", ylabel="PIT test accuracy"))

	fig
end

# ╔═╡ 03763d35-7383-464c-841c-ec955c83a402
md"""
# Control
"""

# ╔═╡ 728a0a72-6e66-4618-9f47-3a21ec1e04d5
@assert all(combine(groupby(control_task_data, :prolific_pid), [:time_elapsed, :trial] => ((x, y) -> all(denserank(x) .== denserank(y))) => :in_order)[:, "in_order"]) "Trial numbers are created *incorrectly* in chronological order"

# ╔═╡ 470ec18e-3c3f-4dab-8c02-1ed5bc2b1bb1
md"""
## Timing in control
"""

# ╔═╡ cd44eab8-151d-40bf-a7d9-364983554cc2
md"""
## Effort in exploration
"""

# ╔═╡ cae03380-56a8-4292-8342-4323d1bf59a5
begin
	threshold_df = (; y = [6, 12, 18], threshold = ["Low", "Mid", "High"])
	spec2 = data(threshold_df) * mapping(:y, color = :threshold) * visual(HLines)
	@chain control_task_data begin
		@filter(trialphase == "control_explore")
		@drop_missing(trial_presses)
		@mutate(trial_stage = ifelse(trial <= maximum(trial)/2, "Trial 1-36", "Trial 37-72"))
		@group_by(prolific_pid, trial_stage, current)
		@summarize(trial_presses = mean(trial_presses))
		@ungroup()
		@arrange(prolific_pid, trial_stage, current)
		data(_) * (
			mapping(:current => nonnumeric, :trial_presses, col=:trial_stage) * visual(RainClouds) +
			mapping(:current => nonnumeric, :trial_presses, col=:trial_stage, group=:prolific_pid) * visual(Lines, alpha = 0.1)
		)
		draw(spec2 + _; axis=(; xlabel = "Current strength", ylabel = "Presses (in explore trials)"))
	end
end

# ╔═╡ 5429031d-e6f3-45b3-9ab3-bff643950a2d
let
	glm_coef(data) = coef(lm(@formula(trial_presses ~ current), data))
	
	split_df = @chain control_task_data begin
		@filter(trialphase == "control_explore")
		@drop_missing(trial_presses)
		@mutate(trial_number = ~denserank(trial))
		@mutate(half = ifelse(trial_number <= maximum(trial_number)/2, "x", "y"))
		groupby([:prolific_pid, :session, :half])
		combine(AsTable([:trial_presses, :current]) => (x -> [glm_coef(x)]) => [:β0, :β_current])
	end

	fig=Figure(;size=(12, 6) .* 144 ./ 2.54)
	workshop_reliability_scatter!(
		fig[1,1];
		df=dropmissing(unstack(split_df, [:prolific_pid, :session], :half, :β0)),
		xlabel="Trial 1-36",
		ylabel="Trial 37-72",
		xcol=:x,
		ycol=:y,
		subtitle="β0",
		correct_r=true
	)
	workshop_reliability_scatter!(
		fig[1,2];
		df=dropmissing(unstack(split_df, [:prolific_pid, :session], :half, :β_current)),
		xlabel="Trial 1-36",
		ylabel="Trial 37-72",
		xcol=:x,
		ycol=:y,
		subtitle="β[Current]",
		correct_r=true
	)
	fig
end

# ╔═╡ 004d7b5f-9ec5-4f4b-8724-02e65040bd8f
@chain control_task_data begin
		@filter(trialphase == "control_explore")
	@count(prolific_pid)
end

# ╔═╡ 80a6e41c-d578-4247-9a7a-b49dab153175
let
	glm_coef(data) = coef(lm(@formula(trial_presses ~ current), data))
	
	split_df = @chain control_task_data begin
		@filter(trialphase == "control_explore")
		@drop_missing(trial_presses)
		@mutate(trial_number = ~denserank(trial))
		@mutate(half = if_else(trial_number % 2 === 0, "x", "y"))
		groupby([:prolific_pid, :session, :half])
		combine(AsTable([:trial_presses, :current]) => (x -> [glm_coef(x)]) => [:β0, :β_current])
	end

	fig=Figure(;size=(12, 6) .* 144 ./ 2.54)
	workshop_reliability_scatter!(
		fig[1,1];
		df=dropmissing(unstack(split_df, [:prolific_pid, :session], :half, :β0)),
		xlabel="Even trials",
		ylabel="Odd trials",
		xcol=:x,
		ycol=:y,
		subtitle="β0",
		correct_r=true
	)
	workshop_reliability_scatter!(
		fig[1,2];
		df=dropmissing(unstack(split_df, [:prolific_pid, :session], :half, :β_current)),
		xlabel="Even trials",
		ylabel="Odd trials",
		xcol=:x,
		ycol=:y,
		subtitle="β[Current]",
		correct_r=true
	)
	fig
end

# ╔═╡ 457ab261-bdfd-4415-a350-c7e0ce9aac63
md"""
## Prediction in control
"""

# ╔═╡ db196a92-f7b3-4293-9788-ee80b57a1d04
# ╠═╡ disabled = true
#=╠═╡
begin
	@chain control_task_data begin
		@filter(trialphase == "control_predict_homebase")
		@mutate(prolific_pid = str_trunc(prolific_pid, 9))
		@drop_missing(correct)
		data(_) * mapping(:trial => nonnumeric, :correct, layout=:prolific_pid) * visual(Lines)
		draw(;axis=(;xlabel = "Trial", ylabel = "Home base predict acc."), figure=(;size=(800, 800)))
	end
end
  ╠═╡ =#

# ╔═╡ 6bb9149f-2428-424c-8f22-166b72cb1fc9
begin
	@chain control_task_data begin
		@filter(trialphase == "control_predict_homebase")
		@drop_missing(correct)
		@group_by(trial, ship)
		@summarize(acc = mean(correct), upper = mean(correct) + std(correct)/sqrt(length(correct)), lower = mean(correct) - std(correct)/sqrt(length(correct)))
		@ungroup
		data(_) * (mapping(:trial, :lower, :upper, color=:ship) * visual(Band, alpha = 0.1) + mapping(:trial, :acc, color=:ship) * visual(ScatterLines))
		draw(;axis=(;xlabel = "Trial", ylabel = "Home base predict acc."), figure=(;size=(600, 400)))
	end
end

# ╔═╡ 1de3f7cb-616e-4624-8adb-8d6796e5c6c4
begin
	transform!(control_report_data, :response => (x -> ifelse.(x == nothing, missing, x)) => :response)
	@chain control_report_data begin
		@filter(trialphase == "control_confidence")
		@drop_missing(response)
		@group_by(trial)
		@summarize(response = mean(response), upper = mean(response) + std(response)/sqrt(length(response)), lower = mean(response) - std(response)/sqrt(length(response)))
		@arrange(trial)
		data(_) * (mapping(:trial => nonnumeric, :lower, :upper) * visual(Band, alpha = 0.1) + mapping(:trial => nonnumeric, :response) * visual(ScatterLines))
		draw(;axis=(;xlabel = "Trial", ylabel = "Confidence rating (0-4)"), figure=(;size=(800, 400)))
	end
end

# ╔═╡ 97b02f97-cab8-4ca7-aeb8-9848491aa5b1
md"""
## Reward in control
"""

# ╔═╡ c427ad21-9e9e-4382-b5ad-bed4daa06aac
@chain control_task_data begin
	@filter(trialphase == "control_reward")
	@drop_missing(response)
	@mutate(correct_choice = ifelse(response == "left", left_viable, right_viable))
	@drop_missing(correct_choice)
	@group_by(trial, island_viable, current)
	@summarize(acc = mean(correct_choice), upper = mean(correct_choice) + std(correct_choice)/sqrt(length(correct_choice)), lower = mean(correct_choice) - std(correct_choice)/sqrt(length(correct_choice)))
	@ungroup
	data(_) * (mapping(:trial => nonnumeric, :lower, :upper) * visual(Band, alpha = 0.1) + mapping(:trial => nonnumeric, :acc) * visual(Lines, linestyle=:dot) + mapping(:trial => nonnumeric, :acc, color=:current => nonnumeric, marker=:island_viable) * visual(Scatter, markersize = 12))
	draw(;axis=(;xlabel = "Trial", ylabel = "Reward trial correct choice rate"), figure=(;size=(800, 400)))
end

# ╔═╡ 4b524fd2-52cc-4e96-8fe9-8dfa9580a457
@chain control_task_data begin
	@filter(trialphase == "control_reward")
	@drop_missing(response)
	@mutate(correct_choice = ifelse(response == "left", left_viable, right_viable))
	@drop_missing(correct_choice)
	@group_by(trial, target)
	@summarize(acc = mean(correct_choice), upper = mean(correct_choice) + std(correct_choice)/sqrt(length(correct_choice)), lower = mean(correct_choice) - std(correct_choice)/sqrt(length(correct_choice)))
	@ungroup
	data(_) * (mapping(:trial => nonnumeric, :lower, :upper, color=:target) * visual(Band, alpha = 0.1) + mapping(:trial => nonnumeric, :acc, color=:target) * visual(ScatterLines, markersize = 12))
	draw(;axis=(;xlabel = "Trial", ylabel = "Reward trial correct choice rate"), figure=(;size=(800, 400)))
end

# ╔═╡ f8b8f06d-c0fa-4741-beb0-837589a9639d
begin
	@chain control_task_data begin
		@filter(trialphase == "control_reward")
		@drop_missing(trial_presses, correct)
		@group_by(prolific_pid, correct, island_viable, current)
		@summarize(trial_presses = mean(trial_presses))
		@ungroup()
		@arrange(prolific_pid, correct, island_viable, current)
		data(_) * (
			mapping(:current => nonnumeric, :trial_presses, col = :island_viable, row = :correct, color = :island_viable) * visual(RainClouds) +
			mapping(:current => nonnumeric, :trial_presses, col = :island_viable, row = :correct, color = :island_viable, group=:prolific_pid) * visual(Lines, alpha = 0.1)
		)
		draw(_, scales(Row = (; categories = [false => "Incorrect choice", true => "Correct choice"]), Col = (; categories = [false => "Island unviable", true => "Island viable"]), Color = (; legend = false)); axis=(; xlabel = "Current strength", ylabel = "Presses (in reward trials)"), figure=(;size=(600, 600)))
	end
end

# ╔═╡ 259a2ada-b821-40ce-81bb-745cd8da34f1
begin
	@chain control_task_data begin
		@filter(trialphase == "control_reward")
		@drop_missing(trial_presses, correct)
		@group_by(prolific_pid, correct, island_viable, current)
		@summarize(trial_presses = mean(trial_presses))
		@ungroup()
		@arrange(prolific_pid, correct, island_viable, current)
		data(_) * (
			mapping(:island_viable, :trial_presses, col = :current => nonnumeric, row = :correct, color = :current => nonnumeric) * visual(RainClouds) +
			mapping(:island_viable, :trial_presses, col = :current => nonnumeric, row = :correct, color = :current => nonnumeric, group=:prolific_pid) * visual(Lines, alpha = 0.1)
		)
		draw(_, scales(Row = (; categories = [false => "Incorrect choice", true => "Correct choice"]), Color = (; legend = false)); axis=(; xlabel = "Island viable?", ylabel = "Presses (in reward trials)"), figure=(;size=(600, 600)))
	end
end

# ╔═╡ ab1a8d54-1a74-44c1-8e27-f0b42bd4238d
begin
	transform!(control_report_data, :response => (x -> ifelse.(x == nothing, missing, x)) => :response)
	@chain control_report_data begin
		@filter(trialphase == "control_controllability")
		@drop_missing(response)
		@group_by(trial)
		@summarize(response = mean(response), upper = mean(response) + std(response)/sqrt(length(response)), lower = mean(response) - std(response)/sqrt(length(response)))
		@arrange(trial)
		data(_) * (mapping(:trial => nonnumeric, :upper, :lower) * visual(Band, alpha = 0.1) + mapping(:trial => nonnumeric, :response) * visual(ScatterLines))
		draw(;figure=(;size=(800, 400)))
	end
end

# ╔═╡ 0dbe2eb5-e83c-4d26-acb1-1151a698a873
"""
    extract_debrief_responses(data::DataFrame) -> DataFrame

Extracts and processes debrief responses from the experimental data. It filters for debrief trials, then parses and expands JSON-formatted Likert scale and text responses into separate columns for each question.

# Arguments
- `data::DataFrame`: The raw experimental data containing participants' trial outcomes and responses, including debrief information.

# Returns
- A DataFrame with participants' debrief responses. The debrief Likert and text responses are parsed from JSON and expanded into separate columns.
"""
function extract_debrief_responses(data::DataFrame)
	# Select trials
	debrief = filter(x -> !ismissing(x.trialphase) && 
		occursin(r"(acceptability|debrief)", x.trialphase) &&
		!(occursin("pre", x.trialphase)), data)


	# Select variables
	select!(debrief, [:prolific_pid, :exp_start_time, :trialphase, :response])

	# Long to wide
	debrief = unstack(
		debrief,
		[:prolific_pid, :exp_start_time],
		:trialphase,
		:response
	)
	

	# Parse JSON and make into DataFrame
	expected_keys = dropmissing(debrief)[1, Not([:prolific_pid, :exp_start_time])]
	expected_keys = Dict([c => collect(keys(JSON.parse(expected_keys[c]))) 
		for c in names(expected_keys)])
	
	debrief_colnames = names(debrief[!, Not([:prolific_pid, :exp_start_time])])
	
	# Expand JSON strings with defaults for missing fields
	expanded = [
	    DataFrame([
	        # Parse JSON or use empty Dict if missing
	        let parsed = ismissing(row[col]) ? Dict() : JSON.parse(row[col])
	            # Fill missing keys with a default value (e.g., `missing`)
	            Dict(key => get(parsed, key, missing) for key in expected_keys[col])
	        end
	        for row in eachrow(debrief)
	    ])
	    for col in debrief_colnames
	]
	expanded = hcat(expanded...)

	# hcat together
	return hcat(debrief[!, Not(debrief_colnames)], expanded)
end

# ╔═╡ 5725d5d4-bb41-4cf1-8256-a2e3132a677b
"""
    summarize_participation(data::DataFrame) -> DataFrame

Summarizes participants' performance in a study based on their trial outcomes and responses, for the purpose of approving and paying bonuses.

This function processes experimental data, extracting key performance metrics such as whether the participant finished the experiment, whether they were kicked out, and their respective bonuses (PILT and vigour). It also computes the number of specific trial types and blocks completed, as well as warnings received. The output is a DataFrame with these aggregated values, merged with debrief responses for each participant.

# Arguments
- `data::DataFrame`: The raw experimental data containing participant performance, trial outcomes, and responses.

# Returns
- A summarized DataFrame with performance metrics for each participant, including bonuses and trial information.
"""
function summarize_participation(data::DataFrame)
	
	participants = combine(groupby(data, [:prolific_pid, :session, :record_id, :exp_start_time]),
		:trialphase => (x -> "experiment_end_message" in x) => :finished,
		:trialphase => (x -> "kick-out" in x) => :kick_out,
		[:trial_type, :trialphase, :block, :n_stimuli] => 
			((t, p, b, n) -> sum((t .== "PILT") .& (.!ismissing.(p) .&& p .!= "PILT_test") .& (typeof.(b) .== Int64) .& (n .== 2))) => :n_trial_PILT,
		[:block, :trial_type, :trialphase, :n_stimuli] => 
			((x, t, p, n) -> length(unique(filter(y -> isa(y, Int64), x[(t .== "PILT") .& (n .== 2) .& (.!ismissing.(p) .&& p .!= "PILT_test")])))) => :n_blocks_PILT,
		# :trialphase => (x -> sum(skipmissing(x .∈ Ref(["control_explore", "control_predict_homebase", "control_reward"])))) => :n_trial_control,
		# :trialPresses => (x -> mean(filter(y -> !ismissing(y), x))) =>  :max_trial_presses,
		:n_warnings => maximum => :n_warnings,
		:time_elapsed => (x -> maximum(x) / 1000 / 60) => :duration,
		:total_bonus => (x -> all(ismissing.(x)) ? missing : only(skipmissing(x))) => :bonus
		# :trialphase => (x -> sum(skipmissing(x .== "control_instruction_quiz_failure"), init=0)) => :n_quiz_failure
	)

	debrief = extract_debrief_responses(data)

	participants = leftjoin(participants, debrief, 
		on = [:prolific_pid, :exp_start_time])

	return participants
end

# ╔═╡ 00d58d1e-9280-4c7e-8399-675db7abb89b
begin
	p_sum = summarize_participation(jspsych_data)
end

# ╔═╡ ff8a5e4f-516e-4483-aaf1-a3d71ed38485
begin
	p_no_double_take = exclude_double_takers(p_sum) |>
		x -> filter(x -> !ismissing(x.finished) && x.finished, x)
end

# ╔═╡ 04049036-613f-4048-9dfd-3b5e4b209b35
begin
	foreach(row -> print(row.prolific_pid * "\r\n"), eachrow(p_no_double_take))
end

# ╔═╡ c3114e6d-fcb4-45c3-8894-6bb2e082f7aa
@chain jspsych_data begin
	semijoin(p_no_double_take, on=:prolific_pid)
	filter(x -> !ismissing(x.trialphase) && x.trialphase in ["control_preload", "control_instruction_end"], _)
	groupby(:prolific_pid)
	combine(:time_elapsed => (x -> (maximum(x) - minimum(x))/60000) => :duration_m)
	describe(cols=:duration_m)
end

# ╔═╡ ff83fc55-e0d3-40e3-9c5e-54fe79bd26de
@chain jspsych_data begin
	semijoin(p_no_double_take, on=:prolific_pid)
	filter(x -> !ismissing(x.trialphase) && x.trialphase in ["control_explore", "control_confidence"], _)
	groupby(:prolific_pid)
	combine(:time_elapsed => (x -> (maximum(x) - minimum(x))/60000) => :duration_m)
	describe(cols=:duration_m)
end

# ╔═╡ f6b1e3e8-8d50-456f-bde8-85d2d29d684c
@chain jspsych_data begin
	semijoin(p_no_double_take, on=:prolific_pid)
	filter(x -> !ismissing(x.trialphase) && x.trialphase in ["control_reward_prompt", "control_reward"], _)
	groupby(:prolific_pid)
	combine(:time_elapsed => (x -> (maximum(x) - minimum(x))/60000) => :duration_m)
	describe(cols=:duration_m)
end

# ╔═╡ 66eedbb3-1602-4086-9cef-0a9876c0d96b
begin
	p_finished = filter(x -> !ismissing(x.finished) && x.finished, p_sum)
	foreach(row -> print(row.prolific_pid * "," * row.bonus * "\r\n"), eachrow(p_finished[:,[:prolific_pid, :bonus]]))
end

# ╔═╡ ce4197d7-33b6-4cee-be22-0cb929627070
antijoin(p_finished, p_no_double_take, on=:prolific_pid)

# ╔═╡ b9a96dd0-419b-405c-9219-1102d7b84bf6
spearman_brown(
r;
n = 2 # Number of splits
) = (n * r) / (1 + (n - 1) * r)

# ╔═╡ Cell order:
# ╠═e953ab1e-2f64-11f0-287e-afd1a5effa1f
# ╠═da326055-094a-4b3a-8a0c-dbb87366b3ae
# ╠═9eb71529-0dc7-4951-9e2e-96ec95f4df89
# ╠═ce32edbe-2883-4b02-ad66-49228b6c2bac
# ╠═b00006c6-ef32-45d3-b5a2-d6614f39bc31
# ╠═00d58d1e-9280-4c7e-8399-675db7abb89b
# ╠═c2896ab1-cf21-4198-bfe8-f672e372d03b
# ╠═ff8a5e4f-516e-4483-aaf1-a3d71ed38485
# ╠═04049036-613f-4048-9dfd-3b5e4b209b35
# ╠═66eedbb3-1602-4086-9cef-0a9876c0d96b
# ╠═ce4197d7-33b6-4cee-be22-0cb929627070
# ╟─781c45d2-d4b6-463f-87bf-a01db1fc0372
# ╠═34e147de-0af9-45a6-aa56-785d7d071153
# ╠═aaf73642-aaec-40a9-95f8-f7f0f25095c7
# ╠═5373906f-51fb-43aa-91f7-20656785a386
# ╟─0b1ead8e-ae59-4add-a1ea-e5420cadae5f
# ╟─e0becc69-04ab-4df0-b56b-df4acc8aca9c
# ╟─a7eb19ed-d7e8-4bb4-a2f3-17426035217d
# ╠═8c4fca45-504b-4be8-bbcc-e48e497b3f0e
# ╠═56a77ffc-be7d-474a-a0f8-5951fc87b424
# ╟─c0f66cdc-4d68-4f2d-b4c6-65de4821091f
# ╠═67b741f1-eb82-4e14-9ba6-79d903e0b2d0
# ╠═6dee7572-0add-4e83-970d-2e629fa7c632
# ╠═67f673b9-58a3-4b1a-a0a2-690758d621c1
# ╠═187e681c-1207-46f3-9cb3-9e6dcc40f625
# ╠═ae08f0a0-d7f3-475a-ac6e-7c6a305737cc
# ╟─03763d35-7383-464c-841c-ec955c83a402
# ╠═728a0a72-6e66-4618-9f47-3a21ec1e04d5
# ╟─470ec18e-3c3f-4dab-8c02-1ed5bc2b1bb1
# ╠═c3114e6d-fcb4-45c3-8894-6bb2e082f7aa
# ╠═ff83fc55-e0d3-40e3-9c5e-54fe79bd26de
# ╠═f6b1e3e8-8d50-456f-bde8-85d2d29d684c
# ╟─cd44eab8-151d-40bf-a7d9-364983554cc2
# ╠═cae03380-56a8-4292-8342-4323d1bf59a5
# ╠═5429031d-e6f3-45b3-9ab3-bff643950a2d
# ╠═004d7b5f-9ec5-4f4b-8724-02e65040bd8f
# ╠═80a6e41c-d578-4247-9a7a-b49dab153175
# ╟─457ab261-bdfd-4415-a350-c7e0ce9aac63
# ╠═db196a92-f7b3-4293-9788-ee80b57a1d04
# ╠═6bb9149f-2428-424c-8f22-166b72cb1fc9
# ╠═1de3f7cb-616e-4624-8adb-8d6796e5c6c4
# ╟─97b02f97-cab8-4ca7-aeb8-9848491aa5b1
# ╠═c427ad21-9e9e-4382-b5ad-bed4daa06aac
# ╠═4b524fd2-52cc-4e96-8fe9-8dfa9580a457
# ╠═f8b8f06d-c0fa-4741-beb0-837589a9639d
# ╠═259a2ada-b821-40ce-81bb-745cd8da34f1
# ╠═ab1a8d54-1a74-44c1-8e27-f0b42bd4238d
# ╠═5725d5d4-bb41-4cf1-8256-a2e3132a677b
# ╠═0dbe2eb5-e83c-4d26-acb1-1151a698a873
# ╠═b9a96dd0-419b-405c-9219-1102d7b84bf6
