### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ ad7f05f1-0e20-4b2a-9dc2-63c5da38bead
begin
	cd("/home/jovyan/")
	import Pkg
	# activate the shared project environment
	Pkg.activate("relmed_environment")
	# instantiate, i.e. make sure that all packages are downloaded
	Pkg.instantiate()
	using Random, DataFrames, JSON, CSV, StatsBase, JLD2, HTTP, CairoMakie, Printf, Distributions, CategoricalArrays, AlgebraOfGraphics, Dates, Turing, SHA
	using Tidier, GLM, MixedModels, ColorSchemes, PlutoUI, LaTeXStrings
	using LogExpFunctions: logistic, logit
	import OpenScienceFramework as OSF
	include("fetch_preprocess_data.jl")
	include("sample_utils.jl")
	include("plotting_utils.jl")
	include("stats_utils.jl")
	include("osf_utils.jl")
	include("vigour_utils.jl")
	Turing.setprogress!(false)
	nothing
end

# ╔═╡ 1afc3d98-3a97-466c-8079-0768e17af03b
TableOfContents(title="📚 Table of Contents", indent=true, depth=4, aside=true)

# ╔═╡ 0a7b56a0-9fbf-4eed-b2e6-f8f2bb86dc38
# Set up saving to OSF
begin
	osf_folder = "/Lab notebook images/PIT/Pilot7/"
	proj = setup_osf("Task development")
	upload = false
end

# ╔═╡ 8a279439-49a3-4aec-adf9-b6b580f81199
begin
	PILT_data, raw_test_data, raw_vigour_data, post_vigour_test_data, raw_PIT_data, WM_data, max_press_data, jspsych_data = load_pilot7_data(; force_download = false, return_version = "7.0")
	nothing
end

# ╔═╡ 24a17406-762f-494e-a13a-ad7266d5f6d9
md"""
Set theme globally
"""

# ╔═╡ 4557a55b-30ca-4e2b-9822-27e1311d3767
begin
	inch = 96 # In reality it shouldn't
	pt = 4/3
	cm = inch / 2.54
	th = merge(theme_minimal(), Theme(
		font = "Helvetica",
		fontsize = 12
	))
	set_theme!(th)
end

# ╔═╡ 802df1f5-ddf5-4b04-8655-55bd03a31772
begin
	data(max_press_data) *
		mapping(:avg_speed) *
		visual(Hist) |> 
		draw(;
			figure=(;size=(8cm, 6cm)), 
			axis=(;xlabel="Average speed (Press/sec)", ylabel="# Participants")
		)
end

# ╔═╡ 457a29ba-d33c-4389-a883-6c5c6ac61954
md"""
## PIT (Pavlovian-instrumental transfer)
"""

# ╔═╡ 1aceb591-9ed1-4a9a-849f-dac14802e5c0
begin
	vigour_unfinished = @chain raw_vigour_data begin
		@count(prolific_pid, exp_start_time)
		@mutate(most_common_n = ~mode(n))
		@filter(n < most_common_n)
	end
	vigour_data = @chain raw_vigour_data begin
		@mutate(
			press_per_sec = trial_presses / trial_duration * 1000,
			ratio = categorical(ratio; levels = [1, 8, 16], ordered=true),
			magnitude = categorical(magnitude; levels = [1, 2, 5], ordered=true)
		)
		@anti_join(vigour_unfinished)
	end
	nothing
end

# ╔═╡ cbde565b-7604-469b-b328-6c6bf84ceeeb
begin
	PIT_unfinished = @chain raw_PIT_data begin
		@count(prolific_pid, exp_start_time)
		@mutate(most_common_n = ~mode(n))
		@filter(n < most_common_n)
	end
	PIT_data = @chain raw_PIT_data begin
		# @filter(trial_presses > 0)
		@mutate(
			press_per_sec = trial_presses / trial_duration * 1000,
			ratio = categorical(ratio; levels = [1, 8, 16], ordered=true),
			magnitude = categorical(magnitude; levels = [1, 2, 5], ordered=true),
			coin_cat = categorical(coin; levels = [-1.0, -0.5, -0.01, 0.0, 0.01, 0.5, 1.0], ordered=true),
			progress = if_else(trial_number <= maximum(trial_number)/2, "Trial 1-36", "Trial 37-72")
		)
		@anti_join(PIT_unfinished)
		@arrange(prolific_pid, progress, magnitude, ratio)
		@mutate(pig = "Mag " * string(magnitude) * ", FR " * string(ratio))
		@mutate(pig=categorical(pig,levels=["Mag 2, FR 16","Mag 2, FR 8","Mag 5, FR 8","Mag 1, FR 1"]))
	end
	pav_eff = @chain PIT_data begin
			@filter(coin != 0)
			@mutate(valence = ifelse(coin > 0, "pos", "neg"))
			@group_by(prolific_pid, valence)
			@summarize(press_per_sec = mean(press_per_sec))
			@ungroup
			@pivot_wider(names_from = valence, values_from = press_per_sec)
			@mutate(diff = pos - neg)
			@mutate(type = if_else(diff > 1, "Pav", "Inst"))
	end
	leftjoin!(PIT_data, pav_eff, on=:prolific_pid)
	nothing;
end

# ╔═╡ f12dfb63-5fc6-48b9-b106-9f8e353a8e5a
@chain raw_PIT_data begin
	@filter(trial_presses == 0)
end

# ╔═╡ b747d881-6515-49eb-8768-e1ed38104e36
md"""
### Instrumental effect
"""

# ╔═╡ 7ca13679-ab22-4e7e-9a9e-573eefea9771
let
	common_rpp = unique(PIT_data.reward_per_press)
	fig = @chain PIT_data begin
		@filter(coin==0)
		# @bind_rows(vigour_data)
		# @mutate(trialphase=categorical(trialphase, levels=["vigour_trial", "pit_trial"], ordered=true))
		# @mutate(trialphase=~recode(trialphase, "vigour_trial" => "Vigour", "pit_trial" => "PIT\nw/o Pav. stim."))
		@filter(reward_per_press in !!common_rpp)
		plot_presses_vs_var(_; x_var=:reward_per_press, y_var=:press_per_sec, grp_var=:type, xlab="Reward/press", ylab = "Press/sec", grplab="Task", combine="average")
	end
	# Save
	filepath = joinpath("results/Pilot7/PIT", "PIT_press_cmp_with_vigour.png")
	save(filepath, fig; px_per_unit = 4)

	if upload
		upload_to_osf(
					filepath,
					proj,
					osf_folder
				)
	end

	fig
end

# ╔═╡ 2c1c0339-a6c4-4e36-81bc-e672ab3b9ebf
md"""
### Pavlovian transfer effect
"""

# ╔═╡ c3dc5eda-1421-4770-a1d3-f08b8c6c2655
begin
	colors=ColorSchemes.PRGn_7.colors;
	colors[4]=colorant"rgb(210, 210, 210)";
	nothing;
end

# ╔═╡ 6431d8ec-b43b-44dc-8631-861200a9a475
let
	pav_eff = @chain PIT_data begin
			@filter(coin != 0)
			@mutate(valence = ifelse(coin > 0, "pos", "neg"))
			@group_by(prolific_pid, valence)
			@summarize(press_per_sec = mean(press_per_sec))
			@ungroup
			@pivot_wider(names_from = valence, values_from = press_per_sec)
			@mutate(diff = pos - neg)
	end
	@info "% ΔValence>0: $(round(mean(pav_eff.diff.>0)...;digits=2))"
	data(pav_eff) *
		mapping(:diff) *
		visual(Hist) |>
	draw(axis=(;xlabel="ΔValence"))
end

# ╔═╡ 8d9858e0-5159-43ef-9a0d-20ba115876f2
grouped_data, avg_w_data = avg_presses_w_fn(PIT_data, [:coin_cat, :progress], :press_per_sec, :type)

# ╔═╡ 5c5f0ce8-c64b-413a-975d-22118f4e1852
let
	grouped_data, avg_w_data = avg_presses_w_fn(PIT_data, [:coin_cat, :progress], :press_per_sec, :type)
	p = data(avg_w_data) * (
		mapping(:coin_cat, :avg_y, col=:progress, group=:type) *
		(
			visual(Errorbars, whiskerwidth=4) *
			mapping(:se_y, color=:coin_cat => :"Coin value") +
			visual(Scatter, markersize=10) *
			mapping(color=:coin_cat => :"Coin value")
		) + 
		mapping(:coin_cat, :avg_y, col=:progress, color=:type => AlgebraOfGraphics.scale(:type_color)) *
		visual(Lines, linewidth=2)
	)
	p_ind = data(@mutate(grouped_data, avg_y = mean_y)) * mapping(:coin_cat, :avg_y, group=:prolific_pid, col=:progress) * visual(Lines, linewidth = 0.5, color=:gray80)
    fig = Figure(size = (12cm, 12cm))
	draw!(fig[1,1], p, scales(Color = (; palette=colors)); axis=(;xlabel="Pavlovian stimuli (coin value)", ylabel="Press/sec"))
	draw!(fig[2,1], p_ind + p, scales(Color = (; palette=colors)); axis=(;xlabel="Pavlovian stimuli (coin value)", ylabel="Press/sec"))
	# legend!(fig[1,2], p)

	# Save
	filepath = joinpath("results/Pilot7/PIT", "PIT_press_by_pavlovian.pdf")
	save(filepath, fig; px_per_unit = 1)

	if upload
		upload_to_osf(
					filepath,
					proj,
					osf_folder
				)
	end

	fig
end

# ╔═╡ 1beaa9b1-73e9-407b-bf5f-a4091f00a17d
let
	df = PIT_data
	colors=ColorSchemes.PRGn_7.colors
	colors[4]=colorant"rgb(210, 210, 210)"
	grouped_data, avg_w_data = avg_presses_w_fn(df, [:coin, :pig, :progress], :press_per_sec)
	p = data(avg_w_data) *
	mapping(:coin=>nonnumeric, :avg_y, col=:pig, row=:progress) *
	(
		visual(Lines, linewidth=2, color=:gray70) +
		visual(Errorbars, whiskerwidth=4) *
		mapping(:se_y, color=:coin => nonnumeric => :"Coin value") +
		visual(Scatter, markersize=10) *
		mapping(color=:coin => nonnumeric => :"Coin value")
	)
	p_ind = data(@mutate(grouped_data, avg_y = mean_y)) * 
		mapping(:coin=>nonnumeric, :avg_y, col=:pig, row=:progress, group=:prolific_pid) * 
		visual(Lines, linewidth = 0.5, color=:gray80)
	fig = Figure(;size=(16cm, 16cm))
	draw!(fig[1,1], p, scales(Color = (; palette=colors)); axis=(;xlabel="Pavlovian stimuli (coin value)", ylabel="Press/sec", xticklabelrotation=pi/4))
	p = draw!(fig[2,1], p_ind + p, scales(Color = (; palette=colors)); axis=(;xlabel="Pavlovian stimuli (coin value)", ylabel="Press/sec", xticklabelrotation=pi/4))
	# legend!(fig[1,2], p)

	# Save
	filepath = joinpath("results/Pilot7/PIT", "PIT_press_by_pavlovian_pig.png")
	save(filepath, fig; px_per_unit = 4)

	if upload
		upload_to_osf(
					filepath,
					proj,
					osf_folder
				)
	end

	fig
end

# ╔═╡ 95fd5935-19de-4a66-881e-77fa276a70af
begin
	PIT_test_unfinished = @chain raw_test_data begin
		@filter(block == "pavlovian")
		@count(prolific_pid, exp_start_time)
		@mutate(most_common_n = ~mode(n))
		@filter(n < most_common_n)
	end
	PIT_test_data = @chain raw_test_data begin
		@filter(block == "pavlovian")
		@mutate(
			magnitude_left = case_when(
				contains(stimulus_left, "PIT4") => -0.01,
				contains(stimulus_left, "PIT6") => -1.0,
				true => magnitude_left),
			magnitude_right = case_when(
				contains(stimulus_right, "PIT4") => -0.01,
				contains(stimulus_right, "PIT6") => -1.0,
				true => magnitude_right)
		)
		@anti_join(PIT_test_unfinished)
	end
	nothing;
end

# ╔═╡ 7fa14255-9f46-466d-baa3-e6fc2eec3347
md"""
#### PIT by test accuracy
"""

# ╔═╡ 58ed1255-6f89-4f8a-b6d6-a8dee840dea2
let
	PIT_acc_df = @chain PIT_test_data begin
		@mutate(valence = ifelse(magnitude_left * magnitude_right < 0, "Different", ifelse(magnitude_left > 0, "Positive", "Negative")))
		@group_by(valence)
		@summarize(acc = mean(skipmissing((magnitude_right > magnitude_left) == right_chosen)))
		@ungroup
	end
	@info "PIT acc for NOT in the same valence: $(round(PIT_acc_df.acc[PIT_acc_df.valence.=="Different"]...; digits=2))"
	@info "PIT acc for in the positive valence: $(round(PIT_acc_df.acc[PIT_acc_df.valence.=="Positive"]...; digits=2))"
	@info "PIT acc for in the negative valence: $(round(PIT_acc_df.acc[PIT_acc_df.valence.=="Negative"]...; digits=2))"

	p = @chain PIT_test_data begin
		@mutate(valence = ifelse(magnitude_left * magnitude_right < 0, "Different", ifelse(magnitude_left > 0, "Positive", "Negative")))
		@mutate(correct = (magnitude_right .> magnitude_left) .== right_chosen)
		@filter(!ismissing(correct))
		@group_by(prolific_pid, valence)
		@summarize(acc = mean(correct))
		@ungroup
		data(_) * mapping(:valence, :acc, color=:valence) * visual(RainClouds; clouds=hist, hist_bins = 20, plot_boxplots = false)
	end
	fig = Figure(;size=(8cm, 6cm))
	p = draw!(fig[1,1], p, scales(Color = (; palette=[colorant"gray", ColorSchemes.Set3_5[[4,5]]...])); axis=(;xlabel="Pavlovian valence", ylabel="PIT test accuracy"))

	# Save
	filepath = joinpath("results/Pilot7/PIT", "PIT_test_acc_by_valence.png")
	save(filepath, fig; px_per_unit = 4)

	if upload
		upload_to_osf(
					filepath,
					proj,
					osf_folder
				)
	end

	fig
end

# ╔═╡ 07321c17-5493-4d1b-a918-2129cab2b0e1
let
	acc_grp_df = @chain PIT_test_data begin
		@group_by(prolific_pid)
		@summarize(acc = mean(skipmissing((magnitude_right > magnitude_left) == right_chosen)))
		@ungroup()
	end
	# acc_quantile = quantile(acc_grp_df.acc, [0.25, 0.5, 0.75])
	# @info "Acc at each quantile: $([@sprintf("%.1f%%", v * 100) for v in acc_quantile])"
	acc_grp_df.acc_grp = cut(acc_grp_df.acc, [0.5, 0.75]; extend=true)

	grouped_data, avg_w_data = avg_presses_w_fn(innerjoin(PIT_data, acc_grp_df, on=[:prolific_pid]), [:coin, :acc_grp], :press_per_sec)
	
	p = data(avg_w_data) *
	mapping(:coin=>nonnumeric, :avg_y, col=:acc_grp) *
	(
		visual(Lines, linewidth=2, color=:gray75) +
		visual(Errorbars, whiskerwidth=4) *
		mapping(:se_y, color=:coin => nonnumeric => :"Coin value") +
		visual(Scatter, markersize=10) *
		mapping(color=:coin => nonnumeric => :"Coin value")
	)
	fig = Figure(;size=(12cm, 6cm))
	p = draw!(fig[1,1], p, scales(Color = (; palette=colors)); axis=(;xlabel="Pavlovian stimuli (coin value)", ylabel="Press/sec", xticklabelrotation=pi/4))
	Label(fig[0,:], "Press Rates by Pavlovian Stimuli Across Test Accuracy", tellwidth = false)
	
	# Save
	filepath = joinpath("results/Pilot7/PIT", "PIT_press_by_pavlovian_acc.png")
	save(filepath, fig; px_per_unit = 4)

	if upload
		upload_to_osf(
					filepath,
					proj,
					osf_folder
				)
	end

	fig
	@count(acc_grp_df, acc_grp)
end

# ╔═╡ 45ebcdfa-e2ef-4df7-a31d-9b5e06044e08
let
	acc_grp_df = @chain PIT_test_data begin
		@group_by(prolific_pid)
		@summarize(acc = mean(skipmissing((magnitude_right > magnitude_left) == right_chosen)))
		@ungroup
	end

	acc_grp_df.acc_grp = cut(acc_grp_df.acc, [0.5, 0.75]; extend=true)

	grouped_data, avg_w_data = avg_presses_w_fn(@filter(innerjoin(PIT_data, acc_grp_df, on=[:prolific_pid])), [:reward_per_press, :acc_grp], :press_per_sec)
	
	p = data(avg_w_data) *
	mapping(:reward_per_press, :avg_y, col=:acc_grp) *
	(
		visual(Lines, linewidth=2, color=:gray75) +
		visual(Errorbars, whiskerwidth=4) *
		mapping(:se_y, color=:reward_per_press) +
		visual(Scatter, markersize=10) *
		mapping(color=:reward_per_press)
	)
	fig = Figure(;size=(12cm, 6cm))
	p = draw!(fig[1,1], p; axis=(;xlabel="Reward/press", ylabel="Press/sec", xticklabelrotation=pi/4))
	Label(fig[0,:], "Press Rates by Reward Rates Across Test Accuracy", tellwidth = false)
	
	# Save
	filepath = joinpath("results/Pilot7/PIT", "PIT_press_by_rpp_acc.png")
	save(filepath, fig; px_per_unit = 4)

	if upload
		upload_to_osf(
					filepath,
					proj,
					osf_folder
				)
	end

	fig
end

# ╔═╡ 19ee8038-242c-4545-ba42-cec1bd6f9b5c
let
	pav_eff = @chain PIT_data begin
			@filter(coin != 0)
			@mutate(valence = ifelse(coin > 0, "pos", "neg"))
			@group_by(prolific_pid, valence)
			@summarize(press_per_sec = mean(press_per_sec))
			@ungroup
			@pivot_wider(names_from = valence, values_from = press_per_sec)
			@mutate(diff = pos - neg)
	end
	acc_grp_df = @chain PIT_test_data begin
		@group_by(prolific_pid)
		@summarize(acc = mean(skipmissing((magnitude_right > magnitude_left) == right_chosen)))
		@ungroup()
	end
	acc_grp_df.acc_grp = cut(acc_grp_df.acc, [0.5, 0.75]; extend=true)
	data(@inner_join(pav_eff, acc_grp_df)) *
		mapping(:acc, :diff) * (visual(Scatter, alpha = 0.3) + AlgebraOfGraphics.linear()) |>
	draw(figure=(;size=(8cm,6cm)), axis=(;xlabel="Post-test accuracy", ylabel="ΔValence"))
end

# ╔═╡ df9a9759-a3a0-491b-9812-62d145bcf3b0
md"""
#### Interaction with time
"""

# ╔═╡ 3a4f9ee8-801e-42a5-9f5c-d05dde038cc6
let
	pav_eff = @chain PIT_data begin
			@filter(coin != 0)
			@mutate(valence = ifelse(coin > 0, "pos", "neg"))
			@group_by(prolific_pid, progress, valence)
			@summarize(press_per_sec = mean(press_per_sec))
			@ungroup
			@pivot_wider(names_from = valence, values_from = press_per_sec)
			@mutate(diff = pos - neg)
	end
	data(pav_eff) *
		mapping(:progress => presorted, :diff => "ΔValence", group=:prolific_pid) *
		visual(Lines, color = :gray80) |>
	draw()
end

# ╔═╡ ffd8653e-ce66-4597-b9d7-c3be2a527eac
md"""
## PIT conflicts
"""

# ╔═╡ 6f5d4458-909b-4818-97de-1de7dd555561
function flatten_dataframe(df, array_column)
    # Use comprehension to create all rows
    rows = [(; zip(propertynames(df), [
        col == array_column ? val : row[col] 
        for col in propertynames(df)
    ])...) for row in eachrow(df) for val in row[array_column]]
    
    return DataFrame(rows)
end

# ╔═╡ 6d55f942-c909-4b99-a4bd-1ae1b5cd0c06
let
	flattened_PIT = flatten_dataframe(PIT_data, :response_times)
	flattened_PIT = @chain flattened_PIT begin
		@group_by(prolific_pid, trial_number)
		@mutate(cumu_time = cumsum(response_times)/1000, cumu_press = 1:n())
		@filter(!(cumu_press == 1 && response_times < 80))
		@mutate(cumu_press = 1:n())
		@ungroup()
		@mutate(speed = cumu_press/cumu_time)
		@mutate(valence = case_when(
			coin == 0 => "Neutral",
			coin > 0 => "Positive",
			coin < 0 => "Negative"
		))
		@filter(valence != "Neutral")
	end
	@filter(flattened_PIT, speed > 10)
	data(flattened_PIT) *
	mapping(:cumu_time, :speed, color=:coin_cat, col=:pig, row=:type) *
	(AlgebraOfGraphics.smooth()) |>
	draw(scales(Color = (; palette=colors)))
end

# ╔═╡ 8e046c5a-1454-4762-a93f-1555d7549931
md"""
## Reliability of PIT effect
"""

# ╔═╡ b6c4383c-98f8-47d1-91e1-369bd9f27aae
md"""
### Valence difference
"""

# ╔═╡ 8e103789-dc11-488f-8671-2222c0360fa3
let
	splithalf_df = @chain PIT_data begin
		@mutate(half = ifelse(trial_number <= maximum(trial_number)/2, "x", "y"))
		@filter(coin != 0)
		@mutate(valence = ifelse(coin > 0, "pos", "neg"))
		@group_by(prolific_pid, half, valence)
		@summarize(press_per_sec = mean(press_per_sec))
		@ungroup
		@pivot_wider(names_from = valence, values_from = press_per_sec)
		@mutate(diff = pos - neg)
		unstack([:prolific_pid], :half, :diff)
	end

	fig=Figure(;size=(8cm, 6cm))
	workshop_reliability_scatter!(
		fig[1,1];
		df=splithalf_df,
		xlabel="ΔValence (Trial 1-36)",
		ylabel="ΔValence (Trial 37-72)",
		xcol=:x,
		ycol=:y,
		subtitle="Valence Difference"
	)
	
	# Save
	filepath = joinpath("results/Pilot7/PIT", "PIT_valdiff_splithalf_firstsecond.png")
	save(filepath, fig; px_per_unit = 4)

	if upload
		upload_to_osf(
			filepath,
			proj,
			osf_folder
		)
	end
	
	fig
end

# ╔═╡ 8e36025a-fe92-4d1a-a7fa-c37918a16dbc
let
	splithalf_df = @chain PIT_data begin
		@mutate(half = ifelse(trial_number % 2 === 0, "x", "y"))
		@filter(coin != 0)
		@mutate(valence = ifelse(coin > 0, "pos", "neg"))
		@group_by(prolific_pid, half, valence)
		@summarize(press_per_sec = mean(press_per_sec))
		@ungroup
		@pivot_wider(names_from = valence, values_from = press_per_sec)
		@mutate(diff = pos - neg)
		unstack([:prolific_pid], :half, :diff)
	end

	fig=Figure(;size=(8cm, 6cm))
	workshop_reliability_scatter!(
		fig[1,1];
		df=splithalf_df,
		xlabel="ΔValence (Trial 2,4,6...)",
		ylabel="ΔValence (Trial 1,3,5...)",
		xcol=:x,
		ycol=:y,
		subtitle="Valence Difference"
	)
	
	# Save
	filepath = joinpath("results/Pilot7/PIT", "PIT_valdiff_splithalf_evenodd.png")
	save(filepath, fig; px_per_unit = 4)

	if upload
		upload_to_osf(
			filepath,
			proj,
			osf_folder
		)
	end

	fig
end

# ╔═╡ fbc4466c-1d20-4359-a893-72a255c903c6
md"""
### Slope: sensitivity to Pavlovian stimuli
"""

# ╔═╡ 22915221-f312-4b6d-8ae6-b0818e06c1a3
let
	glm_coef(dat) = coef(lm(@formula(press_per_sec ~ coin), dat))[2]
	splithalf_df = @chain PIT_data begin
		@mutate(half = ifelse(trial_number <= maximum(trial_number)/2, "x", "y"))
		@filter(coin != 0)
		@mutate(valence = ifelse(coin > 0, "pos", "neg"))
		@group_by(valence)
		@mutate(coin = (coin - mean(coin))/std(coin))
		@ungroup
		groupby([:prolific_pid, :half, :valence])
		combine(AsTable([:press_per_sec, :coin]) => glm_coef => :β)
		unstack([:prolific_pid, :valence], :half, :β)
	end

	fig=Figure(;size=(12cm, 6cm))
	workshop_reliability_scatter!(
		fig[1,1];
		df=@filter(splithalf_df, valence == "pos"),
		xlabel="Trial 1-36",
		ylabel="Trial 37-72",
		xcol=:x,
		ycol=:y,
		subtitle=L"$\beta_{Pos}$"
	)
	workshop_reliability_scatter!(
		fig[1,2];
		df=@filter(splithalf_df, valence == "neg"),
		xlabel="Trial 1-36",
		ylabel="Trial 37-72",
		xcol=:x,
		ycol=:y,
		subtitle=L"$\beta_{Neg}$"
	)
	Label(fig[0,:],"Pavlovian Sensitivity within Valence")
	# Save
	filepath = joinpath("results/Pilot7/PIT", "PIT_slope_splithalf_firstsecond.png")
	save(filepath, fig; px_per_unit = 4)

	if upload
		upload_to_osf(
			filepath,
			proj,
			osf_folder
		)
	end
	
	fig
end

# ╔═╡ 0691061e-d096-46e2-b035-a4ff0dda78a1
let
	glm_coef(dat) = coef(lm(@formula(press_per_sec ~ coin), dat))[2]
	splithalf_df = @chain PIT_data begin
		@mutate(half = ifelse(trial_number % 2 === 0, "x", "y"))
		@filter(coin != 0)
		@mutate(valence = ifelse(coin > 0, "pos", "neg"))
		@group_by(valence)
		@mutate(coin = (coin - mean(coin))/std(coin))
		@ungroup
		groupby([:prolific_pid, :half, :valence])
		combine(AsTable([:press_per_sec, :coin]) => glm_coef => :β)
		unstack([:prolific_pid, :valence], :half, :β)
	end

	fig=Figure(;size=(12cm, 6cm))
	workshop_reliability_scatter!(
		fig[1,1];
		df=@filter(splithalf_df, valence == "pos"),
		xlabel="Trial 2,4,6...",
		ylabel="Trial 1,3,5...",
		xcol=:x,
		ycol=:y,
		subtitle=L"$\beta_{Pos}$"
	)
	workshop_reliability_scatter!(
		fig[1,2];
		df=@filter(splithalf_df, valence == "neg"),
		xlabel="Trial 2,4,6...",
		ylabel="Trial 1,3,5...",
		xcol=:x,
		ycol=:y,
		subtitle=L"$\beta_{Neg}$"
	)
	Label(fig[0,:],"Pavlovian Sensitivity within Valence")
	# Save
	filepath = joinpath("results/Pilot7/PIT", "PIT_slope_splithalf_evenodd.png")
	save(filepath, fig; px_per_unit = 4)
	
	if upload
		upload_to_osf(
			filepath,
			proj,
			osf_folder
		)
	end
	
	fig
end

# ╔═╡ cbbff405-bcf0-4ea8-9683-81080f7b8a9e
md"""
### Asymmetry
"""

# ╔═╡ 8812e1f4-b2b9-4cda-a33a-a12ee278c99b
md"""
$\text{Asymmetry} = \text{Press rate} | \text{Positive} + \text{Press rate} | \text{Negative} - 2 \times \text{Press rate} | \text{Empty}$
"""

# ╔═╡ ec623a4a-c506-47b3-9c17-737fee355511
let
	splithalf_df = @chain PIT_data begin
		@mutate(half = ifelse(trial_number <= maximum(trial_number)/2, "x", "y"))
		@mutate(valence = ifelse(coin == 0, "zero", ifelse(coin > 0, "pos", "neg")))
		@group_by(prolific_pid, half, valence)
		@summarize(press_per_sec = mean(press_per_sec))
		@ungroup
		@pivot_wider(names_from = valence, values_from = press_per_sec)
		@mutate(asymm = pos + neg - 2 * zero)
		unstack([:prolific_pid], :half, :asymm)
	end

	fig=Figure(;size=(8cm, 6cm))
	workshop_reliability_scatter!(
		fig[1,1];
		df=splithalf_df,
		xlabel="Asymmetry (Trial 1-36)",
		ylabel="Asymmetry (Trial 37-72)",
		xcol=:x,
		ycol=:y,
		subtitle="Asymmetry"
	)
	
	# Save
	filepath = joinpath("results/Pilot7/PIT", "PIT_asymm_splithalf_firstsecond.png")
	save(filepath, fig; px_per_unit = 4)
	
	if upload
		upload_to_osf(
			filepath,
			proj,
			osf_folder
		)
	end
	
	fig
end

# ╔═╡ 53979178-5e41-4ef1-82d5-c10193b642ef
let
	splithalf_df = @chain PIT_data begin
		@mutate(half = ifelse(trial_number % 2 === 0, "x", "y"))
		@mutate(valence = ifelse(coin == 0, "zero", ifelse(coin > 0, "pos", "neg")))
		@group_by(prolific_pid, half, valence)
		@summarize(press_per_sec = mean(press_per_sec))
		@ungroup
		@pivot_wider(names_from = valence, values_from = press_per_sec)
		@mutate(asymm = pos + neg - 2 * zero)
		unstack([:prolific_pid], :half, :asymm)
	end

	fig=Figure(;size=(8cm, 6cm))
	workshop_reliability_scatter!(
		fig[1,1];
		df=splithalf_df,
		xlabel="Asymmetry (Trial 2,4,6...)",
		ylabel="Asymmetry (Trial 1,3,5...)",
		xcol=:x,
		ycol=:y,
		subtitle="Asymmetry"
	)
	
	# Save
	filepath = joinpath("results/Pilot7/PIT", "PIT_asymm_splithalf_evenodd.png")
	save(filepath, fig; px_per_unit = 4)
	
	if upload
		upload_to_osf(
			filepath,
			proj,
			osf_folder
		)
	end
	
	fig
end

# ╔═╡ Cell order:
# ╠═ad7f05f1-0e20-4b2a-9dc2-63c5da38bead
# ╠═1afc3d98-3a97-466c-8079-0768e17af03b
# ╠═0a7b56a0-9fbf-4eed-b2e6-f8f2bb86dc38
# ╠═8a279439-49a3-4aec-adf9-b6b580f81199
# ╟─24a17406-762f-494e-a13a-ad7266d5f6d9
# ╠═4557a55b-30ca-4e2b-9822-27e1311d3767
# ╠═802df1f5-ddf5-4b04-8655-55bd03a31772
# ╟─457a29ba-d33c-4389-a883-6c5c6ac61954
# ╠═1aceb591-9ed1-4a9a-849f-dac14802e5c0
# ╠═cbde565b-7604-469b-b328-6c6bf84ceeeb
# ╠═f12dfb63-5fc6-48b9-b106-9f8e353a8e5a
# ╟─b747d881-6515-49eb-8768-e1ed38104e36
# ╠═7ca13679-ab22-4e7e-9a9e-573eefea9771
# ╟─2c1c0339-a6c4-4e36-81bc-e672ab3b9ebf
# ╟─c3dc5eda-1421-4770-a1d3-f08b8c6c2655
# ╠═6431d8ec-b43b-44dc-8631-861200a9a475
# ╠═8d9858e0-5159-43ef-9a0d-20ba115876f2
# ╠═5c5f0ce8-c64b-413a-975d-22118f4e1852
# ╠═1beaa9b1-73e9-407b-bf5f-a4091f00a17d
# ╟─95fd5935-19de-4a66-881e-77fa276a70af
# ╟─7fa14255-9f46-466d-baa3-e6fc2eec3347
# ╠═58ed1255-6f89-4f8a-b6d6-a8dee840dea2
# ╠═07321c17-5493-4d1b-a918-2129cab2b0e1
# ╟─45ebcdfa-e2ef-4df7-a31d-9b5e06044e08
# ╠═19ee8038-242c-4545-ba42-cec1bd6f9b5c
# ╟─df9a9759-a3a0-491b-9812-62d145bcf3b0
# ╠═3a4f9ee8-801e-42a5-9f5c-d05dde038cc6
# ╟─ffd8653e-ce66-4597-b9d7-c3be2a527eac
# ╠═6f5d4458-909b-4818-97de-1de7dd555561
# ╠═6d55f942-c909-4b99-a4bd-1ae1b5cd0c06
# ╟─8e046c5a-1454-4762-a93f-1555d7549931
# ╟─b6c4383c-98f8-47d1-91e1-369bd9f27aae
# ╟─8e103789-dc11-488f-8671-2222c0360fa3
# ╟─8e36025a-fe92-4d1a-a7fa-c37918a16dbc
# ╟─fbc4466c-1d20-4359-a893-72a255c903c6
# ╟─22915221-f312-4b6d-8ae6-b0818e06c1a3
# ╟─0691061e-d096-46e2-b035-a4ff0dda78a1
# ╟─cbbff405-bcf0-4ea8-9683-81080f7b8a9e
# ╟─8812e1f4-b2b9-4cda-a33a-a12ee278c99b
# ╟─ec623a4a-c506-47b3-9c17-737fee355511
# ╟─53979178-5e41-4ef1-82d5-c10193b642ef
