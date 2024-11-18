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
	osf_folder = "/Workshop figures/PIT/"
	proj = setup_osf("Task development")
end

# ╔═╡ 8a279439-49a3-4aec-adf9-b6b580f81199
begin
	# Load data
	_, raw_test_data, raw_vigour_data, _, raw_PIT_data, _,
		_, _ = load_pilot6_data()
	nothing
end

# ╔═╡ 24a17406-762f-494e-a13a-ad7266d5f6d9
md"""
Set theme globally
"""

# ╔═╡ 4557a55b-30ca-4e2b-9822-27e1311d3767
set_theme!(theme_minimal();
		font = "Helvetica",
		fontsize = 16);

# ╔═╡ 457a29ba-d33c-4389-a883-6c5c6ac61954
md"""
## PIT (Pavlovian-instrumental transfer)
"""

# ╔═╡ b747d881-6515-49eb-8768-e1ed38104e36
md"""
### Instrumental effect
"""

# ╔═╡ 1aceb591-9ed1-4a9a-849f-dac14802e5c0
begin
	vigour_unfinished = @chain raw_vigour_data begin
		@count(prolific_pid, exp_start_time)
		@mutate(most_common_n = ~mode(n))
		@filter(n < most_common_n)
	end
	vigour_data = @chain raw_vigour_data begin
		@filter(prolific_pid != "671139a20b977d78ec2ac1e0") # From sess1
		@filter(!(prolific_pid in ["6721ec463c2f6789d5b777b5", "62ae1ecc1bd29fdc6b14f6ea"])) # From sess2
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
		@filter(prolific_pid != "671139a20b977d78ec2ac1e0")
		@filter(prolific_pid != "6721ec463c2f6789d5b777b5")
		@mutate(
			press_per_sec = trial_presses / trial_duration * 1000,
			ratio = categorical(ratio; levels = [1, 8, 16], ordered=true),
			magnitude = categorical(magnitude; levels = [1, 2, 5], ordered=true),
			coin_cat = categorical(coin; levels = [-1.0, -0.5, -0.01, 0.0, 0.01, 0.5, 1.0], ordered=true)
		)
		@anti_join(PIT_unfinished)
	end
	nothing;
end

# ╔═╡ 96c684cf-bf21-468c-8480-98ffeb3cfbf8
let
	fig = @chain PIT_data begin
		@filter(trial_number != 0)
		@ungroup
		plot_presses_vs_var(_; x_var=:reward_per_press, y_var=:press_per_sec, xlab="Reward/press", ylab = "Press/sec", combine="average")
	end
	
	# Save
	filepaths = joinpath("results/workshop/PIT", "PIT_press_by_reward_rate.png")
	save(filepaths, fig; px_per_unit = 4)

	upload_to_osf(
			filepaths,
			proj,
			osf_folder
		)

	fig
end

# ╔═╡ 7ca13679-ab22-4e7e-9a9e-573eefea9771
let
	common_rpp = unique(PIT_data.reward_per_press)
	fig = @chain PIT_data begin
		@filter(coin==0)
		@bind_rows(vigour_data)
		@mutate(trialphase=categorical(trialphase, levels=["vigour_trial", "pit_trial"], ordered=true))
		@mutate(trialphase=~recode(trialphase, "vigour_trial" => "Vigour", "pit_trial" => "PIT\nw/o Pav. stim."))
		@filter(reward_per_press in !!common_rpp)
		plot_presses_vs_var(_; x_var=:reward_per_press, y_var=:press_per_sec, grp_var=:trialphase, xlab="Reward/press", ylab = "Press/sec", grplab="Task", combine="average")
	end
	# Save
	filepaths = joinpath("results/workshop/PIT", "PIT_press_cmp_with_vigour.png")
	save(filepaths, fig; px_per_unit = 4)

	upload_to_osf(
			filepaths,
			proj,
			osf_folder
		)

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

# ╔═╡ 5c5f0ce8-c64b-413a-975d-22118f4e1852
let
	grouped_data, avg_w_data = avg_presses_w_fn(PIT_data, [:coin_cat], :press_per_sec)
	p = data(avg_w_data) *
	mapping(:coin_cat, :avg_y) *
	(
		visual(Lines, linewidth=2, color=:gray75) +
		visual(Errorbars, whiskerwidth=4) *
		mapping(:se_y, color=:coin_cat => :"Coin value") +
		visual(Scatter, markersize=10) *
		mapping(color=:coin_cat => :"Coin value")
	)
	fig = Figure(;size=(8, 6) .* 144 ./ 2.54)
	p = draw!(fig[1,1], p, scales(Color = (; palette=colors)); axis=(;xlabel="Pavlovian stimuli (coin value)", ylabel="Press/sec", xticklabelrotation=pi/4))
	legend!(fig[1,2], p)

	# Save
	filepaths = joinpath("results/workshop/PIT", "PIT_press_by_pavlovian.png")
	save(filepaths, fig; px_per_unit = 4)

	upload_to_osf(
			filepaths,
			proj,
			osf_folder
		)

	fig
end

# ╔═╡ 1beaa9b1-73e9-407b-bf5f-a4091f00a17d
let
	df = @chain PIT_data begin
		@arrange(prolific_pid, magnitude, ratio)
		@mutate(pig = "Mag " * string(magnitude) * ", FR " * string(ratio))
		@mutate(pig=categorical(pig,levels=["Mag 2, FR 16","Mag 2, FR 8","Mag 5, FR 8","Mag 1, FR 1"]))
	end
	colors=ColorSchemes.PRGn_7.colors
	colors[4]=colorant"rgb(210, 210, 210)"
	grouped_data, avg_w_data = avg_presses_w_fn(df, [:coin, :pig], :press_per_sec)
	p = data(avg_w_data) *
	mapping(:coin=>nonnumeric, :avg_y, col=:pig=>nonnumeric) *
	(
		visual(Lines, linewidth=1, color=:gray75) +
		visual(Errorbars, whiskerwidth=4) *
		mapping(:se_y, color=:coin => nonnumeric => :"Coin value") +
		visual(Scatter, markersize=10) *
		mapping(color=:coin => nonnumeric => :"Coin value")
	)
	fig = Figure(;size=(16, 5) .* 144 ./ 2.54)
	p = draw!(fig[1,1], p, scales(Color = (; palette=colors)); axis=(;xlabel="Pavlovian stimuli (coin value)", ylabel="Press/sec", xticklabelrotation=pi/4))
	legend!(fig[1,2], p)

	# Save
	filepaths = joinpath("results/workshop/PIT", "PIT_press_by_pavlovian_pig.png")
	save(filepaths, fig; px_per_unit = 4)

	upload_to_osf(
			filepaths,
			proj,
			osf_folder
		)

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
		@filter(prolific_pid != "671139a20b977d78ec2ac1e0")
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
		@group_by(prolific_pid, exp_start_time, valence)
		@summarize(acc = mean(correct))
		@ungroup
		data(_) * mapping(:valence, :acc, color=:valence) * visual(RainClouds; clouds=hist, hist_bins = 20, plot_boxplots = false)
	end
	fig = Figure(;size=(8, 6) .* 144 ./ 2.54)
	p = draw!(fig[1,1], p, scales(Color = (; palette=[colorant"gray", ColorSchemes.Set3_5[[4,5]]...])); axis=(;xlabel="Pavlovian valence", ylabel="PIT test accuracy"))

	# Save
	filepaths = joinpath("results/workshop/PIT", "PIT_test_acc_by_valence.png")
	save(filepaths, fig; px_per_unit = 4)

	upload_to_osf(
			filepaths,
			proj,
			osf_folder
		)

	fig
end

# ╔═╡ b8df8891-ec81-424e-9e4f-3b1b9c152688
let
	retest_df = @chain PIT_test_data begin
		@group_by(prolific_pid, session)
		@summarize(acc = mean(skipmissing((magnitude_right > magnitude_left) == right_chosen)))
		@ungroup()
		unstack([:prolific_pid], :session, :acc)
		dropmissing
	end
	
	fig=Figure(;size=(8, 6) .* 144 ./ 2.54)
	workshop_reliability_scatter!(
		fig[1,1];
		df=retest_df,
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=Symbol(string(1)),
		ycol=Symbol(string(2)),
		subtitle="Test-retest Pavlovian Test Accuracy",
		correct_r=false
	)
	fig
end

# ╔═╡ 07321c17-5493-4d1b-a918-2129cab2b0e1
let
	acc_grp_df = @chain PIT_test_data begin
		@group_by(prolific_pid, session)
		@summarize(acc = mean(skipmissing((magnitude_right > magnitude_left) == right_chosen)))
		@ungroup()
	end
	# acc_quantile = quantile(acc_grp_df.acc, [0.25, 0.5, 0.75])
	# @info "Acc at each quantile: $([@sprintf("%.1f%%", v * 100) for v in acc_quantile])"
	acc_grp_df.acc_grp = cut(acc_grp_df.acc, [0.5, 0.75]; extend=true)

	grouped_data, avg_w_data = avg_presses_w_fn(innerjoin(PIT_data, acc_grp_df, on=[:prolific_pid,:session]), [:coin, :acc_grp], :press_per_sec)
	
	p = data(avg_w_data) *
	mapping(:coin=>nonnumeric, :avg_y, col=:acc_grp) *
	(
		visual(Lines, linewidth=2, color=:gray75) +
		visual(Errorbars, whiskerwidth=4) *
		mapping(:se_y, color=:coin => nonnumeric => :"Coin value") +
		visual(Scatter, markersize=10) *
		mapping(color=:coin => nonnumeric => :"Coin value")
	)
	fig = Figure(;size=(12, 6) .* 144 ./ 2.54)
	p = draw!(fig[1,1], p, scales(Color = (; palette=colors)); axis=(;xlabel="Pavlovian stimuli (coin value)", ylabel="Press/sec", xticklabelrotation=pi/4))
	Label(fig[0,:], "Press Rates by Pavlovian Stimuli Across Test Accuracy", tellwidth = false)
	# legend!(fig[1,2], p)
	
	# Save
	filepaths = joinpath("results/workshop/PIT", "PIT_press_by_pavlovian_acc.png")
	save(filepaths, fig; px_per_unit = 4)

	upload_to_osf(
			filepaths,
			proj,
			osf_folder
		)

	fig
end

# ╔═╡ 45ebcdfa-e2ef-4df7-a31d-9b5e06044e08
let
	acc_grp_df = @chain PIT_test_data begin
		@group_by(prolific_pid, session)
		@summarize(acc = mean(skipmissing((magnitude_right > magnitude_left) == right_chosen)))
		@ungroup
	end
	# acc_quantile = quantile(acc_grp_df.acc, [0.25, 0.5, 0.75])
	# @info "Acc at each quantile: $([@sprintf("%.1f%%", v * 100) for v in acc_quantile])"
	acc_grp_df.acc_grp = cut(acc_grp_df.acc, [0.5, 0.75]; extend=true)

	grouped_data, avg_w_data = avg_presses_w_fn(@filter(innerjoin(PIT_data, acc_grp_df, on=[:prolific_pid, :session])), [:reward_per_press, :acc_grp], :press_per_sec)
	
	p = data(avg_w_data) *
	mapping(:reward_per_press, :avg_y, col=:acc_grp) *
	(
		visual(Lines, linewidth=2, color=:gray75) +
		visual(Errorbars, whiskerwidth=4) *
		mapping(:se_y, color=:reward_per_press) +
		visual(Scatter, markersize=10) *
		mapping(color=:reward_per_press)
	)
	fig = Figure(;size=(12, 6) .* 144 ./ 2.54)
	p = draw!(fig[1,1], p; axis=(;xlabel="Reward/press", ylabel="Press/sec", xticklabelrotation=pi/4))
	Label(fig[0,:], "Press Rates by Reward Rates Across Test Accuracy", tellwidth = false)
	# legend!(fig[1,2], p)
	
	# # Save
	# filepaths = joinpath("results/workshop/PIT", "PIT_press_by_rpp_acc.png")
	# save(filepaths, fig; px_per_unit = 4)

	# upload_to_osf(
	# 		filepaths,
	# 		proj,
	# 		osf_folder
	# 	)

	fig
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
		@group_by(prolific_pid, session, half, valence)
		@summarize(press_per_sec = mean(press_per_sec))
		@ungroup
		@pivot_wider(names_from = valence, values_from = press_per_sec)
		@mutate(diff = pos - neg)
		unstack([:prolific_pid, :session], :half, :diff)
	end

	figs = []
	for s in unique(PIT_data.session)
		fig=Figure(;size=(8, 6) .* 144 ./ 2.54)
		workshop_reliability_scatter!(
			fig[1,1];
			df=@filter(splithalf_df, session == !!s),
			xlabel="ΔValence (Trial 1-18)",
			ylabel="ΔValence (Trial 19-36)",
			xcol=:x,
			ycol=:y,
			subtitle="Session $(s) Valence Difference"
		)
		
		# Save
		filepaths = joinpath("results/workshop/PIT", "PIT_sess$(s)_valdiff_splithalf_firstsecond.png")
		save(filepaths, fig; px_per_unit = 4)
		
		upload_to_osf(
				filepaths,
				proj,
				osf_folder
			)
		
		push!(figs, fig)
	end
	figs
end

# ╔═╡ 8e36025a-fe92-4d1a-a7fa-c37918a16dbc
let
	splithalf_df = @chain PIT_data begin
		@mutate(half = ifelse(trial_number % 2 === 0, "x", "y"))
		@filter(coin != 0)
		@mutate(valence = ifelse(coin > 0, "pos", "neg"))
		@group_by(prolific_pid, session, half, valence)
		@summarize(press_per_sec = mean(press_per_sec))
		@ungroup
		@pivot_wider(names_from = valence, values_from = press_per_sec)
		@mutate(diff = pos - neg)
		unstack([:prolific_pid, :session], :half, :diff)
	end

	figs = []
	for s in unique(PIT_data.session)
		fig=Figure(;size=(8, 6) .* 144 ./ 2.54)
		workshop_reliability_scatter!(
			fig[1,1];
			df=@filter(splithalf_df, session == !!s),
			xlabel="ΔValence (Trial 2,4,6...)",
			ylabel="ΔValence (Trial 1,3,5...)",
			xcol=:x,
			ycol=:y,
			subtitle="Session $(s) Valence Difference"
		)
		
		# Save
		filepaths = joinpath("results/workshop/PIT", "PIT_sess$(s)_valdiff_splithalf_evenodd.png")
		save(filepaths, fig; px_per_unit = 4)
		
		upload_to_osf(
				filepaths,
				proj,
				osf_folder
			)
		
		push!(figs, fig)
	end
	figs
end

# ╔═╡ 71ce1bf7-8e80-42ff-bee7-0da4ac56ad69
md"""
#### Test-retest
"""

# ╔═╡ 420471d0-5b81-41c5-b79e-a50b338c7d06
let
	retest_df = @chain PIT_data begin
		@filter(coin != 0)
		@mutate(valence = ifelse(coin > 0, "pos", "neg"))
		@group_by(prolific_pid, session, valence)
		@summarize(press_per_sec = mean(press_per_sec))
		@ungroup
		@pivot_wider(names_from = valence, values_from = press_per_sec)
		@mutate(diff = pos - neg)
		unstack([:prolific_pid], :session, :diff)
		dropmissing()
	end
	
	fig=Figure(;size=(8, 6) .* 144 ./ 2.54)
	workshop_reliability_scatter!(
		fig[1,1];
		df=retest_df,
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=Symbol(string(1)),
		ycol=Symbol(string(2)),
		subtitle="Test-retest Valence Difference",
		correct_r=false
	)
	
	# Save
	filepaths = joinpath("results/workshop/PIT", "PIT_retest_valdiff.png")
	save(filepaths, fig; px_per_unit = 4)
	
	upload_to_osf(
			filepaths,
			proj,
			osf_folder
		)

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
		groupby([:prolific_pid, :session, :half, :valence])
		combine(AsTable([:press_per_sec, :coin]) => glm_coef => :β)
		unstack([:prolific_pid, :session, :valence], :half, :β)
	end

	figs = []
	for s in unique(PIT_data.session)
		fig=Figure(;size=(12, 6) .* 144 ./ 2.54)
		workshop_reliability_scatter!(
			fig[1,1];
			df=@filter(splithalf_df, session == !!s, valence == "pos"),
			xlabel=L"Trial 1-18",
			ylabel=L"Trial 19-36",
			xcol=:x,
			ycol=:y,
			subtitle=L"$\beta_{Pos}$"
		)
		workshop_reliability_scatter!(
			fig[1,2];
			df=@filter(splithalf_df, session == !!s, valence == "neg"),
			xlabel=L"Trial 1-18",
			ylabel=L"Trial 19-36",
			xcol=:x,
			ycol=:y,
			subtitle=L"$\beta_{Neg}$"
		)
		Label(fig[0,:],"Session $(s) Pavlovian Sensitivity within Valence")
		# Save
		filepaths = joinpath("results/workshop/PIT", "PIT_sess$(s)_slope_splithalf_firstsecond.png")
		save(filepaths, fig; px_per_unit = 4)
		
		upload_to_osf(
				filepaths,
				proj,
				osf_folder
			)
		
		push!(figs, fig)
	end
	figs
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
		groupby([:prolific_pid, :session, :half, :valence])
		combine(AsTable([:press_per_sec, :coin]) => glm_coef => :β)
		unstack([:prolific_pid, :session, :valence], :half, :β)
	end

	figs = []
	for s in unique(PIT_data.session)
		fig=Figure(;size=(12, 6) .* 144 ./ 2.54)
		workshop_reliability_scatter!(
			fig[1,1];
			df=@filter(splithalf_df, session == !!s, valence == "pos"),
			xlabel="Trial 2,4,6...",
			ylabel="Trial 1,3,5...",
			xcol=:x,
			ycol=:y,
			subtitle=L"$\beta_{Pos}$"
		)
		workshop_reliability_scatter!(
			fig[1,2];
			df=@filter(splithalf_df, session == !!s, valence == "neg"),
			xlabel="Trial 2,4,6...",
			ylabel="Trial 1,3,5...",
			xcol=:x,
			ycol=:y,
			subtitle=L"$\beta_{Neg}$"
		)
		Label(fig[0,:],"Session $(s) Pavlovian Sensitivity within Valence")
		# Save
		filepaths = joinpath("results/workshop/PIT", "PIT_sess$(s)_slope_splithalf_evenodd.png")
		save(filepaths, fig; px_per_unit = 4)
		
		upload_to_osf(
				filepaths,
				proj,
				osf_folder
			)
		
		push!(figs, fig)
	end
	figs
end

# ╔═╡ bd25c85d-7633-4c7f-a08e-611292241af6
md"""
#### Test-retest
"""

# ╔═╡ 416ca0a8-ab33-42fb-96a8-9d28d31d7c1a
let
	glm_coef(dat) = coef(lm(@formula(press_per_sec ~ coin), dat))[2]
	retest_df = @chain PIT_data begin
		@filter(coin != 0)
		@mutate(valence = ifelse(coin > 0, "pos", "neg"))
		@group_by(valence)
		@mutate(coin = (coin - mean(coin))/std(coin))
		@ungroup
		groupby([:prolific_pid, :session, :valence])
		combine(AsTable([:press_per_sec, :coin]) => glm_coef => :β)
		unstack([:prolific_pid, :valence], :session, :β)
		dropmissing()
	end
	fig=Figure(;size=(12, 6) .* 144 ./ 2.54)
	workshop_reliability_scatter!(
		fig[1,1];
		df=@filter(retest_df, valence == "pos"),
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=Symbol(string(1)),
		ycol=Symbol(string(2)),
		subtitle=L"$\beta_{Pos}$",
		correct_r=false
	)
	workshop_reliability_scatter!(
		fig[1,2];
		df=@filter(retest_df, valence == "neg"),
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=Symbol(string(1)),
		ycol=Symbol(string(2)),
		subtitle=L"$\beta_{Neg}$",
		correct_r=false
	)
	Label(fig[0,:], "Test-retest Pavlovian Sensitivity within Valence")
	
	# Save
	filepaths = joinpath("results/workshop/PIT", "PIT_retest_slope.png")
	save(filepaths, fig; px_per_unit = 4)
	
	upload_to_osf(
			filepaths,
			proj,
			osf_folder
		)

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
		@group_by(prolific_pid, session, half, valence)
		@summarize(press_per_sec = mean(press_per_sec))
		@ungroup
		@pivot_wider(names_from = valence, values_from = press_per_sec)
		@mutate(asymm = pos + neg - 2 * zero)
		unstack([:prolific_pid, :session], :half, :asymm)
	end

	figs = []
	for s in unique(PIT_data.session)
		fig=Figure(;size=(8, 6) .* 144 ./ 2.54)
		workshop_reliability_scatter!(
			fig[1,1];
			df=@filter(splithalf_df, session == !!s),
			xlabel="Asymmetry (Trial 1-18)",
			ylabel="Asymmetry (Trial 19-36)",
			xcol=:x,
			ycol=:y,
			subtitle="Session $(s) Asymmetry"
		)
		
		# Save
		filepaths = joinpath("results/workshop/PIT", "PIT_sess$(s)_asymm_splithalf_firstsecond.png")
		save(filepaths, fig; px_per_unit = 4)
		
		upload_to_osf(
				filepaths,
				proj,
				osf_folder
			)
		
		push!(figs, fig)
	end
	figs
end

# ╔═╡ 53979178-5e41-4ef1-82d5-c10193b642ef
let
	splithalf_df = @chain PIT_data begin
		@mutate(half = ifelse(trial_number % 2 === 0, "x", "y"))
		@mutate(valence = ifelse(coin == 0, "zero", ifelse(coin > 0, "pos", "neg")))
		@group_by(prolific_pid, session, half, valence)
		@summarize(press_per_sec = mean(press_per_sec))
		@ungroup
		@pivot_wider(names_from = valence, values_from = press_per_sec)
		@mutate(asymm = pos + neg - 2 * zero)
		unstack([:prolific_pid, :session], :half, :asymm)
	end

	figs = []
	for s in unique(PIT_data.session)
		fig=Figure(;size=(8, 6) .* 144 ./ 2.54)
		workshop_reliability_scatter!(
			fig[1,1];
			df=@filter(splithalf_df, session == !!s),
			xlabel="Asymmetry (Trial 2,4,6...)",
			ylabel="Asymmetry (Trial 1,3,5...)",
			xcol=:x,
			ycol=:y,
			subtitle="Session $(s) Asymmetry"
		)
		
		# Save
		filepaths = joinpath("results/workshop/PIT", "PIT_sess$(s)_asymm_splithalf_evenodd.png")
		save(filepaths, fig; px_per_unit = 4)
		
		upload_to_osf(
				filepaths,
				proj,
				osf_folder
			)
		
		push!(figs, fig)
	end
	figs
end

# ╔═╡ c8256d50-0537-4424-8528-03a5e95f2a08
let
	retest_df = @chain PIT_data begin
		@mutate(valence = ifelse(coin == 0, "zero", ifelse(coin > 0, "pos", "neg")))
		@group_by(prolific_pid, session, valence)
		@summarize(press_per_sec = mean(press_per_sec))
		@ungroup
		@pivot_wider(names_from = valence, values_from = press_per_sec)
		@mutate(asymm = pos + neg - 2 * zero)
		unstack([:prolific_pid], :session, :asymm)
		dropmissing()
	end

	fig=Figure(;size=(8, 6) .* 144 ./ 2.54)
	workshop_reliability_scatter!(
		fig[1,1];
		df=retest_df,
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=Symbol(string(1)),
		ycol=Symbol(string(2)),
		subtitle="Test-retest Valence Asymmetry",
		correct_r=false
	)
	
	# Save
	filepaths = joinpath("results/workshop/PIT", "PIT_retest_asymm.png")
	save(filepaths, fig; px_per_unit = 4)
	
	upload_to_osf(
			filepaths,
			proj,
			osf_folder
		)
		
	fig
end

# ╔═╡ f658b1db-1fbd-4343-ad0c-b507b1c352b2
md"""
## Export PIT measures
"""

# ╔═╡ 10230022-aa20-497d-875c-b073a295e9ea
let
	pit_instrument_w0_pav_df = @chain PIT_data begin
		@filter(coin==0)
		@group_by(prolific_pid, session)
		@summarize(pit_pps = mean(press_per_sec))
		@ungroup
	end
	PIT_acc_df = @chain PIT_test_data begin
		@group_by(prolific_pid, session)
		@summarize(pit_acc = mean(skipmissing((magnitude_right > magnitude_left) == right_chosen)))
		@ungroup
	end
	pit_valence_diff_df = @chain PIT_data begin
		@filter(coin != 0)
		@mutate(valence = ifelse(coin > 0, "pos", "neg"))
		@group_by(prolific_pid, session, valence)
		@summarize(press_per_sec = mean(press_per_sec))
		@ungroup
		@pivot_wider(names_from = valence, values_from = press_per_sec)
		@mutate(diff = pos - neg)
		@select(prolific_pid, session, pit_valence_diff = diff)
	end
	pit_valence_slope_df = let
		glm_coef(dat) = coef(lm(@formula(press_per_sec ~ coin), dat))[2]
		@chain PIT_data begin
			@filter(coin != 0)
			@mutate(valence = ifelse(coin > 0, "pos", "neg"))
			@group_by(valence)
			@mutate(coin = (coin - mean(coin))/std(coin))
			@ungroup
			groupby([:prolific_pid, :session, :valence])
			combine(AsTable([:press_per_sec, :coin]) => glm_coef => :β)
			unstack([:prolific_pid, :session], :valence, :β)
			rename(:neg => :pit_neg_b, :pos => :pit_pos_b)
		end
	end
	pit_asymmetry_df = @chain PIT_data begin
		@mutate(valence = ifelse(coin == 0, "zero", ifelse(coin > 0, "pos", "neg")))
		@group_by(prolific_pid, session, valence)
		@summarize(press_per_sec = mean(press_per_sec))
		@ungroup
		@pivot_wider(names_from = valence, values_from = press_per_sec)
		@mutate(asymm = pos + neg - 2 * zero)
		@select(prolific_pid, session, pit_asymm = asymm)
	end

	all_pit_df = copy(pit_instrument_w0_pav_df)
     for df in [PIT_acc_df, pit_valence_diff_df, pit_valence_slope_df, pit_asymmetry_df]
          leftjoin!(all_pit_df, df, on=[:prolific_pid, :session])
     end
	CSV.write("results/workshop/pit_measures.csv", all_pit_df)
	all_pit_df
end

# ╔═╡ Cell order:
# ╠═ad7f05f1-0e20-4b2a-9dc2-63c5da38bead
# ╠═1afc3d98-3a97-466c-8079-0768e17af03b
# ╠═0a7b56a0-9fbf-4eed-b2e6-f8f2bb86dc38
# ╠═8a279439-49a3-4aec-adf9-b6b580f81199
# ╟─24a17406-762f-494e-a13a-ad7266d5f6d9
# ╠═4557a55b-30ca-4e2b-9822-27e1311d3767
# ╟─457a29ba-d33c-4389-a883-6c5c6ac61954
# ╟─b747d881-6515-49eb-8768-e1ed38104e36
# ╠═1aceb591-9ed1-4a9a-849f-dac14802e5c0
# ╠═cbde565b-7604-469b-b328-6c6bf84ceeeb
# ╟─96c684cf-bf21-468c-8480-98ffeb3cfbf8
# ╠═7ca13679-ab22-4e7e-9a9e-573eefea9771
# ╟─2c1c0339-a6c4-4e36-81bc-e672ab3b9ebf
# ╟─c3dc5eda-1421-4770-a1d3-f08b8c6c2655
# ╠═5c5f0ce8-c64b-413a-975d-22118f4e1852
# ╠═1beaa9b1-73e9-407b-bf5f-a4091f00a17d
# ╟─95fd5935-19de-4a66-881e-77fa276a70af
# ╟─7fa14255-9f46-466d-baa3-e6fc2eec3347
# ╠═58ed1255-6f89-4f8a-b6d6-a8dee840dea2
# ╠═b8df8891-ec81-424e-9e4f-3b1b9c152688
# ╠═07321c17-5493-4d1b-a918-2129cab2b0e1
# ╟─45ebcdfa-e2ef-4df7-a31d-9b5e06044e08
# ╟─8e046c5a-1454-4762-a93f-1555d7549931
# ╟─b6c4383c-98f8-47d1-91e1-369bd9f27aae
# ╠═8e103789-dc11-488f-8671-2222c0360fa3
# ╟─8e36025a-fe92-4d1a-a7fa-c37918a16dbc
# ╟─71ce1bf7-8e80-42ff-bee7-0da4ac56ad69
# ╠═420471d0-5b81-41c5-b79e-a50b338c7d06
# ╟─fbc4466c-1d20-4359-a893-72a255c903c6
# ╟─22915221-f312-4b6d-8ae6-b0818e06c1a3
# ╟─0691061e-d096-46e2-b035-a4ff0dda78a1
# ╟─bd25c85d-7633-4c7f-a08e-611292241af6
# ╠═416ca0a8-ab33-42fb-96a8-9d28d31d7c1a
# ╟─cbbff405-bcf0-4ea8-9683-81080f7b8a9e
# ╟─8812e1f4-b2b9-4cda-a33a-a12ee278c99b
# ╟─ec623a4a-c506-47b3-9c17-737fee355511
# ╟─53979178-5e41-4ef1-82d5-c10193b642ef
# ╠═c8256d50-0537-4424-8528-03a5e95f2a08
# ╟─f658b1db-1fbd-4343-ad0c-b507b1c352b2
# ╠═10230022-aa20-497d-875c-b073a295e9ea
