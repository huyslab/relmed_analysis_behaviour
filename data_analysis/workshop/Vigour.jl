### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ b41e7252-a075-11ef-039c-f532a7fb0a94
# ╠═╡ show_logs = false
begin
	cd("/home/jovyan/")
	import Pkg
	# activate the shared project environment
	Pkg.activate("relmed_environment")
	# instantiate, i.e. make sure that all packages are downloaded
	Pkg.instantiate()
	using Random, DataFrames, JSON, CSV, StatsBase, JLD2, HTTP, CairoMakie, Printf, Distributions, CategoricalArrays, AlgebraOfGraphics, Dates, Turing, SHA
	using Tidier, GLM, MixedModels, PlutoUI, LaTeXStrings, ColorSchemes
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

# ╔═╡ 28b7224d-afb4-4474-b346-7ee353b6d3d3
TableOfContents(title="📚 Table of Contents", indent=true, depth=4, aside=true)

# ╔═╡ 4f9cd22a-c03f-4a95-b631-5bace22aa426
# Set up saving to OSF
begin
	osf_folder = "/Workshop figures/Vigour/"
	proj = setup_osf("Task development")
end

# ╔═╡ 93bc3812-c620-4a8d-a312-de9fd0e55327
begin
	# Load data
	_, _, raw_vigour_data, raw_post_vigour_test_data, _, _,
		_, _ = load_pilot6_data()
	nothing
end

# ╔═╡ 1f8ca836-f2f7-4965-bf07-5656cf6c4ec6
md"""
Set theme globally
"""

# ╔═╡ 1c0196e9-de9a-4dfa-acb5-357c02821c5d
set_theme!(theme_minimal();
		font = "Helvetica",
		fontsize = 16);

# ╔═╡ 8255895b-a337-4c8a-a1a7-0983499f684e
md"""
## Press rate: Trial presses/sec
"""

# ╔═╡ de48ee97-d79a-46e4-85fb-08dd569bf7ef
begin
	vigour_unfinished = @chain raw_vigour_data begin
		@count(prolific_pid, exp_start_time)
		@mutate(most_common_n = ~mode(n))
		@filter(n < most_common_n)
	end
	vigour_data = @chain raw_vigour_data begin
		@filter(prolific_pid != "671139a20b977d78ec2ac1e0") # From sess1
		@filter(prolific_pid != "6721ec463c2f6789d5b777b5") # From sess2
		@mutate(
			press_per_sec = trial_presses / trial_duration * 1000,
			ratio = categorical(ratio; levels = [1, 8, 16], ordered=true),
			magnitude = categorical(magnitude; levels = [1, 2, 5], ordered=true)
		)
		@anti_join(vigour_unfinished)
	end
	nothing
end

# ╔═╡ 99e3d02a-39d2-4c90-97ce-983670c50c38
md"""
### Press rate distribution
"""

# ╔═╡ 4be713bc-4af3-4363-94f7-bc68c71609c2
let
	fig = @chain vigour_data begin
		# @filter(trial_presses > 0)
		@mutate(ratio = ~recode(ratio, 1=>"FR: 1",8=>"FR: 8",16=>"FR: 16"))
		data(_) * mapping(:press_per_sec; color=:magnitude=>"Magnitude", row=:ratio) * AlgebraOfGraphics.density()
		draw(_, scales(Color = (;palette=:Oranges_3)); figure=(;size=(8, 6) .* 144 ./ 2.54), axis=(;xlabel="Press/sec", ylabel="Density"))
	end

	# Save
	filepaths = joinpath("results/workshop/vigour", "Vigour_press_rate_dist.png")
	save(filepaths, fig; px_per_unit = 4)

	upload_to_osf(
			filepaths,
			proj,
			osf_folder
		)

	fig
end

# ╔═╡ bd55dd69-c927-45e2-98cf-04f0aa919853
md"""
### Press rate by reward rate
"""

# ╔═╡ 7e7959a7-d60c-4280-9ec9-269edfc3f2a4
let
	fig = @chain vigour_data begin
		@filter(trial_number != 0)
		@ungroup
		plot_presses_vs_var(_; x_var=:reward_per_press, y_var=:press_per_sec, xlab="Reward/press", ylab = "Press/sec", combine="average")
	end
	
	# Save
	filepaths = joinpath("results/workshop/vigour", "Vigour_press_by_reward_rate.png")
	save(filepaths, fig; px_per_unit = 4)

	upload_to_osf(
			filepaths,
			proj,
			osf_folder
		)

	fig
end

# ╔═╡ 75d4fc7b-63db-4160-9a56-1105244c24f1
md"""
### Press rate by conditions
"""

# ╔═╡ e3faa2fc-c085-4fc1-80ef-307904a38f33
let
	# Group and calculate mean presses for each participant
	grouped_data, avg_w_data = avg_presses_w_fn(@filter(vigour_data, trial_number != 0), [:magnitude, :ratio], :press_per_sec)
	
	# Create the plot for the average line
	average_plot = data(avg_w_data) * 
		mapping(
			:ratio,
			:avg_y,
			color = :magnitude => "Magnitude",
		) * (
			visual(ScatterLines, linewidth = 2) +
			visual(Errorbars, whiskerwidth = 6) *
			mapping(:se_y))
	
	# Set up the axis
	axis = (;
		xlabel = "Fix Ratio",
		ylabel = "Press/sec"
	)
	
	# Draw the plot

	fig = Figure(
		size = (8, 6) .* 144 ./ 2.54, # 72 points per inch, then cm
	)

	# Plot plot plot
	f = draw!(fig[1, 1], average_plot, scales(Color = (; palette = :Oranges_3)); axis)
	legend!(fig[1, 2], f)
	
	# Save
	filepaths = joinpath("results/workshop/vigour", "Vigour_press_by_conds.png")
	save(filepaths, fig; px_per_unit = 4)

	upload_to_osf(
			filepaths,
			proj,
			osf_folder
		)
	
	fig
end

# ╔═╡ 18f08be5-ffbe-455a-a870-57df5c007e01
md"""
## Reliability
"""

# ╔═╡ aa0f06fc-4668-499c-aa81-4069b90076aa
md"""
### Motor-related
"""

# ╔═╡ b9d78883-eb28-4984-af6b-afb76dd85349
let
	splithalf_df = @chain vigour_data begin
		@mutate(half = ifelse(trial_number <= maximum(trial_number)/2, "x", "y"))
		@group_by(prolific_pid, session, half)
		@summarize(n_presses = mean(press_per_sec))
		@ungroup
		@pivot_wider(names_from = half, values_from = n_presses)
	end
	
	figs = []
	for s in unique(vigour_data.session)
		fig=Figure(;size=(8, 6) .* 144 ./ 2.54)
		workshop_reliability_scatter!(
			fig[1,1];
			df=@filter(splithalf_df, session == !!s),
			xlabel="Press/sec (Trial 1-18)",
			ylabel="Press/sec (Trial 19-36)",
			xcol=:x,
			ycol=:y,
			subtitle="Session $(s) Press Rate"
		)
		
		# Save
		filepaths = joinpath("results/workshop/vigour", "Vigour_sess$(s)_motor_splithalf_firstsecond.png")
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

# ╔═╡ f188af11-d2a4-4e1c-9cc7-b63bc386ef57
let
	splithalf_df = @chain vigour_data begin
		@mutate(half = ifelse(trial_number % 2 === 0, "x", "y"))
		@group_by(prolific_pid, session, half)
		@summarize(n_presses = mean(press_per_sec))
		@ungroup
		@pivot_wider(names_from = half, values_from = n_presses)
	end

	figs = []
	for s in unique(vigour_data.session)
		fig=Figure(;size=(8, 6) .* 144 ./ 2.54)
		workshop_reliability_scatter!(
			fig[1,1];
			df=@filter(splithalf_df, session == !!s),
			xlabel="Press/sec (Trial 2,4,6...)",
			ylabel="Press/sec (Trial 1,3,5...)",
			xcol=:x,
			ycol=:y,
			subtitle="Session $(s) Press Rate"
		)
		
		# Save
		filepaths = joinpath("results/workshop/vigour", "Vigour_sess$(s)_motor_splithalf_evenodd.png")
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

# ╔═╡ 9b8909ca-2804-45a4-9085-e6ed5e1f1c49
md"""
#### Test-retest
"""

# ╔═╡ 33a27773-b242-49b3-9318-59c15e9602f9
let
	retest_df = @chain vigour_data begin
		@group_by(prolific_pid, session)
		@summarize(n_presses = mean(press_per_sec))
		@ungroup
		@pivot_wider(names_from = session, values_from = n_presses)
		@drop_missing
	end
	
	fig=Figure(;size=(8, 6) .* 144 ./ 2.54)
	workshop_reliability_scatter!(
		fig[1,1];
		df=retest_df,
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=Symbol(string(1)),
		ycol=Symbol(string(2)),
		subtitle="Test-retest Press Rate (Press/sec)",
		correct_r=false
	)
	
	# Save
	filepaths = joinpath("results/workshop/vigour", "Vigour_retest_motor.png")
	save(filepaths, fig; px_per_unit = 4)
	
	# upload_to_osf(
	# 		filepaths,
	# 		proj,
	# 		osf_folder
	# 	)

	fig
end

# ╔═╡ e6dfc8f4-b0e2-4fe5-9a2d-826e3f505c72
md"""
### Reward rate sensitivity
"""

# ╔═╡ 4fc4a680-0934-49de-a785-08cac3a8be3e
let
	rpp_dff_df = @chain vigour_data begin
		@filter(trial_number != 0)
		@mutate(block = (trial_number - 1) ÷ 9 + 1)
		@mutate(half = if_else(block % 2 === 0, "x", "y"))
		@arrange(prolific_pid, session, reward_per_press)
		@group_by(prolific_pid, session)
		@mutate(low_rpp = if_else(reward_per_press <= median(reward_per_press), "low_rpp", "high_rpp"))
		@ungroup
		@group_by(prolific_pid, session, half, low_rpp)
		@summarize(n_presses = mean(press_per_sec))
		@ungroup
		@pivot_wider(names_from = low_rpp, values_from = n_presses)
		@mutate(low_to_high_diff = low_rpp - high_rpp)
		@select(-ends_with("_rpp"))
		@pivot_wider(names_from = half, values_from = low_to_high_diff)
	end

	figs = []
	for s in unique(vigour_data.session)
		fig=Figure(;size=(8, 6) .* 144 ./ 2.54)
		workshop_reliability_scatter!(
			fig[1,1];
			df=@filter(rpp_dff_df, session == !!s),
			xlabel="ΔRPP in even blocks",
			ylabel="ΔRPP in odd blocks",
			xcol=:x,
			ycol=:y,
			subtitle="Session $(s) Press Rate difference"
		)
		
		# Save
		filepaths = joinpath("results/workshop/vigour", "Vigour_sess$(s)_rppdiff_splithalf_evenodd.png")
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

# ╔═╡ 7b096527-2420-4e0d-9d72-8289a42a78fe
let
	rpp_dff_df = @chain vigour_data begin
		@filter(trial_number != 0)
		@mutate(block = (trial_number - 1) ÷ 9 + 1)
		@mutate(half = if_else(block <= 2, "x", "y"))
		@arrange(prolific_pid, session, reward_per_press)
		@group_by(prolific_pid, session)
		@mutate(low_rpp = if_else(reward_per_press <= median(reward_per_press), "low_rpp", "high_rpp"))
		@ungroup
		@group_by(prolific_pid, session, half, low_rpp)
		@summarize(n_presses = mean(press_per_sec))
		@ungroup
		@pivot_wider(names_from = low_rpp, values_from = n_presses)
		@mutate(low_to_high_diff = low_rpp - high_rpp)
		@select(-ends_with("_rpp"))
		@pivot_wider(names_from = half, values_from = low_to_high_diff)
	end

	figs = []
	for s in unique(vigour_data.session)
		fig=Figure(;size=(8, 6) .* 144 ./ 2.54)
		workshop_reliability_scatter!(
			fig[1,1];
			df=@filter(rpp_dff_df, session == !!s),
			xlabel="ΔRPP in blocks 1-2",
			ylabel="ΔRPP in blocks 3-4",
			xcol=:x,
			ycol=:y,
			subtitle="Session $(s) Press Rate difference"
		)
		
		# Save
		filepaths = joinpath("results/workshop/vigour", "Vigour_sess$(s)_rppdiff_splithalf_firstsecond.png")
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

# ╔═╡ c02b47f4-3e96-4a09-a212-13671b8fad25
md"""
#### Test-retest
"""

# ╔═╡ 4d3da833-7333-442c-96ed-9e2fba0a4298
let
	retest_df = @chain vigour_data begin
		@filter(trial_number != 0)
		@arrange(prolific_pid, session, reward_per_press)
		@group_by(prolific_pid, session)
		@mutate(low_rpp = if_else(reward_per_press <= median(reward_per_press), "low_rpp", "high_rpp"))
		@ungroup
		@group_by(prolific_pid, session, low_rpp)
		@summarize(n_presses = mean(press_per_sec))
		@ungroup
		@pivot_wider(names_from = low_rpp, values_from = n_presses)
		@mutate(low_to_high_diff = low_rpp - high_rpp)
		@select(-ends_with("_rpp"))
		@pivot_wider(names_from = session, values_from = low_to_high_diff)
		@drop_missing
	end
	
	fig=Figure(;size=(8, 6) .* 144 ./ 2.54)
	workshop_reliability_scatter!(
		fig[1,1];
		df=retest_df,
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=Symbol(string(1)),
		ycol=Symbol(string(2)),
		subtitle="Test-retest Press Rate Difference",
		correct_r=false
	)
	
	# Save
	filepaths = joinpath("results/workshop/vigour", "Vigour_retest_rppdiff.png")
	save(filepaths, fig; px_per_unit = 4)
	
	upload_to_osf(
			filepaths,
			proj,
			osf_folder
		)

	fig
end

# ╔═╡ d6a73b37-2079-4ed1-ac49-e7c596fc0997
md"""
## Model of press rate
"""

# ╔═╡ 6f11e67c-84b2-457d-9727-825e0631860b
md"""
### Press rate as a function of (log) reward rate
"""

# ╔═╡ 68985a71-98c4-485b-a800-643aea8b8a5e
let
	glm_predict(data) = predict(glm(@formula(trial_presses ~ log(reward_per_press)), data, Poisson(), LogLink(); offset=log.(data.dur)), data; offset=log.(data.dur))
	
	p = @chain vigour_data begin
		DataFrames.transform(:trial_duration => (x -> x/1000) => :dur)
		groupby([:prolific_pid, :session])
		DataFrames.transform(AsTable([:trial_presses, :reward_per_press, :dur]) => glm_predict => :pred)
		@mutate(pred = pred/dur)
		groupby(:reward_per_press)
		combine([:press_per_sec, :pred] .=> mean; renamecols=false)
		@pivot_longer(-reward_per_press)
		@arrange(reward_per_press)
		data(_) * (mapping(:reward_per_press, :value, color=:variable=>"", linestyle=:variable=>"") * visual(Lines))
	end
	fig = Figure(;size=(8, 6) .* 144 ./ 2.54)
	p = draw!(fig[1,1], p, scales(
		Color=(;palette=reverse(ColorSchemes.Paired_10[1:2]),categories=["pred"=>"Model pred.", "press_per_sec"=>"Data"]),
		LineStyle=(;palette=[:solid,:dash],categories=["pred"=>"Model pred.", "press_per_sec"=>"Data"])
	);axis=(;xlabel="Reward/press", ylabel="Press/sec"))
	legend!(fig[1,2], p
		#; halign=0.95, valign=0.05, tellheight=false, tellwidth=false
	)
	fig
end

# ╔═╡ cea80eac-27cd-4757-ba4b-498f1add5c4f
let
	glm_coef(data) = coef(glm(@formula(trial_presses ~ log(reward_per_press)), data, Poisson(), LogLink(); offset=log.(data.dur)))

	rpp_beta_df = @chain vigour_data begin
		@mutate(block = (trial_number - 1) ÷ 9 + 1)
		@mutate(half = if_else(block % 2 === 0, "x", "y"))
		DataFrames.transform(:trial_duration => (x -> x/1000) => :dur)
		groupby([:prolific_pid, :session, :half])
		combine(AsTable([:trial_presses, :reward_per_press, :dur]) => (x -> [glm_coef(x)]) => [:β0, :β1])
	end

	figs = []
	for s in unique(vigour_data.session)
		fig=Figure(;size=(12, 6) .* 144 ./ 2.54)
		workshop_reliability_scatter!(
			fig[1,1];
			df=unstack(@filter(rpp_beta_df, session == !!s),
						[:prolific_pid, :session], :half, :β0),
			xlabel="Even blocks",
			ylabel="Odd blocks",
			xcol=:x,
			ycol=:y,
			subtitle="β0"
		)
		workshop_reliability_scatter!(
			fig[1,2];
			df=unstack(@filter(rpp_beta_df, session == !!s),
						[:prolific_pid, :session], :half, :β1),
			xlabel="Even blocks",
			ylabel="Odd blocks",
			xcol=:x,
			ycol=:y,
			subtitle="β1"
		)
		Label(fig[0,:], "Session $(s) Press rate ~ β0 + β1 * log(Reward rate)")
		
		# Save
		filepaths = joinpath("results/workshop/vigour", "Vigour_sess$(s)_rpp_beta_splithalf_evenodd.png")
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

# ╔═╡ 81ab693c-431d-4b73-a148-84846e448f4d
let
	glm_coef(data) = coef(glm(@formula(trial_presses ~ log(reward_per_press)), data, Poisson(), LogLink(); offset=log.(data.dur)))

	rpp_beta_df = @chain vigour_data begin
		@mutate(block = (trial_number - 1) ÷ 9 + 1)
		@mutate(half = if_else(block <= 2, "x", "y"))
		DataFrames.transform(:trial_duration => (x -> x/1000) => :dur)
		groupby([:prolific_pid, :session, :half])
		combine(AsTable([:trial_presses, :reward_per_press, :dur]) => (x -> [glm_coef(x)]) => [:β0, :β1])
	end

	figs = []
	for s in unique(vigour_data.session)
		fig=Figure(;size=(12, 6) .* 144 ./ 2.54)
		workshop_reliability_scatter!(
			fig[1,1];
			df=unstack(@filter(rpp_beta_df, session == !!s),
						[:prolific_pid, :session], :half, :β0),
			xlabel="Blocks 1-2",
			ylabel="Blocks 3-4",
			xcol=:x,
			ycol=:y,
			subtitle="β0"
		)
		workshop_reliability_scatter!(
			fig[1,2];
			df=unstack(@filter(rpp_beta_df, session == !!s),
						[:prolific_pid, :session], :half, :β1),
			xlabel="Blocks 1-2",
			ylabel="Blocks 3-4",
			xcol=:x,
			ycol=:y,
			subtitle="β1"
		)
		Label(fig[0,:], "Session $(s) Press rate ~ β0 + β1 * log(Reward rate)")
		
		# Save
		filepaths = joinpath("results/workshop/vigour", "Vigour_sess$(s)_rpp_beta_splithalf_firstsecond.png")
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

# ╔═╡ e8c04ee8-a851-4409-9919-ec5227f96689
md"""
#### Test-retest
"""

# ╔═╡ 198f34e5-c89b-4667-82e0-50164fed3491
let
	glm_coef(data) = coef(glm(@formula(trial_presses ~ log(reward_per_press)), data, Poisson(), LogLink(); offset=log.(data.dur)))

	retest_df = @chain vigour_data begin
		DataFrames.transform(:trial_duration => (x -> x/1000) => :dur)
		groupby([:prolific_pid, :session])
		combine(AsTable([:trial_presses, :reward_per_press, :dur]) => (x -> [glm_coef(x)]) => [:β0, :β1])
	end

	fig=Figure(;size=(12, 6) .* 144 ./ 2.54)
	workshop_reliability_scatter!(
		fig[1,1];
		df=dropmissing(unstack(retest_df, [:prolific_pid], :session, :β0)),
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=Symbol(string(1)),
		ycol=Symbol(string(2)),
		subtitle="β0",
		correct_r=false
	)
	workshop_reliability_scatter!(
		fig[1,2];
		df=dropmissing(unstack(retest_df, [:prolific_pid], :session, :β1)),
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=Symbol(string(1)),
		ycol=Symbol(string(2)),
		subtitle="β1",
		correct_r=false
	)
	Label(fig[0,:], "Press rate ~ β0 + β1 * log(Reward rate)")
	
	# Save
	filepaths = joinpath("results/workshop/vigour", "Vigour_retest_rpp_beta.png")
	save(filepaths, fig; px_per_unit = 4)
	
	upload_to_osf(
			filepaths,
			proj,
			osf_folder
		)
	
	fig
end

# ╔═╡ cc7e08b3-e245-4483-8cac-086a673a2861
md"""
### Press rate as a function of (log) fixed ratio and reward magnitude
"""

# ╔═╡ 42e2e827-253d-4881-bfc9-65d206e6201d
let
	glm_predict(data) = predict(glm(@formula(trial_presses ~ log(ratio) + log(magnitude)), data, Poisson(), LogLink(); offset=log.(data.dur)), data; offset=log.(data.dur))
	
	p = @chain vigour_data begin
		DataFrames.transform(:trial_duration => (x -> x/1000) => :dur)
		DataFrames.transform([:ratio, :magnitude] .=> ByRow(as_float), renamecols=false)
		groupby([:prolific_pid, :session])
		DataFrames.transform(AsTable([:trial_presses, :ratio, :magnitude, :dur]) => glm_predict => :pred)
		@mutate(pred = pred/dur)
		groupby(:reward_per_press)
		combine([:press_per_sec, :pred] .=> mean; renamecols=false)
		@pivot_longer(-reward_per_press)
		@arrange(reward_per_press)
		data(_) * (mapping(:reward_per_press, :value, color=:variable=>"", linestyle=:variable=>"") * visual(Lines))
	end
	fig = Figure(;size=(8, 6) .* 144 ./ 2.54)
	p = draw!(fig[1,1], p, scales(
		Color=(;palette=reverse(ColorSchemes.Paired_10[1:2]),categories=["pred"=>"Model pred.", "press_per_sec"=>"Data"]),
		LineStyle=(;palette=[:solid,:dash],categories=["pred"=>"Model pred.", "press_per_sec"=>"Data"])
	);axis=(;xlabel="Reward/press", ylabel="Press/sec"))
	legend!(fig[1,2], p
		#; halign=0.95, valign=0.05, tellheight=false, tellwidth=false
	)
	fig
end

# ╔═╡ 5d98b42c-2e9a-4111-b2de-5c14d28d4c96
let
	glm_coef(data) = coef(glm(@formula(trial_presses ~ log(ratio) + log(magnitude)), data, Poisson(), LogLink(); offset=log.(data.dur)))

	rpp_beta_df = @chain vigour_data begin
		@mutate(block = (trial_number - 1) ÷ 9 + 1)
		@mutate(half = if_else(block % 2 === 0, "x", "y"))
		DataFrames.transform(:trial_duration => (x -> x/1000) => :dur)
		DataFrames.transform([:ratio, :magnitude] .=> ByRow(as_float), renamecols=false)
		groupby([:prolific_pid, :session, :half])
		combine(AsTable([:trial_presses, :ratio, :magnitude, :dur]) => (x -> [glm_coef(x)]) => [:β0, :β1, :β2])
	end

	figs = []
	for s in unique(vigour_data.session)
		fig=Figure(;size=(15, 6) .* 144 ./ 2.54)
		workshop_reliability_scatter!(
			fig[1,1];
			df=unstack(@filter(rpp_beta_df, session == !!s),
						[:prolific_pid, :session], :half, :β0),
			xlabel="Even blocks",
			ylabel="Odd blocks",
			xcol=:x,
			ycol=:y,
			subtitle="β0"
		)
		workshop_reliability_scatter!(
			fig[1,2];
			df=unstack(@filter(rpp_beta_df, session == !!s),
						[:prolific_pid, :session], :half, :β1),
			xlabel="Even blocks",
			ylabel="Odd blocks",
			xcol=:x,
			ycol=:y,
			subtitle="β1"
		)
		workshop_reliability_scatter!(
			fig[1,3];
			df=unstack(@filter(rpp_beta_df, session == !!s),
						[:prolific_pid, :session], :half, :β2),
			xlabel="Even blocks",
			ylabel="Odd blocks",
			xcol=:x,
			ycol=:y,
			subtitle="β2"
		)
		Label(fig[0,:], "Session $(s) Press rate ~ β0 + β1 * log(Fixed ratio) + β2 * log(Magnitude)")
		
		# Save
		filepaths = joinpath("results/workshop/vigour", "Vigour_sess$(s)_fr_n_rm_beta_splithalf_evenodd.png")
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

# ╔═╡ cb9e5cb5-070c-427e-9895-2e27b0d3344e
let
	glm_coef(data) = coef(glm(@formula(trial_presses ~ log(ratio) + log(magnitude)), data, Poisson(), LogLink(); offset=log.(data.dur)))

	rpp_beta_df = @chain vigour_data begin
		@mutate(block = (trial_number - 1) ÷ 9 + 1)
		@mutate(half = if_else(block <= 2, "x", "y"))
		DataFrames.transform(:trial_duration => (x -> x/1000) => :dur)
		DataFrames.transform([:ratio, :magnitude] .=> ByRow(as_float), renamecols=false)
		groupby([:prolific_pid, :session, :half])
		combine(AsTable([:trial_presses, :ratio, :magnitude, :dur]) => (x -> [glm_coef(x)]) => [:β0, :β1, :β2])
	end

	figs = []
	for s in unique(vigour_data.session)
		fig=Figure(;size=(15, 6) .* 144 ./ 2.54)
		workshop_reliability_scatter!(
			fig[1,1];
			df=unstack(@filter(rpp_beta_df, session == !!s),
						[:prolific_pid, :session], :half, :β0),
			xlabel="Blocks 1-2",
			ylabel="Blocks 3-4",
			xcol=:x,
			ycol=:y,
			subtitle="β0"
		)
		workshop_reliability_scatter!(
			fig[1,2];
			df=unstack(@filter(rpp_beta_df, session == !!s),
						[:prolific_pid, :session], :half, :β1),
			xlabel="Blocks 1-2",
			ylabel="Blocks 3-4",
			xcol=:x,
			ycol=:y,
			subtitle="β1"
		)
		workshop_reliability_scatter!(
			fig[1,3];
			df=unstack(@filter(rpp_beta_df, session == !!s),
						[:prolific_pid, :session], :half, :β2),
			xlabel="Blocks 1-2",
			ylabel="Blocks 3-4",
			xcol=:x,
			ycol=:y,
			subtitle="β2"
		)
		Label(fig[0,:], "Session $(s) Press rate ~ β0 + β1 * log(Fixed ratio) + β2 * log(Magnitude)")
		
		# Save
		filepaths = joinpath("results/workshop/vigour", "Vigour_sess$(s)_fr_n_rm_beta_splithalf_evenodd.png")
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

# ╔═╡ 0e33dab6-6824-4883-8c47-5dd69aa288df
let
	glm_coef(data) = coef(glm(@formula(trial_presses ~ log(ratio) + log(magnitude)), data, Poisson(), LogLink(); offset=log.(data.dur)))

	retest_df = @chain vigour_data begin
		DataFrames.transform(:trial_duration => (x -> x/1000) => :dur)
		DataFrames.transform([:ratio, :magnitude] .=> ByRow(as_float), renamecols=false)
		groupby([:prolific_pid, :session])
		combine(AsTable([:trial_presses, :ratio, :magnitude, :dur]) => (x -> [glm_coef(x)]) => [:β0, :β1, :β2])
	end

	fig=Figure(;size=(15, 6) .* 144 ./ 2.54)
	workshop_reliability_scatter!(
		fig[1,1];
		df=dropmissing(unstack(retest_df, [:prolific_pid], :session, :β0)),
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=Symbol(string(1)),
		ycol=Symbol(string(2)),
		subtitle="β0",
		correct_r=false
	)
	workshop_reliability_scatter!(
		fig[1,2];
		df=dropmissing(unstack(retest_df, [:prolific_pid], :session, :β1)),
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=Symbol(string(1)),
		ycol=Symbol(string(2)),
		subtitle="β1",
		correct_r=false
	)
	workshop_reliability_scatter!(
		fig[1,3];
		df=dropmissing(unstack(retest_df, [:prolific_pid], :session, :β2)),
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=Symbol(string(1)),
		ycol=Symbol(string(2)),
		subtitle="β2",
		correct_r=false
	)
	Label(fig[0,:], "Press rate ~ β0 + β1 * log(Fixed ratio) + β2 * log(Magnitude)")
	
	# Save
	filepaths = joinpath("results/workshop/vigour", "Vigour_retest_fr_n_rm_beta.png")
	save(filepaths, fig; px_per_unit = 4)
	
	upload_to_osf(
			filepaths,
			proj,
			osf_folder
		)
	
	fig
end

# ╔═╡ 7ad7369a-f063-4270-8859-2e23d6c4ea94
md"""
## Post-vigour test
"""

# ╔═╡ c0ae5758-efef-42fa-9f46-1ec4e231c550
# ╠═╡ show_logs = false
let
    # 1. Identify and remove incomplete test data
    vigour_test_unfinished = @chain raw_post_vigour_test_data begin
        @count(prolific_pid, exp_start_time)
        @mutate(most_common_n = ~mode(n))
        @filter(n < most_common_n)
    end

    # 2. Clean and preprocess the data
    post_vigour_test_data = @chain raw_post_vigour_test_data begin
        @filter(prolific_pid != "671139a20b977d78ec2ac1e0")
        @anti_join(vigour_test_unfinished)
        @mutate(
            # Calculate difference in relative physical payment
            diff_rpp = (left_magnitude/left_ratio) - (right_magnitude/right_ratio),
            chose_left = Int(response === "ArrowLeft")
        )
    end

    # 3. Calculate accuracy metrics
    test_acc_df = @chain post_vigour_test_data begin
        @group_by(prolific_pid, session)
        @select(diff_rpp, chose_left)
        @mutate(
            chose_left = chose_left * 2 - 1,  # Convert to {-1, 1}
            truth = sign(diff_rpp)
        )
        @mutate(acc = chose_left == truth)
        @summarize(acc = mean(acc))
        @ungroup
    end

    # 4. Create accuracy summary for plotting
    avg_acc_df = @chain test_acc_df begin
        @summarize(acc = mean(acc))
        @mutate(
            x = maximum((!!post_vigour_test_data).diff_rpp) * 0.5,
            y = 0.5,
            acc_text = string("Accuracy: ", round(acc; digits = 2))
        )
    end

    # 5. Fit mixed-effects model
    test_mixed_rpp = fit(
        GeneralizedLinearMixedModel,
        @formula(chose_left ~ diff_rpp + (diff_rpp | prolific_pid)),
        post_vigour_test_data,
        Binomial()
    )

    # 6. Generate prediction data
    diff_rpp_range = LinRange(
        minimum(post_vigour_test_data.diff_rpp),
        maximum(post_vigour_test_data.diff_rpp),
        100
    )

	pred_effect_rpp = crossjoin(
        DataFrame.((
            pairs((;
                prolific_pid="New",
                chose_left=0.5,
                diff_rpp=diff_rpp_range
            ))...,
        ))...
    )

    pred_effect_rpp.pred = predict(
        test_mixed_rpp,
        pred_effect_rpp;
        new_re_levels=:population,
        type=:response
    )

	# 7. Create and display the plot
		fig=Figure(;size=(8, 6) .* 144 ./ 2.54)
	    rpp_pred_plot = 
	        data(@filter(post_vigour_test_data, chose_left == 0)) *
	            mapping(:chose_left, :diff_rpp) *
	            visual(RainClouds;
					markersize=0,
					color=RGBAf(7/255, 68/255, 31/255, 0.5),
					plot_boxplots = false,
					cloud_width=0.2,
					clouds=hist,
					orientation = :horizontal) +
			data(@filter(post_vigour_test_data, chose_left == 1)) *
	            mapping(:chose_left, :diff_rpp) *
	            visual(RainClouds;
					markersize=0,
					color=RGBAf(63/255, 2/255, 73/255, 0.5),
					plot_boxplots = false,
					side=:right,
					cloud_width=0.2,
					clouds=hist,
					orientation = :horizontal) +
	        data(pred_effect_rpp) *
	            mapping(:diff_rpp, :pred) *
	            visual(Lines)
	    
	    draw!(fig[1,1],
			rpp_pred_plot; 
			axis=(; xlabel="ΔRPP: Left option − Right option", ylabel="P(Choose left)"))
		Label(
			fig[1,1], 
			"Accuracy = " * string(round(avg_acc_df[!, :acc]...; digits=2)),
			fontsize = 16,
			font = :bold,
			halign = 0.975,
			valign = 0.025,
			tellheight = false,
			tellwidth = false
		);
	
		# Save
		filepaths = joinpath("results/workshop/vigour", "Vigour_test_acc_by_reward_diff.png")
		save(filepaths, fig; px_per_unit = 4)
	
		upload_to_osf(
				filepaths,
				proj,
				osf_folder
			)
		
		fig
end

# ╔═╡ Cell order:
# ╠═b41e7252-a075-11ef-039c-f532a7fb0a94
# ╠═28b7224d-afb4-4474-b346-7ee353b6d3d3
# ╠═4f9cd22a-c03f-4a95-b631-5bace22aa426
# ╠═93bc3812-c620-4a8d-a312-de9fd0e55327
# ╟─1f8ca836-f2f7-4965-bf07-5656cf6c4ec6
# ╠═1c0196e9-de9a-4dfa-acb5-357c02821c5d
# ╟─8255895b-a337-4c8a-a1a7-0983499f684e
# ╠═de48ee97-d79a-46e4-85fb-08dd569bf7ef
# ╟─99e3d02a-39d2-4c90-97ce-983670c50c38
# ╠═4be713bc-4af3-4363-94f7-bc68c71609c2
# ╟─bd55dd69-c927-45e2-98cf-04f0aa919853
# ╠═7e7959a7-d60c-4280-9ec9-269edfc3f2a4
# ╟─75d4fc7b-63db-4160-9a56-1105244c24f1
# ╟─e3faa2fc-c085-4fc1-80ef-307904a38f33
# ╟─18f08be5-ffbe-455a-a870-57df5c007e01
# ╟─aa0f06fc-4668-499c-aa81-4069b90076aa
# ╟─b9d78883-eb28-4984-af6b-afb76dd85349
# ╟─f188af11-d2a4-4e1c-9cc7-b63bc386ef57
# ╟─9b8909ca-2804-45a4-9085-e6ed5e1f1c49
# ╠═33a27773-b242-49b3-9318-59c15e9602f9
# ╟─e6dfc8f4-b0e2-4fe5-9a2d-826e3f505c72
# ╟─4fc4a680-0934-49de-a785-08cac3a8be3e
# ╟─7b096527-2420-4e0d-9d72-8289a42a78fe
# ╟─c02b47f4-3e96-4a09-a212-13671b8fad25
# ╠═4d3da833-7333-442c-96ed-9e2fba0a4298
# ╟─d6a73b37-2079-4ed1-ac49-e7c596fc0997
# ╟─6f11e67c-84b2-457d-9727-825e0631860b
# ╟─68985a71-98c4-485b-a800-643aea8b8a5e
# ╟─cea80eac-27cd-4757-ba4b-498f1add5c4f
# ╟─81ab693c-431d-4b73-a148-84846e448f4d
# ╟─e8c04ee8-a851-4409-9919-ec5227f96689
# ╠═198f34e5-c89b-4667-82e0-50164fed3491
# ╟─cc7e08b3-e245-4483-8cac-086a673a2861
# ╟─42e2e827-253d-4881-bfc9-65d206e6201d
# ╟─5d98b42c-2e9a-4111-b2de-5c14d28d4c96
# ╟─cb9e5cb5-070c-427e-9895-2e27b0d3344e
# ╠═0e33dab6-6824-4883-8c47-5dd69aa288df
# ╟─7ad7369a-f063-4270-8859-2e23d6c4ea94
# ╟─c0ae5758-efef-42fa-9f46-1ec4e231c550
