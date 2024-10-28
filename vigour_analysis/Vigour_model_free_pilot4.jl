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

# ╔═╡ 05c22183-d42d-4d27-8b21-33f6842e0291
begin
	cd("/home/jovyan/")
    import Pkg
    # activate the shared project environment
    Pkg.activate("relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	Pkg.add("MixedModels")
	Pkg.add("TidierData")
	using Random, DataFrames, JSON, CSV, StatsBase, JLD2, HTTP, Dates, AlgebraOfGraphics, CairoMakie, TidierData, MixedModels, Printf, CategoricalArrays
	include("fetch_preprocess_data.jl")
	include("vigour_analysis/Vigour_utlis_fn.jl")
	set_theme!(theme_minimal())
end

# ╔═╡ c5be4a4b-d800-4ba4-abde-c9607cefe261
begin
	Pkg.add("RCall")
	using RCall
end

# ╔═╡ 33e63f0d-8b59-46a5-af57-b45eb4b8b531
md"""
## Load data from REDCap or from local files
"""

# ╔═╡ be928c7e-363b-45cd-971b-4fad90335b06
md"""
### From REDCap
"""

# ╔═╡ 02713d34-3f91-4a60-b499-93975cede337
md"""
Testing removed in `load_pilot4_data()`
"""

# ╔═╡ 485584ee-2efc-4ed1-b58d-2a73ec9c8ae1
begin
	_, _, raw_vigour_data, raw_vigour_test_data, raw_pit_data, _, jspsych_data = load_pilot4_data();
	transform!(raw_vigour_data, [:trial_presses, :trial_duration] => ((x, y) -> x .* 1000 ./ y) => :press_per_sec)
	transform!(raw_pit_data, [:trial_presses, :trial_duration] => ((x, y) -> x .* 1000 ./ y) => :press_per_sec)
end

# ╔═╡ e8568096-8344-43cf-94fa-22219b3a7c7b
begin
	n_miss_df = combine(groupby(raw_vigour_data, [:prolific_id, :version]), :trial_presses => (x -> sum(x .== 0)) => :n_miss)
	many_miss = filter(x -> x.n_miss > 9, n_miss_df)
end

# ╔═╡ cd445ddb-a22a-415c-8aae-6fc1eb84585b
md"""
### Read demographic data
"""

# ╔═╡ 51515262-32d8-4c9e-8a9c-9b4fd4eb4f09
begin
	info_files = filter(x -> endswith(x, ".csv"), readdir(joinpath("data", "prolific_participant_info"); join = true))
	info_data = @chain info_files begin
		CSV.read(DataFrame, select = ["Participant id", "Age"])
		@rename(prolific_id = var"Participant id", age = var"Age")
		@filter(age != "CONSENT_REVOKED")
		@mutate(age = as_float(age))
	end
end

# ╔═╡ 45d95aea-7751-4f37-9558-de4c55a0881e
begin
	vigour_data = antijoin(raw_vigour_data, many_miss; on = :prolific_id)
	leftjoin!(vigour_data, info_data; on = :prolific_id)
	# filter!(x -> !ismissing(x.age), vigour_data)
    vigour_test_data = antijoin(raw_vigour_test_data, many_miss; on = :prolific_id)
end

# ╔═╡ e80172e8-0055-4179-b1a9-ce38da6b3fc4
combine(groupby(vigour_data, [:version]), :age => (x -> DataFrame([summarystats(x)])) => AsTable)

# ╔═╡ 7fadad91-a4b1-4a44-9491-2d389303d405
describe(vigour_data)

# ╔═╡ 5ae2870a-408f-4aac-8f12-5af081a250d8
@count(vigour_data, prolific_id, exp_start_time, version) |>
	x -> @count(x, version)

# ╔═╡ bda488ea-faa3-4308-ac89-40b8481363bf
filter(x -> x.trial_presses .== 0, vigour_data)

# ╔═╡ ac1399c4-6a2f-446f-8687-059fa5da9473
md"""
## Vigour trial analyses
"""

# ╔═╡ 9cc0560b-5e3b-45bc-9ba8-605f8111840b
md"""
### Response time
"""

# ╔═╡ 2391eb75-db59-4156-99ae-abb8f1508037
	let
		# Group and calculate mean presses for each participant
		grouped_data = vigour_data |>
			x -> transform(x, :response_times => ByRow(safe_mean) => :avg_rt) |>
			x -> groupby(x, [:prolific_id, :reward_per_press]) |>
			x -> combine(x, :avg_rt => mean => :avg_rt)
		sort!(grouped_data, [:prolific_id, :reward_per_press])
		
		# Calculate the average across all participants
		avg_data = combine(groupby(grouped_data, :reward_per_press), :avg_rt => (x -> median(skipmissing(x))) => :avg_rt)
		
		# Create the plot for individual participants
		individual_plot = data(grouped_data) * 
		    mapping(
		        :reward_per_press => "Reward/press",
		        :avg_rt => "Response times (ms)",
		        group = :prolific_id
		    ) * 
		    visual(Lines, color = (:gray, 0.3), linewidth = 1)
		
		# Create the plot for the average line
		average_plot = data(avg_data) * 
		    mapping(
		        :reward_per_press => "Reward/press",
		        :avg_rt => "Response times (ms)"
		    ) * 
		    visual(ScatterLines, color = :dodgerblue, linewidth = 2)
		
		# Combine the plots
		final_plot = individual_plot + average_plot
		fig = Figure()

		# Draw the individual lines plot
		draw!(fig[1, 1], individual_plot)
		
		# Draw the average line plot
		draw!(fig[1, 2], average_plot)
		
		# Add a overall title
		Label(fig[0, :], "Response times vs Reward/press")

		fig
	end

# ╔═╡ b9f4babc-8da0-40f2-b0d2-95f915989faf
md"""
### Number of key presses
"""

# ╔═╡ fd68ab0d-ca1b-437b-8c55-24f900a0628d
begin
	groupby(vigour_data, Cols(:prolific_id, :version)) |>
			x -> combine(x, [:trial_presses, :trial_duration] => ((x, y) -> mean(x .* 1000 ./ y)) => :n_presses) |>
			x -> sort(x, Cols(:prolific_id, :version)) |>
			x -> groupby(x, :version) |>
			x -> combine(x, :n_presses => mean => :n_presses)
end

# ╔═╡ 9180c237-a966-4d0c-b92c-7c70eedabc1a
let
	df = @chain vigour_data begin
	@filter(trial_number != 0)
	@group_by(prolific_id, reward_per_press)
	@summarize(press_per_sec = mean(press_per_sec))
	@ungroup
	@arrange(prolific_id, reward_per_press)
	@mutate(short_id = last.(prolific_id, 5))
	end
	fig = Figure(size = (1200, 1200))
	axis = (xlabel = "Fixed ratio", ylabel = "Press/sec")
	p = data(df) * mapping(:reward_per_press, :press_per_sec, layout = :short_id) * visual(ScatterLines; markersize = 5)
	draw!(fig, p; axis=axis)
	fig
end

# ╔═╡ 342feddc-494a-4807-b4d5-4425d7ca3657
md"""
#### First four trials
"""

# ╔═╡ 2576dc02-f6c6-4c05-8d36-f20e341825c7
let
	first_four_df = @chain vigour_data begin
		@filter(trial_number <= 4)
	end
	fig = Figure()
	axis = (;
		xlabel="Trial number",
		ylabel="Press/sec",
		ygridvisible=true,
		yminorgridvisible=true,
		width=600,
		height=450
	)
	p = data(first_four_df) *
		mapping(:trial_number, :press_per_sec, color = :reward_per_press, col = :version) *
		visual(RainClouds)
	fig = draw(p, scales(Color = (; colormap = :blues)); axis=axis, colorbar=(;label = "Reawrd/press", height = 200))
	fig
end

# ╔═╡ 366287ba-7c26-497c-aada-ec8a131ad22e
sort(unique(vigour_data.reward_per_press))

# ╔═╡ 4401e0dc-fcc2-4561-ae82-afb890b9d22b
[1/16, 1/8, 2/16, 2/8, 5/16, 5/8, 1/1, 2/1, 5/1]

# ╔═╡ e39b43eb-7695-4d64-8b6b-cd5afb0616f1
unique(vigour_data.trial_number[vigour_data.reward_per_press .== 1])

# ╔═╡ 00f78d00-ca54-408a-a4a6-ad755566052a
plot_presses_vs_var(@filter(vigour_data, trial_number != 1); x_var=:reward_per_press, y_var=:press_per_sec, grp_var=:version, xlab="Reward/press", ylab = "Press/sec", combine=false)

# ╔═╡ 89d68509-bcf0-44aa-9ddc-a685131ce146
plot_presses_vs_var(vigour_data; x_var=:reward_per_press, y_var=:press_per_sec, grp_var=:version, ylab="Press/sec", xlab = "Reward/press", combine=true)

# ╔═╡ ef4c0f64-2f16-4700-8fc4-703a1b858c37
plot_presses_vs_var(vigour_data; x_var=:ratio, y_var=:press_per_sec, grp_var=:version, ylab="Press/sec", xlab = "Fixed ratio", combine=false)

# ╔═╡ eab61744-af48-4789-ba2f-a92d73527962
plot_presses_vs_var(vigour_data; x_var=:ratio, y_var=:press_per_sec, grp_var=:version, ylab="Press/sec", xlab = "Fixed ratio", combine=true)

# ╔═╡ c4e7d7ec-8da8-4a1f-8da3-0f2d29be6db7
plot_presses_vs_var(vigour_data; x_var=:magnitude, y_var=:press_per_sec, grp_var=:version, ylab="Press/sec", xlab = "Reward magnitude", combine=false)

# ╔═╡ 09a6f213-f70a-42c2-8eb7-8689d40d140b
plot_presses_vs_var(vigour_data; x_var=:magnitude, y_var=:press_per_sec, grp_var=:version, ylab="Press/sec", xlab = "Reward magnitude", combine=true)

# ╔═╡ 14d72296-7392-4375-9748-a8ee84d34701
CSV.write("data/pilot4_vigour_data.csv", vigour_data)

# ╔═╡ 1066a576-c7bf-42cc-8097-607c663dcdac
let
	# Group and calculate mean presses for each participant
	grouped_data, avg_w_data = avg_presses_w_fn(vigour_data, [:magnitude, :ratio], :press_per_sec, :version)
	
	# Create the plot for the average line
	average_plot = data(avg_w_data) * 
		mapping(
			:ratio,
			:avg_y,
			col = :version,
			color = :magnitude => nonnumeric => "Reward\nMagnitude",
		) * (
			visual(ScatterLines, linewidth = 2) +
			visual(Errorbars, whiskerwidth = 6) *
			mapping(:se_y))
	
	# Set up the axis
	axis = (
		xlabel = "Fix Ratio",
		ylabel = "Number of Presses",
		xticks = [1, 4, 8, 12, 16]
	)
	
	# Draw the plot

	fig = Figure(
		size = (12.2, 7.6) .* 144 ./ 2.54, # 72 points per inch, then cm
	)

	# Plot plot plot
	f = draw!(fig[1, 1], average_plot, scales(Color = (; palette = from_continuous(:viridis))); axis)
	legend!(fig[1, 2], f)
	fig
end

# ╔═╡ 128e1be1-191d-4095-bfa9-cd538b913181
md"""
### Response-missing trials
"""

# ╔═╡ 302e4f7a-fd09-48c3-89ac-a80a63841ee7
begin
	let
		# Create the plot
		plot = data(n_miss_df) * 
		       mapping(:n_miss, ) * 
		       histogram(bins=20)
		
		# Set up the axis
		axis = (
		    title = "Some participants didn't respond in many trials",
		    xlabel = "# No-response trials",
		    ylabel = "Count (# participants)",
		    yticks = 0:10:50
		)
		
		# Draw the plot
		draw(plot; axis)
	end
end

# ╔═╡ b0c1d8fb-8f95-4a40-b760-6584022867ce
begin
	missed_trial_df = @chain raw_vigour_data begin
		@filter(trial_presses == 0)
		@group_by(prolific_id, reward_per_press)
		@summarize(n_miss = n())
		@ungroup()
		@mutate(short_id = last.(prolific_id, 5))
		@arrange(prolific_id, reward_per_press)
	end
	data(missed_trial_df) *
		mapping(:reward_per_press, :n_miss, layout = :short_id) *
		visual(ScatterLines) |>
	x -> draw(x)
end

# ╔═╡ 9a8d96d7-13bd-4f70-9344-6569dfa39ab6
md"""
## Reliability analysis
"""

# ╔═╡ 16154073-51c3-4a37-b40f-8c4f6044702c
md"""
#### Split-half reliability: First-second
"""

# ╔═╡ 08fe8e09-51e5-4d54-a0c2-7c6ac8269a00
@count(vigour_data, prolific_id, version)

# ╔═╡ b5c429ee-9f31-4815-98b3-2536952b6881
begin
	transform!(vigour_data, :trial_number => (x -> ifelse.(x .> maximum(x)/2, "second", "first")) => :half)
	first_second_df = @chain vigour_data begin
		@mutate(press_per_sec = trial_presses .* 1000 ./ trial_duration)
		@group_by(prolific_id, version, half)
		@summarize(n_presses = mean(press_per_sec))
		@ungroup
		@pivot_wider(names_from = half, values_from = n_presses)
	end
	ρ_12_df = @chain first_second_df begin
		@group_by(version)
		@summarize(ρ_12 = ~ cor(first, second))
		@mutate(ρ_12 = (2 * ρ_12) / (1 + ρ_12))
		@mutate(ρ_text = string("ρ: ", round(ρ_12; digits = 2)))
		@mutate(x = 6, y = [1.5])
		@ungroup
	end
	first_second_plot = 
		data(first_second_df) *
		mapping(:first, :second, color = :version) *
		visual(Scatter) +
		data(ρ_12_df) *
		mapping(:x, :y, color = :version; text = :ρ_text => verbatim) *
		visual(Makie.Text, fontsize = 16, font = :bold) +
		data(DataFrame(intercept = 0, slope = 1)) *
		mapping(:intercept, :slope) *
		visual(ABLines, linestyle = :dash, color = :gray70)
	draw(
		first_second_plot;
		axis=(;xlabel="First half trials",ylabel="Second half trials")
	)
end

# ╔═╡ 00d6678d-4e26-4fe6-9937-b1055f78cad5
let
	df = @chain vigour_data begin
		@mutate(press_per_sec = trial_presses .* 1000 ./ trial_duration)
		@group_by(prolific_id, version, half, magnitude, ratio)
		@summarize(n_presses = mean(press_per_sec))
		@ungroup
		@pivot_wider(names_from = half, values_from = n_presses)
		@mutate(magnitude = nonnumeric(magnitude), 
		ratio = nonnumeric(ratio))
	end
	version_df = @chain df begin
		@group_by(magnitude, ratio, version)
		@summarize(ρ_12 = ~ cor(first, second))
		@mutate(ρ_12 = (2 * ρ_12) / (1 + ρ_12))
		@mutate(ρ_text = string("ρ: ", round(ρ_12; digits = 2)))
		@mutate(x = 0, y = [6.5])
		@ungroup
	end
	p = data(df) * 
			mapping(:first, :second, col = :magnitude, row = :ratio, color = :version) *
			visual(Scatter, alpha = 0.3) +
		data(version_df) *
			mapping(:x, :y, color = :version; text = :ρ_text => verbatim, col = :magnitude, row = :ratio) *
			visual(Makie.Text, fontsize = 12) +
		data(DataFrame(intercept = 0, slope = 1)) *
			mapping(:intercept, :slope) *
			visual(ABLines, linestyle = :dash, color = :gray70)
	draw(p;
		axis=(;xlabel="First half trials",ylabel="Second half trials"))
end

# ╔═╡ 50e8263e-a593-47d3-abc4-aceeeb68ba58
ρ_by_conds = @chain vigour_data begin
	@group_by(prolific_id, half, magnitude, ratio, version)
	@summarize(n_presses = mean(skipmissing(trial_presses)))
	@ungroup
	@pivot_wider(names_from = half, values_from = n_presses)
	@group_by(magnitude, ratio, version)
	@summarize(ρ = ~ cor(first, second))
	@mutate(ρ = 2 * ρ / (1 + ρ))
	@ungroup
end

# ╔═╡ a2adae6f-d651-4f0c-89f3-38cf357c943e
md"""
### Split-half reliability: Even-odd
"""

# ╔═╡ 1adc75da-c6c4-4b11-81ff-227851a18076
begin
	even_odd_df = @chain vigour_data begin
		@mutate(press_per_sec = trial_presses .* 1000 ./ trial_duration)
		@mutate(even = ifelse(trial_number % 2 === 0, "even", "odd"))
		@group_by(prolific_id, even, version)
		@summarize(n_presses = mean(press_per_sec))
		@ungroup
		@pivot_wider(names_from = even, values_from = n_presses)
	end
	ρ_01_df = @chain even_odd_df begin
		@group_by(version)
		@summarize(ρ_01 = ~ cor(odd, even))
		@mutate(ρ_01 = (2 * ρ_01) / (1 + ρ_01))
		@mutate(ρ_text = string("ρ: ", round(ρ_01; digits = 2)))
		@mutate(x = 5, y = [1.5])
		@ungroup
	end
	even_odd_plot = 
		data(even_odd_df) *
			mapping(:even, :odd, color = :version) *
			visual(Scatter) +
		data(ρ_01_df) *
			mapping(:x, :y, color = :version; text = :ρ_text => verbatim) *
			visual(Makie.Text, fontsize = 16, font = :bold) +
		data(DataFrame(intercept = 0, slope = 1)) *
			mapping(:intercept, :slope) *
			visual(ABLines, linestyle = :dash, color = :gray70)
	draw(
		even_odd_plot;
		axis=(;xlabel="Even trials",ylabel="Odd trials")
	)	
end

# ╔═╡ 387e33da-31fd-4214-b819-bb4f2e43ab23
md"""
### Reliability of reward/press difference
"""

# ╔═╡ 26282185-43f7-44a1-936d-b4598be46cb4
md"""
Since the task by nature has four blocks, we could also have first-second and even-odd split-half by blocks.

- Even-odd by blocks
"""

# ╔═╡ 3028ec29-09b5-4754-a318-0b9aaf44e76f
begin
	let
	rpp_dff_df = @chain vigour_data begin
		@mutate(press_per_sec = trial_presses .* 1000 ./ trial_duration)
		@mutate(block = (trial_number - 1) ÷ 9 + 1)
		@mutate(even = if_else(block % 2 === 0, "even", "odd"),
				first = if_else(block <= 2, "first", "second"))
		@arrange(prolific_id, version, reward_per_press)
		@group_by(prolific_id, version)
		@filter(reward_per_press != median(reward_per_press))
		@mutate(low_rpp = if_else(reward_per_press < median(reward_per_press), "low_rpp", "high_rpp"))
		@ungroup
		@group_by(prolific_id, version, even, low_rpp)
		@summarize(n_presses = mean(press_per_sec))
		@ungroup
		@pivot_wider(names_from = low_rpp, values_from = n_presses)
		@mutate(low_to_high_diff = low_rpp - high_rpp)
		@select(-ends_with("_rpp"))
		@pivot_wider(names_from = even, values_from = low_to_high_diff)
	end
	ρ_rpp_diff_df = @chain rpp_dff_df begin
		@group_by(version)
		@summarize(ρ = ~ cor(odd, even))
		@mutate(ρ = (2 * ρ) / (1 + ρ))
		@mutate(ρ_text = string("ρ: ", round(ρ; digits = 2)))
		@mutate(x = -0.5, y = [-4])
		@ungroup
	end
	rpp_diff_plot = 
		data(rpp_dff_df) *
			mapping(:even, :odd, color = :version) *
			visual(Scatter) +
		data(ρ_rpp_diff_df) *
			mapping(:x, :y, color = :version; text = :ρ_text => verbatim) *
			visual(Makie.Text, fontsize = 16, font = :bold) +
		data(DataFrame(intercept = 0, slope = 1)) *
			mapping(:intercept, :slope) *
			visual(ABLines, linestyle = :dash, color = :gray70)
	draw(rpp_diff_plot; axis=(;xlabel="ΔRPP in even blocks", ylabel="ΔRPP in odd blocks"))
	end
end

# ╔═╡ e8719da6-89e0-4e25-acfc-2161642d6395
md"""
- First-second by blocks
"""

# ╔═╡ f07d048f-121b-4675-b9d0-edc0553b05fa
begin
	let
	rpp_dff_df = @chain vigour_data begin
		@mutate(press_per_sec = trial_presses .* 1000 ./ trial_duration)
		@mutate(block = (trial_number - 1) ÷ 9 + 1)
		@mutate(even = if_else(block % 2 === 0, "even", "odd"),
				first = if_else(block <= 2, "first", "second"))
		@arrange(prolific_id, version, reward_per_press)
		@group_by(prolific_id, version)
		@filter(reward_per_press != median(reward_per_press))
		@mutate(low_rpp = if_else(reward_per_press < median(reward_per_press), "low_rpp", "high_rpp"))
		@ungroup
		@group_by(prolific_id, version, first, low_rpp)
		@summarize(n_presses = mean(press_per_sec))
		@ungroup
		@pivot_wider(names_from = low_rpp, values_from = n_presses)
		@mutate(low_to_high_diff = low_rpp - high_rpp)
		@select(-ends_with("_rpp"))
		@pivot_wider(names_from = first, values_from = low_to_high_diff)
	end
	ρ_rpp_diff_df = @chain rpp_dff_df begin
		@group_by(version)
		@summarize(ρ = ~ cor(first, second))
		@mutate(ρ = (2 * ρ) / (1 + ρ))
		@mutate(ρ_text = string("ρ: ", round(ρ; digits = 2)))
		@mutate(x = -0.5, y = [-4])
		@ungroup
	end
	rpp_diff_plot = 
		data(rpp_dff_df) *
			mapping(:first, :second, color = :version) *
			visual(Scatter) +
		data(ρ_rpp_diff_df) *
			mapping(:x, :y, color = :version; text = :ρ_text => verbatim) *
			visual(Makie.Text, fontsize = 16, font = :bold) +
		data(DataFrame(intercept = 0, slope = 1)) *
			mapping(:intercept, :slope) *
			visual(ABLines, linestyle = :dash, color = :gray70)
	draw(rpp_diff_plot; axis=(;xlabel="ΔRPP in first two blocks", ylabel="ΔRPP in second two blocks"))
	end
end

# ╔═╡ 9a6438cc-d758-42a6-8c0b-36cb7dddacf9
md"""
## Post-vigour test
"""

# ╔═╡ 096ffb0d-cddb-445a-8887-d27dcd16ea47
begin
	diff_vigour_test_data = @chain vigour_test_data begin
		@mutate(
			diff_mag = left_magnitude - right_magnitude,
			diff_fr_rel = log2(left_ratio/right_ratio),
			diff_fr_abs = left_ratio - right_ratio,
			diff_rpp = (left_magnitude/left_ratio) - (right_magnitude/right_ratio),
			chose_left = Int(response === "ArrowLeft")
		)
		@mutate(across(starts_with("diff_"), zscore))
	end
end

# ╔═╡ 223d7ed5-8621-4f6a-905e-cfbd27849686
CSV.write("data/pilot4_vigour_test_data.csv", diff_vigour_test_data)

# ╔═╡ b3f76fad-a1c6-4fc0-afbf-7ce9736986e8
md"""
### Separate models for each effect

Since only RPP was balanced, perhaps it would be best to focus on and only on its effect.

#### RPP
"""

# ╔═╡ 7f3c680a-970b-483d-b3df-ed22b6dd8d81
# ╠═╡ show_logs = false
begin
	test_mixed_rpp = fit(GeneralizedLinearMixedModel,
		@formula(chose_left ~ diff_rpp +
		(diff_rpp | prolific_id)),
		diff_vigour_test_data, Binomial())
end

# ╔═╡ 341083df-98ee-4758-85a6-8a070b662b86
diff_vigour_test_data

# ╔═╡ 2f8f197a-77c1-48ac-b18a-d2e3a9006bfb
begin
	test_acc_df = @chain diff_vigour_test_data begin
		@group_by(prolific_id, version)
		@select(diff_rpp, chose_left)
		@mutate(chose_left = chose_left * 2 - 1, truth = sign(diff_rpp))
		@mutate(acc = chose_left == truth)
		@summarize(acc = mean(acc))
		@ungroup
	end
	@printf "%.2f" mean(test_acc_df.acc)
	data(test_acc_df) *
	mapping(:acc) *
	visual(Hist) |>
	draw()
end

# ╔═╡ 96bcba8a-e8a1-49c4-b60e-65aa322ca321
 @chain diff_vigour_test_data begin
	@mutate(chose_left = chose_left * 2 - 1, truth = sign(diff_rpp))
	@mutate(correct = chose_left == truth)
	@mutate(easy = (left_ratio <= right_ratio && left_magnitude >= right_magnitude) || (left_ratio >= right_ratio && left_magnitude <= right_magnitude))
	@group_by(prolific_id, easy)
	@summarize(acc = mean(correct))
	@ungroup
	@mutate(easy = ifelse(easy, "Easy pairs", "Hard pairs"))
	data(_) * mapping(:acc, col = :easy) * visual(Hist)
	draw(_, axis=(; xlabel = "Accuracy"))
 end

# ╔═╡ 6b8abbbb-bb91-49ae-8df0-aeebddb55a7f
@chain diff_vigour_test_data begin
	@mutate(chose_left = chose_left * 2 - 1, truth = sign(diff_rpp))
	@mutate(correct = chose_left == truth)
	@mutate(easy = (left_ratio <= right_ratio && left_magnitude >= right_magnitude) || (left_ratio >= right_ratio && left_magnitude <= right_magnitude))
	@group_by(prolific_id, easy)
	@summarize(acc = mean(correct))
	@ungroup
	@mutate(easy = ifelse(easy, "Easy pairs", "Hard pairs"))
	@group_by(easy)
	@summarize(acc = mean(acc))
end

# ╔═╡ 779fac61-ef41-47ab-b461-ddb7fb930011
 @chain diff_vigour_test_data begin
	@mutate(chose_left = chose_left * 2 - 1, truth = sign(diff_rpp))
	@mutate(correct = chose_left == truth)
	@mutate(easy = (left_ratio <= right_ratio && left_magnitude >= right_magnitude) || (left_ratio >= right_ratio && left_magnitude <= right_magnitude))
	@summarize(prop_easy = mean(easy))
 end

# ╔═╡ c686ea0b-7654-408f-8d9c-cabbef803d70
 @chain diff_vigour_test_data begin
	@mutate(chose_left = chose_left * 2 - 1, truth = sign(diff_rpp))
	@mutate(correct = chose_left == truth)
	@mutate(easy = (left_ratio <= right_ratio && left_magnitude >= right_magnitude) || (left_ratio >= right_ratio && left_magnitude <= right_magnitude))
	@filter(.!easy)
	data(_) *
	mapping(:diff_rpp) *
	visual(Hist)
	draw(_)
 end

# ╔═╡ 24545992-c191-43c3-8741-22175650f580
begin
	avg_acc_version_df = @chain test_acc_df begin
		@group_by(version)
		@summarize(acc = mean(acc))
		@ungroup
		@mutate(x = 2, y = [0.2],
		acc_text = string("Accuracy: ", round(acc; digits = 2)))
	end
	diff_rpp_range = LinRange(minimum(diff_vigour_test_data.diff_rpp), maximum(diff_vigour_test_data.diff_rpp), 100)
	pred_effect_rpp = crossjoin(DataFrame.((pairs((;prolific_id="New",chose_left=0.5,diff_rpp=diff_rpp_range,version=["4.0"]))...,))...)
	pred_effect_rpp.pred = predict(test_mixed_rpp, pred_effect_rpp; new_re_levels=:population, type=:response)
	rpp_pred_plot = 
		data(diff_vigour_test_data) *
			mapping(:diff_rpp, :chose_left, color=:version => "Version") *
			visual(Scatter, alpha = 0.15) +
		data(pred_effect_rpp) *
			mapping(:diff_rpp, :pred, color=:version => "Version") *
			visual(Lines, color = :royalblue) +
		data(avg_acc_version_df) *
			mapping(:x, :y, color = :version => "Version"; text = :acc_text => verbatim) *
			visual(Makie.Text, fontsize = 16, font = :bold)
	draw(rpp_pred_plot;axis=(;xlabel="ΔRPP: Left−Right", ylabel="P(Choose left)"))
end

# ╔═╡ 54178f5c-6950-4aca-97e4-aeb56c176c74
md"""
#### Reward magnitude
"""

# ╔═╡ 74882f58-32bc-4eaa-b9e6-087390f8fe65
begin
	test_mixed_mag = fit(GeneralizedLinearMixedModel,
		@formula(chose_left ~ diff_mag +
		(diff_mag | prolific_id)),
		diff_vigour_test_data, Binomial())
end

# ╔═╡ e36d7568-2951-48e8-876b-a86b1a09c23a
begin
	diff_mag_range = LinRange(minimum(diff_vigour_test_data.diff_mag), maximum(diff_vigour_test_data.diff_mag), 100)
	pred_effect_mag = DataFrame(
		prolific_id = "New",
		chose_left = 0.5,
		diff_mag = diff_mag_range
	)
	pred_effect_mag.pred = predict(test_mixed_mag, pred_effect_mag; new_re_levels=:population, type=:response)
	mag_pred_plot = 
		data(diff_vigour_test_data) *
			mapping(:diff_mag, :chose_left) *
			visual(Scatter, alpha = 0.01) +
		data(pred_effect_mag) *
			mapping(:diff_mag, :pred) *
			visual(Lines, color = :royalblue)
	draw(mag_pred_plot)
end

# ╔═╡ 664e3e14-c757-4cd1-9df4-3f530d7a33a7
md"""
#### Fixed ratio
"""

# ╔═╡ 25dc8353-d6f8-4503-bda3-6b288630b770
# ╠═╡ show_logs = false
begin
	test_mixed_fr_rel = fit(GeneralizedLinearMixedModel,
		@formula(chose_left ~ diff_fr_rel_zscore +
		(diff_fr_rel_zscore | prolific_id)),
		diff_vigour_test_data, Binomial())
end

# ╔═╡ eebe4ca4-d3ba-4bce-a8e9-310a53569019
begin
	let
	diff_fr_zscore_range = LinRange(minimum(diff_vigour_test_data.diff_fr_rel_zscore), maximum(diff_vigour_test_data.diff_fr_rel_zscore), 100)
		
	diff_fr_range = std(diff_vigour_test_data.diff_fr_rel) * diff_fr_zscore_range .+ mean(diff_vigour_test_data.diff_fr_rel)

	pred_effect_fr = DataFrame(
		prolific_id = "New",
		chose_left = 0.5,
		diff_fr_rel = diff_fr_range,
		diff_fr_rel_zscore = diff_fr_zscore_range
	)
		
	pred_effect_fr.pred = predict(test_mixed_fr_rel, pred_effect_fr; new_re_levels=:population, type=:response)
		
	fr_pred_plot = 
		data(diff_vigour_test_data) *
			mapping(:diff_fr_rel, :chose_left) *
			visual(Scatter, alpha = 0.01) +
		data(pred_effect_fr) *
			mapping(:diff_fr_rel, :pred) *
			visual(Lines, color = :royalblue)
	draw(fr_pred_plot)
	end
end

# ╔═╡ 6a8d4aac-7c87-476c-8259-116d4dcf24aa
md"""
### Combined models
"""

# ╔═╡ 385ecb85-ea97-45e4-8766-b99969ab12a0
# ╠═╡ show_logs = false
begin
	test_mixed_combined = fit(GeneralizedLinearMixedModel,
		@formula(chose_left ~ diff_mag_zscore + diff_fr_rel_zscore + diff_rpp_zscore +
		(diff_mag_zscore + diff_fr_rel_zscore + diff_rpp_zscore | prolific_id)),
		diff_vigour_test_data, Binomial())
end

# ╔═╡ 837cb513-f4ed-4bb5-8f12-80a717a2dec4
md"""
## PIT
"""

# ╔═╡ 8625eba8-94e7-4b04-9050-7e5b122d6094
md"""
### Keypresses in general
"""

# ╔═╡ b62c9843-4c4d-4c79-a066-c2186a594794
pit_data = antijoin(raw_pit_data, many_miss; on = :prolific_id)

# ╔═╡ 61d553b8-b365-47ea-b4cc-f21a2db9b7f9
plot_presses_vs_var(pit_data; x_var=:reward_per_press, y_var=:press_per_sec, grp_var=:version, xlab="Reward/press", ylab = "Press/sec", combine=false)

# ╔═╡ 74c61b1a-741b-4e89-8314-7559a43fa2c2
unique(pit_data.reward_per_press)

# ╔═╡ 78069573-8451-4411-b2e5-9908263b83e8
let
	common_rpp = unique(pit_data.reward_per_press)
	p = @chain vigour_data begin
		@filter(trial_number != 1)
		@bind_rows(pit_data)
		@filter(reward_per_press in !!common_rpp)
		plot_presses_vs_var(x_var=:reward_per_press, y_var=:press_per_sec, grp_var=:trialphase, xlab="Reward/press", ylab = "Press/sec", grplab = "Task", combine=false)
	end
end

# ╔═╡ 51918379-dc4d-418b-8bf1-d13e4cc18119
md"""
### Pavlovian transfer
"""

# ╔═╡ 0b0faf93-858a-43f2-99e8-70fae15bda9e
plot_presses_vs_var(@mutate(pit_data, coin = categorical(coin)); x_var=:coin, y_var=:press_per_sec, grp_var=:version, xlab="Pavlovian stimuli (coin)", ylab = "Press/sec", combine=false)

# ╔═╡ 9bb2610c-8611-465b-a46e-e4c81f7b98d5
let
	grouped_data, avg_w_data = avg_presses_w_fn(@filter(pit_data, trial_number != 0), [:coin, :reward_per_press], :press_per_sec)
	p = data(avg_w_data) *
	mapping(:reward_per_press, :avg_y, col=:coin=>nonnumeric) *
	(
	visual(Errorbars, whiskerwidth=4) *
	mapping(:se_y) +
	visual(ScatterLines, linewidth=2)
	)
	draw(p; axis=(;xlabel="Pavlovian stimuli (coin)", ylabel="Press/sec", width=150, height=150))
end

# ╔═╡ 3c541186-1087-4339-aba4-1834b636cecf
let
	df = @chain pit_data begin
		@arrange(prolific_id, magnitude, ratio)
		@mutate(pig = "Mag " * string(magnitude) * ", FR " * string(ratio))
	end
	grouped_data, avg_w_data = avg_presses_w_fn(df, [:coin, :pig, :trial_number], :press_per_sec)
	p = data(avg_w_data) *
	mapping(:coin, :avg_y, col=:pig=>nonnumeric) *
	(
	visual(Lines, linewidth=1, color=:gray) +
	visual(Errorbars, whiskerwidth=4) *
	mapping(:se_y, color=:trial_number) +
	visual(Scatter) *
	mapping(color=:trial_number)
	)
	draw(p; axis=(;xlabel="Pavlovian stimuli (coin)", ylabel="Press/sec", width=150, height=150))
end

# ╔═╡ 18726a34-9b24-466a-8f70-6657afb8ceef
CSV.write("data/pit_data.csv", pit_data)

# ╔═╡ 1b81dc3c-4eef-42ac-a931-61b6ea2794c8
# ╠═╡ show_logs = false
# ╠═╡ disabled = true
#=╠═╡
let
	pit_data_cat = @mutate(pit_data,
		reward_per_press = categorical(reward_per_press),
		coin = categorical(coin)
	)
	test_pav_trans = fit(LinearMixedModel,
		@formula(press_per_sec ~ reward_per_press * coin +
		(reward_per_press + coin | prolific_id)),
		pit_data_cat, contrasts=Dict(:reward_per_press => EffectsCoding(base=2.0), :coin => EffectsCoding(base=1.0), :prolific_id => MixedModels.Grouping()))
end
  ╠═╡ =#

# ╔═╡ bcce8e35-588d-45c0-a13d-154def2a7b47
md"""
### Reliability of Pavlovian bias
"""

# ╔═╡ 2f25f144-4b3f-4c2a-bcbb-89bba69bb2c0
pit_data

# ╔═╡ 7c471d3d-3146-4ed2-9bed-538d4d543d42
begin
	pav_12_df = @chain pit_data begin
		@mutate(half = if_else(trial_number <= maximum(trial_number)/2, "first", "second"))
		@group_by(prolific_id, half, coin, reward_per_press)
		@summarize(press_per_sec = mean(press_per_sec))
		@summarize(press_per_sec = mean(press_per_sec))
		@ungroup
		@pivot_wider(names_from = coin, values_from = press_per_sec)
		@mutate(coin_diff = `1.0` - `-0.01`)
		@select(prolific_id, half, coin_diff)
		@pivot_wider(names_from = half, values_from = coin_diff)
	end
	pav_ρ_12_df = @chain pav_12_df begin
		@summarize(ρ_12 = ~ cor(first, second))
		@mutate(ρ_12 = (2 * ρ_12) / (1 + ρ_12))
		@mutate(ρ_text = string("ρ: ", round(ρ_12; digits = 2)))
		@mutate(x = 1, y = [-1.5])
		@ungroup
	end
	pav_ρ_12_plot = 
		data(pav_12_df) *
		mapping(:first, :second) *
		visual(Scatter) +
		data(pav_ρ_12_df) *
		mapping(:x, :y; text = :ρ_text => verbatim) *
		visual(Makie.Text, fontsize = 16, font = :bold) +
		data(DataFrame(intercept = 0, slope = 1)) *
		mapping(:intercept, :slope) *
		visual(ABLines, linestyle = :dash, color = :gray70)
	draw(
		pav_ρ_12_plot;
		axis=(;xlabel="1P - (-0.01P) in first half",ylabel="1P - (-0.01P) in second half")
	)
end

# ╔═╡ 7f614961-a172-4ebf-a08b-1bb660702f05
# ╠═╡ disabled = true
#=╠═╡
begin
	pav_avg_12_df = @chain pit_data begin
		@mutate(half = if_else(trial_number <= maximum(trial_number)/2, "first", "second"), reward_pav = if_else(coin < 0, "pun", "rew"))
		@group_by(prolific_id, half, reward_pav)
		@summarize(press_per_sec = mean(press_per_sec))
		@ungroup
		@pivot_wider(names_from = reward_pav, values_from = press_per_sec)
		@mutate(rew_pun = rew - pun)
		@select(prolific_id, half, rew_pun)
		@pivot_wider(names_from = half, values_from = rew_pun)
	end
	pav_avg_ρ_12_df = @chain pav_avg_12_df begin
		@summarize(ρ_12 = ~ cor(first, second))
		@mutate(ρ_12 = (2 * ρ_12) / (1 + ρ_12))
		@mutate(ρ_text = string("ρ: ", round(ρ_12; digits = 2)))
		@mutate(x = 1, y = [-1])
		@ungroup
	end
	pav_avg_ρ_12_plot = 
		data(pav_avg_12_df) *
		mapping(:first, :second) *
		visual(Scatter) +
		data(pav_avg_ρ_12_df) *
		mapping(:x, :y; text = :ρ_text => verbatim) *
		visual(Makie.Text, fontsize = 16, font = :bold) +
		data(DataFrame(intercept = 0, slope = 1)) *
		mapping(:intercept, :slope) *
		visual(ABLines, linestyle = :dash, color = :gray70)
	draw(
		pav_avg_ρ_12_plot;
		axis=(;xlabel="Rew - Pun in first half",ylabel="Rew - Pun in second half")
	)
end
  ╠═╡ =#

# ╔═╡ a3099b93-c5a5-4ac2-9ca6-b57647dbf708
md"""
### Call R for help!
"""

# ╔═╡ 9026cd8b-5b99-40be-a93b-0d994db5397a
R"""
library(tidyverse)
library(afex)
library(emmeans)
set_sum_contrasts()
theme_set(theme_minimal())

pit_data <- as_tibble($(pit_data))
pit_fit <- pit_data %>% 
    group_by(prolific_id, reward_per_press, coin) %>%
    summarize(press_per_sec = mean(press_per_sec)) %>%
    ungroup() %>%
    mutate(across(c(coin, reward_per_press), as.factor)) %>%
    aov_4(press_per_sec ~ reward_per_press * coin + (reward_per_press + coin | prolific_id), data = .)
pit_fit
"""

# ╔═╡ 2118f30b-cc13-468d-a439-7fe25eba9a18
R"""
emmeans(pit_fit, pairwise ~ reward_per_press)
"""

# ╔═╡ 5675e782-3def-4857-86a2-01e091a5c108
R"""
emmeans(pit_fit, pairwise ~ coin)
"""

# ╔═╡ 71a4e5c5-70ad-4b48-b692-5cb17d5236ef
R"""
emmeans(pit_fit, pairwise ~ coin, lmer.df = "asymptotic") %>%
    contrast(list(pun_vs_rew = c(-1/3, -1/3, -1/3, 1/3, 1/3, 1/3),
				  sub_vs_opt = c(-1/2, 0, 1/2, -1/2, 0, 1/2)))
"""

# ╔═╡ 6aedc1bf-d9bb-40b1-a5ff-fdd864946880
R"""
emmeans(pit_fit, ~ coin + reward_per_press) %>%
	joint_tests(by = "coin")
"""

# ╔═╡ Cell order:
# ╠═05c22183-d42d-4d27-8b21-33f6842e0291
# ╟─33e63f0d-8b59-46a5-af57-b45eb4b8b531
# ╟─be928c7e-363b-45cd-971b-4fad90335b06
# ╟─02713d34-3f91-4a60-b499-93975cede337
# ╠═485584ee-2efc-4ed1-b58d-2a73ec9c8ae1
# ╠═e8568096-8344-43cf-94fa-22219b3a7c7b
# ╟─cd445ddb-a22a-415c-8aae-6fc1eb84585b
# ╠═51515262-32d8-4c9e-8a9c-9b4fd4eb4f09
# ╠═45d95aea-7751-4f37-9558-de4c55a0881e
# ╠═e80172e8-0055-4179-b1a9-ce38da6b3fc4
# ╠═7fadad91-a4b1-4a44-9491-2d389303d405
# ╠═5ae2870a-408f-4aac-8f12-5af081a250d8
# ╠═bda488ea-faa3-4308-ac89-40b8481363bf
# ╟─ac1399c4-6a2f-446f-8687-059fa5da9473
# ╟─9cc0560b-5e3b-45bc-9ba8-605f8111840b
# ╠═2391eb75-db59-4156-99ae-abb8f1508037
# ╟─b9f4babc-8da0-40f2-b0d2-95f915989faf
# ╠═fd68ab0d-ca1b-437b-8c55-24f900a0628d
# ╠═9180c237-a966-4d0c-b92c-7c70eedabc1a
# ╟─342feddc-494a-4807-b4d5-4425d7ca3657
# ╠═2576dc02-f6c6-4c05-8d36-f20e341825c7
# ╠═366287ba-7c26-497c-aada-ec8a131ad22e
# ╠═4401e0dc-fcc2-4561-ae82-afb890b9d22b
# ╠═e39b43eb-7695-4d64-8b6b-cd5afb0616f1
# ╠═00f78d00-ca54-408a-a4a6-ad755566052a
# ╠═89d68509-bcf0-44aa-9ddc-a685131ce146
# ╠═ef4c0f64-2f16-4700-8fc4-703a1b858c37
# ╠═eab61744-af48-4789-ba2f-a92d73527962
# ╠═c4e7d7ec-8da8-4a1f-8da3-0f2d29be6db7
# ╠═09a6f213-f70a-42c2-8eb7-8689d40d140b
# ╠═14d72296-7392-4375-9748-a8ee84d34701
# ╠═1066a576-c7bf-42cc-8097-607c663dcdac
# ╟─128e1be1-191d-4095-bfa9-cd538b913181
# ╠═302e4f7a-fd09-48c3-89ac-a80a63841ee7
# ╠═b0c1d8fb-8f95-4a40-b760-6584022867ce
# ╟─9a8d96d7-13bd-4f70-9344-6569dfa39ab6
# ╟─16154073-51c3-4a37-b40f-8c4f6044702c
# ╠═08fe8e09-51e5-4d54-a0c2-7c6ac8269a00
# ╠═b5c429ee-9f31-4815-98b3-2536952b6881
# ╠═00d6678d-4e26-4fe6-9937-b1055f78cad5
# ╠═50e8263e-a593-47d3-abc4-aceeeb68ba58
# ╟─a2adae6f-d651-4f0c-89f3-38cf357c943e
# ╠═1adc75da-c6c4-4b11-81ff-227851a18076
# ╟─387e33da-31fd-4214-b819-bb4f2e43ab23
# ╟─26282185-43f7-44a1-936d-b4598be46cb4
# ╠═3028ec29-09b5-4754-a318-0b9aaf44e76f
# ╟─e8719da6-89e0-4e25-acfc-2161642d6395
# ╠═f07d048f-121b-4675-b9d0-edc0553b05fa
# ╟─9a6438cc-d758-42a6-8c0b-36cb7dddacf9
# ╠═096ffb0d-cddb-445a-8887-d27dcd16ea47
# ╠═223d7ed5-8621-4f6a-905e-cfbd27849686
# ╟─b3f76fad-a1c6-4fc0-afbf-7ce9736986e8
# ╠═7f3c680a-970b-483d-b3df-ed22b6dd8d81
# ╠═341083df-98ee-4758-85a6-8a070b662b86
# ╠═2f8f197a-77c1-48ac-b18a-d2e3a9006bfb
# ╠═96bcba8a-e8a1-49c4-b60e-65aa322ca321
# ╠═6b8abbbb-bb91-49ae-8df0-aeebddb55a7f
# ╠═779fac61-ef41-47ab-b461-ddb7fb930011
# ╠═c686ea0b-7654-408f-8d9c-cabbef803d70
# ╠═24545992-c191-43c3-8741-22175650f580
# ╟─54178f5c-6950-4aca-97e4-aeb56c176c74
# ╠═74882f58-32bc-4eaa-b9e6-087390f8fe65
# ╠═e36d7568-2951-48e8-876b-a86b1a09c23a
# ╟─664e3e14-c757-4cd1-9df4-3f530d7a33a7
# ╠═25dc8353-d6f8-4503-bda3-6b288630b770
# ╠═eebe4ca4-d3ba-4bce-a8e9-310a53569019
# ╟─6a8d4aac-7c87-476c-8259-116d4dcf24aa
# ╠═385ecb85-ea97-45e4-8766-b99969ab12a0
# ╟─837cb513-f4ed-4bb5-8f12-80a717a2dec4
# ╟─8625eba8-94e7-4b04-9050-7e5b122d6094
# ╠═b62c9843-4c4d-4c79-a066-c2186a594794
# ╠═61d553b8-b365-47ea-b4cc-f21a2db9b7f9
# ╠═74c61b1a-741b-4e89-8314-7559a43fa2c2
# ╠═78069573-8451-4411-b2e5-9908263b83e8
# ╟─51918379-dc4d-418b-8bf1-d13e4cc18119
# ╠═0b0faf93-858a-43f2-99e8-70fae15bda9e
# ╠═9bb2610c-8611-465b-a46e-e4c81f7b98d5
# ╠═3c541186-1087-4339-aba4-1834b636cecf
# ╠═18726a34-9b24-466a-8f70-6657afb8ceef
# ╠═1b81dc3c-4eef-42ac-a931-61b6ea2794c8
# ╟─bcce8e35-588d-45c0-a13d-154def2a7b47
# ╠═2f25f144-4b3f-4c2a-bcbb-89bba69bb2c0
# ╠═7c471d3d-3146-4ed2-9bed-538d4d543d42
# ╠═7f614961-a172-4ebf-a08b-1bb660702f05
# ╟─a3099b93-c5a5-4ac2-9ca6-b57647dbf708
# ╠═c5be4a4b-d800-4ba4-abde-c9607cefe261
# ╠═9026cd8b-5b99-40be-a93b-0d994db5397a
# ╠═2118f30b-cc13-468d-a439-7fe25eba9a18
# ╠═5675e782-3def-4857-86a2-01e091a5c108
# ╠═71a4e5c5-70ad-4b48-b692-5cb17d5236ef
# ╠═6aedc1bf-d9bb-40b1-a5ff-fdd864946880
