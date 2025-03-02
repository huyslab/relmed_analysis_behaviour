### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ 05c22183-d42d-4d27-8b21-33f6842e0291
begin
	cd("/home/jovyan/")
    import Pkg
    # activate the shared project environment
    Pkg.activate("relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	Pkg.add("MixedModels")
	using Random, DataFrames, JSON, CSV, StatsBase, JLD2, HTTP, Dates, AlgebraOfGraphics, CairoMakie, Tidier, MixedModels, Printf, CategoricalArrays
	include("fetch_preprocess_data.jl")
	include("vigour_utils.jl")
	set_theme!(theme_minimal())
end

# ╔═╡ 33e63f0d-8b59-46a5-af57-b45eb4b8b531
md"""
## Load data from REDCap or from local files
"""

# ╔═╡ be928c7e-363b-45cd-971b-4fad90335b06
md"""
### From REDCap
"""

# ╔═╡ 83858169-2022-4da9-8c28-a1685ffb51f3
function load_pilot3_data()
	datafile = "data/pilot3.1.jld2"
	
	# Load data or download from REDCap
	if !isfile(datafile)
		jspsych_json, records = get_REDCap_data("pilot3.1"; file_field = "file_data")
	
		jspsych_data = REDCap_data_to_df(jspsych_json, records)

		remove_testing!(jspsych_data)

		JLD2.@save datafile jspsych_data
	else
		JLD2.@load datafile jspsych_data
	end

	### Completion by vigour bonus
	complete_sub = @chain jspsych_data begin
		@group_by(prolific_pid)
		@filter(!ismissing(vigour_bonus))
		@ungroup()
	end
	complete_jspsych_data = semijoin(jspsych_data, complete_sub, on = :prolific_pid)
	
	### Vigour task here
	vigour_data = prepare_vigour_data(complete_jspsych_data) |>
		x -> exclude_vigour_trials(x, 36)

    ### Vigour test data
		vigour_test_data = prepare_post_vigour_test_data(complete_jspsych_data)
	
	return vigour_data, vigour_test_data, complete_jspsych_data
end

# ╔═╡ 485584ee-2efc-4ed1-b58d-2a73ec9c8ae1
raw_vigour_data, raw_vigour_test_data, jspsych_data = load_pilot3_data();

# ╔═╡ 97257b1e-48dc-4be3-98c7-5872af7d3458
md"""
### Add back version for the last pilot
"""

# ╔═╡ 7eafecc7-81c4-4702-b0cf-6f591cacb6cd
begin
	transform!(raw_vigour_data, :version => (x -> replace(x, missing => "3.1")) => :version)
	transform!(raw_vigour_data, [:trial_presses, :trial_duration] => ((x, y) -> x .* 1000 ./ y) => :press_per_sec)
	transform!(raw_vigour_test_data, :version => (x -> replace(x, missing => "3.1")) => :version)
end

# ╔═╡ e8568096-8344-43cf-94fa-22219b3a7c7b
begin
	n_miss_df = combine(groupby(raw_vigour_data, [:prolific_pid, :version]), :trial_presses => (x -> sum(x .== 0)) => :n_miss)
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
		@rename(prolific_pid = var"Participant id", age = var"Age")
		@filter(age != "CONSENT_REVOKED")
		@mutate(age = as_float(age))
	end
end

# ╔═╡ 45d95aea-7751-4f37-9558-de4c55a0881e
begin
	vigour_data = antijoin(raw_vigour_data, many_miss; on = :prolific_pid)
	leftjoin!(vigour_data, info_data; on = :prolific_pid)
	# filter!(x -> !ismissing(x.age), vigour_data)
	vigour_data_2 = @filter(vigour_data, version === "3.2")
	vigour_test_data = antijoin(raw_vigour_test_data, many_miss; on = :prolific_pid)
	vigour_test_data_2 = @filter(vigour_test_data, version === "3.2")
end

# ╔═╡ e80172e8-0055-4179-b1a9-ce38da6b3fc4
combine(groupby(vigour_data, [:version]), :age => (x -> DataFrame([summarystats(x)])) => AsTable)

# ╔═╡ 5ae2870a-408f-4aac-8f12-5af081a250d8
@count(raw_vigour_data, prolific_pid, exp_start_time, version) |>
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
begin
	let
		# Group and calculate mean presses for each participant
		grouped_data = vigour_data_2 |>
			x -> transform(x, :response_times => ByRow(safe_mean) => :avg_rt) |>
			x -> groupby(x, [:prolific_pid, :reward_per_press]) |>
			x -> combine(x, :avg_rt => mean => :avg_rt)
		sort!(grouped_data, [:prolific_pid, :reward_per_press])
		
		# Calculate the average across all participants
		avg_data = combine(groupby(grouped_data, :reward_per_press), :avg_rt => (x -> median(skipmissing(x))) => :avg_rt)
		
		# Create the plot for individual participants
		individual_plot = data(grouped_data) * 
		    mapping(
		        :reward_per_press => "Reward/press",
		        :avg_rt => "Response times (ms)",
		        group = :prolific_pid
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
end

# ╔═╡ b9f4babc-8da0-40f2-b0d2-95f915989faf
md"""
### Number of key presses
"""

# ╔═╡ fd68ab0d-ca1b-437b-8c55-24f900a0628d
begin
	groupby(vigour_data, Cols(:prolific_pid, :version)) |>
			x -> combine(x, [:trial_presses, :trial_duration] => ((x, y) -> mean(x .* 1000 ./ y)) => :n_presses) |>
			x -> sort(x, Cols(:prolific_pid, :version)) |>
			x -> groupby(x, :version) |>
			x -> combine(x, :n_presses => mean => :n_presses)
end

# ╔═╡ a6f79793-358f-410f-bba2-de8b2a721420
let
	first_four_df = @chain vigour_data begin
		@filter(trial_number <= 4)
		@filter(version === "3.2")
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

# ╔═╡ 1281426b-2f18-42d1-b6f2-a72bde6381e8
[1/16, 1/8, 2/16, 2/8, 5/16, 5/8, 1/1, 2/1, 5/1]

# ╔═╡ 00f78d00-ca54-408a-a4a6-ad755566052a
plot_presses_vs_var(@filter(vigour_data, trial_number != 0); x_var=:reward_per_press, y_var=:press_per_sec, grp_var=:version, xlab="Reward/press", ylab = "Press/sec", combine=false)

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
CSV.write("data/pilot3.2_vigour_data.csv", vigour_data)

# ╔═╡ 1066a576-c7bf-42cc-8097-607c663dcdac
let
	# Group and calculate mean presses for each participant
	grouped_data, avg_w_data = avg_presses_w_fn(@filter(vigour_data, trial_number != 0), [:magnitude, :ratio], :press_per_sec, :version)
	
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
		@group_by(prolific_pid, reward_per_press)
		@summarize(n_miss = n())
		@ungroup()
		@mutate(short_id = last.(prolific_pid, 5))
		@arrange(prolific_pid, reward_per_press)
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
@count(vigour_data, prolific_pid, version)

# ╔═╡ b5c429ee-9f31-4815-98b3-2536952b6881
begin
	transform!(vigour_data, :trial_number => (x -> ifelse.(x .> maximum(x)/2, "second", "first")) => :half)
	first_second_df = @chain vigour_data begin
		@mutate(press_per_sec = trial_presses .* 1000 ./ trial_duration)
		@group_by(prolific_pid, version, half)
		@summarize(n_presses = mean(press_per_sec))
		@ungroup
		@pivot_wider(names_from = half, values_from = n_presses)
	end
	ρ_12_df = @chain first_second_df begin
		@group_by(version)
		@summarize(ρ_12 = ~ cor(first, second))
		@mutate(ρ_12 = (2 * ρ_12) / (1 + ρ_12))
		@mutate(ρ_text = string("ρ: ", round(ρ_12; digits = 2)))
		@mutate(x = 6, y = [1.5, 2])
		@ungroup
	end
	first_second_plot = 
		data(first_second_df) *
		mapping(:first, :second, color = :version) *
		visual(Scatter) +
		data(ρ_12_df) *
		mapping(:x, :y, color = :version; text = :ρ_text => AlgebraOfGraphics.AlgebraOfGraphics.verbatim) *
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
		@group_by(prolific_pid, version, half, magnitude, ratio)
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
		@mutate(x = 0, y = [6.5, 7.5])
		@ungroup
	end
	p = data(df) * 
			mapping(:first, :second, col = :magnitude, row = :ratio, color = :version) *
			visual(Scatter, alpha = 0.3) +
		data(version_df) *
			mapping(:x, :y, color = :version; text = :ρ_text => AlgebraOfGraphics.AlgebraOfGraphics.verbatim, col = :magnitude, row = :ratio) *
			visual(Makie.Text, fontsize = 12) +
		data(DataFrame(intercept = 0, slope = 1)) *
			mapping(:intercept, :slope) *
			visual(ABLines, linestyle = :dash, color = :gray70)
	draw(p;
		axis=(;xlabel="First half trials",ylabel="Second half trials"))
end

# ╔═╡ 50e8263e-a593-47d3-abc4-aceeeb68ba58
ρ_by_conds = @chain vigour_data begin
	@group_by(prolific_pid, half, magnitude, ratio, version)
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
		@group_by(prolific_pid, even, version)
		@summarize(n_presses = mean(press_per_sec))
		@ungroup
		@pivot_wider(names_from = even, values_from = n_presses)
	end
	ρ_01_df = @chain even_odd_df begin
		@group_by(version)
		@summarize(ρ_01 = ~ cor(odd, even))
		@mutate(ρ_01 = (2 * ρ_01) / (1 + ρ_01))
		@mutate(ρ_text = string("ρ: ", round(ρ_01; digits = 2)))
		@mutate(x = 6, y = [1.5, 2])
		@ungroup
	end
	even_odd_plot = 
		data(even_odd_df) *
			mapping(:even, :odd, color = :version) *
			visual(Scatter) +
		data(ρ_01_df) *
			mapping(:x, :y, color = :version; text = :ρ_text => AlgebraOfGraphics.AlgebraOfGraphics.verbatim) *
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
		@arrange(prolific_pid, version, reward_per_press)
		@group_by(prolific_pid, version)
		@filter(reward_per_press != median(reward_per_press))
		@mutate(low_rpp = if_else(reward_per_press < median(reward_per_press), "low_rpp", "high_rpp"))
		@ungroup
		@group_by(prolific_pid, version, even, low_rpp)
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
		@mutate(x = -0.5, y = [-4, -3.75])
		@ungroup
	end
	rpp_diff_plot = 
		data(rpp_dff_df) *
			mapping(:even, :odd, color = :version) *
			visual(Scatter) +
		data(ρ_rpp_diff_df) *
			mapping(:x, :y, color = :version; text = :ρ_text => AlgebraOfGraphics.AlgebraOfGraphics.verbatim) *
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
		@arrange(prolific_pid, version, reward_per_press)
		@group_by(prolific_pid, version)
		@filter(reward_per_press != median(reward_per_press))
		@mutate(low_rpp = if_else(reward_per_press < median(reward_per_press), "low_rpp", "high_rpp"))
		@ungroup
		@group_by(prolific_pid, version, first, low_rpp)
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
		@mutate(x = -0.5, y = [-4, -3.75])
		@ungroup
	end
	rpp_diff_plot = 
		data(rpp_dff_df) *
			mapping(:first, :second, color = :version) *
			visual(Scatter) +
		data(ρ_rpp_diff_df) *
			mapping(:x, :y, color = :version; text = :ρ_text => AlgebraOfGraphics.AlgebraOfGraphics.verbatim) *
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
CSV.write("data/pilot3.2_vigour_test_data.csv", diff_vigour_test_data)

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
		@formula(chose_left ~ diff_rpp * version +
		(diff_rpp | prolific_pid)),
		diff_vigour_test_data, Binomial())
end

# ╔═╡ 341083df-98ee-4758-85a6-8a070b662b86
diff_vigour_test_data

# ╔═╡ 2f8f197a-77c1-48ac-b18a-d2e3a9006bfb
begin
	test_acc_df = @chain diff_vigour_test_data begin
		@group_by(prolific_pid, version)
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
	@group_by(prolific_pid, easy)
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
	@group_by(prolific_pid, easy)
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
		@mutate(x = 2, y = [0.2, 0.1],
		acc_text = string("Accuracy: ", round(acc; digits = 2)))
	end
	diff_rpp_range = LinRange(minimum(diff_vigour_test_data.diff_rpp), maximum(diff_vigour_test_data.diff_rpp), 100)
	pred_effect_rpp = crossjoin(DataFrame.((pairs((;prolific_pid="New",chose_left=0.5,diff_rpp=diff_rpp_range,version=["3.1", "3.2"]))...,))...)
	pred_effect_rpp.pred = predict(test_mixed_rpp, pred_effect_rpp; new_re_levels=:population, type=:response)
	rpp_pred_plot = 
		data(diff_vigour_test_data) *
			mapping(:diff_rpp, :chose_left, color=:version => "Version") *
			visual(Scatter, alpha = 0.15) +
		data(pred_effect_rpp) *
			mapping(:diff_rpp, :pred, color=:version => "Version") *
			visual(Lines, color = :royalblue) +
		data(avg_acc_version_df) *
			mapping(:x, :y, color = :version => "Version"; text = :acc_text => AlgebraOfGraphics.AlgebraOfGraphics.verbatim) *
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
		(diff_mag | prolific_pid)),
		diff_vigour_test_data, Binomial())
end

# ╔═╡ e36d7568-2951-48e8-876b-a86b1a09c23a
begin
	diff_mag_range = LinRange(minimum(diff_vigour_test_data.diff_mag), maximum(diff_vigour_test_data.diff_mag), 100)
	pred_effect_mag = DataFrame(
		prolific_pid = "New",
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
		(diff_fr_rel_zscore | prolific_pid)),
		diff_vigour_test_data, Binomial())
end

# ╔═╡ eebe4ca4-d3ba-4bce-a8e9-310a53569019
begin
	let
	diff_fr_zscore_range = LinRange(minimum(diff_vigour_test_data.diff_fr_rel_zscore), maximum(diff_vigour_test_data.diff_fr_rel_zscore), 100)
		
	diff_fr_range = std(diff_vigour_test_data.diff_fr_rel) * diff_fr_zscore_range .+ mean(diff_vigour_test_data.diff_fr_rel)

	pred_effect_fr = DataFrame(
		prolific_pid = "New",
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
		(diff_mag_zscore + diff_fr_rel_zscore + diff_rpp_zscore | prolific_pid)),
		diff_vigour_test_data, Binomial())
end

# ╔═╡ Cell order:
# ╠═05c22183-d42d-4d27-8b21-33f6842e0291
# ╟─33e63f0d-8b59-46a5-af57-b45eb4b8b531
# ╟─be928c7e-363b-45cd-971b-4fad90335b06
# ╠═83858169-2022-4da9-8c28-a1685ffb51f3
# ╠═485584ee-2efc-4ed1-b58d-2a73ec9c8ae1
# ╟─97257b1e-48dc-4be3-98c7-5872af7d3458
# ╠═7eafecc7-81c4-4702-b0cf-6f591cacb6cd
# ╠═e8568096-8344-43cf-94fa-22219b3a7c7b
# ╟─cd445ddb-a22a-415c-8aae-6fc1eb84585b
# ╠═51515262-32d8-4c9e-8a9c-9b4fd4eb4f09
# ╠═45d95aea-7751-4f37-9558-de4c55a0881e
# ╠═e80172e8-0055-4179-b1a9-ce38da6b3fc4
# ╠═5ae2870a-408f-4aac-8f12-5af081a250d8
# ╠═bda488ea-faa3-4308-ac89-40b8481363bf
# ╟─ac1399c4-6a2f-446f-8687-059fa5da9473
# ╟─9cc0560b-5e3b-45bc-9ba8-605f8111840b
# ╠═2391eb75-db59-4156-99ae-abb8f1508037
# ╟─b9f4babc-8da0-40f2-b0d2-95f915989faf
# ╠═fd68ab0d-ca1b-437b-8c55-24f900a0628d
# ╠═a6f79793-358f-410f-bba2-de8b2a721420
# ╠═366287ba-7c26-497c-aada-ec8a131ad22e
# ╠═1281426b-2f18-42d1-b6f2-a72bde6381e8
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
