### A Pluto.jl notebook ###
# v0.19.46

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
	using Random, DataFrames, JSON, CSV, StatsBase, JLD2, HTTP, Dates, AlgebraOfGraphics, CairoMakie, TidierData
	include("fetch_preprocess_data.jl")
	include("vigour_analysis/Vigour_utlis_fn.jl")
	set_theme!(theme_minimal())
	nothing
end

# ╔═╡ 33e63f0d-8b59-46a5-af57-b45eb4b8b531
md"""
## Load data from REDCap or from local files
"""

# ╔═╡ 73f5f721-f907-4c1e-94b4-953f63f6775e
md"""
### From local JSON files
"""

# ╔═╡ be928c7e-363b-45cd-971b-4fad90335b06
md"""
### From REDCap
"""

# ╔═╡ 230df26d-e547-42d1-bb71-1bfa7a1dd25c
function load_pilot3_data()
	datafile = "data/pilot3.jld2"

	# Load data or download from REDCap
	if !isfile(datafile) || true
		jspsych_json, records = get_REDCap_data("pilot3"; file_field = "file_data")
	
		jspsych_data = REDCap_data_to_df(jspsych_json, records)

		remove_testing!(jspsych_data)

		JLD2.@save datafile jspsych_data
	else
		JLD2.@load datafile jspsych_data
	end

	### Vigour task here
	vigour_data = prepare_vigour_data(jspsych_data) |>
		x -> exclude_vigour_trials(x, 36)
	return vigour_data, jspsych_data
end

# ╔═╡ 485584ee-2efc-4ed1-b58d-2a73ec9c8ae1
raw_vigour_data, jspsych_data = load_pilot3_data();

# ╔═╡ e8568096-8344-43cf-94fa-22219b3a7c7b
begin
	n_miss_df = combine(groupby(raw_vigour_data, :prolific_id), :trial_presses => (x -> sum(x .== 0)) => :n_miss)
	many_miss = filter(x -> x.n_miss > 9, n_miss_df)
end

# ╔═╡ 45d95aea-7751-4f37-9558-de4c55a0881e
begin
	transform!(raw_vigour_data, [:trial_presses, :trial_duration] => ((x, y) -> x .* 1000 ./ y) => :press_per_sec)
	vigour_data = antijoin(raw_vigour_data, many_miss; on = :prolific_id);
	@count(vigour_data, prolific_id)
end

# ╔═╡ 2391eb75-db59-4156-99ae-abb8f1508037
begin
	let
		# Group and calculate mean presses for each participant
		grouped_data = vigour_data |>
			x -> transform(x, :response_times => ByRow(safe_median) => :avg_rt) |>
			x -> groupby(x, [:prolific_id, :reward_per_press]) |>
			x -> combine(x, :avg_rt => mean => :avg_rt)
		sorted_data = sort(grouped_data, [:prolific_id, :reward_per_press])
		
		# Calculate the average across all participants
		avg_data = combine(groupby(sorted_data, :reward_per_press), :avg_rt => (x -> median(skipmissing(x))) => :avg_rt)
		
		# Create the plot for individual participants
		individual_plot = data(sorted_data) * 
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
		fig = Figure(
			size = (12.2, 7.6) .* 144 ./ 2.54, # 144 points per inch, then cm
		)
		# Draw the individual lines plot
		draw!(fig[1, 1], individual_plot)
		
		# Draw the average line plot
		draw!(fig[1, 2], average_plot)
		
		# Add a overall title
		Label(fig[0, :], "Response times vs Reward/press")

		fig
	end
end

# ╔═╡ 89d68509-bcf0-44aa-9ddc-a685131ce146
plot_presses_vs_var(vigour_data; combine = false)

# ╔═╡ 50b3cb4b-8cbb-45d2-89c0-231e63a0ba20
plot_presses_vs_var(vigour_data; x_var=:ratio, combine = false)

# ╔═╡ 3a424bc7-1d6b-4670-8e18-03a79b43a9f6
plot_presses_vs_var(vigour_data; x_var=:magnitude, combine = false)

# ╔═╡ 1066a576-c7bf-42cc-8097-607c663dcdac
begin
	let
		# Group and calculate mean presses for each participant
		grouped_data, avg_w_data = avg_presses_w_fn(vigour_data, [:magnitude, :ratio], :press_per_sec)
		
		# Create the plot for the average line
		average_plot = data(avg_w_data) * 
		    mapping(
		        :ratio,
		        :avg_y,
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
end

# ╔═╡ 302e4f7a-fd09-48c3-89ac-a80a63841ee7
begin
	let
		# Create the plot
		plot = data(n_miss_df) * 
		       mapping(:n_miss) * 
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

# ╔═╡ Cell order:
# ╠═05c22183-d42d-4d27-8b21-33f6842e0291
# ╟─33e63f0d-8b59-46a5-af57-b45eb4b8b531
# ╟─73f5f721-f907-4c1e-94b4-953f63f6775e
# ╟─be928c7e-363b-45cd-971b-4fad90335b06
# ╠═230df26d-e547-42d1-bb71-1bfa7a1dd25c
# ╠═485584ee-2efc-4ed1-b58d-2a73ec9c8ae1
# ╠═e8568096-8344-43cf-94fa-22219b3a7c7b
# ╠═45d95aea-7751-4f37-9558-de4c55a0881e
# ╠═2391eb75-db59-4156-99ae-abb8f1508037
# ╠═89d68509-bcf0-44aa-9ddc-a685131ce146
# ╠═50b3cb4b-8cbb-45d2-89c0-231e63a0ba20
# ╠═3a424bc7-1d6b-4670-8e18-03a79b43a9f6
# ╠═1066a576-c7bf-42cc-8097-607c663dcdac
# ╠═302e4f7a-fd09-48c3-89ac-a80a63841ee7
