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
	using Random, DataFrames, JSON, CSV, StatsBase, JLD2, HTTP, Dates, CairoMakie, AlgebraOfGraphics, Tidier
	include("fetch_preprocess_data.jl")
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

# ╔═╡ 21f66f03-61ec-4d43-a4fe-8de250bc8455
PLT_data, test_data, vigour_data, jspsych_data = load_pilot2_data();

# ╔═╡ 9afd25b7-0b29-4b78-b81f-45831b417dfb
# ╠═╡ disabled = true
#=╠═╡
CSV.write("data/pilot2_vigour_data.csv", vigour_data)
  ╠═╡ =#

# ╔═╡ 392006ba-b905-4a51-a02e-796e19fbc24a
@chain vigour_data begin
	@group_by(prolific_id, exp_start_time)
	@summarize(n = n(), n_miss = sum(trial_presses .== 0))
	@ungroup
end

# ╔═╡ eab61744-af48-4789-ba2f-a92d73527962
begin
	let
	# Assume `data` is a DataFrame with columns `prolific_id` and `trial_presses`
	# Group by prolific_id and calculate the number of misses
	grouped_data = combine(groupby(vigour_data, :prolific_id), :trial_presses => (x -> sum(x .== 0)) => :n_miss)

	# Plot the histogram of the number of misses
	fig = data(grouped_data) *
		mapping(:n_miss) *
		visual(Hist)
	draw(fig; axis=(xlabel = "# Missing trials", ylabel = ""))
	end
end

# ╔═╡ 63bf3524-5fcb-4e3b-9dfc-dbc2f2a079c9
begin
	let
		# Assuming `data` is a DataFrame with columns: `trial_presses`, `ratio`, and `magnitude`
		# Step 1: Filter data for rows where trial_presses == 0
		filtered_data = filter(row -> row.trial_presses == 0, vigour_data)

		# Step 2: Create a Figure with subplots (faceting by `magnitude`)
		fig = Figure(size = (800, 400))
		unique_magnitudes = unique(filtered_data.magnitude)

		# Step 3: Loop through magnitudes and create subplots
		for (i, mag) in enumerate(unique_magnitudes)
		    ax = Axis(fig[1, i], title = "Magnitude = $mag", xlabel = "FR", ylabel = "# Missing trials",
		              xticks = [1; 2:2:16])  # Custom x-axis ticks

		    # Step 4: Filter data for this magnitude
		    mag_data = filter(row -> row.magnitude == mag, filtered_data)

		    # Step 5: Plot the histogram for `ratio` in this facet
		    hist!(ax, mag_data.ratio)
		end
		fig
	end
end

# ╔═╡ Cell order:
# ╠═05c22183-d42d-4d27-8b21-33f6842e0291
# ╟─33e63f0d-8b59-46a5-af57-b45eb4b8b531
# ╟─73f5f721-f907-4c1e-94b4-953f63f6775e
# ╟─be928c7e-363b-45cd-971b-4fad90335b06
# ╠═21f66f03-61ec-4d43-a4fe-8de250bc8455
# ╠═9afd25b7-0b29-4b78-b81f-45831b417dfb
# ╠═392006ba-b905-4a51-a02e-796e19fbc24a
# ╠═eab61744-af48-4789-ba2f-a92d73527962
# ╠═63bf3524-5fcb-4e3b-9dfc-dbc2f2a079c9
