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
	using Tidier
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

# ╔═╡ 4f9cd22a-c03f-4a95-b631-5bace22aa426
# Set up saving to OSF
begin
	osf_folder = "Workshop figures/Vigour"
	proj = setup_osf("Task development")
end

# ╔═╡ 93bc3812-c620-4a8d-a312-de9fd0e55327
begin
	# Load data
	_, _, raw_vigour_data, raw_post_vigour_test_data, _, _,
		_, _ = load_pilot6_data(;force_download=false)
	nothing
end

# ╔═╡ 1f8ca836-f2f7-4965-bf07-5656cf6c4ec6
md"""
Set theme globally
"""

# ╔═╡ 1c0196e9-de9a-4dfa-acb5-357c02821c5d
set_theme!(theme_minimal());

# ╔═╡ 8255895b-a337-4c8a-a1a7-0983499f684e
md"""
## Press rate: Trial presses/sec
"""

# ╔═╡ de48ee97-d79a-46e4-85fb-08dd569bf7ef
begin
	vigour_data = @chain raw_vigour_data begin
		@filter(prolific_pid != "671139a20b977d78ec2ac1e0")
		@mutate(
			press_per_sec = trial_presses / trial_duration * 1000,
			ratio = categorical(ratio; levels = [1, 8, 16], ordered=true),
			magnitude = categorical(magnitude; levels = [1, 2, 5], ordered=true)
		)
	end
	nothing
end

# ╔═╡ 4be713bc-4af3-4363-94f7-bc68c71609c2
let
	fig = @chain vigour_data begin
		# @filter(trial_presses > 0)
		@mutate(ratio = ~recode(ratio, 1=>"FR: 1",8=>"FR: 8",16=>"FR: 16"))
		data(_) * mapping(:press_per_sec; color=:magnitude=>"Magnitude", row=:ratio) * AlgebraOfGraphics.density()
		draw(_, scales(Color = (;palette=:Oranges_3)); figure=(;size=(4, 4) .* 300 ./ 2.54), axis=(;xlabel="Press/sec", ylabel="Density"))
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

# ╔═╡ Cell order:
# ╠═b41e7252-a075-11ef-039c-f532a7fb0a94
# ╠═4f9cd22a-c03f-4a95-b631-5bace22aa426
# ╠═93bc3812-c620-4a8d-a312-de9fd0e55327
# ╟─1f8ca836-f2f7-4965-bf07-5656cf6c4ec6
# ╠═1c0196e9-de9a-4dfa-acb5-357c02821c5d
# ╟─8255895b-a337-4c8a-a1a7-0983499f684e
# ╠═de48ee97-d79a-46e4-85fb-08dd569bf7ef
# ╠═4be713bc-4af3-4363-94f7-bc68c71609c2
