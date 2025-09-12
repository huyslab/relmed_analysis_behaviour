### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ ff8c6dd6-7b4a-11ef-3148-ad43f7d07801
begin
	cd("/home/jovyan/")
    import Pkg
    # activate the shared project environment
    Pkg.activate("relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	using Random, DataFrames, JSON, CSV, StatsBase, JLD2, HTTP, CairoMakie, Printf, Distributions, CategoricalArrays, AlgebraOfGraphics
	using LogExpFunctions: logistic, logit
	include("fetch_preprocess_data.jl")
	include("sample_utils.jl")
	include("plotting_utils.jl")
	nothing
end

# ╔═╡ 3f653d16-588f-4a30-8746-105c93305156
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

# ╔═╡ 2b7c2f4c-36cc-4008-bb44-0e34d40e3d92
# Load data
begin
	PLT_data, test_data, vigour_data, jspsych_data = load_pilot2_data()
	nothing
end

# ╔═╡ 60559c55-840f-49cb-9537-1aceaaf5e658
# Plot PLT accuracy curve
let

	f = Figure()

	ax = plot_group_accuracy!(
		f[1,1],
		group = :n_pairs,
		PLT_data
	)

	ax.xticks = [1, 10, 20, 30]

	f

end

# ╔═╡ cb470684-4557-4cb4-a6c2-90a207d4dcc6
# Plot PLT accuracy curve
let

	PLT_remmaped = copy(PLT_data)

	transform!(
		groupby(PLT_remmaped, [:session, :prolific_pid, :exp_start_time, :block, :stimulus_pair]),
		:trial => (x -> 1:length(x)) => :trial
	)
	
	f = Figure()

	ax = plot_group_accuracy!(
		f[1,1],
		group = :n_pairs,
		PLT_remmaped
	)

	ax.xticks = [1, 10, 20, 30]

	f

end

# ╔═╡ Cell order:
# ╠═ff8c6dd6-7b4a-11ef-3148-ad43f7d07801
# ╠═3f653d16-588f-4a30-8746-105c93305156
# ╠═2b7c2f4c-36cc-4008-bb44-0e34d40e3d92
# ╠═60559c55-840f-49cb-9537-1aceaaf5e658
# ╠═cb470684-4557-4cb4-a6c2-90a207d4dcc6
