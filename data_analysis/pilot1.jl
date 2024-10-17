### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ bbaf0748-7f6b-11ef-3597-5197791d694f
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

# ╔═╡ 1bfec8c2-7e2a-45ed-bbc6-df613fcb50f9
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

	ppt_th = Theme(
		font = "Helvetica",
		fontsize = 20,
		Axis = (
			xgridvisible = false,
			ygridvisible = false,
			rightspinevisible = false,
			topspinevisible = false,
			xticklabelsize = 18,
			yticklabelsize = 18,
			spinewidth = 2,
			xtickwidth = 2,
			ytickwidth = 2
		)
	)
	set_theme!(th)
end

# ╔═╡ 67070788-2176-4ddb-bb24-f4d515f3ad36
# Load data
begin
	PLT_data = load_pilot1_data()
	nothing
end

# ╔═╡ 0dc65bca-d972-4988-a26f-7cb43671cc26
# Plot PLT accuracy curve
let
	with_theme(ppt_th) do
		f = Figure(
			size = (13.24, 6.7) .* 72 ./ 2.54,
			figure_padding = (0, 12, 0, 0)
		)
	
		ax = plot_group_accuracy!(
			f[1,1],
			PLT_data,
			ylabel = "Prop. optimal\nchoice"
		)

		save("results/pilot1_acc_ppt.pdf", f, pt_per_unit = 1)
	
		f
	end

end

# ╔═╡ 737b1aa0-918c-4a95-a26d-4e2515d6209b
# Plot PLT accuracy curve by valence
let
	with_theme(ppt_th) do
		f = Figure(
			size = (13.24, 6.7) .* 72 ./ 2.54,
			figure_padding = (0, 12, 0, 0)
		)
	
		ax = plot_group_accuracy!(
			f[1,1],
			PLT_data,
			group = :valence,
			ylabel = "Prop. optimal\nchoice"
		)

		Legend(
			f[1,1],
			[PolyElement(color = c) for c in Makie.wong_colors()[1:2]],
			["Punishment", "Reward"],
			framevisible = false,
			tellwidth = false,
			labelsize = 18,
			halign = :right,
			valign = :bottom
		)
		
		save("results/pilot1_acc_by_valence_ppt.pdf", f, pt_per_unit = 1)
	
		f
	end

end

# ╔═╡ Cell order:
# ╠═bbaf0748-7f6b-11ef-3597-5197791d694f
# ╠═1bfec8c2-7e2a-45ed-bbc6-df613fcb50f9
# ╠═67070788-2176-4ddb-bb24-f4d515f3ad36
# ╠═0dc65bca-d972-4988-a26f-7cb43671cc26
# ╠═737b1aa0-918c-4a95-a26d-4e2515d6209b
