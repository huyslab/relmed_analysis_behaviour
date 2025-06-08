### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ 95c436fe-446c-11f0-0460-d56b7f17790b
begin
	cd("/home/jovyan/")
    import Pkg
    # activate the shared project environment
    Pkg.activate("relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()

	using AlgebraOfGraphics, DataFrames, CairoMakie

end

# ╔═╡ 11845224-c0aa-41f9-93aa-abe0ecfc6430
set_theme!(
	merge(
		theme_minimal(),
		Theme(
		font = "Helvetica",
		fontsize = 28,
		Axis = (
			xticklabelsize = 24,
			yticklabelsize = 24,
			spinewidth = 3,
			xtickwidth = 3,
			ytickwidth = 3
		)
	)
	)
)

# ╔═╡ b105085f-9de7-43e3-9210-c1db82be994e
let
	dat = DataFrame(
		Common = 1 .- vcat([10, 10, 10, 10, 9, 9, 9, 9], fill(8, 12)) ./ 10.,
		block = 1:20
	)
	

	f = Figure(size = (3.5, 2.) .* 72, figure_padding = (1, 2, 1, 10))
	
	mp = data(dat) * mapping(
		:block => "Block",
		:Common => "Noise"
	) * visual(BarPlot)

	plt = draw!(f[1,1], mp; axis = (; yticks = [0., 0.1, 0.2]))

	save("results/2025_rldm/PILT_shaping.pdf", f, pt_per_unit = 1)
	
	f
end

# ╔═╡ Cell order:
# ╠═95c436fe-446c-11f0-0460-d56b7f17790b
# ╠═11845224-c0aa-41f9-93aa-abe0ecfc6430
# ╠═b105085f-9de7-43e3-9210-c1db82be994e
