### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ 626673da-2397-4e50-97a9-bbf8c0da127a
begin
	cd("/home/jovyan/")
    import Pkg
    # activate the shared project environment
    Pkg.activate("relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()

	using AlgebraOfGraphics, DataFrames, CairoMakie

end

# ╔═╡ 104f5bf9-7179-489a-baa5-e0f6050e60b6
set_theme!(
	merge(
		theme_minimal(),
		Theme(
		font = "Helvetica",
		fontsize = 16,
		Axis = (
			xticklabelsize = 14,
			yticklabelsize = 14,
			spinewidth = 1.5,
			xtickwidth = 1.5,
			ytickwidth = 1.5
		)
	)
	)
)

# ╔═╡ 77589c65-e851-4bee-a01b-6eb9e1b870e8
let
	dat = DataFrame(
		Common = vcat([10, 10, 10, 10, 9, 9, 9, 9], fill(8, 12)),
		block = 1:20
	)
	
	dat.Confusing = 10 .- dat.Common

	dat = stack(
		dat,
		[:Common, :Confusing],
		:block
	)


	f = Figure(size = (26.21, 10.26) .* 72 ./ 2.54 ./ 2, figure_padding = 0)
	
	mp = data(dat) * mapping(
		:block,
		:value => "# trials",
		stack = :variable => "Feedback",
		color = :variable => "Feedback"
	) * visual(BarPlot)

	plt = draw!(f[1,1], mp)

	legend!(f[0,1], plt, orientation = :horizontal, titleposition = :left)

	save("results/workshop/PILT_shaping.png", f, pt_per_unit = 1)
	
	f
end

# ╔═╡ Cell order:
# ╠═626673da-2397-4e50-97a9-bbf8c0da127a
# ╠═104f5bf9-7179-489a-baa5-e0f6050e60b6
# ╠═77589c65-e851-4bee-a01b-6eb9e1b870e8
