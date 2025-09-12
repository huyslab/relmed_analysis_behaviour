### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ 93a9cf8e-d32c-11ef-18dc-53cc297a09d1
begin
	cd("/home/jovyan/")
    import Pkg
    # activate the shared project environment
    Pkg.activate("relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	using Random, DataFrames, JSON, CSV, StatsBase, JLD2, HTTP, CairoMakie, Printf, Distributions, CategoricalArrays, AlgebraOfGraphics, Dates, Turing, SHA, HypothesisTests, Tidier
	using LogExpFunctions: logistic, logit
	import OpenScienceFramework as OSF
	include("fetch_preprocess_data.jl")
	include("sample_utils.jl")
	include("plotting_utils.jl")
	include("stats_utils.jl")
	include("osf_utils.jl")
	include("model_utils.jl")
	include("PILT_models.jl")
	Turing.setprogress!(false)
end

# ╔═╡ 5f59cfe3-add9-4bd2-93c0-64feb09007ab
context = "poster" # or "abstract"

# ╔═╡ a557600f-d217-4f0d-b004-cb8b17e94471
begin
	# Set theme
	inter_bold = assetpath(pwd() * "/fonts/Inter/Inter-Bold.ttf")

	inch = 96
	pt = 4/3
	mm = inch / 25.4

	sizes = Dict(
		"abstract" => (46.5mm, 31.3mm),
		"poster" => (22.86, 11.42) .* inch ./ 2.54
	)


	
	th =  Dict()
	
	th["abstract"] = merge(theme_minimal(), Theme(
		font = "Helvetica",
		fontsize = 7pt,
		Axis = (
			xticklabelsize = 5pt,
			yticklabelsize = 5pt,
			spinewidth = 0.5pt,
			xlabelpadding = 0,
			ylabelpadding = 0
		)
	))

	th["poster"] = merge(
			theme_minimal(),
			Theme(
			font = "Helvetica",
			fontsize = 28pt,
			Axis = (
				xticklabelsize = 24pt,
				yticklabelsize = 24pt,
				spinewidth = 3pt,
				xtickwidth = 3pt,
				ytickwidth = 3pt
			)
		)
	)
	
	set_theme!(th[context])

end

# ╔═╡ b75e74aa-5e7d-40e7-9a06-fdc3dfd2fafd
begin
	# Load data
	PILT_data, _, _, _, _, _,
		_, _ = load_pilot6_data()
	nothing
end

# ╔═╡ 832ca25e-8ee5-45e8-a0f1-b22126a80bc1
PILT_data_clean = let
	# Clean data
	PILT_data_clean = exclude_PLT_sessions(PILT_data, required_n_blocks = 20)
	PILT_data_clean = filter(x -> x.response != "noresp", PILT_data_clean)
end

# ╔═╡ a0c56dbd-f328-4bdd-b803-e3b727a0b524
function reorder_bands_lines!(f::GridPosition)

	# Get plots
	plots = extract_axis(f).scene.plots

	# Select only plots of type Band or Lines
	plots = filter(p -> isa(p, Band) || isa(p, Lines), plots)

	n = length(plots)

	@assert allequal(typeof.(plots[1:(n ÷ 2)])) && allequal(typeof.(plots[(n ÷ 2 + 1):n])) "Expecting plots to be ordered by type"

	# Reorder bands and lines
	new_idx = vcat(1:2:n, 2:2:n)

	# Apply reordering via z-value
	for (i, p) in enumerate(plots)
		translate!(p, 0, 0, new_idx[i])
	end

end


# ╔═╡ 74d5d010-04e7-43d1-a55d-ee42de085c06
# Accuracy curveֿ by valence
let

	# Sumarrize by participant, valence, and trial
	acc_curve = combine(
		groupby(PILT_data_clean, [:prolific_pid, :valence, :trial]),
		:response_optimal => mean => :acc
	)

	sort!(acc_curve, [:prolific_pid, :valence, :trial])

	# Summarize by trial and valence
	acc_curve_sum = combine(
		groupby(acc_curve, [:valence, :trial]),
		:acc => mean => :acc,
		:acc => sem => :se
	)

	# Compute bounds
	insertcols!(
		acc_curve_sum,
		:lb => acc_curve_sum.acc .- acc_curve_sum.se,
		:ub => acc_curve_sum.acc .+ acc_curve_sum.se
	)

	# Labels for valence
	acc_curve_sum.val_lables = CategoricalArray(
		ifelse.(
			acc_curve_sum.valence .> 0,
			fill("Reward", nrow(acc_curve_sum)),
			fill("Punishment", nrow(acc_curve_sum))
		),
		levels = ["Reward", "Punishment"]
	)

	# Create plot mapping
	lws = Dict(
		"abstract" => 1pt,
		"poster" => 3pt
	)

	mp = data(acc_curve_sum) * (
	# Error band
		mapping(
		:trial => "Trial #",
		:lb => "Prop. optimal choice",
		:ub => "Prop. optimal choice",
		color = :val_lables => ""
	) * visual(Band, alpha = 0.5, legend = (; height = 1)) +
	# Average line	
		mapping(
		:trial => "Trial #",
		:acc => "Prop. optimal choice",
		color = :val_lables => ""
	) * visual(Lines, linewidth = lws[context])
	) + mapping([5.]) * visual(
		VLines, 
		color = :lightgrey, 
		linestyle = :dash,
		linewidth = lws[context]) +
	mapping([.5]) * visual(HLines, color = :lightgrey, linestyle = :dash,
		linewidth = lws[context])

	# Plot whole figure
	
	f1 = Figure(
		size = sizes[context],
		figure_padding = context == "abstract" ? 8 : 19
	)
	
	plt1 = draw!(f1[1,1], mp, scales(Color = (; palette = [
		colorant"#009ADE",  # Blue
		colorant"#FF1F5B" # Red
	  ])); 
		axis = (; xautolimitmargin = (0, 0)))

	ps = Dict(
		"abstract" => (7,7),
		"poster" => (20,20)
	)

	legend!(
		f1[1,1], 
		plt1,
		tellwidth = false,
		tellheight = false,
		framevisible = false,
		valign = 0.25,
		halign = 0.9,
		patchsize = ps[context]
	)

	# Fix order of layers
	reorder_bands_lines!(f1[1,1])

	# Save
	filepath = joinpath("results/rldm", "poster_PILT_acc_curve_valence.pdf")

	save(filepath, f1)
	
	f1
end

# ╔═╡ 2c410400-dd38-4a4e-a163-14c6654d1d36
# Tell fitting functions the column names
pilt_columns = Dict(
	"block" => :cblock,
	"trial" => :trial,
	"feedback_optimal" => :feedback_optimal,
	"feedback_suboptimal" => :feedback_suboptimal,
	"choice" => :response_optimal
)

# ╔═╡ 2d3bfc2a-009d-432d-8ff8-6fdebf8d7e44
# Prepare data for fit
function prepare_data(
	PILT_data_clean::DataFrame
)
	
	# Make sure sorted
	forfit = sort(PILT_data_clean, [:prolific_pid, :block, :trial])

	# Make sure block is Int64
	forfit.block = Int.(forfit.block)

	# Create feedback_optimal and feedback_suboptimal
	forfit.feedback_optimal = ifelse.(
		forfit.optimal_right .== 1,
		forfit.feedback_right,
		forfit.feedback_left
	)

	forfit.feedback_suboptimal = ifelse.(
		forfit.optimal_right .== 0,
		forfit.feedback_right,
		forfit.feedback_left
	)

	# Cumulative block number
	forfit.cblock = forfit.block .+ (parse.(Int, forfit.session) .- 1) .* maximum(forfit.block)

	# Split for reliability
	forfit.half = ifelse.(
		forfit.block .< median(unique(forfit.block)),
		fill(1, nrow(forfit)),
		fill(2, nrow(forfit))
	)

	sort!(forfit, [:prolific_pid, :cblock, :trial])

	return forfit

end

# ╔═╡ 7a8845f4-491e-4792-b4f7-4ca4e543f550
# Fit by session
fits_retest = let	
	fits = optimize_multiple_by_factor(
		prepare_data(PILT_data_clean);
		model = single_p_QL_recip,
		factor = :session,
		priors = Dict(
			:ρ => truncated(Normal(0, 5.), lower = 0.),
			:a => Normal(0, 2.)
		),
		unpack_function = unpack_single_p_QL,
		remap_columns = pilt_columns
	)
end

# ╔═╡ fabf50f6-6c73-444e-96d9-665da5f76bbe
function RLDM_reliability_scatter!(
		f::GridPosition;
		df::AbstractDataFrame,
		xlabel::AbstractString,
		ylabel::AbstractString,
		xcol::Symbol = :x,
		ycol::Symbol = :y,
		subtitle::AbstractString = "",
		tickformat::Union{Function, Makie.Automatic} = Makie.automatic,
		correct_r::Bool = true, # Whether to apply Spearman Brown
		markersize::Union{Int64,Float64} = 9,
		label_valign = 0.025
	)	

		# Compute correlation
		r = cor(df[!, xcol], df[!, ycol])
		
		# Spearman-Brown correction
		if correct_r
			r = spearman_brown(r)
		end

		# Text
		r_text = "n = $(nrow(df)),$(correct_r ? " SB" : "") r = $(round(r; digits = 2))"

		# Plot
		mp = data(df) *
				mapping(xcol, ycol) *
				(visual(Scatter; markersize = markersize, alpha = 0.75) + linear()) +
			mapping([0], [1]) *
				visual(ABLines, linestyle = :dash, color = :gray70)
		
		draw!(f, mp; axis=(;
			xlabel = xlabel, 
			ylabel = ylabel,
			xtickformat = tickformat,
			ytickformat = tickformat,
			subtitle = subtitle
		))

		if r > 0
			Label(
				f,
				r_text,
				fontsize = 24pt,
				font = :bold,
				halign = 0.975,
				valign = label_valign,
				tellheight = false,
				tellwidth = false
			)
		end

	end

# ╔═╡ 5ccf1422-d3a0-4c07-afec-cfc210398569
# Test retest of parameters
let
	f = Figure(
		size = sizes[context],
		figure_padding = (1, 11, 10, 5)
	)

	# Run over parameters
	for (i, (p, st, tf)) in enumerate(zip(
		[:a, :ρ], 
		["Learning rate", "Reward sensitivity"],
		[x -> string.(round.(a2α.(x), digits = 1)), Makie.automatic]
	))


		# Long to wide
		this_retest = unstack(
			fits_retest,
			:prolific_pid,
			:session,
			p,
			renamecols = (x -> "$(p)_$x")
		)

		# Plot
		RLDM_reliability_scatter!(
			f[1, i];
			df = dropmissing!(this_retest),
			xcol = Symbol("$(p)_1"),
			ycol = Symbol("$(p)_2"),
			xlabel = "Session 1",
			ylabel = ["Session 2", ""][i],
			subtitle = "$st",
			tickformat = tf,
			correct_r = false,
			markersize = context == "abstract" ? 2 : 9,
			label_valign= :top
		)

	end

	colgap!(f.layout, 10)

	save("results/rldm/$(context)_PILT_test_retest.pdf", f)
	
	f
end

# ╔═╡ Cell order:
# ╠═93a9cf8e-d32c-11ef-18dc-53cc297a09d1
# ╠═5f59cfe3-add9-4bd2-93c0-64feb09007ab
# ╠═a557600f-d217-4f0d-b004-cb8b17e94471
# ╠═b75e74aa-5e7d-40e7-9a06-fdc3dfd2fafd
# ╠═832ca25e-8ee5-45e8-a0f1-b22126a80bc1
# ╠═74d5d010-04e7-43d1-a55d-ee42de085c06
# ╠═a0c56dbd-f328-4bdd-b803-e3b727a0b524
# ╠═2c410400-dd38-4a4e-a163-14c6654d1d36
# ╠═7a8845f4-491e-4792-b4f7-4ca4e543f550
# ╠═5ccf1422-d3a0-4c07-afec-cfc210398569
# ╠═2d3bfc2a-009d-432d-8ff8-6fdebf8d7e44
# ╠═fabf50f6-6c73-444e-96d9-665da5f76bbe
