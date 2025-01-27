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

# ╔═╡ a557600f-d217-4f0d-b004-cb8b17e94471
begin
	# Set theme
	inter_bold = assetpath(pwd() * "/fonts/Inter/Inter-Bold.ttf")

	inch = 96
	pt = 4/3
	mm = inch / 25.4

	
	th = merge(theme_minimal(), Theme(
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
	set_theme!(th)

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
	mp = (
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
	) * visual(Lines, linewidth = 1)
	)

	# Plot whole figure
	f1 = Figure(
		size = (46.5mm, 31.3mm),
		figure_padding = 8
	)
	
	plt1 = draw!(f1[1,1], data(acc_curve_sum) * mp; 
		axis = (; xautolimitmargin = (0, 0)))

	legend!(
		f1[1,1], 
		plt1,
		tellwidth = false,
		tellheight = false,
		framevisible = false,
		valign = 0.1,
		halign = 0.3,
		patchsize = (7, 7)
	)

	# Fix order of layers
	reorder_bands_lines!(f1[1,1])

	# Save
	filepath = joinpath("results/rldm", "PILT_acc_curve_valence.pdf")

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
	markersize::Int64 = 4pt,
	label_halign::Union{Float64, Symbol} = 0.975,
	label_valign::Union{Float64, Symbol} = 0.025,
	label_fontsize = 5pt
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
			(visual(Scatter; markersize = markersize) + linear()) +
		mapping([0], [1]) *
			visual(ABLines, linestyle = :dash, color = :gray70)
	
	draw!(f, mp; axis=(;
		xlabel = xlabel, 
		ylabel = ylabel,
        xticklabelsize = 5pt,
        yticklabelsize = 5pt,
		xtickformat = tickformat,
		ytickformat = tickformat,
		subtitle = subtitle,
		ylabelpadding = 0,
		xlabelpadding = 1
	))

	if r > 0
		Label(
			f,
			r_text,
			fontsize = label_fontsize,
			font = :bold,
			halign = label_halign,
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
		size = (46.5mm, 31.3mm),
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
			markersize = 2,
			label_valign = :top
		)

	end

	colgap!(f.layout, 10)

	save("results/rldm/PILT_test_retest.pdf", f)
	
	f
end

# ╔═╡ Cell order:
# ╠═93a9cf8e-d32c-11ef-18dc-53cc297a09d1
# ╠═a557600f-d217-4f0d-b004-cb8b17e94471
# ╠═b75e74aa-5e7d-40e7-9a06-fdc3dfd2fafd
# ╠═832ca25e-8ee5-45e8-a0f1-b22126a80bc1
# ╠═74d5d010-04e7-43d1-a55d-ee42de085c06
# ╠═2c410400-dd38-4a4e-a163-14c6654d1d36
# ╠═7a8845f4-491e-4792-b4f7-4ca4e543f550
# ╠═5ccf1422-d3a0-4c07-afec-cfc210398569
# ╠═2d3bfc2a-009d-432d-8ff8-6fdebf8d7e44
# ╠═fabf50f6-6c73-444e-96d9-665da5f76bbe
