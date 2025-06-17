begin
	cd("/home/jovyan/")
    import Pkg
    # activate the shared project environment
    Pkg.activate("relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	using Random, DataFrames, JSON, CSV, StatsBase, JLD2, HTTP, CairoMakie, Printf, Distributions, CategoricalArrays, AlgebraOfGraphics, Dates, Turing, SHA, HypothesisTests, GLM, MixedModels
	using LogExpFunctions: logistic, logit
	import OpenScienceFramework as OSF
	include("$(pwd())/fetch_preprocess_data.jl")
	include("$(pwd())/sample_utils.jl")
	include("$(pwd())/plotting_utils.jl")
	include("$(pwd())/stats_utils.jl")
	include("$(pwd())/model_utils.jl")
	include("$(pwd())/PILT_models.jl")
	Turing.setprogress!(false)
end

context = "poster" # "abstract" or "poster"

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


begin
	# Load data
	_, _, _, _, _, WM_data, _, _ = load_pilot6_data()
	nothing
end

begin
    # Clean data
	WM_data_clean = exclude_PLT_sessions(WM_data, required_n_blocks = 10)
	WM_data_clean = filter(x -> x.response != "noresp", WM_data_clean)

	# Create appearance variable
	sort!(WM_data_clean, [:prolific_pid, :session, :block, :trial])
	
	DataFrames.transform!(
		groupby(WM_data_clean, [:prolific_pid, :exp_start_time, :session, :block, :stimulus_group]),
		:trial => (x -> 1:length(x)) => :appearance
	)

	# Summarize by appearance
	app_curve = combine(
		groupby(WM_data_clean, [:prolific_pid, :appearance, :n_groups]),
		:response_optimal => mean => :acc
	)

	# Summarize by apperance and n_groups
	app_curve_sum = combine(
		groupby(app_curve, [:appearance, :n_groups]),
		:acc => mean => :acc,
		:acc => sem => :se
	)

	# Compute bounds
	app_curve_sum.lb = app_curve_sum.acc .- app_curve_sum.se
	app_curve_sum.ub = app_curve_sum.acc .+ app_curve_sum.se

	# Sort
	sort!(app_curve_sum, [:n_groups, :appearance])

    # Create figure
	f = Figure(size = sizes[context],
    figure_padding = 8)

	# Create mapping
	lws = Dict(
		"abstract" => 1pt,
		"poster" => 3pt
	)

	mp2 = (data(app_curve_sum) * (
		mapping(
		:appearance => "Apperance #",
		:lb,
		:ub,
		group = :n_groups => nonnumeric => "Set size",
		color = :n_groups => nonnumeric => "Set size"
	) * visual(Band, alpha = 0.5) +
		mapping(
		:appearance => "Apperance #",
		:acc => "Prop. optimal choice",
		group = :n_groups => nonnumeric => "Set size",
		color = :n_groups => nonnumeric => "Set size"
	) * visual(Lines, linewidth = lws[context]))) +
	mapping([1/3]) * visual(HLines, color = :lightgray, linewidth = lws[context], linestyle = :dash)
	
	# Plot
	plt2 = draw!(f[1,1], mp2, scales(Color = (; palette = [
		colorant"#009ADE",  # Blue
		colorant"#FF1F5B" # Red
	  ])); 
		axis = (; ylabel = "Prop. optimal choice"),
	)

	ps = Dict(
		"abstract" => (7,7),
		"poster" => (20,20)
	)
    legend!(f[1,1], plt2, tellwidth = false, tellheight = false, halign = :right, valign = 0.15, 
        orientation = :horizontal, framevisible = false, titleposition = :left, patchsize = ps[context])

	save("results/rldm/$(context)_wm_acc.pdf", f)

	f
end

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

begin

    retest_df = DataFrame(CSV.File("results/rldm/wm_test_retest_df.csv"))

    dropmissing!(retest_df)

	fig=Figure(; 
		size = sizes[context],
        figure_padding = (0, 11.5, 10, 5)
	)

	ms = Dict(
		"abstract" => 2,
		"poster" => 9
	)

    RLDM_reliability_scatter!(
		fig[1,1]; df=retest_df, xlabel="Session 1", ylabel="Session 2", 
		xcol=:a_pos, ycol=:a_pos_1, subtitle="QL learning rate", markersize = ms[context], correct_r = false
	)
	RLDM_reliability_scatter!(
		fig[1,2]; df=retest_df, xlabel="Session 1", ylabel="Session 2", 
		xcol=:w0, ycol=:w0_1, subtitle="WM weighting", markersize = ms[context], correct_r = false
	)

    save("results/rldm/$(context)_wm_retest.pdf", fig)

	fig

end