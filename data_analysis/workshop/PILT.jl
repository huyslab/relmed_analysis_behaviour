### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ 8cf30b5e-a020-11ef-23b2-2da6e9116b54
begin
	cd("/home/jovyan/")
    import Pkg
    # activate the shared project environment
    Pkg.activate("relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	using Random, DataFrames, JSON, CSV, StatsBase, JLD2, HTTP, CairoMakie, Printf, Distributions, CategoricalArrays, AlgebraOfGraphics, Dates, Turing, SHA
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

# ╔═╡ 82ef300e-536f-40ce-9cde-72056e6f4b5e
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

# ╔═╡ 595c642e-32df-448e-81cc-6934e2152d70
# Set up saving to OSF
begin
	osf_folder = "Workshop figures/PILT"
	proj = setup_osf("Task development")
end

# ╔═╡ 14a292db-43d4-45d8-97a5-37ffc03bdc5c
begin
	# Load data
	PILT_data, _, _, _, _, _,
		_, _ = load_pilot6_data()
	nothing
end

# ╔═╡ 6ed82686-35ab-4afd-a1b2-6fa19ae67168
PILT_data_clean = let
	# Clean data
	PILT_data_clean = exclude_PLT_sessions(PILT_data, required_n_blocks = 20)
	PILT_data_clean = filter(x -> x.response != "noresp", PILT_data_clean)
end

# ╔═╡ b5b75f4e-7b91-4287-a409-6f0ebdf20f4e
# Accuracy curveֿ
let

	# Sumarrize by participant and trial
	acc_curve = combine(
		groupby(PILT_data_clean, [:prolific_pid, :trial]),
		:response_optimal => mean => :acc
	)

	sort!(acc_curve, [:prolific_pid, :trial])

	# Summarize by trial
	acc_curve_sum = combine(
		groupby(acc_curve, :trial),
		:acc => mean => :acc,
		:acc => sem => :se
	)

	# Compute bounds
	insertcols!(
		acc_curve_sum,
		:lb => acc_curve_sum.acc .- acc_curve_sum.se,
		:ub => acc_curve_sum.acc .+ acc_curve_sum.se
	)

	# Create plot mapping
	mp = (
	# Error band
		mapping(
		:trial => "Trial #",
		:lb => "Prop. optimal choice",
		:ub => "Prop. optimal choice"
	) * visual(Band) +
	# Average line	
		mapping(
		:trial => "Trial #",
		:acc => "Prop. optimal choice"
	) * visual(Lines, linewidth = 4)
	)

	# Mathcing line
	matching = mapping([0.86]) * visual(HLines, linestyle = :dash)

	# Plot whole data
	f1 = Figure()
	
	draw!(f1, data(acc_curve_sum) * mp + matching;
		axis = (; xautolimitmargin = (0, 0))
	)

	# Plot up to trial 5
	f2 = Figure()
	
	draw!(
		f2, 
		data(filter(x -> x.trial <= 5, acc_curve_sum)) * mp;
		axis = (; xautolimitmargin = (0, 0))
	)


	# Plot with matching level
	f3 = Figure()
	
	draw!(
		f3, 
		data(filter(x -> x.trial <= 5, acc_curve_sum)) * 
		mp + 
		matching;
		axis = (; xautolimitmargin = (0, 0))
	)

	# Link axes
	linkaxes!(contents(f1[1,1])[1], contents(f2[1,1])[1], contents(f3[1,1])[1])

	# Save
	filepaths = [joinpath("results/workshop", "PILT_acc_curve_$k.png") for k in ["full", "partial", "partial_line"]]

	save.(filepaths, [f1, f2, f3])

	for fp in filepaths
		upload_to_osf(
			fp,
			proj,
			osf_folder
		)
	end

	f1, f2, f3
	
end

# ╔═╡ 18b19cd7-8af8-44ad-8b92-d40a2cfff8b4
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
	) * visual(Band, alpha = 0.5) +
	# Average line	
		mapping(
		:trial => "Trial #",
		:acc => "Prop. optimal choice",
		color = :val_lables => ""
	) * visual(Lines, linewidth = 4)
	)

	# Plot whole figure
	f1 = Figure()
	
	plt1 = draw!(f1[1,1], data(acc_curve_sum) * mp; 
		axis = (; xautolimitmargin = (0, 0)))

	legend!(
		f1[0,1], 
		plt1,
		tellwidth = false,
		framevisible = false,
		orientation = :horizontal
	)

	# Fix order of layers
	reorder_bands_lines!(f1[1,1])

	# Plot first five trials
	f2 = Figure()
	
	draw!(f2[1,1], data(filter(x -> x.trial <= 5, acc_curve_sum)) * mp; 
		axis = (; xautolimitmargin = (0, 0)))

	legend!(
		f2[0,1], 
		plt1,
		tellwidth = false,
		framevisible = false,
		orientation = :horizontal
	)


	# Fix order of layers
	reorder_bands_lines!(f2[1,1])

	
	# Link axes
	linkaxes!(extract_axis(f1[1,1]), extract_axis(f2[1,1]))

	# Save
	filepaths = [joinpath("results/workshop", "PILT_acc_curve_valence_$k.png") for k in ["full", "partial"]]

	save.(filepaths, [f1, f2])

	for fp in filepaths
		upload_to_osf(
			fp,
			proj,
			osf_folder
		)
	end

	f1, f2
	
end

# ╔═╡ d26f4afb-d734-40af-97aa-9604db2a335a
function fit_by_factor(
	PILT_data_clean::DataFrame;
	model::Function,
	factor::Symbol,
	priors::Dict,
	unpack_function::Function,
	remap_columns::Dict
)

	fits = []

	levels = sort(unique(PILT_data_clean[!, factor]))

	for l in levels

		# Subset data
		gdf = filter(x -> x[factor] == l, PILT_data_clean)

		# Fit data
		fit = optimize_multiple(
			gdf;
			model = model,
			unpack_function = df -> unpack_function(df; columns = remap_columns),
		    priors = priors,
			grouping_col = :prolific_pid,
			n_starts = 10
		)

		# Add factor variable
		insertcols!(fit, factor => l)

		push!(fits, fit)
	end

	# Combine to single DataFrame
	fits = vcat(fits...)

	# Sort
	sort!(fits, [factor, :prolific_pid])

	return fits

end

# ╔═╡ Cell order:
# ╠═8cf30b5e-a020-11ef-23b2-2da6e9116b54
# ╠═82ef300e-536f-40ce-9cde-72056e6f4b5e
# ╠═595c642e-32df-448e-81cc-6934e2152d70
# ╠═14a292db-43d4-45d8-97a5-37ffc03bdc5c
# ╠═6ed82686-35ab-4afd-a1b2-6fa19ae67168
# ╠═b5b75f4e-7b91-4287-a409-6f0ebdf20f4e
# ╠═18b19cd7-8af8-44ad-8b92-d40a2cfff8b4
# ╠═d26f4afb-d734-40af-97aa-9604db2a335a
