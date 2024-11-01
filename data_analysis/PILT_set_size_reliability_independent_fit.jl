### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ ede40f7a-96cb-11ef-24e5-aba8853e00f7
begin
	cd("/home/jovyan/")
    import Pkg
    # activate the shared project environment
    Pkg.activate("relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	using Random, DataFrames, JSON, CSV, StatsBase, JLD2, HTTP, CairoMakie, Printf, Distributions, CategoricalArrays, AlgebraOfGraphics, Dates, Turing
	using LogExpFunctions: logistic, logit
	import OpenScienceFramework as OSF
	include("fetch_preprocess_data.jl")
	include("sample_utils.jl")
	include("plotting_utils.jl")
	include("stats_utils.jl")
	include("osf_utils.jl")
	include("model_utils.jl")
	include("PILT_models.jl")
	nothing
end

# ╔═╡ d3bc3151-4b95-4644-a72e-6ad6f94a06b9
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

# ╔═╡ c69ff811-cfe5-43b1-9652-b822ca5bafd8
# Set up saving to OSF
begin
	osf_folder = "Lab notebook images/PILT/"
	proj = setup_osf("Task development")
end

# ╔═╡ d4442c94-6ecd-4f57-9e93-0235163197d1
begin
	pilot_4_1 , _, _, _ = load_pilot4x_data()
	filter!(x -> x.version == "4.1", pilot_4_1)
	pilot_4_1 = exclude_PLT_sessions(pilot_4_1, required_n_blocks = 18)
	filter!(x -> x.choice != "noresp", pilot_4_1)
end

# ╔═╡ 3c3149f8-c79a-459b-916a-bf7220ca848e
begin
	pilot_4 , _, _, _ = load_pilot4_data()
	filter!(x -> x.version == "4.0", pilot_4)
	pilot_4 = exclude_PLT_sessions(pilot_4, required_n_blocks = 18)
	filter!(x -> x.choice != "noresp", pilot_4)
end

# ╔═╡ e2d7c238-a6cc-4e8f-9e23-466955bbfd9d
remap_columns = Dict(
	"block" => :stimulus_pair_id,
	"trial" => :appearance,
	"feedback_optimal" => :feedback_optimal,
	"feedback_suboptimal" => :feedback_suboptimal,
	"choice" => :isOptimal
)

# ╔═╡ fbdc0f7c-a416-487d-9a87-3067bf3cbf53
# Auxillary variables
function prepare_data(
	PILT_data_clean::DataFrame
)
	# Make sure sorted
	sort!(PILT_data_clean, [:prolific_pid, :block, :trial])

	# Create appearance variable
	DataFrames.transform!(
		groupby(PILT_data_clean, [:prolific_pid, :block, :stimulus_pair]),
		:trial => (x -> 1:length(x)) => :appearance
	)

	# Create feedback_optimal and feedback_suboptimal
	PILT_data_clean.feedback_optimal = ifelse.(
		PILT_data_clean.optimalRight .== 1,
		PILT_data_clean.outcomeRight,
		PILT_data_clean.outcomeLeft
	)

	PILT_data_clean.feedback_suboptimal = ifelse.(
		PILT_data_clean.optimalRight .== 0,
		PILT_data_clean.outcomeRight,
		PILT_data_clean.outcomeLeft
	)

	# Split for reliability
	PILT_data_clean.half = ifelse.(
		PILT_data_clean.block .< median(unique(PILT_data_clean.block)),
		fill(1, nrow(PILT_data_clean)),
		fill(2, nrow(PILT_data_clean))
	)

	return PILT_data_clean

end

# ╔═╡ d2c1e2fb-766e-4e88-a6c6-8deea007b2aa
function fit_by_set_size(
	PILT_data_clean::DataFrame;
	model::Function,
	priors::Dict,
	unpack_function::Function,
	remap_columns::Dict
)

	fits = []

	set_sizes = sort(unique(PILT_data_clean.n_pairs))

	for s in set_sizes

		# Subset data
		gdf = filter(x -> x.n_pairs == s, PILT_data_clean)

		# Fit data
		fit = optimize_multiple(
			gdf;
			model = model,
			unpack_function = df -> unpack_function(df; columns = remap_columns),
		    priors = priors,
			grouping_col = :prolific_pid,
			n_starts = 10
		)

		# Add set size variable
		insertcols!(fit, :n_pairs => s)

		push!(fits, fit)
	end

	# Combine to single DataFrame
	fits = vcat(fits...)

	# Sort
	sort!(fits, [:n_pairs, :prolific_pid])

	return fits

end

# ╔═╡ f639f2c3-5382-4d5a-99fa-b8d8083e3d5c
function plot_fit_by_set_size(
	set_size_fits::DataFrame;
	model_name::String,
	exp_name::String
)
	params = Symbol.(names(set_size_fits[!, Not([:prolific_pid, :lp, :n_pairs])]))

	ptransform = [a2α, identity]

	f = Figure(size = (700, 600))

	for (i, p) in enumerate(params)

		mp1 = data(set_size_fits) * 
			mapping(
				:n_pairs => nonnumeric => "Set size",
				p => ptransform[i],
				color = :n_pairs => nonnumeric => "Set size"
			) *
			visual(RainClouds)

		draw!(f[1, i], mp1)

		colors = filter(x -> x.n_pairs == 1, set_size_fits)[!, [:prolific_pid, p]]
		rename!(colors, p => :color)

		forplot = innerjoin(set_size_fits, colors, on = :prolific_pid)

		mp2 = data(forplot) * 
			mapping(
				:n_pairs => nonnumeric => "Set size",
				p => ptransform[i],
				color = :color,
				group = :prolific_pid
			) *
			visual(ScatterLines)

		draw!(f[2, i], mp2)

	end

	filepath = "results/$(exp_name)_$(model_name)_by_set_size.png"

	save(filepath, f, pt_per_unit = 1)

	upload_to_osf(
		filepath,
		proj,
		osf_folder
	)

	f
end

# ╔═╡ ef149638-2d75-42dc-b337-5fadaf8e477d
function fit_split_by_set_size(
	PILT_data_clean::DataFrame;
	model::Function,
	priors::Dict,
	unpack_function::Function,
	remap_columns::Dict
) 

	fits = []

	set_sizes = sort(unique(PILT_data_clean.n_pairs))
	splits = sort(unique(PILT_data_clean.half))

	for sp in splits
		for s in set_sizes
	
			# Subset data
			gdf = filter(x -> (x.n_pairs == s) && (x.half == sp), PILT_data_clean)
	
			# Fit data
			fit = optimize_multiple(
				gdf;
				model = model,
				unpack_function = df -> unpack_function(df; columns = remap_columns),
			    priors = priors,
				grouping_col = :prolific_pid,
				n_starts = 10
			)
	
			# Add set size variable
			insertcols!(fit, 
				:n_pairs => s,
				:half => sp
			)
	
			push!(fits, fit)
		end
	end

	# Combine to single DataFrame
	fits = vcat(fits...)

	# Long to wide for each parameter
	params = Symbol.(names(fits[!, Not([:prolific_pid, :lp, :n_pairs, :half])]))
	set_size_pairs = [(a, b) for (i, a) in enumerate(set_sizes) 
		for b in set_sizes[(i+1):end]]
	
	wides = []
	for p in params
		wide = unstack(fits,
			[:prolific_pid, :half],
			:n_pairs,
			p;
			renamecols = x -> "$(p)_$x"
		)

		# Take differences
		insertcols!(
			wide,
			[Symbol("$(p)_$b - $a") => 
				wide[!, Symbol("$(p)_$b")] .- wide[!, Symbol("$(p)_$a")] 
				for (a,b) in set_size_pairs
			]...
		)
		
		push!(wides, wide)
	end

	# Join together
	if length(wides) > 1
		fits = innerjoin(
			wides..., 
			on = [:prolific_pid, :half]
		)
	else
		fits = wides[1]
	end

	return fits
	
end

# ╔═╡ e591efb9-1e37-4ee4-9941-f77b7316eccb
function plot_split_by_set_size(
	set_size_diff::DataFrame;
	exp_name::String,
	model_name::String
) 

	f = Figure(size = (700, 700))

	cols = filter(x -> x ∉ ["prolific_pid", "half"], names(set_size_diff))
	
	for (i, c) in enumerate(cols)

		wide = unstack(
			set_size_diff[!, [:prolific_pid, :half, Symbol(c)]],
			:prolific_pid,
			:half,
			c
		)

		mp = data(wide) * mapping(
			Symbol(1) => "First half", 
			Symbol(2) => "Second half"
		) * (visual(Scatter) + linear())


		draw!(
			f[div(i-1, 3) + 1, i - div(i-1, 3) * 3],
			mp,
			axis = (; subtitle = c)
		)

		r = cor(wide[!, Symbol(1)], wide[!, Symbol(2)])

		if r > 0
			Label(
				f[div(i-1, 3) + 1, i - div(i-1, 3) * 3],
				"r=$(round(spearman_brown(r), digits = 2))",
				color = :blue,
				tellheight = false,
				tellwidth = false,
				halign = 0.1,
				valign = 1.
			)
		end

	end

	filepath = "results/$(exp_name)_$(model_name)_reliability_by_set_size.png"

	save(filepath, f, pt_per_unit = 1)

	upload_to_osf(
		filepath,
		proj,
		osf_folder
	)

			
	f

end

# ╔═╡ b2d0789c-3e10-4c47-9ae1-2f6ad0633ec7
function set_size_effect_reliability(
	PILT_data_clean::DataFrame;
	model::Function,
	priors::Dict,
	unpack_function::Function,
	remap_columns::Dict,
	exp_name::String,
	model_name::String
)

	# Prepare data
	PILT_data_clean = prepare_data(PILT_data_clean)

	# Fit by set size
	set_size_fits = fit_by_set_size(
		PILT_data_clean;
		model = model,
		priors = priors,
		unpack_function = unpack_function,
		remap_columns = remap_columns
	)

	# Plot by set size
	f1 = plot_fit_by_set_size(
		set_size_fits;
		model_name = model_name,
		exp_name = exp_name
	)

	# Fit spilt to half by set size
	set_size_diff = fit_split_by_set_size(
		PILT_data_clean;
		model = model,
		priors = priors,
		unpack_function = unpack_function,
		remap_columns = remap_columns
	) 

	# Plot set size differences reliability
	f2 = plot_split_by_set_size(
		set_size_diff;
		model_name = model_name,
		exp_name = exp_name
	) 

	return f1, f2
end

# ╔═╡ 444e4a45-3c23-415a-84c5-9b1aeb8d8757
set_size_effect_reliability(
	pilot_4_1;
	model = single_p_QL,
	priors = Dict(
		:ρ => truncated(Normal(0., 5.), lower = 0.),
		:a => Normal(0., 2.)
	),
	unpack_function = unpack_single_p_QL,
	remap_columns = remap_columns,
	exp_name = "pilot_4.1",
	model_name = "Q_learning"
)

# ╔═╡ eb0db6b9-b5d9-449c-8881-ef4e7ebabfbe
set_size_effect_reliability(
	pilot_4_1;
	model = single_p_QL_recip,
	priors = Dict(
		:ρ => truncated(Normal(0., 5.), lower = 0.),
		:a => Normal(0., 2.)
	),
	unpack_function = unpack_single_p_QL,
	remap_columns = remap_columns,
	exp_name = "pilot_4.1",
	model_name = "Q_learning_recip"
)

# ╔═╡ ac8a775d-d73b-4b8b-bc76-85c70ef12b5b
set_size_effect_reliability(
	pilot_4;
	model = single_p_QL,
	priors = Dict(
		:ρ => truncated(Normal(0., 5.), lower = 0.),
		:a => Normal(0., 2.)
	),
	unpack_function = unpack_single_p_QL,
	remap_columns = remap_columns,
	exp_name = "pilot_4",
	model_name = "Q_learning"
)

# ╔═╡ c9f8f5f9-7807-48ba-b1a6-485ebf27f436
set_size_effect_reliability(
	pilot_4;
	model = single_p_QL,
	priors = Dict(
		:ρ => truncated(Normal(0., 5.), lower = 0.),
		:a => Normal(0., 2.)
	),
	unpack_function = unpack_single_p_QL,
	remap_columns = remap_columns,
	exp_name = "pilot_4",
	model_name = "Q_learning_recip"
)

# ╔═╡ Cell order:
# ╠═ede40f7a-96cb-11ef-24e5-aba8853e00f7
# ╠═d3bc3151-4b95-4644-a72e-6ad6f94a06b9
# ╠═c69ff811-cfe5-43b1-9652-b822ca5bafd8
# ╠═d4442c94-6ecd-4f57-9e93-0235163197d1
# ╠═3c3149f8-c79a-459b-916a-bf7220ca848e
# ╠═e2d7c238-a6cc-4e8f-9e23-466955bbfd9d
# ╠═444e4a45-3c23-415a-84c5-9b1aeb8d8757
# ╠═eb0db6b9-b5d9-449c-8881-ef4e7ebabfbe
# ╠═ac8a775d-d73b-4b8b-bc76-85c70ef12b5b
# ╠═c9f8f5f9-7807-48ba-b1a6-485ebf27f436
# ╠═b2d0789c-3e10-4c47-9ae1-2f6ad0633ec7
# ╠═fbdc0f7c-a416-487d-9a87-3067bf3cbf53
# ╠═d2c1e2fb-766e-4e88-a6c6-8deea007b2aa
# ╠═f639f2c3-5382-4d5a-99fa-b8d8083e3d5c
# ╠═ef149638-2d75-42dc-b337-5fadaf8e477d
# ╠═e591efb9-1e37-4ee4-9941-f77b7316eccb
