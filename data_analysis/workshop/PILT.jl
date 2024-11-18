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
	using Random, DataFrames, JSON, CSV, StatsBase, JLD2, HTTP, CairoMakie, Printf, Distributions, CategoricalArrays, AlgebraOfGraphics, Dates, Turing, SHA, HypothesisTests
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

	# for fp in filepaths
	# 	upload_to_osf(
	# 		fp,
	# 		proj,
	# 		osf_folder
	# 	)
	# end

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

	# for fp in filepaths
	# 	upload_to_osf(
	# 		fp,
	# 		proj,
	# 		osf_folder
	# 	)
	# end

	f1, f2
	
end

# ╔═╡ 6ea0b5b3-d3b0-47c6-a8a2-b2c82200e7b0
# Acc test-retes
f_acc_retest, acc_sum = let

	acc_sum = combine(
		groupby(PILT_data_clean, [:prolific_pid, :session]),
		:response_optimal => mean => :response_optimal
	)

	# Long to wide
	acc_sum_wide = unstack(
		acc_sum,
		:prolific_pid,
		:session,
		:response_optimal
	)

	f = Figure()

	# Plot
	workshop_reliability_scatter!(
		f[1, 1];
		df = dropmissing!(acc_sum_wide),
		xcol = Symbol("1"),
		ycol = Symbol("2"),
		xlabel = "Session 1",
		ylabel = "Session 2",
		subtitle = "Proportion of optimal choices"
	)

	filepath = "results/workshop/PILT_acc_test_retest.png"
	save(filepath, f, pt_per_unit = 1)

	#  upload_to_osf(
	# 		filepath,
	# 		proj,
	# 		osf_folder
	# )

	f, acc_sum

end

# ╔═╡ c40ea9ef-0d50-4889-a28a-778a14b0dec7
# Tell fitting functions the column names
pilt_columns = Dict(
	"block" => :cblock,
	"trial" => :trial,
	"feedback_optimal" => :feedback_optimal,
	"feedback_suboptimal" => :feedback_suboptimal,
	"choice" => :response_optimal
)

# ╔═╡ a9f674dc-6cb8-43a8-82fd-6faa8a5e8396
priors = Dict(
	:ρ => truncated(Normal(0., 5.), lower = 0.),
	:a => Normal(0., 2.)
)

# ╔═╡ 7b1a8fcf-66f5-4c5a-a991-e3007819675c
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

# ╔═╡ ee0d1649-9917-4393-a300-ec48cc47360f
# Acc splithalf
let

	fs = []
	for s in unique(PILT_data_clean.session)

		acc_sum = combine(
			groupby(
				filter(x -> x.session == s, prepare_data(PILT_data_clean)), 
				[:prolific_pid, :half]
			),
			:response_optimal => mean => :response_optimal
		)
	
		# Long to wide
		acc_sum_wide = unstack(
			acc_sum,
			:prolific_pid,
			:half,
			:response_optimal
		)
	
		f = Figure()
	
		# Plot
		workshop_reliability_scatter!(
			f[1, 1];
			df = dropmissing!(acc_sum_wide),
			xcol = Symbol("1"),
			ycol = Symbol("2"),
			xlabel = "First half",
			ylabel = "Second half",
			subtitle = "Session $(s) proportion of optimal choices"
		)

		# Save plot
		filepath = "results/workshop/PILT_acc_sess$(s)_splithalf.png"
		save(filepath, f, pt_per_unit = 1)

		#  upload_to_osf(
		# 		filepath,
		# 		proj,
		# 		osf_folder
		# )

	
		push!(fs, f)

	end

	fs

end

# ╔═╡ a5b29872-3854-4566-887f-35d6e53479f6
fits_by_valence = let	
	# Fit data
	fits = optimize_multiple_by_factor(
		prepare_data(PILT_data_clean);
		model = single_p_QL_recip,
		factor = :valence,
		priors = priors,
		unpack_function = unpack_single_p_QL,
		remap_columns = pilt_columns
	)
end

# ╔═╡ 4b9732c4-e74e-4775-8ff0-531be8576c30
# Plot parameters by valence
let fits = fits_by_valence
	# Perform t tests
	pvals = []
	for p in [:ρ, :a]
	
		ρ_wide = unstack(
			fits,
			:prolific_pid,
			:valence,
			p,
			renamecols = (x -> "valence_$x")
		) |> disallowmissing

		ttest = OneSampleTTest(ρ_wide.valence_1, ρ_wide[!, Symbol("valence_-1")])

		@info "$p: $ttest"
		push!(pvals, 
			pvalue(ttest)
		)
	end

	# Translate p_value to stars
	star_pval(p) = ["***", "**", "*", "ns."][findfirst(p .< [0.001, 0.01, 0.05, 1.])]

	# Tranform a to α
	fits.α = a2α.(fits.a)

	# Labels for valence
	fits.val_lables = CategoricalArray(
		ifelse.(
			fits.valence .> 0,
			fill("Reward", nrow(fits)),
			fill("Punishment", nrow(fits))
		),
		levels = ["Reward", "Punishment"]
	)

	# Plot
	f1 = Figure()

	for (i, (p, l)) in 
		enumerate(zip([:ρ, :α], ["Reward sensitivity", "Learning rate"]))
		
		mp = data(fits) * 
		(mapping(
			:valence => (x -> sign.(x) .* (abs.(x) .- 0.2)) => "",
			p => l,
			group = :prolific_pid
		) * visual(Lines, color = :grey, alpha = 0.1) +
		mapping(
			:valence => (x -> sign.(x) .* (abs.(x) .- 0.2)) => "",
			p => l,
			color = :valence => (x -> nonnumeric(sign.(x) .* (abs.(x) .- 0.2))) => "",
			group = :prolific_pid
		) * visual(Scatter, alpha = 0.3, markersize = 6) +
		mapping(
			:valence => (x -> sign.(x) .* (abs.(x) .+ 0.2)) => "",
			p => l,
			color = :valence => (x -> nonnumeric(sign.(x) .* (abs.(x) .- 0.2))) => "",
		) * visual(BoxPlot, width = 0.5, show_notch = true, outliercolor = :white))
	
	
		draw!(f1[1, i], mp,
			scales(Color = (; palette = Makie.wong_colors()[[2,1]])); 
			axis = (; 
				xticks = ([-1, 1],  ["Punishment", "Reward"]), 
				xreversed = true,
				subtitle = star_pval(pvals[i])
			)
		)
	end

	# Save
	filepath = "results/workshop/PILT_valence_parameters.png"

	save(filepath, f1)

	# upload_to_osf(
	# 		filepath,
	# 		proj,
	# 		osf_folder
	# )

	f1
end

# ╔═╡ d4ee7c24-5d83-4e07-acca-13006ae4278a
# Fit by halves
fits_splithalf = let	
	fits = Dict(s => optimize_multiple_by_factor(
		prepare_data(filter(x -> x.session == s, PILT_data_clean));
		model = single_p_QL_recip,
		factor = :half,
		priors = priors,
		unpack_function = unpack_single_p_QL,
		remap_columns = pilt_columns
	) for s in unique(PILT_data_clean.session))
end

# ╔═╡ ad2b69d5-9582-4cf3-ab9b-e6c3463f1435
# Split half reliability of parameters
let
	fs = []

	# Run over sessions and parameters
	for (s, fits) in fits_splithalf

		for (p, st, tf) in zip(
			[:a, :ρ], 
			["learning rate", "reward sensitivity"],
			[x -> string.(round.(a2α.(x), digits = 2)), Makie.automatic]
		)

			f = Figure()

			# Long to wide
			splithalf = unstack(
				fits_splithalf[s],
				:prolific_pid,
				:half,
				p,
				renamecols = (x -> "$(p)_$x")
			)

			# Plot
			workshop_reliability_scatter!(
				f[1, 1];
				df = splithalf,
				xcol = Symbol("$(p)_1"),
				ycol = Symbol("$(p)_2"),
				xlabel = "First half",
				ylabel = "Second half",
				subtitle = "Session $s $st",
				tickformat = tf
			)

			# Save
			filepath = "results/workshop/PILT_sess$(s)_$(string(p))_split_half.png"
			save(filepath, f)

			# Push for plotting in notebook
			push!(fs, f)
		end
	end

	fs
end

# ╔═╡ 47046875-475e-4ff1-b626-7bc285f0aac7
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

# ╔═╡ 6e965be9-5e8e-43ed-a711-c5845705bdc3
# Test retest of parameters
let
	fs = []

	# Run over parameters
	for (p, st, tf) in zip(
		[:a, :ρ], 
		["Learning rate", "Reward Sensitivity"],
		[x -> string.(round.(a2α.(x), digits = 2)), Makie.automatic]
	)

		f = Figure()

		# Long to wide
		this_retest = unstack(
			fits_retest,
			:prolific_pid,
			:session,
			p,
			renamecols = (x -> "$(p)_$x")
		)

		# Plot
		workshop_reliability_scatter!(
			f[1, 1];
			df = dropmissing!(this_retest),
			xcol = Symbol("$(p)_1"),
			ycol = Symbol("$(p)_2"),
			xlabel = "First session",
			ylabel = "Second session",
			subtitle = st,
			tickformat = tf,
			correct_r = false
		)

		# Save
		filepath = "results/workshop/PILT_$(string(p))_test_retest.png"
		save(filepath, f)

		# Push for plotting in notebook
		push!(fs, f)
	end
	fs
end

# ╔═╡ 0b3de2ef-84a7-4cf3-8aff-e587190060e1
# Save params file
let
	params = outerjoin(
		select(
			acc_sum,
			:prolific_pid,
			:session => (x -> parse.(Int, x)) => :session,
			:response_optimal => :PILT_prop_optimal_chosen
		),
		select(
			fits_retest,
			:prolific_pid,
			:session => (x -> parse.(Int, x)) => :session,
			:a => :PILT_learning_rate_unconstrained,
			:ρ => :PILT_reward_sensitivity
		),
		on = [:prolific_pid, :session]
	)

	CSV.write("results/workshop/PILT_params.csv", params)

	params

end

# ╔═╡ c8a9802b-a1db-47d8-9719-89f1eadd11f7
# Fit by valence and half
fits_splithalf_valence = let
	# Prepare data for fit
	forfit = prepare_data(PILT_data_clean)

	# Split half by valence
	DataFrames.transform!(
		groupby(forfit, [:prolific_pid, :session, :valence]),
		:cblock => (x -> ifelse.(
			x .< median(unique(x)), 
			fill(1, length(x)), 
			fill(2, length(x))
		)) => :half
	)

	@assert nrow(unique(forfit[!, [:cblock, :valence, :half]])) == maximum(forfit.cblock) "Problem with splitting to two halves"

	# Sort for fitting
	sort!(forfit, [:prolific_pid, :cblock, :trial])

	fits = optimize_multiple_by_factor(
		forfit;
		model = single_p_QL_recip,
		factor = [:session, :half, :valence],
		priors = priors,
		unpack_function = unpack_single_p_QL,
		remap_columns = pilt_columns
	)
end

# ╔═╡ f8e79974-c3a2-46f6-a760-f558733d9226
# Plot valence splithalf
let 

	fs = []
	# Run over sessions and parameters
	for s in sort(unique(fits_splithalf_valence.session))

		for (p, st, tf) in zip(
			[:a, :ρ], 
			["learning rate", "reward sensitivity"],
			[x -> string.(round.(a2α.(x), digits = 2)), Makie.automatic]
		)
			# Long to wide
			fp = unstack(
				filter(x -> x.session == s, fits_splithalf_valence),
				[:prolific_pid, :half],
				:valence,
				p,
				renamecols = x -> "$(p)_$x"
			)
		
			# Compute difference
			fp.diff = fp[!, Symbol("$(p)_1")] .- fp[!, Symbol("$(p)_-1")]
		
			# Long to wide again
			fp = unstack(
				fp,
				:prolific_pid,
				:half,
				:diff,
				renamecols = x -> "$(diff)_$x"
			) 
		
			# Plot
			f = Figure()
			
			workshop_reliability_scatter!(
				f[1, 1];
				df = fp,
				xcol = Symbol("$(diff)_1"),
				ycol = Symbol("$(diff)_2"),
				xlabel = "First half",
				ylabel = "Second half",
				subtitle = "Session $s $st reward - punishment"
			)

			# Save
			filepath = "results/workshop/PILT_sess$(s)_$(string(p))_valence_diff_split_half.png"
			save(filepath, f)

			# Push for plotting in notebook
			push!(fs, f)
	
		end
	end

	fs

end

# ╔═╡ ac8a4d61-ba61-4635-96fa-4e6ee9769e5e
# Fit by valence and session
fits_retest_valence = let
	# Prepare data for fit
	forfit = prepare_data(PILT_data_clean)

	# Sort for fitting
	sort!(forfit, [:prolific_pid, :cblock, :trial])

	fits = optimize_multiple_by_factor(
		forfit;
		model = single_p_QL,
		factor = [:session, :valence],
		priors = priors,
		unpack_function = unpack_single_p_QL,
		remap_columns = pilt_columns
	)
end

# ╔═╡ 6f48c845-c2ec-4fd9-ad5b-eb5d7ba49a45
# Plot valence test_retest
let 

	fs = []
	for (p, st, tf) in zip(
		[:a, :ρ], 
		["Learning rate", "Reward sensitivity"],
		[x -> string.(round.(a2α.(x), digits = 2)), Makie.automatic]
	)
		# Long to wide
		fp = unstack(
			fits_retest_valence,
			[:prolific_pid, :session],
			:valence,
			p,
			renamecols = x -> "$(p)_$x"
		)
	
		# Compute difference
		fp.diff = fp[!, Symbol("$(p)_1")] .- fp[!, Symbol("$(p)_-1")]
	
		# Long to wide again
		fp = unstack(
			fp,
			:prolific_pid,
			:session,
			:diff,
			renamecols = x -> "$(diff)_$x"
		) 
	
		# Plot
		f = Figure()
		
		workshop_reliability_scatter!(
			f[1, 1];
			df = dropmissing!(fp),
			xcol = Symbol("$(diff)_1"),
			ycol = Symbol("$(diff)_2"),
			xlabel = "Session 1",
			ylabel = "Session 2",
			subtitle = "$st reward - punishment"
		)

		# Save
		filepath = "results/workshop/PILT_$(string(p))_valence_diff_test_retest.png"
		save(filepath, f)

		# Push for plotting in notebook
		push!(fs, f)

	end
	fs

end

# ╔═╡ Cell order:
# ╠═8cf30b5e-a020-11ef-23b2-2da6e9116b54
# ╠═82ef300e-536f-40ce-9cde-72056e6f4b5e
# ╠═595c642e-32df-448e-81cc-6934e2152d70
# ╠═14a292db-43d4-45d8-97a5-37ffc03bdc5c
# ╠═6ed82686-35ab-4afd-a1b2-6fa19ae67168
# ╟─b5b75f4e-7b91-4287-a409-6f0ebdf20f4e
# ╟─18b19cd7-8af8-44ad-8b92-d40a2cfff8b4
# ╠═ee0d1649-9917-4393-a300-ec48cc47360f
# ╠═6ea0b5b3-d3b0-47c6-a8a2-b2c82200e7b0
# ╠═c40ea9ef-0d50-4889-a28a-778a14b0dec7
# ╠═a9f674dc-6cb8-43a8-82fd-6faa8a5e8396
# ╠═a5b29872-3854-4566-887f-35d6e53479f6
# ╠═4b9732c4-e74e-4775-8ff0-531be8576c30
# ╠═d4ee7c24-5d83-4e07-acca-13006ae4278a
# ╠═ad2b69d5-9582-4cf3-ab9b-e6c3463f1435
# ╠═47046875-475e-4ff1-b626-7bc285f0aac7
# ╠═6e965be9-5e8e-43ed-a711-c5845705bdc3
# ╠═c8a9802b-a1db-47d8-9719-89f1eadd11f7
# ╠═f8e79974-c3a2-46f6-a760-f558733d9226
# ╠═ac8a4d61-ba61-4635-96fa-4e6ee9769e5e
# ╠═6f48c845-c2ec-4fd9-ad5b-eb5d7ba49a45
# ╠═0b3de2ef-84a7-4cf3-8aff-e587190060e1
# ╠═7b1a8fcf-66f5-4c5a-a991-e3007819675c
