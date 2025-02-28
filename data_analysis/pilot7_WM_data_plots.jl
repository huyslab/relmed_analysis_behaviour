### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ fb93a03b-8ae6-4395-bae3-4183fbe45cc5
begin
	cd("/home/jovyan/")
    import Pkg
    # activate the shared project environment
    Pkg.activate("relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	using Random, DataFrames, JSON, CSV, StatsBase, JLD2, HTTP, CairoMakie, Printf, Distributions, CategoricalArrays, AlgebraOfGraphics, Dates
	using LogExpFunctions: logistic, logit
	using Tidier
	include("fetch_preprocess_data.jl")
	include("sample_utils.jl")
	include("plotting_utils.jl")
	nothing
end

# ╔═╡ b4067011-f52b-4da7-a278-063d8743bcaf
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

# ╔═╡ 86ac0107-f0cd-46ad-94f6-219638c80711
begin
	_, test_data, _, _, _, WM_data, _, _ = load_pilot7_data(; force_download = false, return_version = "7.0")
end

# ╔═╡ f081f2d7-bef3-476c-ad14-ebc4f0a35e77
md"""## Test Performance"""

# ╔═╡ 6ab2f6d7-186d-435d-9bc1-a50b079e4eda
filter(x -> x.block == 6, test_data)

# ╔═╡ 01255688-ee82-4f77-95ba-5fb2c1d5a717
	function equi_groups(x::AbstractVector; n::Int = 3, labels = ["Early", "Mid", "Late"])
	    min_x, max_x = extrema(x)
	    edges = range(min_x, max_x, length=n+1)[2:end]  # Define n equal-width bins
	    return [findfirst(v ≤ edge for edge in edges) === nothing ? labels[end] : labels[findfirst(v ≤ edge for edge in edges)] for v in x]
	end


# ╔═╡ ff992cf9-e256-4ebc-a661-e13780cb99a0
# Recoding function
recoder = (x, edges, labels) -> ([findfirst(v ≤ edge for edge in edges) === nothing ? labels[end] : labels[findfirst(v ≤ edge for edge in edges)] for v in x])

# ╔═╡ 8f52df1d-598a-43e3-bd07-2e2bdf458c93
function compute_delays(vec::AbstractVector)
    last_seen = Dict{Any, Int}()
    delays = zeros(Int, length(vec))

    for (i, val) in enumerate(vec)
        delays[i] = haskey(last_seen, val) ? (i - last_seen[val]) : 0
        last_seen[val] = i
    end

    return delays
end

# ╔═╡ a2216e2d-7890-4c23-90dc-3ad8a07e6485
WM_data_clean = let
	# Clean data
	WM_data_clean = exclude_PLT_sessions(WM_data, required_n_blocks = 1)

	# Sort
	sort!(
		WM_data_clean,
		[:prolific_pid, :session, :block, :trial]
	)

	# Apperance number
	transform!(
		groupby(WM_data_clean, [:prolific_pid, :exp_start_time, :session, :block, :stimulus_group]),
		:trial => (x -> 1:length(x)) => :appearance
	)

	# Compute delays
	DataFrames.transform!(
		groupby(
			WM_data_clean,
			:prolific_pid
		),
		:stimulus_group => compute_delays => :delay,
	) 

	WM_data_clean = filter(x -> x.response != "noresp", WM_data_clean)

	# Previous correct
	DataFrames.transform!(
		groupby(
			WM_data_clean,
			[:prolific_pid, :stimulus_group]
		),
		:response_optimal => lag => :previous_optimal,
	)

	

end

# ╔═╡ e7e5c7c9-4038-4c8c-876c-d30104445c02
let df = WM_data_clean

	# Sumarrize by participant, trial, n_groups
	acc_curve = combine(
		groupby(df, [:prolific_pid, :trial]),
		:response_optimal => mean => :acc
	)

	# Summarize by trial and n_groups
	acc_curve_sum = combine(
		groupby(acc_curve, [:trial]),
		:acc => mean => :acc,
		:acc => sem => :se
	)

	# Compute bounds
	acc_curve_sum.lb = acc_curve_sum.acc .- acc_curve_sum.se
	acc_curve_sum.ub = acc_curve_sum.acc .+ acc_curve_sum.se

	# Sort
	sort!(acc_curve_sum, [:trial])

	# Create figure
	f = Figure(size = (700, 350))

	# Create mapping
	mp1 = (data(acc_curve_sum) * (
		mapping(
		:trial => "Trial #",
		:lb,
		:ub
	) * visual(Band, alpha = 0.5) +
		mapping(
		:trial => "Trial #",
		:acc => "Prop. optimal choice"
	) * visual(Lines)))
	
	# Plot
	plt1 = draw!(f[1,1], mp1, axis = (; ylabel = "Prop. optimal choice"))
	legend!(f[0,1:2], plt1, tellwidth = false, halign = 0.5, orientation = :horizontal, framevisible = false, titleposition = :left)

	
	# Summarize by appearance
	app_curve = combine(
		groupby(df, [:prolific_pid, :appearance]),
		:response_optimal => mean => :acc
	)

	# Summarize by apperance and n_groups
	app_curve_sum = combine(
		groupby(app_curve, [:appearance]),
		:acc => mean => :acc,
		:acc => sem => :se
	)

	# Compute bounds
	app_curve_sum.lb = app_curve_sum.acc .- app_curve_sum.se
	app_curve_sum.ub = app_curve_sum.acc .+ app_curve_sum.se

	# Sort
	sort!(app_curve_sum, [:appearance])

	# Create mapping
	mp2 = (data(app_curve_sum) * (
		mapping(
		:appearance => "Apperance #",
		:lb,
		:ub
	) * visual(Band, alpha = 0.5) +
		mapping(
		:appearance => "Apperance #",
		:acc => "Prop. optimal choice"
	) * visual(Lines)))
	
	# Plot
	plt2 = draw!(f[1,2], mp2)

	f
end

# ╔═╡ 4386f2f7-f4dc-4377-980a-af6f11253336
let

	cor_sum = combine(
		groupby(
			WM_data_clean,
			:prolific_pid
		),
		[:trial, :delay] => ((x,y) -> cor(x,y)) => :cor_trial,
		[:appearance, :delay] => ((x,y) -> cor(x,y)) => :cor_appearance,
		:trial => length => :n
	)

	# @info "Correation between trial number and delay: $(round(mean(cor_sum.cor), digits = 2)), range $(round.(extrema(cor_sum.cor), digits = 2))"


end

# ╔═╡ 14a4593b-9fd0-4d2f-964f-32d5b1530067
let df = WM_data_clean


	df.delay_bin = recoder(df.delay, [0, 1, 5, 9, 13], ["0", "1", "2-5", "6-9", "10-13"])
	
	# Summarize by participant
	app_curve = combine(
		groupby(df, [:prolific_pid, :delay_bin, :appearance]),
		:response_optimal => mean => :acc
	)

	# Summarize across participants
	app_curve_sum = combine(
		groupby(app_curve, [:delay_bin, :appearance]),
		:acc => mean => :acc,
		:acc => sem => :se
	)

	# Compute bounds
	app_curve_sum.lb = app_curve_sum.acc .- app_curve_sum.se
	app_curve_sum.ub = app_curve_sum.acc .+ app_curve_sum.se

	# Sort
	sort!(app_curve_sum, [:delay_bin, :appearance])

	# Create mapping
	mp2 = (data(app_curve_sum) * (
		mapping(
		:appearance => "Apperance #",
		:lb,
		:ub,
		color = :delay_bin  => "Delay"
	) * visual(Band, alpha = 0.5) +
		mapping(
		:appearance => "Apperance #",
		:acc => "Prop. optimal choice",
		color = :delay_bin  => "Delay"
	) * visual(Lines))) + (
		data(filter(x -> x.delay_bin == "0", app_curve_sum)) *
		(mapping(
			:appearance  => "Apperance #",
			:acc,
			:se,
			color = :delay_bin => "Delay"
		) * visual(Errorbars) +
		mapping(
			:appearance  => "Apperance #",
			:acc,
			color = :delay_bin  => "Delay"
		) * visual(Scatter))
	)

	f = Figure()

	plt = draw!(f[1,1], mp2)

	legend!(f[0,1], plt, tellwidth = false, halign = 0.5, orientation = :horizontal, framevisible = false, titleposition = :left)

	f
end

# ╔═╡ 88591a1a-f440-11ef-2b4f-9f1d7635cbe4
let df = WM_data_clean,
	appearance_breaks = [1, 2, 4, 14, 24]
	appearance_labels = ["1", "2", "3-4", "5-14", "15-24"]

	df.learning_phase = recoder(df.appearance, appearance_breaks, appearance_labels) 

	df.learning_phase = CategoricalArray(df.learning_phase, levels = appearance_labels)

	delay_sum = combine(
		groupby(
			df,
			[:prolific_pid, :learning_phase, :delay]
		),
		:response_optimal => mean => :acc
	)

	delay_sum = combine(
		groupby(
			delay_sum,
			[:delay, :learning_phase]
		),
		:acc => mean => :acc,
		:acc => sem => :se
	)

	sort!(delay_sum, [:delay, :learning_phase])

	mp = data(delay_sum) * (
		mapping(
			:delay, 
			:acc, 
			color = :learning_phase => nonnumeric => "Learning phase"
		) * visual(Lines) +
		mapping(
			:delay, 
			:acc, 
			color = :learning_phase => nonnumeric =>  "Learning phase"
		) * visual(Scatter) +
		mapping(
			:delay, 
			:acc, 
			:se, 
			color = :learning_phase  => nonnumeric =>  "Learning phase"
		) * visual(Errorbars)
	)

	f = Figure()

	plt = draw!(f[1,1], mp, axis = (; 
		ylabel = "Prop. optimal choice",
		xlabel = "Delay"
	))

	legend!(f[0,1], plt, tellwidth = false, halign = 0.5, orientation = :horizontal, framevisible = false, titleposition = :left)
		
	f
	
end

# ╔═╡ e72ee428-01bc-4cab-aa4c-1de1d7e60693
let df = WM_data_clean

	# Divide into learning phases
	df.learning_phase = equi_groups(
		df.appearance;
		n=2,
		labels = ["1-12", "13-24"]
	)

	# Summarize by participant
	delay_sum = combine(
		groupby(
			df,
			[:prolific_pid, :learning_phase, :previous_optimal, :delay]
		),
		:response_optimal => mean => :acc
	)

	# Summarize across participants
	delay_sum = combine(
		groupby(
			delay_sum,
			[:delay, :learning_phase, :previous_optimal]
		),
		:acc => mean => :acc,
		:acc => sem => :se,
		:acc => length => :n
	)
	
	# Sort for plotting
	sort!(delay_sum, [:delay, :learning_phase, :previous_optimal])

	# Filter
	filter!(x -> x.n > 10, delay_sum)

	# Label previous_optimal
	delay_sum.previous_optimal = passmissing(ifelse).(
		delay_sum.previous_optimal,
		fill("Correct", nrow(delay_sum)),
		fill("Error", nrow(delay_sum))
	)

	mp = data(delay_sum) * (
		mapping(
			:delay, 
			:acc, 
			color = :learning_phase  => "Apperances",
			group = :previous_optimal => "Previous choice"
		) * visual(Lines) +
		mapping(
			:delay, 
			:acc, 
			color = :learning_phase  => "Apperances",
			marker = :previous_optimal => "Previous choice"
		) * visual(Scatter) +
		mapping(
			:delay, 
			:acc, 
			:se, 
			color = :learning_phase => "Apperances",
			group = :previous_optimal => "Previous choice"
		) * visual(Errorbars)
	)

	f = Figure()

	plt = draw!(f[1,1], mp, axis = (; 
		ylabel = "Prop. optimal choice",
		xlabel = "Delay"
	))

	legend!(f[0,1], plt, tellwidth = false, halign = 0.5, orientation = :horizontal, framevisible = false, titleposition = :left, nbanks = 2)
		
	f
	
end

# ╔═╡ 0da6af42-ef20-4d28-8818-92abd077ecf7
value_sum = let df = WM_data_clean

	value_sum = vcat(
		[select(
			df,
			:prolific_pid,
			:stimulus_group_id,
			Symbol("stimulus_$side") => :stimulus,
			Symbol("feedback_$side") => :feedback
		) for side in ["left", "middle", "right"]]...
	)

	value_sum = combine(
		groupby(
			value_sum,
			[:prolific_pid, :stimulus_group_id, :stimulus]),
		:feedback => mean => :value
	)

end

# ╔═╡ 9382247e-d825-4e3f-89f8-959683b0fe62
value_test_data = let vdf = value_sum,
	tdf = filter(x -> x.block == 6, test_data)

	# Select relevant columns
	select!(
		tdf,
		:prolific_pid,
		:trial,
		:stimulus_left,
		:stimulus_right,
		:right_chosen
	)

	# Merge values from learning phase
	for side in ["left", "right"]
	
		leftjoin!(
			tdf,
			select(
				vdf,
				:prolific_pid,
				:stimulus => Symbol("stimulus_$side"),
				:value => Symbol("value_$side")
			),
			on = [:prolific_pid, Symbol("stimulus_$side")]
		)
	end

	# Whether high value chosen
	tdf.high_chosen = ifelse.(
		tdf.value_right .> tdf.value_left,
		tdf.right_chosen,
		.!tdf.right_chosen
	)

	# Value difference
	tdf.Δ_val = tdf.value_right - tdf.value_left

	tdf

end

# ╔═╡ 7b97683e-ba34-4e48-8fca-fd863dde343e
let tdf = value_test_data
	f = Figure()

	# Histogram of values --------------------
	ax = Axis(
		f[1,1],
		yautolimitmargin = (0,0),
		xlabel = "Average observed outcome",
		xticks = [0.01, 0.5, 0.75, 1.]
	)
	
	hist!(ax, value_sum.value; bins = vcat([0.01 - 0.025 / 2, 0.01 + 0.025 / 2], range(0.75, 1., 11)), color = :grey)

	hideydecorations!(ax)

	hidespines!(ax, :l)

	# Choice by value ----------------------

	# Remove missing
	dropmissing!(tdf)

	# Bin value
	vbins = vcat(
		[-1., -0.8],
		range(-0.2, 0.2, 6),
		[0.8, 1.]
	)

	vlabels = [mean([vbins[i], vbins[i+1]]) for i in 1:(length(vbins)-1)]

	tdf.Δ_val_bin = recoder(
		tdf.Δ_val,
		vbins[2:end],
		vlabels
	)

	# Average by participant
	tsum = combine(
		groupby(
			tdf,
			[:prolific_pid, :Δ_val_bin]),
		:right_chosen => mean => :right_chosen
	)

	# Average across participants
	tsum = combine(
		groupby(
			tsum,
			:Δ_val_bin),
		:right_chosen => mean => :right_chosen,
		:right_chosen => sem => :se,
		:right_chosen => length => :n
	)

	# Remove rare
	filter!(x -> x.n > 10, tsum)

	# Plot
	vmap = data(tsum) * (mapping(
		:Δ_val_bin,
		:right_chosen,
		:se
	) * visual(Errorbars) + mapping(
		:Δ_val_bin,
		:right_chosen
	) * visual(Scatter))

	draw!(
		f[1,2],
		vmap;
		axis = (;
			xlabel = "Δ average outcome\nright − left",
			ylabel = "Prop. right chosen",
			xticks = sort(unique(tsum.Δ_val_bin)),
			xticklabelrotation = 45,
			limits = (nothing, (0., 1.))
		)
	)
	

	f

end

# ╔═╡ Cell order:
# ╠═fb93a03b-8ae6-4395-bae3-4183fbe45cc5
# ╠═b4067011-f52b-4da7-a278-063d8743bcaf
# ╠═86ac0107-f0cd-46ad-94f6-219638c80711
# ╠═a2216e2d-7890-4c23-90dc-3ad8a07e6485
# ╠═e7e5c7c9-4038-4c8c-876c-d30104445c02
# ╠═4386f2f7-f4dc-4377-980a-af6f11253336
# ╠═14a4593b-9fd0-4d2f-964f-32d5b1530067
# ╠═88591a1a-f440-11ef-2b4f-9f1d7635cbe4
# ╠═e72ee428-01bc-4cab-aa4c-1de1d7e60693
# ╟─f081f2d7-bef3-476c-ad14-ebc4f0a35e77
# ╠═0da6af42-ef20-4d28-8818-92abd077ecf7
# ╠═9382247e-d825-4e3f-89f8-959683b0fe62
# ╠═7b97683e-ba34-4e48-8fca-fd863dde343e
# ╠═6ab2f6d7-186d-435d-9bc1-a50b079e4eda
# ╠═01255688-ee82-4f77-95ba-5fb2c1d5a717
# ╠═ff992cf9-e256-4ebc-a661-e13780cb99a0
# ╠═8f52df1d-598a-43e3-bd07-2e2bdf458c93
