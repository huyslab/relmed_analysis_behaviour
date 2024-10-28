### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ d1b4ae5e-952a-11ef-24af-bb7226096cd5
begin
	cd("/home/jovyan/")
    import Pkg
    # activate the shared project environment
    Pkg.activate("relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	using Random, DataFrames, JSON, CSV, StatsBase, JLD2, HTTP, CairoMakie, Printf, Distributions, CategoricalArrays, AlgebraOfGraphics, Dates
	using LogExpFunctions: logistic, logit
	include("fetch_preprocess_data.jl")
	include("sample_utils.jl")
	include("plotting_utils.jl")
	nothing
end

# ╔═╡ 6dc7628c-5230-4562-a452-be9c61238b00
begin
	# Load data
	PILT_data, test_data, _, _, _, _, jspsych_data = load_pilot4x_data()

	# Keep only 4.1
	filter!(x -> x.version == "4.1", jspsych_data)
	filter!(x -> x.version == "4.1", PILT_data)
	filter!(x -> x.version == "4.1", test_data)
	
end

# ╔═╡ 91475e7f-b52a-48e4-a72d-4298b3544bf4
begin
	# Plot accuracy for a group, divided by condition / group
	function plot_group_accuracy!(
	    f::GridPosition,
	    data::Union{DataFrame, SubDataFrame};
	    group::Union{Symbol, Missing} = missing,
	    pid_col::Symbol = :prolific_pid,
	    acc_col::Symbol = :isOptimal,
	    colors = Makie.wong_colors(),
	    title::String = "",
	    legend::Union{Dict, Missing} = missing,
	    legend_title::String = "",
	    backgroundcolor = :white,
	    ylabel::Union{String, Makie.RichText}="Prop. optimal choice",
	    levels::Union{AbstractVector, Missing} = missing,
		error_band::Union{Bool, String} = "se",
		linewidth::Float64 = 3.,
		plw::Float64 = 1.
	    )
	
		# Set up axis
		ax = Axis(f[1,1],
	        xlabel = "Trial #",
	        ylabel = ylabel,
	        xautolimitmargin = (0., 0.),
	        backgroundcolor = backgroundcolor,
	        title = title
	    )
	
		plot_group_accuracy!(
			ax,
			data;
			group = group,
			pid_col = pid_col,
			acc_col = acc_col,
			colors = colors,
			title = title,
			legend = legend,
			legend_title = legend_title,
			backgroundcolor = backgroundcolor,
			ylabel = ylabel,
			levels = levels,
			error_band = error_band,
			linewidth = linewidth,
			plw = plw
			)
	
		# Legend
		if !ismissing(legend)
			group_levels = ismissing(levels) ? unique(data[!, group]) : levels
			elements = [PolyElement(color = colors[i]) for i in 1:length(group_levels)]
			labels = [legend[g] for g in group_levels]
			
			Legend(f[0,1],
				elements,
				labels,
				legend_title,
				framevisible = false,
				tellwidth = false,
				orientation = :horizontal,
				titleposition = :left
			)
			# rowsize!(f.layout, 0, Relative(0.1))
		end
		
	
		return ax
	
	end
	
	function plot_group_accuracy!(
	    ax::Axis,
	    data::Union{DataFrame, SubDataFrame};
	    group::Union{Symbol, Missing} = missing,
	    pid_col::Symbol = :prolific_pid,
	    acc_col::Symbol = :isOptimal,
	    colors = Makie.wong_colors(),
	    title::String = "",
	    legend::Union{Dict, Missing} = missing,
	    legend_title::String = "",
	    backgroundcolor = :white,
	    ylabel::Union{String, Makie.RichText}="Prop. optimal choice",
	    levels::Union{AbstractVector, Missing} = missing,
		error_band::Union{String, Bool} = "se", # Whether and which type of error band to plot
		linewidth::Float64 = 3.,
		plw::Float64 = 1. # Line width for per participant traces
	    )
	
	    # Default group value
	    tdata = copy(data)
	    if ismissing(group)
	        tdata.group .= 1
	        group = :group
	    else
	        tdata.group = tdata[!, group]
	    end
	
	
	    # Summarize into proportion of participants choosing optimal
	    sum_data = combine(
	        groupby(tdata, [pid_col, :group, :trial]),
	        acc_col => mean => :acc
	    )
	
		# Unstack per participant data for Stim A
		p_data = unstack(sum_data, [:group, :trial], 
			pid_col,
			:acc)
	
	    sum_data = combine(
	        groupby(sum_data, [:group, :trial]),
	        :acc => mean => :acc,
	        :acc => sem => :acc_sem,
			:acc => lb => :acc_lb,
			:acc => ub => :acc_ub,
			:acc => llb => :acc_llb,
			:acc => uub => :acc_uub
	    )
	
		# Set axis xticks
		ax.xticks = range(1, round(Int64, maximum(sum_data.trial)), 4)
	
	    group_levels = ismissing(levels) ? unique(sum_data.group) : levels
	    for (i,g) in enumerate(group_levels)
	        gdat = filter(:group => (x -> x==g), sum_data)
			sort!(gdat, [:trial])
	
			dropmissing!(gdat)
	
			g_p_dat = filter(:group => (x -> x == g), p_data)
	
	
	        # Plot line
			mc = length(colors)
	
			if typeof(error_band) == String
				if error_band == "PI"
					band!(ax,
						gdat.trial,
						gdat.acc_llb,
						gdat.acc_uub,
						color = (colors[rem(i - 1, mc) + 1], 0.1)
					)
				end 
	
				if error_band in ["se", "PI"]
					band!(ax,
						gdat.trial,
						error_band == "se" ? gdat.acc - gdat.acc_sem : gdat.acc_lb,
						error_band == "se" ? gdat.acc + gdat.acc_sem : gdat.acc_ub,
						color = (colors[rem(i - 1, mc) + 1], 0.3)
					)
				elseif error_band == "traces"
					series!(ax, transpose(Matrix(g_p_dat[!, 3:end])), 
						solid_color = (colors[rem(i - 1, mc) + 1], 0.1),
						linewidth = plw)
				end
			end
	        
	        lines!(ax, 
	            gdat.trial, 
	            gdat.acc, 
	            color = colors[rem(i - 1, mc) + 1],
	            linewidth = linewidth)
	    end
	        
	end
end

# ╔═╡ 518161d7-137e-4e63-9961-b365c0adead2
let
	PILT_data = exclude_PLT_sessions(PILT_data, required_n_blocks = 18)
	PILT_data_clean = filter(x -> x.choice != "noresp", PILT_data)

	@info "Plotting data from $(length(unique(PILT_data_clean.prolific_pid))) participants"

	f = Figure()

	ax = plot_group_accuracy!(
		f[1,1],
		group = :n_pairs,
		PILT_data_clean,
		legend = Dict(i => "$i" for i in sort(unique(PILT_data_clean.n_pairs))),
		legend_title = "Set size",
		error_band = false
	)

	ax.xticks = [1, 10, 20, 30]

	f


end

# ╔═╡ 4055b14d-7db9-4231-84c6-7b82b7e17c03
"""
    extract_debrief_responses(data::DataFrame) -> DataFrame

Extracts and processes debrief responses from the experimental data. It filters for debrief trials, then parses and expands JSON-formatted Likert scale and text responses into separate columns for each question.

# Arguments
- `data::DataFrame`: The raw experimental data containing participants' trial outcomes and responses, including debrief information.

# Returns
- A DataFrame with participants' debrief responses. The debrief Likert and text responses are parsed from JSON and expanded into separate columns.
"""
function extract_debrief_responses(data::DataFrame)
	# Select trials
	debrief = filter(x -> !ismissing(x.trialphase) && 
		occursin(r"(acceptability|debrief)(?!.*pre)", x.trialphase), data)


	# Select variables
	select!(debrief, [:prolific_pid, :exp_start_time, :trialphase, :response])

	# Long to wide
	debrief = unstack(
		debrief,
		[:prolific_pid, :exp_start_time],
		:trialphase,
		:response
	)

	# Function to rename column only if it exists
	function safe_rename(
		data::AbstractDataFrame,
		pairs...
	)

		safe_pairs = [p for p in pairs if p.first in Symbol.(names(data))]
		
		rename(
			data,
			safe_pairs...
		)

	end

	# Parse JSON and make into DataFrame
	debrief_colnames = ["acceptability_pilt", "debrief_pilt", "debrief_instructions"]
	expanded = [
		safe_rename(DataFrame([JSON.parse(row[col]) for row in eachrow(debrief)]),
			:vigour_strategy => :pilt_strategy
		) 
			for col in debrief_colnames
		]

	expanded = hcat(expanded...)

	# hcat together
	return hcat(debrief[!, Not(debrief_colnames)], expanded)
end

# ╔═╡ b6661ceb-bf2c-46a1-bd47-91e945e1530a
"""
    summarize_participation(data::DataFrame) -> DataFrame

Summarizes participants' performance in a study based on their trial outcomes and responses, for the purpose of approving and paying bonuses.

This function processes experimental data, extracting key performance metrics such as whether the participant finished the experiment, whether they were kicked out, and their respective bonuses (PILT and vigour). It also computes the number of specific trial types and blocks completed, as well as warnings received. The output is a DataFrame with these aggregated values, merged with debrief responses for each participant.

# Arguments
- `data::DataFrame`: The raw experimental data containing participant performance, trial outcomes, and responses.

# Returns
- A summarized DataFrame with performance metrics for each participant, including bonuses and trial information.
"""
function summarize_participation(data::DataFrame)

	function extract_PILT_bonus(outcome)

		if all(ismissing.(outcome)) # Return missing if participant didn't complete
			return missing
		else # Parse JSON
			bonus = filter(x -> !ismissing(x), unique(outcome))[1]
			bonus = JSON.parse(bonus)[1] 
			return bonus
		end

	end
	
	participants = combine(groupby(data, [:prolific_pid, :record_id, :exp_start_time]),
		:trialphase => (x -> "experiment_end_message" in x) => :finished,
		:trialphase => (x -> "kick-out" in x) => :kick_out,
		:outcomes => extract_PILT_bonus => :PILT_bonus,
		[:trial_type, :block] => ((t, b) -> sum((t .== "PLT") .& (typeof.(b) .== Int64))) => :n_trial_PLT,
		[:block, :trial_type] => ((x, t) -> length(unique(filter(y -> isa(y, Int64), x[t .== "PLT"])))) => :n_blocks_PLT,
		:n_warnings => maximum => :n_warnings
	)

	debrief = extract_debrief_responses(data)

	participants = leftjoin(participants, debrief, 
		on = [:prolific_pid, :exp_start_time])

	return participants
end

# ╔═╡ 4e0246af-545d-410b-9c32-c4f43d441194
p_sum = summarize_participation(jspsych_data)

# ╔═╡ fd38388f-7a09-4281-b473-38c5547116af
for r in eachrow(p_sum)
	if !ismissing(r.PILT_bonus)
		println("$(r.prolific_pid), $(r.PILT_bonus)")
	end
end

# ╔═╡ 6654d1dc-b8b1-4794-81b5-57c92814e020


# ╔═╡ Cell order:
# ╠═d1b4ae5e-952a-11ef-24af-bb7226096cd5
# ╠═6dc7628c-5230-4562-a452-be9c61238b00
# ╠═4e0246af-545d-410b-9c32-c4f43d441194
# ╠═fd38388f-7a09-4281-b473-38c5547116af
# ╠═518161d7-137e-4e63-9961-b365c0adead2
# ╠═91475e7f-b52a-48e4-a72d-4298b3544bf4
# ╠═b6661ceb-bf2c-46a1-bd47-91e945e1530a
# ╠═4055b14d-7db9-4231-84c6-7b82b7e17c03
# ╠═6654d1dc-b8b1-4794-81b5-57c92814e020
