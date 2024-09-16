function plot_prior_expectations!(
	f::GridLayout,
	data::DataFrame;
	colA::Symbol = :EV_A,
	colB::Symbol = :EV_B,
    norm::Symbol = :ρ,
	ylab::String = "Q value",
	ylims::Union{Tuple{Float64, Float64}, Nothing} = nothing,
	group::Symbol = :group,
	legend::Bool = true,
    legend_title::String = "",
    legend_rows::Int64 = 1,
	colors = Makie.wong_colors(),
	backgroundcolor::Symbol = :white,
	plw = 0.2
)

	p_data = copy(data)

	# Normalize by ρ
	p_data.valA_s = p_data[!, colA] ./ p_data[!, norm]
	p_data.valB_s = p_data[!, colB] ./ p_data[!, norm]

	# Summarize into proportion of participants choosing optimal
	p_data = combine(
		groupby(p_data, [group, :trial, :PID]),
		:valA_s => mean => colA,
		:valB_s => mean => colB
	)

	# Unstack per participant data for Stim A
	p_data_A = unstack(p_data, [group, :trial], 
		:PID,
		colA)

	# Unstack per participant data for Stim B
	p_data_B = unstack(p_data, [group, :trial], 
		:PID,
		colB)

	# Summarize per group
	sum_data = combine(
		groupby(p_data, [group, :trial]),
		colA => mean => colA,
		colB => mean => colB
	)


	# Set up axis
	ax = Axis(f[1,1],
		xlabel = "Trial #",
		ylabel = ylab,
		limits = (nothing, ylims),
		xautolimitmargin = (0., 0.),
		yautolimitmargin = (0., 0.),
		xticks = range(1, round(Int64, maximum(sum_data.trial)), 4),
		backgroundcolor = backgroundcolor)

	# Plot line
	for (i, g) in enumerate(unique(sum_data[!, group]))

		# Subset per group
		gsum_dat = filter(group => (x -> x == g), sum_data)
		gp_dat_A = filter(group => (x -> x == g), p_data_A)
		gp_dat_B = filter(group => (x -> x == g), p_data_B)

		# Plot group means
		lines!(ax, gsum_dat.trial, gsum_dat[!, colA], 
			color = colors[i],
			linewidth = 3)
		lines!(ax, gsum_dat.trial, gsum_dat[!, colB],
			color = colors[i],
			linestyle = :dash,
			linewidth = 3)

		# Plot per participant
		series!(ax, transpose(Matrix(gp_dat_A[!, 3:end])), 
			solid_color = (colors[i], 0.1),
			linewidth = plw)

		series!(ax, transpose(Matrix(gp_dat_B[!, 3:end])), 
			solid_color = (colors[i], 0.1),
			linewidth = plw,
			linestyle = :dash)
	end
    if legend
        group_levels = unique(data[!, group])
		elements = [PolyElement(color = colors[i]) for i in 1:length(group_levels)]
		labels = [g for g in group_levels]
		
		Legend(f[1,1],
            elements,
            labels,
            legend_title,
            framevisible = false,
            tellheight = false,
            tellwidth = false,
            margin = (10, 10, 10, 10),
            orientation = :horizontal,
            halign = :center,
            valign = :top,
            nbanks = legend_rows
        )
    end
end

function plot_prior_accuracy!(
    f::GridLayout,
    data::Union{DataFrame, SubDataFrame};
    group::Symbol = :group,
    pid_col::Symbol = :PID,
    acc_col::Symbol = :isOptimal,
    colors = Makie.wong_colors(),
    title::String = "",
    legend::Bool = true,
    legend_title::String = "",
    legend_rows::Int64 = 1,
    backgroundcolor = :white,
    ylabel::Union{String, Makie.RichText}="Prop. optimal choice",
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

    group_levels = unique(sum_data.group)
    for (i,g) in enumerate(group_levels)
        gdat = filter(:group => (x -> x==g), sum_data)
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

	# Legend
	if legend
		group_levels = unique(data[!, group])
		elements = [PolyElement(color = colors[i]) for i in 1:length(group_levels)]
		labels = [g for g in group_levels]
		
		Legend(f[1,1],
			elements,
			labels,
			legend_title,
            framevisible = false,
            tellheight = false,
            tellwidth = false,
            margin = (10, 10, 10, 10),
            orientation = :horizontal,
            halign = :center,
            valign = :top,
            nbanks = legend_rows
		)
		# rowsize!(f.layout, 0, Relative(0.1))
	end

	return ax

end

# This function makes density plots for posteriors, plus true value if needed
function plot_posteriors(draws::Vector{DataFrame},
	params::Vector{String};
	labels::AbstractVector = params,
	true_values::Union{Vector{Float64}, Nothing} = nothing,
	colors::AbstractVector = Makie.wong_colors()[1:length(draws)],
	nrows::Int64=1,
	scale_col::Union{Symbol, Nothing} = nothing,
	mean_col::Union{Symbol, Nothing} = nothing,
	model_labels::AbstractVector = repeat([nothing], length(draws))
)

	# Plot
	f_sim = Figure(size = (700, 20 + 230 * nrows))

	n_per_row = ceil(Int64, length(params) / nrows)

	for (i, p) in enumerate(params)

		# Set up axis
		ax = Axis(
			f_sim[div(i-1, n_per_row) + 1, rem(i - 1, n_per_row) + 1],
			xlabel = labels[i]
		)

		hideydecorations!(ax)
		hidespines!(ax, :l)

		for (j, d) in enumerate(draws)

			# Scale and add mean
			dp = copy(d[!, Symbol(p)])

			if !isnothing(scale_col)
				dp .*= d[!, scale_col]
			end

			if !isnothing(mean_col)
				dp .+= d[!, mean_col]
			end

			# Plot posterior density
			density!(ax,
				dp,
				color = (:black, 0.),
				strokewidth = 2,
				strokecolor = colors[j]
			)

			# Plot 95% PI
			linesegments!(ax,
				[(quantile(dp, 0.025), 0.),
				(quantile(dp, 0.975), 0.)],
				linewidth = 4,
				color = colors[j]
			)
		end

		# Plot true value
		if !isnothing(true_values)
			vlines!(ax,
				true_values[i],
				color = :gray,
				linestyle = :dash)
		end

	end

	draw_legend = length(draws) > 1 & !all(isnothing.(model_labels))
	if draw_legend
		Legend(f_sim[0, 1:n_per_row],
			[LineElement(color = Makie.wong_colors()[i]) for i in 1:length(draws)],
			model_labels,
			nbanks = length(draws),
			framevisible = false,
			valign = :bottom
		)
	end

	Label(f_sim[0, 1:n_per_row],
		rich(rich("Group-level posteriors\n", fontsize = 18),
			rich("95% PI and true value marked as dashed line", fontsize = 14)),
		valign = :top
		)

	rowsize!(f_sim.layout, 0, Relative(draw_legend ? 0.4 / nrows : 0.2 / nrows))

	return f_sim
end

# Methods for MCMCChains
function plot_posteriors(draws::AbstractVector,
	params::Vector{String};
	labels::AbstractVector = params,
	true_values::Union{Vector{Float64}, Nothing} = nothing,
	colors::AbstractVector = Makie.wong_colors()[1:length(draws)],
	nrows::Int64=1,
	scale_col::Union{Symbol, Nothing} = nothing,
	mean_col::Union{Symbol, Nothing} = nothing,
	model_labels::AbstractVector = repeat([nothing], length(draws))
)

	draws_dfs = [
		DataFrame(Array(chains), names(chains, :parameters)) for chains in draws
	]

	plot_posteriors(draws_dfs,
		params;
		labels = labels,
		true_values = true_values,
		colors = colors,
		nrows = nrows,
		scale_col = scale_col,
		mean_col = mean_col,
		model_labels = model_labels
	)	

end

# Plot unit line
unit_line!(ax; color = :grey, linestyle = :dash, linewidth = 2) = ablines!(
	0., 
	1.,
	color = color,
	linestyle = linestyle,
	linewidth = linewidth
	)

# Plot calibration for optimization methods
function optimization_calibration(
	prior_sample::DataFrame,
	optimize_func::Function;
	estimate::String = "MLE",
    model::Function = RL_ss,
    set_size::Union{Vector{Int64}, Nothing} = nothing,
    parameters::Vector{Symbol} = [:ρ, :a], # Group-level parameters to estimate
	initial_params::Union{AbstractVector,Nothing}=[mean(truncated(Normal(0., 2.), lower = 0.)), 0.5],
    transformed::Dict{Symbol, Symbol} = Dict(:a => :α), # Transformed parameters
	priors::Dict{Symbol, Distribution{Univariate, Continuous}} = Dict(
		:ρ => truncated(Normal(0., 2.), lower = 0.),
		:a => Normal(0., 1.)
	),
	ms::Float64 = 4.
)
	# Initial value for Q values
	aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])

	MLEs = optimize_func(
		prior_sample;
		initV = aao,
		estimate = estimate,
        model = model,
        set_size = set_size,
        include_true = true,
        parameters = parameters,
        initial_params = initial_params,
		transformed = transformed,
		priors = priors
	)[1]

    other_pars = setdiff(parameters, [:a, :ρ])
	f = length(other_pars) > 0 ? Figure(size = (900, 400)) : Figure(size = (900, 200))

	# Plot a
	ax_a = Axis(
		f[1,1],
		xlabel = "True a",
		ylabel = "$estimate a",
		aspect = 1.
	)

	scatter!(
		ax_a,
		MLEs.true_a,
		MLEs.MLE_a,
		markersize = ms
	)

	unit_line!(ax_a)

	# Plot ρ
	ax_ρ = Axis(
		f[1,2],
		xlabel = "True ρ",
		ylabel = "$estimate ρ",
		aspect = 1.
	)

	scatter!(
		ax_ρ,
		MLEs.true_ρ,
		MLEs.MLE_ρ,
		markersize = ms
	)

	unit_line!(ax_ρ)

	# Plot bivariate
	ax_aρ = Axis(
		f[1,3],
		xlabel = "$estimate a",
		ylabel = "$estimate ρ",
		aspect = 1.
	)

	scatter!(
		ax_aρ,
		MLEs.MLE_a,
		MLEs.MLE_ρ,
		markersize = ms
	)

	# Plot ground truth
	ax_taρ = Axis(
		f[1,4],
		xlabel = "True a",
		ylabel = "True ρ",
		aspect = 1.
	)

	scatter!(
		ax_taρ,
		MLEs.true_a,
		MLEs.true_ρ,
		markersize = ms
	)

    if length(other_pars) > 0
        for (i, p) in enumerate(other_pars)
            ax = Axis(
                f[2, i],
                xlabel = "True $p",
                ylabel = "$estimate $p",
                aspect = 1.
            )

            scatter!(
                ax,
                MLEs[!, Symbol("true_$p")],
                MLEs[!, Symbol("MLE_$p")],
                markersize = ms
            )

            unit_line!(ax)
        end
    end

	f
end