# Script for estimating discount faction using logistic regression method described in 
# Wileyto, E. P., Audrain-McGovern, J., Epstein, L. H., & Lerman, C. (2004). Using logistic regression to estimate delay-discounting functions. Behavior Research Methods, Instruments, & Computers, 36(1), 41-51.
# https://link.springer.com/content/pdf/10.3758/BF03195548.pdf

using DataFrames, CairoMakie, AlgebraOfGraphics, CSV

function preprocess_delay_discounting_data(
    df::DataFrame;
    participant_id_column::Symbol = :participant_id
)

    # Select variables
    forfit = select(
        df,
        participant_id_column,
        :delay,
        :sum_today,
        :sum_later,
        :response
    )

    # Remove missing response trials
    filter!(row -> !ismissing(row.response) && !isnothing(row.response), forfit)

    # Calculate transformed ratio
    forfit.ip1 = 1 .- 1 ./ (forfit.sum_today ./ forfit.sum_later) # 1 - (1 / R), R = VI / VD

    return forfit
end

function fit_dd_logistic_regression(
    df::DataFrame;
    participant_id_column::Symbol = :participant_id,
    force::Bool = false,
    output_file::String = "tmp/delay_discounting_model"
)

    # Save to temporary CSV file for R processing
    isdir("tmp") || mkpath("tmp")
    CSV.write("tmp/delay_discounting_for_fit.csv", df)

    r_script = """
    library(cmdstanr)
    library(brms)

    # Load data
    dat <- read.csv("tmp/delay_discounting_for_fit.csv")

    # formula: no intercept, group-varying slopes for ip1 and time
    bf_model <- bf(response ~ 0 + ip1 + delay + (0 + ip1 + delay | $participant_id_column),
                family = bernoulli(link = "logit"))

    # sensible weakly-informative priors; 
    priors <- c(
        prior(normal(0, 2), class = "b"),   # fixed-effect priors
        prior(student_t(3, 0, 2), class = "sd")           # group sd priors
    )

    # fit
    (fit <- brm(
        formula = bf_model,
        data = dat,
        prior = priors,
        backend = "cmdstanr",
        threads = threading(2),
        chains = 4, cores = 4, iter = 4000, warmup = 2000,
        control = list(adapt_delta = 0.9),
        seed = 2025
    ))
    write.csv(as.data.frame(fit), file = "$output_file.csv")

    # Save coefficients and draws
    coefs <- coef(
        fit,
        summary = FALSE
    )\$$participant_id_column

    write.csv(coefs, file = "$(output_file)_coefs.csv")
    """

    # Run R script
    if !isfile("$(output_file).csv") || !isfile("$(output_file)_coefs.csv") || force
        write("tmp/run_delay_discounting_model.R", r_script)
        run(`Rscript tmp/run_delay_discounting_model.R`)
    end

    # Load model coefficients
    draws = DataFrame(CSV.File("$(output_file).csv"))
    coefs = DataFrame(CSV.File("$(output_file)_coefs.csv"))

    return draws, coefs

end

function post_process_dd_logistic_regression(
    draws::DataFrame,
    coefs::DataFrame;
    participant_id_column::Symbol = :participant_id,
    summarize::Bool = true
)

    # Separate coefficient draws and melt
    function shape_coefs(
        coefs::DataFrame;
        col::String,
    )
        cols = filter(x -> endswith(x, ".$col") || x == "Column1", names(coefs))
        out = select(coefs, cols)
        out = stack(out, Not("Column1"), variable_name = participant_id_column, value_name = Symbol(col))
        out[!, participant_id_column] = replace.(out[!, participant_id_column], ".$col" => "")
        out[!, participant_id_column] = replace.(out[!, participant_id_column], "." => "-")

        return out
    end

    ip1 = shape_coefs(coefs; col = "ip1")
    delay = shape_coefs(coefs; col = "delay")

    # Join
    coef_draws = innerjoin(ip1, delay, on = [participant_id_column, :Column1])

    # Compute discount factor k from delay coefficient
    coef_draws.k = coef_draws.delay ./ coef_draws.ip1

    # Compute group level k
    draws.k = draws.b_delay ./ draws.b_ip1
    draws[!, participant_id_column] .= "group"

    # Join
    coef_draws = vcat(select(coef_draws, :Column1, participant_id_column, :k), select(draws, [:Column1, participant_id_column, :k]))

    if summarize
        return combine(
            groupby(coef_draws, participant_id_column),
            :k => mean => :k_mean,
            :k => std => :k_sd
        )
    else
        return coef_draws
    end
end

hyperbolic_discount_function(k, delay) = 1 ./ (1 .+ k .* delay)

function plot_value_ratio_as_function_of_delay!(
    f::Figure,
    coef_draws::DataFrame,
    df::DataFrame;
    participant_id_column::Symbol = :participant_id,
    facet::Symbol = :session
)

    # Whether facet is used or not, it must be in both dataframes or neither 
    if ((facet ∉ names(df)) || (facet ∉ names(coef_draws))) && ! (((facet ∉ names(df)) && (facet ∉ names(coef_draws))))
        error("Facet column $facet found in one dataframe, but not the other.")
    end

    if facet ∉ names(df)
        df[!, facet] .= "all"
        coef_draws[!, facet] .= "all"
    end

    # Create plots
    xrange = range(0, stop = maximum(df.delay), length = 100)

    # Predicted value
    ys = combine(
        groupby(coef_draws, [participant_id_column, facet]),
        :k_mean => (x -> xrange) => :x,
        :k_mean => (k -> hyperbolic_discount_function.(k, xrange)) => :m
    )

    ys.lw = ifelse.(ys[!, participant_id_column] .== "group", 4, 1)

    # Proportion chosen later
    df.ratio = df.sum_today ./ df.sum_later
    chosen_later = combine(
        groupby(df, [participant_id_column, facet, :delay, :ratio]),
        :response => mean => :response
    )

    chosen_later = combine(
        groupby(chosen_later, [facet, :delay, :ratio]),
        :response => mean => :response
    )

    # Check if facet has only one unique value
    facet_levels = unique(ys[!, facet])
    use_facet = length(facet_levels) > 1


    mp1 = data(ys) *
    mapping(
        :x,
        :m,
        group = participant_id_column,
        linewidth = :lw => verbatim,
        color = participant_id_column => scale(:color_lines)
    ) 

    if use_facet
        mp1 = mp1 * mapping(layout = facet)
    end

    mp1 = mp1 * visual(Lines)
    
    mp2 = data(chosen_later) *
    mapping(
        :delay,
        :ratio,
        color = :response => scale(:color_scatter),
    ) 

    if use_facet
        mp2 = mp2 * mapping(layout = facet)
    end

    mp2 = mp2 * visual(
            Scatter; 
            strokecolor = :black, 
            marker = :circle,
            markersize = 10,
            strokewidth = 0.5
        )

    colors =  ["group" => :black, Makie.wong_colors()...]

    plt = draw!(f[1,1], mp1+mp2, scales(color_lines = (; palette = colors), color_scatter = (; colormap = :greys)); 
        axis = (; 
            xlabel = "Delay (days)",
            ylabel = "Value ratio (immediate / later)",
            xautolimitmargin = (0., 0.05),)
    )

     # Add k value annotation at the end of the group line
    group_data = filter(row -> row[participant_id_column] == "group", ys)
    if !isempty(group_data)
        # Get the k value for the group
        group_k = filter(row -> row[participant_id_column] == "group", coef_draws).k_mean[1]
        
        # Find the last two points to calculate slope
        sorted_data = sort(group_data, :x)
        n_points = nrow(sorted_data)
        
        # Get last two points for slope calculation
        point_diff = 1
        x2, y2 = sorted_data[n_points, :x], sorted_data[n_points, :m]
        x1, y1 = sorted_data[n_points-point_diff, :x], sorted_data[n_points-point_diff, :m]

        # Calculate angle in radians
        angle = atan((y2 - y1) / (point_diff / n_points))
        println("Angle (radians): ", angle)
        # Get the axis from the figure and add text annotation
        ax = current_axis()
        text!(ax, x2, y2; 
            text = "k = $(round(group_k, digits=3))",
            align = (:right, :bottom),
            offset = (0, 10),  # offset in pixels
            fontsize = 16,
            color = :black,
            rotation = angle
        )
    end

    colorbar!(f[1,2], plt; label = "Prop. chosen later")

    return f
end
