# Script for estimating discount factor using logistic regression method described in 
# Wileyto, E. P., Audrain-McGovern, J., Epstein, L. H., & Lerman, C. (2004). Using logistic regression to estimate delay-discounting functions. Behavior Research Methods, Instruments, & Computers, 36(1), 41-51.
# https://link.springer.com/content/pdf/10.3758/BF03195548.pdf

using DataFrames, CairoMakie, AlgebraOfGraphics, CSV

"""
    preprocess_delay_discounting_data(df; participant_id_column=:participant_id)

Preprocess delay discounting data for logistic regression analysis.
Calculates the transformed ratio (1 - 1/R) where R = immediate_value / delayed_value.

# Arguments
- `df::DataFrame`: Raw delay discounting data
- `participant_id_column::Symbol`: Column name for participant IDs

# Returns
- `DataFrame`: Preprocessed data with transformed ratio column
"""
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

"""
    fit_dd_logistic_regression(df; participant_id_column=:participant_id, force=false, output_file="tmp/delay_discounting_model")

Fit a Bayesian logistic regression model to delay discounting data using R/brms.
Uses a hierarchical model with group-varying slopes for transformed ratio and delay.

# Arguments
- `df::DataFrame`: Preprocessed delay discounting data
- `participant_id_column::Symbol`: Column name for participant IDs
- `force::Bool`: Whether to refit model even if output files exist
- `output_file::String`: Base filename for model output (without extension)

# Returns
- `Tuple{DataFrame, DataFrame}`: Model draws and coefficient draws
"""
function fit_dd_logistic_regression(
    df::DataFrame;
    participant_id_column::Symbol = :participant_id,
    force::Bool = false,
    output_file::String = "tmp/delay_discounting_model"
)

    # Save to temporary CSV file for R processing
    isdir("tmp") || mkpath("tmp")
    CSV.write("tmp/delay_discounting_for_fit.csv", df)

    # R script for Bayesian logistic regression using brms
    r_script = """
    library(cmdstanr)
    library(brms)

    # Load data
    dat <- read.csv("tmp/delay_discounting_for_fit.csv")

    # Formula: no intercept, group-varying slopes for ip1 and delay
    bf_model <- bf(response ~ 0 + ip1 + delay + (0 + ip1 + delay | $participant_id_column),
                family = bernoulli(link = "logit"))

    # Sensible weakly-informative priors
    priors <- c(
        prior(normal(0, 2), class = "b"),   # fixed-effect priors
        prior(student_t(3, 0, 2), class = "sd")           # group sd priors
    )

    # Fit the model
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

    # Run R script only if output files don't exist or if forced
    if !isfile("$(output_file).csv") || !isfile("$(output_file)_coefs.csv") || force
        write("tmp/run_delay_discounting_model.R", r_script)
        run(`Rscript tmp/run_delay_discounting_model.R`)
    end

    # Load model coefficients
    draws = DataFrame(CSV.File("$(output_file).csv"))
    coefs = DataFrame(CSV.File("$(output_file)_coefs.csv"))

    return draws, coefs

end

"""
    post_process_dd_logistic_regression(draws, coefs; participant_id_column=:participant_id, summarize=true)

Post-process the outputs from the delay discounting logistic regression model.
Extracts coefficients, computes discount factor k, and optionally summarizes results.

# Arguments
- `draws::DataFrame`: Model draws from brms fit
- `coefs::DataFrame`: Coefficient draws from brms fit
- `participant_id_column::Symbol`: Column name for participant IDs
- `summarize::Bool`: Whether to return summary statistics or full draws

# Returns
- `DataFrame`: Either summary statistics (mean, sd) or full coefficient draws
"""
function post_process_dd_logistic_regression(
    draws::DataFrame,
    coefs::DataFrame;
    participant_id_column::Symbol = :participant_id,
    summarize::Bool = true
)

    # Separate coefficient draws and melt to long format
    function shape_coefs(
        coefs::DataFrame;
        col::String,
    )
        # Select columns ending with the coefficient name
        cols = filter(x -> endswith(x, ".$col") || x == "Column1", names(coefs))
        out = select(coefs, cols)
        # Reshape from wide to long format
        out = stack(out, Not("Column1"), variable_name = participant_id_column, value_name = Symbol(col))
        # Clean participant IDs by removing coefficient suffix
        out[!, participant_id_column] = replace.(out[!, participant_id_column], ".$col" => "")
        out[!, participant_id_column] = replace.(out[!, participant_id_column], "." => "-")

        return out
    end

    # Extract ip1 and delay coefficients
    ip1 = shape_coefs(coefs; col = "ip1")
    delay = shape_coefs(coefs; col = "delay")

    # Join coefficient draws by participant and draw number
    coef_draws = innerjoin(ip1, delay, on = [participant_id_column, :Column1])

    # Compute discount factor k from coefficients (k = delay coefficient / ip1 coefficient)
    coef_draws.k = coef_draws.delay ./ coef_draws.ip1

    # Compute group-level k from fixed effects
    draws.k = draws.b_delay ./ draws.b_ip1
    draws[!, participant_id_column] .= "group"

    # Combine individual and group-level draws
    coef_draws = vcat(select(coef_draws, :Column1, participant_id_column, :k), select(draws, [:Column1, participant_id_column, :k]))

    if summarize
        # Return summary statistics
        return combine(
            groupby(coef_draws, participant_id_column),
            :k => mean => :k_mean,
            :k => std => :k_sd
        )
    else
        # Return full draws
        return coef_draws
    end
end

"""
    hyperbolic_discount_function(k, delay)

Calculate the hyperbolic discount function: 1 / (1 + k * delay).

# Arguments
- `k`: Discount rate parameter
- `delay`: Time delay

# Returns
- Discounted value
"""
hyperbolic_discount_function(k, delay) = 1 ./ (1 .+ k .* delay)

"""
    plot_value_ratio_as_function_of_delay!(f, coef_draws, df; participant_id_column=:participant_id, facet=:session)

Create a plot showing value ratio as a function of delay with model predictions and observed data.

# Arguments
- `f::Figure`: Makie figure to plot into
- `coef_draws::DataFrame`: Model coefficient draws with k values
- `df::DataFrame`: Original data with responses
- `participant_id_column::Symbol`: Column name for participant IDs
- `facet::Symbol`: Column to facet by (optional)

# Returns
- `Figure`: Updated figure with plot
"""
function plot_value_ratio_as_function_of_delay!(
    f::Figure,
    coef_draws::DataFrame,
    df::DataFrame;
    participant_id_column::Symbol = :participant_id,
    facet::Symbol = :session
)

    # Ensure facet column is in both dataframes or neither
    if ((facet ∉ names(df)) || (facet ∉ names(coef_draws))) && ! (((facet ∉ names(df)) && (facet ∉ names(coef_draws))))
        error("Facet column $facet found in one dataframe, but not the other.")
    end

    # Add facet column if not present in both dataframes
    if facet ∉ names(df)
        df[!, facet] .= "all"
        coef_draws[!, facet] .= "all"
    end

    # Create prediction range for x-axis
    xrange = range(0, stop = maximum(df.delay), length = 100)

    # Generate predicted values using hyperbolic discount function
    ys = combine(
        groupby(coef_draws, [participant_id_column, facet]),
        :k_mean => (x -> xrange) => :x,
        :k_mean => (k -> hyperbolic_discount_function.(k, xrange)) => :m
    )

    # Set line width (group line is thicker)
    ys.lw = ifelse.(ys[!, participant_id_column] .== "group", 4, 1)

    # Calculate proportion chosen later for observed data
    df.ratio = df.sum_today ./ df.sum_later
    chosen_later = combine(
        groupby(df, [participant_id_column, facet, :delay, :ratio]),
        :response => mean => :response
    )

    # Aggregate across participants
    chosen_later = combine(
        groupby(chosen_later, [facet, :delay, :ratio]),
        :response => mean => :response
    )

    # Check if facet should be used (has multiple levels)
    facet_levels = unique(ys[!, facet])
    use_facet = length(facet_levels) > 1

    # Create mapping for predicted lines
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
    
    # Create mapping for observed data points
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

    # Set color palette
    colors =  ["group" => :black, Makie.wong_colors()...]

    # Create the plot
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
        
        # Find the last two points to calculate slope for text rotation
        sorted_data = sort(group_data, :x)
        n_points = nrow(sorted_data)
        
        # Get last two points for slope calculation
        point_diff = 1
        x2, y2 = sorted_data[n_points, :x], sorted_data[n_points, :m]
        x1, y1 = sorted_data[n_points-point_diff, :x], sorted_data[n_points-point_diff, :m]

        # Calculate angle in radians for text rotation
        angle = atan((y2 - y1) / (point_diff / n_points))
        println("Angle (radians): ", angle)
        
        # Add k value annotation
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

    # Add colorbar for observed data
    colorbar!(f[1,2], plt; label = "Prop. chosen later")

    return f
end
