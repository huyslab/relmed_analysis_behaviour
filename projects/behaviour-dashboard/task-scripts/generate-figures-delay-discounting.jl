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
