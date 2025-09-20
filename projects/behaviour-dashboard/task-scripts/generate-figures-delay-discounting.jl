# Script for estimating discount faction using logistic regression method described in 
# Wileyto, E. P., Audrain-McGovern, J., Epstein, L. H., & Lerman, C. (2004). Using logistic regression to estimate delay-discounting functions. Behavior Research Methods, Instruments, & Computers, 36(1), 41-51.
# https://link.springer.com/content/pdf/10.3758/BF03195548.pdf

using DataFrames, CairoMakie, AlgebraOfGraphics

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