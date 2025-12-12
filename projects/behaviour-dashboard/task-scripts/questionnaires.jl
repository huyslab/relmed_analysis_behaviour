using DataFrames, CairoMakie, AlgebraOfGraphics, CategoricalArrays
include("$(pwd())/core/questionnaire_scores.jl")

"""
    plot_questionnaire_histograms!(f::Figure, df::DataFrame; kwargs...)

Create histograms for multiple questionnaire scores in a grid layout.

# Arguments
- `f::Figure`: The figure to plot into
- `df::DataFrame`: Raw data containing questionnaire responses
- `columns::Vector{Symbol}`: Column names for questionnaire scores to plot
- `labels::Vector{String}`: Human-readable labels for each questionnaire
- `bins::Int`: Number of bins for histograms (default: 20)
- `experiment::ExperimentInfo`: Experiment configuration (default: TRIAL1)

# Returns
- `Figure`: The modified figure with histogram plots
"""
function plot_questionnaire_histograms!(
    f::Figure,
    df::DataFrame;
    columns::Vector{Symbol} = [:phq_total, :gad_total, :wsas_total, :icecap_total, 
        :bfi_E, :bfi_A, :bfi_C, :bfi_N, :bfi_O, :pvss_total, :bads_total, :hopeless_total, 
        :pers_negact_total, :rrs_brooding_total],
    labels::Vector{String} = ["PHQ-9", "GAD-7", "WSAS", "ICECAP", 
        "BFI Extraversion", "BFI Agreeableness", "BFI Conscientiousness", "BFI Neuroticism", "BFI Openness", 
        "PVSS", "BADS", "Hopelessness", "PERS negative affect", "RRS Brooding"],
    experiment::ExperimentInfo = TRIAL1,
    bins::Int = 15
)
    
    # Compute total scores for all questionnaires
    scores = compute_questionnaire_scores(df; experiment = experiment)

    # Reshape data from wide to long format for plotting
    keep_indices = findall(x -> x in names(scores), string.(columns))
    measures = string.(columns)[keep_indices]
    labels = labels[keep_indices]

    long_scores = stack(scores, measures, [experiment.participant_id_column, :session]; 
                       variable_name = :questionnaire, value_name = :score)

    # Remove rows with missing scores (occurs when session doesn't include a questionnaire)
    filter!(row -> !ismissing(row.score), long_scores)

    # Map questionnaire column names to human-readable labels
    question_map = Dict(string.(measures) .=> labels)
    long_scores.questionnaire = map(x -> question_map[x], long_scores.questionnaire)

    # Set up color scheme for different sessions
    all_sessions = sort(unique(long_scores.session))
    session_colors = Makie.wong_colors()[1:length(all_sessions)]
    color_map = [all_sessions[i] => session_colors[i] for i in eachindex(all_sessions)]

    # Calculate grid layout dimensions (roughly square)
    n_questionnaires = length(measures)
    n_cols = ceil(Int, sqrt(n_questionnaires))

    # Plot each questionnaire separately
    for (i, label) in enumerate(labels)
        # Filter data for this questionnaire
        questionnaire_data = filter(row -> row.questionnaire == label, long_scores)
        
        # Skip if no data
        if nrow(questionnaire_data) == 0
            continue
        end
        
        # Calculate subplot position
        row = div(i - 1, n_cols) + 1
        col = mod(i - 1, n_cols) + 1
        
        # Create mapping for this questionnaire with explicit colors
        mp = data(questionnaire_data) *
        mapping(
            :score,
            color = :session => "Session"
        ) * histogram(bins=bins)
        
        # Draw to specific subplot
        draw!(f[row, col], mp, scales(Color = (; palette = color_map));
            axis = (; xlabel = label))
                
    end

    # Create manual legend showing all sessions
    legend_elements = [PolyElement(polycolor = session_colors[i]) for i in 1:length(all_sessions)]
    legend_labels = [string(session) for session in all_sessions]
    
    # Add horizontal legend at the top of the figure
    Legend(f[0, :], legend_elements, legend_labels, "Session"; 
           orientation = :horizontal, framevisible = false,
           tellwidth = false, titleposition = :left)

    return f
end

function plot_questionnaire_histograms_offset!(
    f::Figure,
    df::DataFrame;
    columns::Vector{Symbol} = [:phq_total, :gad_total, :wsas_total, :icecap_total, 
        :bfi_E, :bfi_A, :bfi_C, :bfi_N, :bfi_O, :pvss_total, :bads_total, :hopeless_total, 
        :pers_negact_total, :rrs_brooding_total],
    labels::Vector{String} = ["PHQ-9", "GAD-7", "WSAS", "ICECAP", 
        "BFI Extraversion", "BFI Agreeableness", "BFI Conscientiousness", "BFI Neuroticism", "BFI Openness", 
        "PVSS", "BADS", "Hopelessness", "PERS negative affect", "RRS Brooding"],
    experiment::ExperimentInfo = TRIAL1,
    bins::Int = 15
)
    
    # Compute total scores for all questionnaires
    scores = compute_questionnaire_scores(df; experiment = experiment)

    # Reshape data from wide to long format for plotting
    keep_indices = findall(x -> x in names(scores), string.(columns))
    measures = string.(columns)[keep_indices]
    labels = labels[keep_indices]

    long_scores = stack(scores, measures, [experiment.participant_id_column, :session]; 
                       variable_name = :questionnaire, value_name = :score)

    # Remove rows with missing scores (occurs when session doesn't include a questionnaire)
    filter!(row -> !ismissing(row.score), long_scores)

    # Map questionnaire column names to human-readable labels
    question_map = Dict(string.(measures) .=> labels)
    long_scores.questionnaire = map(x -> question_map[x], long_scores.questionnaire)

    # Set up color scheme for different sessions
    all_sessions = sort(unique(long_scores.session))
    session_colors = Makie.wong_colors()[1:length(all_sessions)]
    color_map = [all_sessions[i] => session_colors[i] for i in eachindex(all_sessions)]

    # Calculate grid layout dimensions (roughly square)
    n_questionnaires = length(measures)
    n_cols = ceil(Int, sqrt(n_questionnaires))

    # Plot each questionnaire separately
    for (i, label) in enumerate(labels)
        # Filter data for this questionnaire
        questionnaire_data = filter(row -> row.questionnaire == label, long_scores)
        
        # Skip if no data
        if nrow(questionnaire_data) == 0
            continue
        end
        
        # Calculate subplot position
        row = div(i - 1, n_cols) + 1
        col = mod(i - 1, n_cols) + 1
        
        ax = Axis(f[row, col], xlabel = label, yticks = (1:length(all_sessions), string.(all_sessions)))
        for i in 1:length(all_sessions)
            session_df = filter(x -> x.session == all_sessions[i], questionnaire_data)
            if nrow(session_df) == 0
                continue
            end
            hist!(ax, session_df[!, :score], scale_to=0.9, offset=i, direction=:y, color=color_map[i][2], bins=bins, normalization=:none)
        end
    end

    # Create manual legend showing all sessions
    legend_elements = [PolyElement(polycolor = session_colors[i]) for i in 1:length(all_sessions)]
    legend_labels = [string(session) for session in all_sessions]
    
    # Add horizontal legend at the top of the figure
    Legend(f[0, :], legend_elements, legend_labels, "Session"; 
           orientation = :horizontal, framevisible = false,
           tellwidth = false, titleposition = :left)

    return f
end

function plot_demographics!(
    f::Figure,
    df::DataFrame;
    columns::Vector{Symbol} = [:age_group, :sex, :gender, :education, :employment, :income, :financial, :menstrual_since_first_day, :menstrual_cycle_length],
    labels::Vector{String} = ["Age Group", "Sex", "Gender", "Education", "Employment", "Income", "Financial", "Menstrual Since First Day", "Menstrual Cycle Length"],
    complex_columns::Vector{Symbol} = [:menstrual_since_first_day, :menstrual_cycle_length],
    experiment::ExperimentInfo = TRIAL1,
    factor::Symbol = :sex
)

    df = compute_questionnaire_scores(df; experiment = experiment)
    
    # Helper: Create horizontal barplot with consistent styling
    function create_horizontal_barplot(count_df, has_factor, factor_name)
        flip_threshold = maximum(count_df.count) * 0.85
        bar_style = (bar_labels = :y, direction = :x, color_over_background = :black, dodge_gap = 0,
                     color_over_bar = :white, flip_labels_at = flip_threshold, label_size = 11,
                     label_formatter = x -> string(Int(round(x))))

        if has_factor
            return data(count_df) *
                   mapping(:category => nonnumeric => "", :count => "",
                          dodge = :factor, color = :factor => factor_name) *
                   visual(BarPlot; bar_style...)
        else
            return data(count_df) *
                   mapping(:category => nonnumeric => "", :count => "") *
                   visual(BarPlot, color = :dodgerblue2; bar_style...)
        end
    end

    # Helper: Prepare categorical data with optional factor grouping
    function prepare_count_data(values, source_df, has_factor)
        plot_df = DataFrame(category = categorical(values, ordered = false))
        if has_factor
            plot_df.factor = categorical(source_df[!, factor], ordered = false)
            return combine(groupby(plot_df, [:category, :factor], sort = false), nrow => :count)
        else
            return combine(groupby(plot_df, :category, sort = false), nrow => :count)
        end
    end

    # Calculate grid layout
    n_simple = count(col -> !(col in complex_columns), columns)
    n_complex = count(col -> col in complex_columns, columns)
    total_plots = n_simple + 2 * n_complex
    n_cols = min(4, total_plots)

    # Helper to get grid position
    get_position(idx) = (div(idx - 1, n_cols) + 1, mod(idx - 1, n_cols) + 1)

    # Axis configuration for barplots
    barplot_axis = (titlealign = :left, xticksvisible = false, xticklabelsvisible = false,
                   xlabelvisible = false, bottomspinevisible = false)

    plot_idx = 0
    started_complex = false  # Track if we've started plotting complex columns

    for (i, col) in enumerate(columns)
        label = labels[i]

        # Skip if column doesn't exist
        !(String(col) in names(df)) && continue

        # Filter valid data
        valid_rows = .!ismissing.(df[!, col])
        has_factor = String(factor) in names(df)
        has_factor && (valid_rows = valid_rows .& .!ismissing.(df[!, factor]))

        valid_df = df[valid_rows, :]
        nrow(valid_df) == 0 && continue

        if col in complex_columns
            # Start a new row before first complex column
            if !started_complex && mod(plot_idx, n_cols) != 0
                plot_idx = ceil(Int, plot_idx / n_cols) * n_cols
            end
            started_complex = true

            # Complex column: separate numeric and categorical
            raw_values = valid_df[!, col]
            numeric_mask = [tryparse(Float64, string(v)) !== nothing for v in raw_values]

            # Plot 1: Barplot showing all response types (numeric + categorical)
            plot_idx += 1
            row, col_idx = get_position(plot_idx)

            # Prepare count data including both numeric and categorical responses
            if has_factor
                # Create dataframe with category labels
                response_type = [numeric_mask[i] ? "Numeric response" : string(raw_values[i]) for i in 1:length(raw_values)]
                count_df = DataFrame(category = categorical(response_type, ordered = false),
                                   factor = categorical(valid_df[!, factor], ordered = false))
                count_df = combine(groupby(count_df, [:category, :factor], sort = false), nrow => :count)
            else
                response_type = [numeric_mask[i] ? "Numeric response" : string(raw_values[i]) for i in 1:length(raw_values)]
                count_df = DataFrame(category = categorical(response_type, ordered = false))
                count_df = combine(groupby(count_df, :category, sort = false), nrow => :count)
            end

            mp = create_horizontal_barplot(count_df, has_factor, string(factor))
            draw!(f[row, col_idx], mp; axis = (; title = "$label (response type)", barplot_axis...))

            # Plot 2: Histogram for numeric values
            if any(numeric_mask)
                plot_idx += 1
                row, col_idx = get_position(plot_idx)

                numeric_df = DataFrame(value = [parse(Float64, string(v)) for v in raw_values[numeric_mask]])

                if has_factor
                    numeric_df.factor = categorical(valid_df[numeric_mask, factor], ordered = false)
                    mp = data(numeric_df) * mapping(:value => label, color = :factor => string(factor)) * histogram(bins = 15)
                else
                    mp = data(numeric_df) * mapping(:value => label) * histogram(bins = 15)
                end

                draw!(f[row, col_idx], mp; axis = (; xlabel = label, ylabel = "Count"))
            end
        else
            # Simple categorical column: horizontal barplot
            plot_idx += 1
            row, col_idx = get_position(plot_idx)

            count_df = prepare_count_data(valid_df[!, col], valid_df, has_factor)
            mp = create_horizontal_barplot(count_df, has_factor, string(factor))

            draw!(f[row, col_idx], mp; axis = (; title = label, barplot_axis...))
        end
    end

    return f
end