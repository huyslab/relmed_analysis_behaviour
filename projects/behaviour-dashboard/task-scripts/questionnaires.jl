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
    long_scores = stack(scores, columns, [experiment.participant_id_column, :session]; 
                       variable_name = :questionnaire, value_name = :score)

    # Remove rows with missing scores (occurs when session doesn't include a questionnaire)
    filter!(row -> !ismissing(row.score), long_scores)

    # Map questionnaire column names to human-readable labels
    question_map = Dict(string.(columns) .=> labels)
    long_scores.questionnaire = map(x -> question_map[x], long_scores.questionnaire)

    # Set up color scheme for different sessions
    all_sessions = sort(unique(long_scores.session))
    session_colors = Makie.wong_colors()[1:length(all_sessions)]
    color_map = [all_sessions[i] => session_colors[i] for i in eachindex(all_sessions)]

    # Calculate grid layout dimensions (roughly square)
    n_questionnaires = length(labels)
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