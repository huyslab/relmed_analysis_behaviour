using DataFrames

"""
    reindex_questions(question_string::String) -> String

Re-index Sam Zorrowitz's question numbering to match jsPsych question numbering.
Converts 1-based indexing (Q1, Q2, ...) to 0-based indexing (Q0, Q1, ...).
"""
function reindex_questions(question_string::String)
    if startswith(question_string, "Q")
        # Extract the number part, convert to integer, then back to string with Q prefix
        number_part = question_string[2:end]
        number = parse(Int, number_part)
        return "Q$(number - 1)"  # Zero-index by subtracting 1
    else
        return question_string  # Return unchanged if it doesn't start with Q
    end
end

"""
    parse_response(df::AbstractDataFrame) -> AbstractDataFrame

Helper function to parse response values into integers where possible.
Attempts to convert string responses to integers, leaving non-numeric values unchanged.
"""
parse_response = x -> transform(x, :response => (vals -> [tryparse(Int, val) !== nothing ? parse(Int, val) : val for val in vals]) => :response)

"""
    prepare_questionnaire_data(df::AbstractDataFrame, participant_id_column::Symbol) -> AbstractDataFrame

Prepare questionnaire data by standardizing question numbering across different survey plugins.
Converts Sam Zorrowitz's 1-based question indexing to jsPsych's 0-based indexing for consistency.
"""
function prepare_questionnaire_data(df::AbstractDataFrame, participant_id_column::Symbol)
    questionnaire_data = copy(df)
    questionnaire_data.question = ifelse.(
        questionnaire_data.trial_type .== "survey-template", # SZ plugin
        reindex_questions.(questionnaire_data.question),
        questionnaire_data.question,                        # jsPsych plugin
    )
    return questionnaire_data
end

"""
    compute_PHQ_scores(questionnaire_data::AbstractDataFrame, participant_id_column::Symbol) -> AbstractDataFrame

Compute PHQ-9 (Patient Health Questionnaire) depression scores.
Higher scores indicate more severe depression symptoms (9 items × 3-point scale).
Includes catch question validation: "Experiencing sadness or a sense of despair" vs "Feeling down, depressed, or hopeless".
"""
function compute_PHQ_scores(questionnaire_data::AbstractDataFrame, participant_id_column::Symbol)
    # Catch question processing: Q1 and Q8 should have similar responses
    PHQ_catch = filter(x -> (x.trialphase .== "PHQ" && x.question in ["Q1", "Q8"]), questionnaire_data) |>
         parse_response |>
         x -> unstack(x, [participant_id_column, :module_start_time, :session], :question, :response) |>
              x -> DataFrames.transform(x, [:Q8, :Q1] => ByRow((x, y) -> abs(x - y) > 1) => :phq_fail_catch) |>
                   x -> select(x, Not([:Q8, :Q1]))
    
    # Main PHQ scoring: exclude catch question Q8 from total
    PHQ = filter(x -> (x.trialphase .== "PHQ" && x.question .!= "Q8"), questionnaire_data) |>
         parse_response |>
          x -> groupby(x, [participant_id_column, :module_start_time, :session]) |>
               x -> combine(x, :response => sum => :phq_total, :response => length => :phq_n)
    
    leftjoin!(PHQ, PHQ_catch, on=[participant_id_column, :module_start_time, :session])
    return PHQ
end

"""
    compute_GAD_scores(questionnaire_data::AbstractDataFrame, participant_id_column::Symbol) -> AbstractDataFrame

Compute GAD-7 (Generalized Anxiety Disorder) scores.
Higher scores indicate more severe anxiety symptoms (7 items × 3-point scale).
Includes catch question validation: "Worrying about the 1974 Eurovision Song Contest".
"""
function compute_GAD_scores(questionnaire_data::AbstractDataFrame, participant_id_column::Symbol)
    # Catch question processing: Q6 should be answered with 0 (not worried about Eurovision)
    GAD_catch = filter(x -> (x.trialphase .== "GAD" && x.question in ["Q6"]), questionnaire_data) |>
         parse_response |>
         x -> unstack(x, [participant_id_column, :module_start_time, :session], :question, :response) |>
         x -> DataFrames.transform(x, [:Q6] => ByRow(!=(0)) => :gad_fail_catch) |>
         x -> select(x, Not([:Q6]))
    
    # Main GAD scoring: exclude catch question Q6 from total
    GAD = filter(x -> (x.trialphase .== "GAD" && x.question .!= "Q6"), questionnaire_data) |>
         parse_response |>
          x -> groupby(x, [participant_id_column, :module_start_time, :session]) |>
               x -> combine(x, :response => sum => :gad_total, :response => length => :gad_n)
    
    leftjoin!(GAD, GAD_catch, on=[participant_id_column, :module_start_time, :session])
    return GAD
end

"""
    compute_WSAS_scores(questionnaire_data::AbstractDataFrame, participant_id_column::Symbol) -> AbstractDataFrame

Compute WSAS (Work and Social Adjustment Scale) functional impairment scores.
Higher scores indicate more impairment (5 items × 8-point scale). Excludes work item for 
non-working participants and scales to match total scores.

Follows approach from: Skelton, M., et al. (2023). Trajectories of depression symptoms, 
anxiety symptoms and functional impairment during internet-enabled cognitive-behavioural 
therapy. Behaviour Research and Therapy, 169, 104386.
"""
function compute_WSAS_scores(questionnaire_data::AbstractDataFrame, participant_id_column::Symbol)
    WSAS_nojob = filter(x -> (x.trialphase .== "WSAS" && x.question == "retired_check"), questionnaire_data) |>
         x -> select(x, participant_id_column, :module_start_time, :session, :response => :WSAS_no_job)
    
    WSAS = filter(x -> (x.trialphase .== "WSAS" && x.question .!= "retired_check"), questionnaire_data)
    leftjoin!(WSAS, WSAS_nojob, on=[participant_id_column, :module_start_time, :session])

    # Keep all rows except those where the question is "Q0" and the participant is not working (WSAS_no_job is true).
    # In other words, exclude "Q0" rows only for participants who are not working; include all other rows.
    # Keep all rows except those where the question is "Q0" and the participant is not working (WSAS_no_job is true).
    # In other words, exclude "Q0" rows only for participants who are not working; include all other rows.
    WSAS = filter(x -> (x.question != "Q0") || (!x.WSAS_no_job), WSAS) |>
         parse_response |>
         x -> groupby(x, [participant_id_column, :module_start_time, :session, :WSAS_no_job]) |>
         x -> combine(x, :response => (x -> mean(x) * 5) => :wsas_total, :response => length => :wsas_n)
    return WSAS
end

"""
    compute_ICECAP_scores(questionnaire_data::AbstractDataFrame, participant_id_column::Symbol) -> AbstractDataFrame

Compute ICECAP-A (ICEpop CAPability measure for Adults) quality of life scores.
Higher scores indicate better quality of life (5 items scored using tariff values).
"""
function compute_ICECAP_scores(questionnaire_data::AbstractDataFrame, participant_id_column::Symbol)
    # Tariff values for ICECAP-A capability states
    multichoice_ICECAP = Dict(
         "Q0" => Dict(
              "I am able to feel settled and secure in <b>all</b> areas of my life" => 0.222,
              "I am able to feel settled and secure in <b>many</b> areas of my life" => 0.191,
              "I am able to feel settled and secure in <b>a few</b> areas of my life" => 0.101,
              "I am <b>unable</b> to feel settled and secure in <b>any</b> areas of my life" => -0.001,
         ),
         "Q1" => Dict(
              "I can have <b>a lot</b> of love, friendship and support" => 0.228,
              "I can have <b>quite a lot</b> of love, friendship and support" => 0.189,
              "I can have <b>a little</b> love, friendship and support" => 0.096,
              "I <b>cannot</b> have <b>any</b> love, friendship and support" => -0.024,
         ),
         "Q2" => Dict(
              "I am able to be <b>completely</b> independent" => 0.188,
              "I am able to be independent in <b>many</b> things" => 0.156,
              "I am able to be independent in <b>a few</b> things" => 0.084,
              "I am <b>unable</b> to be at all independent" => 0.006,
         ),
         "Q3" => Dict(
              "I can achieve and progress in <b>all</b> aspects of my life" => 0.181,
              "I can achieve and progress in <b>many</b> aspects of my life" => 0.159,
              "I can achieve and progress in <b>a few</b> aspects of my life" => 0.091,
              "I <b>cannot</b> achieve and progress in <b>any</b> aspects of my life" => 0.021,
         ),
         "Q4" => Dict(
              "I can have <b>a lot</b> of enjoyment and pleasure" => 0.181,
              "I can have <b>quite a lot</b> of enjoyment and pleasure" => 0.154,
              "I can have a <b>little</b> enjoyment and pleasure" => 0.069,
              "I <b>cannot</b> have <b>any</b> enjoyment and pleasure" => -0.003,
         ),
    )
    ICECAP =
         filter(x -> (x.trialphase .== "ICECAP"), questionnaire_data) |>
         parse_response |>
         x ->
              DataFrames.transform(x, [:question, :response] => ByRow((x, y) -> multichoice_ICECAP[x][y]) => :tariff_score) |>
              x -> groupby(x, [participant_id_column, :module_start_time, :session]) |>
                   x -> combine(x, :tariff_score => sum => :icecap_total, :tariff_score => length => :icecap_n)
    return ICECAP
end

"""
    compute_BFI_scores(questionnaire_data::AbstractDataFrame, participant_id_column::Symbol) -> AbstractDataFrame

Compute BFI-10 (Big Five Inventory) personality scores.
Returns 5 personality dimensions (Extraversion, Agreeableness, Conscientiousness, Neuroticism, Openness).
Each dimension calculated from 2 items on a 1-5 scale, with some items reverse-scored.
"""
function compute_BFI_scores(questionnaire_data::AbstractDataFrame, participant_id_column::Symbol)
    BFI =
         filter(x -> (x.trialphase .== "BFI"), questionnaire_data) |>
         parse_response |>
         x ->
              DataFrames.transform(x, [:question, :response] => ByRow((x, y) -> ifelse(x in ["Q0", "Q6", "Q2", "Q3", "Q4"], 5 - y, y + 1)) => :score) |>
              x ->
                   unstack(x, [participant_id_column, :module_start_time, :session], :question, :score) |>
                   x -> DataFrames.transform(x,
                        [:Q0, :Q5] => (+) => :bfi_E,
                        [:Q1, :Q6] => (+) => :bfi_A,
                        [:Q2, :Q7] => (+) => :bfi_C,
                        [:Q3, :Q8] => (+) => :bfi_N,
                        [:Q4, :Q9] => (+) => :bfi_O,
                   ) |>
                        x -> select(x, Not(r"^Q"))
    return BFI
end

"""
    compute_PVSS_scores(questionnaire_data::AbstractDataFrame, participant_id_column::Symbol) -> AbstractDataFrame

Compute PVSS (Positive Valence Systems Scale) scores.
Higher scores indicate more positive affect and motivation (21 items × 1-9-point scale).
Includes 6 subscales: valuation, expectancy, effort, anticipation, responsiveness, and satiation.
Includes catch question validation: "I wished to engage in enjoyable activities with people I'm close to" vs "I <u>wanted</u> to participate in a fun activity with friends".
"""
function compute_PVSS_scores(questionnaire_data::AbstractDataFrame, participant_id_column::Symbol)
    PVSS_catch =
         filter(x -> (x.trialphase .== "PVSS"), questionnaire_data) |>
         parse_response |>
         x ->
              DataFrames.transform(x, :response => (x -> x .+ 1) => :score) |>
              x ->
                   unstack(x, [participant_id_column, :module_start_time, :session], :question, :score) |>
                   x ->
                        DataFrames.transform(x,
                             [:Q19, :Q13] => ByRow((x, y) -> abs(x - y) > 2) => :pvss_fail_catch,
                             [:Q3, :Q11, :Q13] => (+) => :pvss_valuation,
                             [:Q6, :Q8, :Q12, :Q18] => (+) => :pvss_expectancy,
                             [:Q1, :Q14, :Q20, :Q21] => (+) => :pvss_effort,
                             [:Q7, :Q10, :Q15] => (+) => :pvss_anticipation,
                             [:Q0, :Q2, :Q5, :Q16] => (+) => :pvss_responsiveness,
                             [:Q4, :Q9, :Q17] => (+) => :pvss_satiation,
                        ) |>
                        x -> select(x, Not(r"^Q"))
    PVSS =
         filter(x -> (x.trialphase .== "PVSS" && x.question .!= "Q19"), questionnaire_data) |>
         parse_response |>
         x -> DataFrames.transform(x, :response => (x -> x .+ 1) => :score) |>
              x -> groupby(x, [participant_id_column, :module_start_time, :session]) |>
                   x -> combine(x, :score => sum => :pvss_total, :score => length => :pvss_n)
    leftjoin!(PVSS, PVSS_catch, on=[participant_id_column, :module_start_time, :session])
    return PVSS
end

"""
    compute_BADS_scores(questionnaire_data::AbstractDataFrame, participant_id_column::Symbol) -> AbstractDataFrame

Compute BADS (Behavioral Activation for Depression Scale) scores.
Higher scores indicate better behavioral activation (9 items × 0-6 scale).
Includes subscales for activation (AC) and avoidance (AV).
Includes catch question validation: "I was able to lift my coffee cup or water glass when drinking."
"""
function compute_BADS_scores(questionnaire_data::AbstractDataFrame, participant_id_column::Symbol)
    BADS_subscale =
         filter(x -> (x.trialphase .== "BADS"), questionnaire_data) |>
         parse_response |>
         x ->
              DataFrames.transform(x, [:question, :response] => ByRow((x, y) -> ifelse(x in ["Q0", "Q5", "Q7", "Q8"], 6 - y, y)) => :score) |>
              x ->
                   unstack(x, [participant_id_column, :module_start_time, :session], :question, :score) |>
                   x -> DataFrames.transform(x,
                        [:Q6] => ByRow(<(5)) => :bads_fail_catch,
                        [:Q0, :Q1, :Q2, :Q3, :Q4, :Q9] => (+) => :bads_ac,
                        [:Q5, :Q7, :Q8] => (+) => :bads_av,
                   ) |>
                        x -> select(x, Not(r"^Q"))
    BADS =
         filter(x -> (x.trialphase .== "BADS" && x.question .!= "Q6"), questionnaire_data) |>
         parse_response |>
         x -> DataFrames.transform(x, [:question, :response] => ByRow((x, y) -> ifelse(x in ["Q0", "Q5", "Q7", "Q8"], 6 - y, y)) => :score) |>
              x -> groupby(x, [participant_id_column, :module_start_time, :session]) |>
                   x -> combine(x, :score => sum => :bads_total, :score => length => :bads_n)
    leftjoin!(BADS, BADS_subscale, on=[participant_id_column, :module_start_time, :session])
    return BADS
end

"""
    compute_Hopelessness_scores(questionnaire_data::AbstractDataFrame, participant_id_column::Symbol) -> AbstractDataFrame

Compute Hopelessness scores.
Higher scores indicate more helplessness (2 items × 1-5 scale, reverse-coded).
"""
function compute_Hopelessness_scores(questionnaire_data::AbstractDataFrame, participant_id_column::Symbol)
    Hopelessness =
         filter(x -> (x.trialphase .== "Hopelessness"), questionnaire_data) |>
         parse_response |>
         x -> DataFrames.transform(x, :response => (x -> 5 .- x) => :score) |>
              x -> groupby(x, [participant_id_column, :module_start_time, :session]) |>
                   x -> combine(x, :score => sum => :hopeless_total, :score => length => :hopeless_n)
    return Hopelessness
end

"""
    compute_PERS_negAct_scores(questionnaire_data::AbstractDataFrame, participant_id_column::Symbol) -> AbstractDataFrame

Compute PERS (Personality and Emotional Reactivity Scale) negative activation scores.
Higher scores indicate more negative emotional activation (5 items × 1-5 scale).
"""
function compute_PERS_negAct_scores(questionnaire_data::AbstractDataFrame, participant_id_column::Symbol)
    PERS =
         filter(x -> (x.trialphase .== "PERS_negAct"), questionnaire_data) |>
         parse_response |>
         x -> DataFrames.transform(x, :response => (x -> x .+ 1) => :score) |>
              x -> groupby(x, [participant_id_column, :module_start_time, :session]) |>
                   x -> combine(x, :score => sum => :pers_negact_total, :score => length => :pers_n)
    return PERS
end

"""
    compute_RRS_brooding_scores(questionnaire_data::AbstractDataFrame, participant_id_column::Symbol) -> AbstractDataFrame

Compute RRS (Ruminative Responses Scale) brooding subscale scores.
Higher scores indicate more ruminative brooding (5 items × 1-4 scale).
"""
function compute_RRS_brooding_scores(questionnaire_data::AbstractDataFrame, participant_id_column::Symbol)
    RRS =
         filter(x -> (x.trialphase .== "RRS_brooding"), questionnaire_data) |>
         parse_response |>
         x -> DataFrames.transform(x, :response => (x -> x .+ 1) => :score) |>
              x -> groupby(x, [participant_id_column, :module_start_time, :session]) |>
                   x -> combine(x, :score => sum => :rrs_brooding_total, :score => length => :rrs_brooding_n)
    return RRS
end

# Dictionary mapping questionnaire names to their corresponding functions
const QUESTIONNAIRE_FUNCTIONS = Dict(
    "PHQ" => compute_PHQ_scores,
    "GAD" => compute_GAD_scores,
    "WSAS" => compute_WSAS_scores,
    "ICECAP" => compute_ICECAP_scores,
    "BFI" => compute_BFI_scores,
    "PVSS" => compute_PVSS_scores,
    "BADS" => compute_BADS_scores,
    "Hopelessness" => compute_Hopelessness_scores,
    "PERS_negAct" => compute_PERS_negAct_scores,
    "RRS_brooding" => compute_RRS_brooding_scores
)

"""
    compute_questionnaire_scores(df::AbstractDataFrame; experiment::ExperimentInfo = TRIAL1) -> AbstractDataFrame

Compute scores for all questionnaires specified in the experiment configuration.

# Arguments
- `df::AbstractDataFrame`: Raw questionnaire response data containing trial phases, questions, and responses
- `experiment::ExperimentInfo`: Experiment configuration specifying which questionnaires to process

# Returns
- `AbstractDataFrame`: Combined questionnaire scores with participant identifiers, timestamps, and all computed scores

Processes only questionnaires listed in `experiment.questionnaire_names`. Each questionnaire is scored 
using its dedicated function from `QUESTIONNAIRE_FUNCTIONS`. Results are joined on participant ID, 
module start time, and session.
"""
function compute_questionnaire_scores(
    df::AbstractDataFrame;
    experiment::ExperimentInfo = TRIAL1
    )

    participant_id_column = experiment.participant_id_column
    
    # Return empty DataFrame if no questionnaire names specified
    if isnothing(experiment.questionnaire_names) || isempty(experiment.questionnaire_names)
        return DataFrame()
    end

    # Prepare questionnaire data
    questionnaire_data = prepare_questionnaire_data(df, participant_id_column)

    # Initialize results with first questionnaire
    first_questionnaire = experiment.questionnaire_names[1]
    
    if !haskey(QUESTIONNAIRE_FUNCTIONS, first_questionnaire)
        @warn "Questionnaire function not found: $first_questionnaire"
        questionnaire_score_data = DataFrame()
    else
        questionnaire_score_data = QUESTIONNAIRE_FUNCTIONS[first_questionnaire](questionnaire_data, participant_id_column)
    end

    # Process remaining questionnaires
    for questionnaire_name in experiment.questionnaire_names[2:end]
        if haskey(QUESTIONNAIRE_FUNCTIONS, questionnaire_name)
            questionnaire_result = QUESTIONNAIRE_FUNCTIONS[questionnaire_name](questionnaire_data, participant_id_column)
            if !isempty(questionnaire_result)
                if isempty(questionnaire_score_data)
                    questionnaire_score_data = questionnaire_result
                else
                    leftjoin!(questionnaire_score_data, questionnaire_result, on=[participant_id_column, :module_start_time, :session])
                end
            end
        else
            @warn "Questionnaire function not found: $questionnaire_name"
        end
    end

    return questionnaire_score_data
end