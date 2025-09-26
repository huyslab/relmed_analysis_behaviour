using DataFrames

function compute_questionnaire_scores(
    df::AbstractDataFrame;
    experiment::ExperimentInfo = TRIAL1
    )

    participant_id_column = experiment.participant_id_column

    # Re-index Sam Zorrowitz's question numbering to match jsPsych question numbering
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


    questionnaire_data = copy(df)
    questionnaire_data.question = ifelse.(
        questionnaire_data.trial_type .== "survey-template", # SZ plugin
        reindex_questions.(questionnaire_data.question),
        questionnaire_data.question,                        # jsPsych plugin
    )

     # PHQ: Higher, Severer; 9 * 3
     # Catch question: "Experiencing sadness or a sense of despair" => "Feeling down, depressed, or hopeless"
     PHQ_catch =
          filter(x -> (x.trialphase .== "PHQ" && x.question in ["Q1", "Q8"]), questionnaire_data) |>
          x -> transform(x, :response => (x -> parse.(Int, x)) => :response) |>
          x -> unstack(x, [participant_id_column, :module_start_time, :session], :question, :response) |>
               x -> DataFrames.transform(x, [:Q8, :Q1] => ByRow((x, y) -> abs(x - y) > 1) => :phq_fail_catch) |>
                    x -> select(x, Not([:Q8, :Q1]))
     PHQ = filter(x -> (x.trialphase .== "PHQ" && x.question .!= "Q8"), questionnaire_data) |>
           x -> groupby(x, [participant_id_column, :module_start_time, :session]) |>
                x -> combine(x, :response => sum => :phd_total, :response => length => :phq_n)
     leftjoin!(PHQ, PHQ_catch, on=[participant_id_column, :module_start_time, :session])

     # GAD: Higher, Severer; 7 * 3
     # Catch question: "Worrying about the 1974 Eurovision Song Contest"
     GAD_catch = filter(x -> (x.trialphase .== "GAD" && x.question in ["Q6"]), questionnaire_data) |>
                 x -> unstack(x, :question, :response) |>
                      x -> DataFrames.transform(x, [:Q6] => ByRow(!=(0)) => :gad_fail_catch) |>
                           x -> select(x, Not([:trialphase, :Q6]))
     GAD = filter(x -> (x.trialphase .== "GAD" && x.question .!= "Q6"), questionnaire_data) |>
           x -> groupby(x, [participant_id_column, :module_start_time, :session]) |>
                x -> combine(x, :response => sum => :gad_total, :response => length => :gad_n)
     leftjoin!(GAD, GAD_catch, on=[participant_id_column, :module_start_time, :session])

     # WSAS: Higher, more impaired; 5 * 8
     WSAS_nojob = filter(x -> (x.trialphase .== "WSAS" && x.question in ["Q0"]), questionnaire_data) |>
                  x -> unstack(x, :question, :response) |>
                       x -> select(x, Not([:trialphase, :Q0]), :Q0 => ByRow(==(0)) => :WSAS_nojob)
     WSAS = filter(x -> (x.trialphase .== "WSAS" && x.question .!= "Q0"), questionnaire_data) |>
            x -> groupby(x, [participant_id_column, :module_start_time, :session]) |>
                 x -> combine(x, :response => sum => :wsas_total, :response => length => :wsas_n)
     leftjoin!(WSAS, WSAS_nojob, on=[participant_id_column, :module_start_time, :session])

     # ICECAP: Higher, better quality; 5 * Tariff
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
          x ->
               DataFrames.transform(x, [:question, :response] => ByRow((x, y) -> multichoice_ICECAP[x][y]) => :tariff_score) |>
               x -> groupby(x, [participant_id_column, :module_start_time, :session]) |>
                    x -> combine(x, :tariff_score => sum => :icecap_total, :tariff_score => length => :icecap_n)

     # BFI: Big five; 5 subscales (* 2 items) * 1-5
     BFI =
          filter(x -> (x.trialphase .== "BFI"), questionnaire_data) |>
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

     # PVSS: Higher, more positive; 21 * 1-9
     # Catch question: "I wished to engage in enjoyable activities with people I'm close to" => "I <u>wanted</u> to participate in a fun activity with friends"
     PVSS_catch =
          filter(x -> (x.trialphase .== "PVSS"), questionnaire_data) |>
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
          x -> DataFrames.transform(x, :response => (x -> x .+ 1) => :score) |>
               x -> groupby(x, [participant_id_column, :module_start_time, :session]) |>
                    x -> combine(x, :score => sum => :pvss_total, :score => length => :pvss_n)
     leftjoin!(PVSS, PVSS_catch, on=[participant_id_column, :module_start_time, :session])

     # BADS: Higher, better behavioral activation; 9 * 0-6
     # Catch question: "I was able to lift my coffee cup or water glass when drinking."
     BADS_subscale =
          filter(x -> (x.trialphase .== "BADS"), questionnaire_data) |>
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
          x -> DataFrames.transform(x, [:question, :response] => ByRow((x, y) -> ifelse(x in ["Q0", "Q5", "Q7", "Q8"], 6 - y, y)) => :score) |>
               x -> groupby(x, [participant_id_column, :module_start_time, :session]) |>
                    x -> combine(x, :score => sum => :bads_total, :score => length => :bads_n)
     leftjoin!(BADS, BADS_subscale, on=[participant_id_column, :module_start_time, :session])

     # Hopelessness: Higher, more helplessness; 2 * 1-5
     Hopelessness =
          filter(x -> (x.trialphase .== "Hopelessness"), questionnaire_data) |>
          x -> DataFrames.transform(x, :response => (x -> 5 .- x) => :score) |>
               x -> groupby(x, [participant_id_column, :module_start_time, :session]) |>
                    x -> combine(x, :score => sum => :hopeless_total, :score => length => :hopeless_n)

     # PERS: Higher, more negative activation; 5 * 1-5
     PERS =
          filter(x -> (x.trialphase .== "PERS_negAct"), questionnaire_data) |>
          x -> DataFrames.transform(x, :response => (x -> x .+ 1) => :score) |>
               x -> groupby(x, [participant_id_column, :module_start_time, :session]) |>
                    x -> combine(x, :score => sum => :pers_negact_total, :score => length => :pers_n)

     # RRS: Higher, more rumination - brooding; 5 * 1-4
     RRS =
          filter(x -> (x.trialphase .== "RRS_brooding"), questionnaire_data) |>
          x -> DataFrames.transform(x, :response => (x -> x .+ 1) => :score) |>
               x -> groupby(x, [participant_id_column, :module_start_time, :session]) |>
                    x -> combine(x, :score => sum => :rrs_brooding_total, :score => length => :rrs_brooding_n)

     # Questionnaire response time in minute
     questionnaire_time_data = raw_questionnaire_data |>
                               x -> select(x, participant_id_column, :module_start_time, :session, :trialphase => :questionnaire, :rt => (x -> x ./ 60000) => :response_time)

     # Merge all questionnaire data
     questionnaire_score_data = copy(PHQ)
     for df in [GAD, WSAS, ICECAP, BFI, PVSS, BADS, Hopelessness, PERS, RRS]
          leftjoin!(questionnaire_score_data, df, on=[participant_id_column, :module_start_time, :session])
     end

     return questionnaire_score_data, questionnaire_time_data
end