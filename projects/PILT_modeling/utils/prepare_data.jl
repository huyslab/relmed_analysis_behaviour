using CSV

include("$(pwd())/core/experiment-registry.jl")

# Which experiment to generate the dashboard for
experiment_name = length(ARGS) > 0 ? ARGS[1] : "NORMING"
manual_download = length(ARGS) > 1 ? parse(Bool, ARGS[2]) : true
experiment = eval(Meta.parse(experiment_name))

# Setup
begin
    cd("/home/jovyan")

    using DataFrames, CairoMakie, Dates, CategoricalArrays

    # Include data scripts
    include("$(pwd())/core/preprocess_data.jl")

end

# Load and preprocess data
begin 
    dat = preprocess_project(experiment; force_download = false, delay_ms = 65, use_manual_download = manual_download)
end


# Generate PILT learning curve by session
let 

    PILT_main_sessions = filter(x -> x.session != "screening", dat.PILT)

    select!(
        PILT_main_sessions,
        :participant_id,
        :session,
        :module,
        :module_start_time,
        :trialphase,
        :block,
        :valence,
        :trial,
        :stimulus_left,
        :stimulus_right,
        :feedback_left,
        :feedback_right,
        :optimal_right,
        :response,
        :rt,
        :response_optimal,
        :chosen_feedback,
        :chosen_stimulus
    )

    CSV.write("projects/PILT_modeling/data/NORMING_PILT_main_sessions.csv", PILT_main_sessions)

end
