# This file contains a registry of all RELMED pilot and trial experiments, with information needed to load and analyse data.
# Version: 1.0.1
# Last Modified: 2025-09-28
using Dates

"""
Represents metadata for a single experiment.
"""
struct ExperimentInfo
    project::String
    tasks_included::Vector{String}
    questionnaire_names::Union{Vector{String}, Nothing}
    participant_id_column::Symbol
    module_column::Symbol
    exclude_testing_participants::Function # Must accept the ExperimentInfo object as argument
    date_collected::Union{Date, Nothing}
    notes::Union{String, Nothing}
end

TRIAL1 = ExperimentInfo(
    "trial1",
    ["reversal", "max_press", "PILT", "vigour", "PIT", "vigour_test", "PILT_test", "control", "WM", "WM_test", "delay_discounting", "open_text", "questionnaire", "pavlovian_lottery"],
    ["PHQ", "GAD", "WSAS", "ICECAP", "BFI", "PVSS", "BADS", "Hopelessness", "RRS_brooding", "PERS_negAct"],
    :participant_id,
    :task,
    (data::DataFrame; experiment::ExperimentInfo) -> begin
        participant_id_column = experiment.participant_id_column
        # Exclude participant IDs matching test/demo patterns
        filter!(x -> !ismissing(x[participant_id_column]) && !occursin(r"haoyang|yaniv|tore|demo|simulate|debug|REL-LON-000", x[participant_id_column]), data)
        # Exclude participant IDs with length <= 10
        filter!(x -> length(x[participant_id_column]) > 10, data)
    end,
    Date(2025, 6, 4),
    "First RELMED trial with participants.",
)

NORMING = ExperimentInfo(
    "norming",
    ["reversal", "max_press", "PILT", "control", "questionnaire"],
    ["demographics", "PHQ", "WSAS", "ICECAP", "BFI"],
    :PROLIFIC_PID,
    :module,
    (data::DataFrame; experiment::ExperimentInfo) -> begin
        participant_id_column = experiment.participant_id_column
        
        # Exclude participant IDs matching test/demo patterns
        filter!(x -> !ismissing(x[participant_id_column]) && !occursin(r"simulate|debug", x[participant_id_column]), data)
        # Exclude participant IDs with length <= 10
        filter!(x -> length(x[participant_id_column]) > 10, data)
    end,
    Date(2025, 10, 21),
    "General population norming sample of RELMED battery.",
)