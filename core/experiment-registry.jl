# This file contains a registry of all RELMED pilot and trial experiments, with information needed to load and analyse data.
using Dates

"""
Represents metadata for a single experiment.
"""
struct ExperimentInfo
    project::String
    tasks_included::Vector{String}
    questionnaire_names::Union{Vector{String}, Nothing}
    participant_id_column::Symbol
    date_collected::Union{Date, Nothing}
    notes::Union{String, Nothing}
end

TRIAL1 = ExperimentInfo(
    "trial1",
    ["reversal", "max_press", "PILT", "vigour", "PIT", "vigour_test", "PILT_test", "control", "WM", "WM_test", "delay_discounting", "open_text", "questionnaire", "pavlovian_lottery"],
    ["PHQ", "GAD", "WSAS", "ICECAP", "BFI", "PVSS", "BADS", "Hopelessness", "RRS_brooding", "PERS_negAct"],
    :participant_id,
    Date(2025, 6, 4),
    "First RELMED trial with participants.",
)