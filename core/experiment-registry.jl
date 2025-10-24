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
    date_collected::Union{Date, Nothing}
    notes::Union{String, Nothing}
end

TRIAL1 = ExperimentInfo(
    "trial1",
    ["reversal", "max_press", "PILT", "vigour", "PIT", "vigour_test", "PILT_test", "control", "WM", "WM_test", "delay_discounting", "open_text", "questionnaire", "pavlovian_lottery"],
    ["PHQ", "GAD", "WSAS", "ICECAP", "BFI", "PVSS", "BADS", "Hopelessness", "RRS_brooding", "PERS_negAct"],
    :participant_id,
    :task,
    Date(2025, 6, 4),
    "First RELMED trial with participants.",
)

NORMING = ExperimentInfo(
    "norming",
    ["reversal", "max_press", "PILT", "vigour_test", "PILT_test", "WM", "WM_test", "delay_discounting", "open_text", "questionnaire", "pavlovian_lottery"],
    ["PHQ", "GAD", "WSAS", "ICECAP", "BFI", "PVSS", "BADS", "Hopelessness", "RRS_brooding", "PERS_negAct"],
    :PROLIFIC_PID,
    :module,
    Date(2025, 10, 21),
    "General population norming sample of RELMED battery.",
)