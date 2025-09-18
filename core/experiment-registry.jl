# This file contains a registry of all RELMED pilot and trial experiments, with information needed to load and analyse data.

"""
Represents metadata for a single experiment.
"""
struct ExperimentInfo
    project::String
    tasks_included::Vector{String}
    participant_id_field::Symbol
    date_collected::Union{Date, Nothing}
    notes::Union{String, Nothing}
end

const TRIAL1 = ExperimentInfo(
    "trial1",
    ["reversal", "max_press", "PILT", "vigour", "PIT", "vigour_test", "post_PILT_test", "control", "WM", "delay_discounting", "open_text"],
    :participant_id,
    Date(2025, 6, 4),
    "First RELMED trial with participants.",
)