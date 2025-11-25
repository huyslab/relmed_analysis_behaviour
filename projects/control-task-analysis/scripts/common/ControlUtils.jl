module ControlUtils

using RCall, DataFrames, JSON

export gg_show_save, summarize_participation, extract_debrief_responses,
       spearman_brown

"""
    gg_show_save(plot_obj; width=6, height=4, dpi=150)

Saves R plot to PNG and displays it in VSCode using standard Julia I/O.
Requires NO extra Julia packages (no ImageMagick/FileIO).
"""
function gg_show_save(plot_obj::RObject, filename::String; width=6, height=4, dpi=300, scale=1)
    # 1. Save the real PDF file (Persistent)
    # PDF support is built-in to R, so this works without svglite
    R"ggsave($filename, plot=$plot_obj, width=$width, height=$height, device='pdf', scale=$scale)"
    println("✅ PDF saved to: ", abspath(filename))

    # 2. Define temp file
    temp_file = tempname() * ".png"

    # 3. Save PNG using R's built-in device
    # 'type="cairo"' is safer for Docker if you have libcairo installed, 
    # otherwise remove it.
    R"ggsave($temp_file, plot=$plot_obj, width=$width, height=$height, dpi=$dpi, device='png')"

    # 3. Read raw bytes from disk
    # We don't decode the image; we just grab the binary data
    image_data = read(temp_file)

    # 4. Send raw bytes to VSCode with the correct label (MIME type)
    display("image/png", image_data)

    # 5. Cleanup
    rm(temp_file)
end

function summarize_participation(data::DataFrame)
    participants = combine(groupby(data, [:prolific_pid, :session, :record_id, :exp_start_time]),
        :trialphase => (x -> "experiment_end_message" in x) => :finished,
        :trialphase => (x -> "kick-out" in x) => :kick_out,
        [:trial_type, :trialphase, :block, :n_stimuli] =>
            ((t, p, b, n) -> sum((t .== "PILT") .& (.!ismissing.(p) .&& p .!= "PILT_test") .& (typeof.(b) .== Int64) .& (n .== 2))) => :n_trial_PILT,
        [:block, :trial_type, :trialphase, :n_stimuli] =>
            ((x, t, p, n) -> length(unique(filter(y -> isa(y, Int64), x[(t.=="PILT").&(n.==2).&(.!ismissing.(p).&&p.!="PILT_test")])))) => :n_blocks_PILT,
        # :trialphase => (x -> sum(skipmissing(x .∈ Ref(["control_explore", "control_predict_homebase", "control_reward"])))) => :n_trial_control,
        # :trialPresses => (x -> mean(filter(y -> !ismissing(y), x))) =>  :max_trial_presses,
        :n_warnings => maximum => :n_warnings,
        :time_elapsed => (x -> maximum(x) / 1000 / 60) => :duration,
        :total_bonus => (x -> all(ismissing.(x)) ? missing : only(skipmissing(x))) => :bonus
        # :trialphase => (x -> sum(skipmissing(x .== "control_instruction_quiz_failure"), init=0)) => :n_quiz_failure
    )

    debrief = extract_debrief_responses(data)

    participants = leftjoin(participants, debrief,
        on=[:prolific_pid, :exp_start_time])

    return participants
end

function extract_debrief_responses(data::DataFrame)
    # Select trials
    debrief = filter(x -> !ismissing(x.trialphase) &&
                              occursin(r"(acceptability|debrief)", x.trialphase) &&
                              !(occursin("pre", x.trialphase)), data)


    # Select variables
    select!(debrief, [:prolific_pid, :exp_start_time, :trialphase, :response])

    # Long to wide
    debrief = unstack(
        debrief,
        [:prolific_pid, :exp_start_time],
        :trialphase,
        :response
    )


    # Parse JSON and make into DataFrame
    expected_keys = dropmissing(debrief)[1, Not([:prolific_pid, :exp_start_time])]
    expected_keys = Dict([c => collect(keys(JSON.parse(expected_keys[c])))
                          for c in names(expected_keys)])

    debrief_colnames = names(debrief[!, Not([:prolific_pid, :exp_start_time])])

    # Expand JSON strings with defaults for missing fields
    expanded = [
        DataFrame([
            # Parse JSON or use empty Dict if missing
            let parsed = ismissing(row[col]) ? Dict() : JSON.parse(row[col])
                # Fill missing keys with a default value (e.g., `missing`)
                Dict(key => get(parsed, key, missing) for key in expected_keys[col])
            end
            for row in eachrow(debrief)
        ])
        for col in debrief_colnames
    ]
    expanded = hcat(expanded...)

    # hcat together
    return hcat(debrief[!, Not(debrief_colnames)], expanded)
end

spearman_brown(
    r;
    n=2 # Number of splits
) = (n * r) / (1 + (n - 1) * r)

end # module