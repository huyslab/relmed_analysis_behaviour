# Data Dictionary for Control Data

The current version (Pilot 8) has 192 trials in total, including 144 explore trials, 24 home base prediction trials, and 24 reward trials.

The current task structure is: 6 explore trials (#trials 1-6) —> 2 prediction trials (#trials 7-8; prediction + confidence rating + prediction + confidence rating) —> 6 explore trials (#trials 9-12) —> 2 reward trials(#trials 13-14; controllability rating + reward + reward) —> …

## For Task Data (i.e., not including self-report)

| Variable                | Description                                                  |
| ----------------------- | ------------------------------------------------------------ |
| **exp_start_time**      | Timestamp recording when the experiment started              |
| **prolific_pid**        | Participant ID for Prolific platform users; used when context is "prolific" and taken from PROLIFIC_PID URL parameter |
| **record_id**           | Unique identifier for participant's data record              |
| **session**             | Identifier for experimental session number; retrieved from URL parameter "session" |
| **task**                | Identifier for the specific task being performed; possible values: "pilt", "pilt-to-test", "vigour", "pit", "ltm", "wm", "wm_only", "reversal", "control", "quests", and **"pilot8"** |
| **time_elapsed**        | Time elapsed since experiment start (in milliseconds)        |
| **rt**                  | Response time for participant actions (in milliseconds); measures time from stimulus presentation to participant's response<br />**Note**: for explore and reward trials, rt is the response time of the first choice (left or right) |
| **trialphase**          | Identifier for the current phase/stage of a trial; possible values in this dataset include: "control_explore", "control_predict_homebase", "control_reward" |
| **response**            | Participant's response choice; possible values: "left", "right", or numerical indices representing choices<br />**Note**: "left" or "right" are only available for explore and reward trials; in prediction trials, it will be 0, 1, 2, or 3, corresponding to each home base button (see variable `button`) |
| **trial_presses**       | Count of key presses made by participant during effort-based tasks; used to measure effort level in control tasks<br />**Note**: only available in explore and reward trials; this doesn't include the first decision response (i.e., choosing between left and right option) |
| **button**              | The specific button/option selected by participant; contains actual value (e.g., island name: "banana", "coconut", "grape", "orange") rather than position<br />**Note**: only available in prediction trials |
| **trial**               | Trial number within the experiment<br />**Note**: self-reports (confidence and controllability) are included as a part of prediction and reward trials; thus, not counted as a separate trial |
| **left**                | Color/type of the ship on the left side; possible values: "green", "blue", "red", "yellow" |
| **right**               | Color/type of the ship on the right side; possible values: "green", "blue", "red", "yellow" |
| **near**                | Type of the near island (i.e., the current state) in control explore and reward trials; possible values: "banana", "coconut", "grape", "orange" |
| **current**             | Strength of ocean current in control task; range: 1-3 (1=Low, 2=Mid, 3=High)<br />**Note**: 6/12/18 presses for Low/Mid/High threshold |
| **ship**                | Color/type of the ship to predict the home base in prediction tasks; possible values: "green", "blue", "red", "yellow" |
| **island_viable**       | Whether the current near island is a viable destination, only available in reward trials; values: true/false<br />**Note**: when true, it means participant could actually reach the target island only by drifting with the current (i.e., using base uncontrollable rule) |
| **target**              | Target island for delivery in reward trials; possible values: "banana", "coconut", "grape", "orange" |
| **right_viable**        | Whether the right ship choice is viable for goal in reward trials; values: true/false |
| **left_viable**         | Whether the left ship choice is viable for goal in reward trials; values: true/false |
| **response_times**      | Array of response times for repeated key presses in explore and reawrd trials; captures timing between successive key presses during effort phases |
| **control_rule_used**   | Rule system used to determine destination island in explore and reawrd trials; possible values: "base", "control"; depends on effort level and current strength |
| **destination_island**  | Island where ship actually docks after sailing in explore and reward trials; possible values: "banana", "coconut", "grape", "orange" |
| **probability_control** | Probability of using control rule based on effort and current; range: 0.0-1.0; based on sigmoid function of effort relative to threshold<br />**Note**: only available in explore trials (but it should easily available for reward trials) |
| **ship_color**          | Color of the chosen ship in explore and reward trials; possible values: "green", "blue", "red", "yellow" |
| **correct**             | Whether participant's action achieved the desired outcome; used in prediction tasks or reward tasks |

## For Self-Report Data

| Variable           | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| **exp_start_time** | Timestamp recording when the experiment started              |
| **prolific_pid**   | Participant ID for Prolific platform users; used when context is "prolific" and taken from PROLIFIC_PID URL parameter |
| **record_id**      | Unique identifier for participant's data record              |
| **session**        | Identifier for experimental session number; retrieved from URL parameter "session" |
| **task**           | Identifier for the specific task being performed             |
| **time_elapsed**   | Time elapsed since experiment start (in milliseconds)        |
| **trialphase**     | Identifier for the current phase/stage of a trial; in self-report context, this typically indicates question type: control_confidence (comes after each prediction trial), and control_controllability (comes before the first of two consecutive reward trials) |
| **trial**          | Trial number within the experiment; overlapped with trial numbers from control_task_data |
| **rt**             | Response time for answering a question (in milliseconds); measures time from question presentation to participant's response |
| **response**       | Participant's response to the self-report item; Likert scale values from 0 to 4<br />In confidence rating: 0 - Not at all, 4 - Very confidence<br />In controllability rationg: 0 - Not at all, 2 - I don't know, 4 - Completely in control |