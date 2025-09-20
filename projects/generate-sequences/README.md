# Generate Sequences Project

This project generates experimental sequences for the RELMED task battery, specifically for PILT (Probabilistic Instrumental Learning Task), RLWM (Reinforcement Learning Working Memory), and reversal learning tasks.

## Prerequisites

Before running the sequence generation scripts, you need to have the **THINGS stimulus folder** present in the project directory. Download it from [https://things-initiative.org/](https://things-initiative.org/) and name it `THINGS`.

### Directory Structure
```
/home/jovyan/projects/generate-sequences/
├── THINGS/                     # Required: THINGS stimulus database folder
├── data/                       # Contains CSV files with stimulus metadata
├── results/                    # Output directory for generated sequences
├── trial1_stimuli/            # Output directory for processed stimuli
├── sequence_utils.jl          # Utility functions for sequence generation
├── trial1_process_THINGS.jl   # Image processing script
└── trial1.jl                  # Main sequence generation script
```

### Required Data Files
The following CSV files are in the `data/` directory:
- `Concept_to_category_linking.csv`
- `THINGS_Memorability_Scores.csv` 
- `wm_stimulus_sequence.csv`

## How to Run

### Step 1: Process THINGS Stimuli
First, run the image processing script to select and prepare stimuli from the THINGS database:

```bash
julia trial1_process_THINGS.jl
```

**What this script does:**
- Selects appropriate stimulus categories based on memorability scores
- Filters out dangerous, disgusting, scary, or inappropriate content
- Copies and resizes selected images to 500x500 pixels
- Creates a `stimuli.csv` file with metadata
- Outputs processed images to `trial1_stimuli/` folder

**Parameters configured in the script:**
- `n_categories_PILT = 21` (categories for PILT task)
- `n_categories_WM = 36` (categories for working memory task)  
- `n_categories_LTM = 12` (categories for long-term memory task)
- `n_sessions = 6` (number of experimental sessions)
- `n_extra = 50` (extra categories as buffer)

### Step 2: Generate Experimental Sequences
After processing the stimuli, run the main sequence generation script:

```bash
julia trial1.jl
```

**What this script does:**
- Generates optimized trial sequences for multiple experimental tasks:
  - **PILT (Probabilistic Instrumental Learning Task)**: Reward learning with probabilistic feedback
  - **RLWM (Reinforcement Learning Working Memory)**: Three-choice learning with working memory delays
  - **Reversal Learning**: Two-choice learning with reward contingency reversals
- Creates balanced experimental designs across 6 sessions (screening, wk0, wk2, wk4, wk24, wk28)
- Optimizes sequences for statistical power and counterbalancing
- Exports results as both CSV files and JavaScript files for web-based experiments

## Output Files

After running both scripts, you'll find the generated sequences in the `results/` directory:

### Main Sequence Files
- `trial1_PILT.csv` / `trial1_*_sequences.js` - PILT task sequences
- `trial1_PILT_test.csv` / `trial1_*_sequences.js` - PILT test sequences  
- `trial1_WM.csv` / `trial1_*_sequences.js` - Working memory task sequences
- `trial1_WM_test.csv` / `trial1_*_sequences.js` - Working memory test sequences

### Visualization Files
- `trial1_pilt_trial_plan.png` - Visual summary of PILT trial structure
- `trial1_reversal_sequence.png` - Visual summary of reversal learning structure

### Processed Stimuli
- `trial1_stimuli/` - Folder containing all processed stimulus images
- `trial1_stimuli/stimuli.csv` - Metadata for all selected stimuli

## Task Descriptions

### PILT (Probabilistic Instrumental Learning Task)
- Two-choice learning task with probabilistic rewards/punishments
- Participants learn which stimulus is better through trial and error
- Includes both deterministic and probabilistic blocks
- Tests ability to learn from feedback under uncertainty

### RLWM (Reinforcement Learning Working Memory) 
- Three-choice learning task with working memory delays
- Single stimulus per trial with left/middle/right response options
- Tests interaction between reinforcement learning and working memory

### Reversal Learning
- Two-choice task where reward contingencies reverse across blocks
- Tests cognitive flexibility and adaptation to changing environments
- 30 blocks with varying reversal criteria and feedback reliability

## Dependencies

The scripts require Julia with the following packages:
- `CairoMakie`, `CSV`, `DataFrames`, `Combinatorics`, `StatsBase`, `Random`
- `CategoricalArrays`, `AlgebraOfGraphics`, `IterTools`
- `Images`, `ImageTransformations` (for image processing)
- `JSON`, `JLD2` (for data export)

These dependencies are managed through the shared project environment at `/home/jovyan/environment/`.

## Customization

Key parameters can be modified in the scripts:
- Number of categories per task type (`trial1_process_THINGS.jl`)
- Trial counts and block structure (`trial1.jl`)
- Feedback probabilities and reward magnitudes
- Session timing and counterbalancing schemes

## Troubleshooting

**Common issues:**
1. **Missing THINGS folder**: Ensure the THINGS stimulus database is properly downloaded and placed in the project directory
2. **Missing data files**: Check that all required CSV files are present in the `data/` folder  
3. **Memory issues**: The image processing step may require substantial RAM for large stimulus sets
4. **Permission errors**: Ensure write permissions for the `results/` and `trial1_stimuli/` directories

For additional help, check the inline documentation in the Julia scripts or examine the generated log outputs.
