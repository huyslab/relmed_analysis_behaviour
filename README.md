# RELMED behaviour analysis repository

Code in this repository runs simulations and analyses behavioural data from RELMED.

The code runs on a Docker container. To lauch the docker container, first you need to create a local version of the file env.list (which won't be tracked by git). This file contains settings that are specific to your machine, as well as passwords:

1. `JULIA_NUM_THREADS=11`: Number of cores available on your machine. Make sure that Docker desktop is set to allow you to use this number of cores.
2. `REDCap_url=https://redcap.slms.ucl.ac.uk/api/`: Url of REDCap database to download pilot1 data.
3. `REDCap_token=YOURTOKEN`: Your URL token for REDCap.

(If you don't want to download data from REDCap, you don't need the latter two. The data is saved on osf in preprocessed format which you can download.)

Then run the following command in the Mac terminal:

```
docker run -it --rm --name relmed -p 8888:8888 -v $(pwd):/home/jovyan --env-file env.list yanivabir/relmed:v1.1ma
```

Copy the URL for Jupyter in the resulting ouptut, navigate to it in your browser. Launch a Pluto.jl instance from Jupyter to view the Pluto notebooks in the repository.

## File map
```
.
├── PLT_task_functions.jl - Functions to simulate PLT.
├── plotting_functions.jl - Function to help plot simulations and data.
├── fetch_preprocess_data.jl - Functions for fetching data from REDCap database and preprocess.
├── fisher_information_functions.jl - Functions to compute and analyse FI for simulations.
├── stan_functions.jl - Functions to run Stan models and work with posteriors
├── env.list - Environment variables needed to interact with APIs.
├── scripts_pilot1\ - Notebooks running anlysis of Pilot 1.
│   ├── data_plots.jl - Plotting raw data
│   └── whitelist.jl - Computing bonus and preparing approval and second session invitation lists.
├── scripts_simulation - Notebooks running simulations.
│   ├── interactive_simulate_plot.jl - Interactive plots of Q-Learning simulations in PLT.
│   ├── simulate_plot_fisher_information.jl
│   ├── stan_QL_single_participant.jl - Model development single participant PLT.
│   ├── stan_QL_group.jl - Model development group PLT.
│   ├── create_exp_sequences.jl - Select PLT feedback sequences for pilot1 based on FI.
│   └── free_oprant.jl - Simulations of free operant paradigm.
├── scripts_conferences\ - Notebooks used to make figures for conferences and talks.
├── data\ - Downloaded and preprocessed data 
├── relmed_environment\ - Files for building Docker container. This is a git subtree.
└── saved_models\ - Not tracked on git.
```
