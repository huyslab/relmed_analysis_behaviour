# Environment

This folder contains Docker files and environment setup scripts to ensure reproducibility and consistent development environments across projects.

## Overview

The environment is built using Docker and provides a complete data science stack with:
- **Julia 1.10.3** with packages for statistical modeling, optimization, and data analysis
- **R** with Stan ecosystem (RStan, CmdStan, brms) for Bayesian modeling
- **Python** via Jupyter Data Science notebook stack
- **Jupyter Lab** with support for Julia, R, and Python kernels
- **Pluto.jl** for interactive Julia notebooks

## Files and Components

### Core Files

- **`Dockerfile`**: Main Docker configuration that builds the complete environment
- **`Project.toml`** & **`Manifest.toml`**: Julia package environment specification ensuring reproducible Julia dependencies
- **`docker_setup/`**: Directory containing installation scripts for different components

### Setup Scripts

#### Julia Setup
- **`install-julia.bash`**: Installs Julia 1.10.3 and configures system paths
- **`install-julia-environment.bash`**: Installs Julia packages and sets up Jupyter kernels

#### R Setup  
- **`install_r.R`**: Installs R packages including:
  - Statistical packages: `data.table`, `psych`, `GPArotation`, `afex`, `emmeans`
  - Bayesian modeling: `rstan`, `brms`
  - Visualization: `ggplot2`, `cowplot`, `GGally`
  - Data manipulation: `tidyverse`
  - Performance: `qs` for fast serialization

#### Stan/CmdStan Setup
- **`install_cmdstan.R`**: Installs CmdStanR and CmdStan 2.34.1 for high-performance Bayesian modeling

## Key Julia Packages

The environment includes comprehensive Julia packages for:

- **Data handling**: `CSV`, `DataFrames`, `HDF5`, `JLD2`
- **Statistical modeling**: `GLM`, `MixedModels`, `Distributions`, `HypothesisTests`
- **Bayesian analysis**: `Turing`, `MCMCDiagnosticTools`, `ParetoSmooth`
- **Optimization**: `Optim`, `JuMP`, `HiGHS`, `Cbc`
- **Visualization**: `Plots`, `CairoMakie`, `AlgebraOfGraphics`, `StatsPlots`
- **Interactive computing**: `Pluto`, `IJulia`, `PlutoUI`
- **Development tools**: `Revise`, `Debugger`

## Usage

### Building the Docker Environment

```bash
# Build the Docker image for Apple Silicone and amd64 simultaneously
docker buildx build --push --platform linux/arm64/v8,linux/amd64 -t user/tag:version --progress=plain . 2>&1 | tee build.log

# Run the container with Jupyter Lab
docker run -it --rm --name relmed -p 8888:8888 -v $(pwd):/home/jovyan --env-file env.list user/tag:version
```

### Environment Features

1. **Reproducible Dependencies**: Fixed versions for Julia (Manifest.toml), R (dated CRAN repo), and system packages
4. **Bayesian Modeling**: Complete Stan ecosystem with CmdStan 2.34.1
5. **Interactive Development**: Jupyter Lab + Pluto.jl for exploratory analysis

## Version Information

- **Base**: Jupyter Data Science Notebook (2024-04-29)
- **Julia**: 1.10.3
- **R**: Latest from conda-forge
- **CmdStan**: 2.34.1
- **CRAN snapshot**: 2024-05-03