# RELMED Behaviour Analysis Repository

This repository contains code for running simulations and analyzing behavioural data from the RELMED project. The repository is organized using a modular structure to support reproducible research and collaborative development.

## Repository Structure

```
.
├── core/                    # Shared core functions and utilities
│   ├── models/             # Model implementations
│   └── model_utils.jl      # Utility functions for computational modeling
├── projects/               # Individual project directories (manuscripts, analyses)
├── templates/              # Project templates for new analyses
├── docs/                   # Documentation and protocols
├── tests/                  # Repository-wide integration tests
├── environment/            # Docker environment and setup scripts
├── archive/                # Completed/published projects
├── OLD/                    # Legacy code (pre-refactoring)
├── env.list               # Environment variables (local, not tracked by git)
└── README.md              # This file
```

## Getting Started

### Prerequisites

- Docker Desktop with sufficient memory and CPU allocation
- Git for version control

### Environment Setup

The code runs in a Docker container that provides a complete data science environment with Julia, R, Python, and Jupyter Lab. 

**Create local environment file**: First, create a local `env.list` file (not tracked by git) with machine-specific settings:

```bash
# Copy the example and edit with your settings
cp env.list.example env.list
```

The `env.list` file should contain:
- `JULIA_NUM_THREADS=8`: Number of CPU cores available (adjust based on your machine and Docker settings)
- `REDCAP_URL=https://redcap.slms.ucl.ac.uk/api/`: REDCap database URL (if accessing data)
- `REDCAP_TOKEN_{PROJECT}=YOURTOKEN`: Your REDCap access token to a given project (if accessing pilot data; replacing {PROJECT} with project name, e.g. pilot1 or trial1)

*Note: REDCap credentials are only needed if downloading raw data. Preprocessed data is available on OSF.*

*Currently, only a Mac Silicone docker image is available. If you require an amd64 image, please let Yaniv or Haoyang know.*

### Option 1: Using Jupyter Lab

1. **Launch Docker container**:

```bash
docker run -it --rm --name relmed -p 8888:8888 -v $(pwd):/home/jovyan --env-file env.list yanivabir/relmed:v1.12
```

2. **Access Jupyter Lab**: Copy the URL from the Docker output and navigate to it in your browser. You can launch Pluto.jl notebooks from within Jupyter Lab for interactive analysis.

### Option 2: Using VS Code

1. **Launch container with bash**:

```bash
docker run -it --rm --name relmed -v $(pwd):/home/jovyan --env-file env.list yanivabir/relmed:v1.12 /bin/bash
```

2. **Attach VS Code to container**: 
   - Install the "Dev Containers" extension in VS Code
   - Open VS Code and use `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
   - Run "Dev Containers: Attach to Running Container..."
   - Select the `relmed` container from the list
   - VS Code will open a new window connected to the container

3. **Open workspace**: Navigate to `/home/jovyan` in the VS Code file explorer to access your project files

## Development Environment

The Docker environment includes:

- **Julia 1.10.3** with packages for statistical modeling, optimization, and data analysis
- **R** with Stan ecosystem (RStan, CmdStan, brms) for Bayesian modeling  
- **Python** via Jupyter Data Science notebook stack
- **Jupyter Lab** with multi-language kernel support
- **Pluto.jl** for interactive Julia notebooks

See [`environment/README.md`](environment/README.md) for detailed information about the environment setup and included packages.

## Working with the Repository

### Core Functions

The [`core/`](core/) directory contains shared utilities used across projects:
- **`model_utils.jl`**: Functions for computational modeling, parameter handling, and model fitting
- **`models/`**: Reusable model implementations

⚠️ **Important**: Changes to core functions require GitHub issues and collaborative review to maintain compatibility across projects.

### Starting a New Project

1. Copy an appropriate template from [`templates/`](templates/)
2. Create a new directory in [`projects/`](projects/) 
3. Follow the project structure guidelines in the template
4. Document your analysis approach and findings

### Documentation

- [`docs/`](docs/): Contains data dictionaries, analysis protocols, and collaboration guidelines
- [`docs/REFACTORING_GUIDE.md`](docs/REFACTORING_GUIDE.md): Information about the repository restructuring

### Testing

Run repository tests to ensure core functions work correctly:

```julia
# In Julia REPL or notebook
using Pkg
Pkg.test()
```

## Data Access

- **REDCap**: Raw pilot data (requires credentials in `env.list`)
- **OSF**: Preprocessed data available for download
- **Local storage**: Downloaded data should be stored within individual project directories

## Legacy Code

The [`OLD/`](OLD/) directory contains code from before the repository restructuring. This is maintained for reference and reproducibility of previous analyses.

## Contributing

1. Create feature branches for new work
2. Follow the modular project structure
3. Document changes and analysis approaches
4. Test core function changes thoroughly
5. Tag relevant collaborators for review

## Environment Updates

The Docker environment is maintained in the [`environment/`](environment/) directory.

## Support

- Check [`docs/`](docs/) for protocols and guidelines
- Review existing projects in [`projects/`](projects/) for examples
- Consult [`OLD/`](OLD/) for legacy implementation references
- Create GitHub issues for bug reports or feature requests
