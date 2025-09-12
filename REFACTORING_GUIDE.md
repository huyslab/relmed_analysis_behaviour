# Repository Refactoring Guide: Multi-Project Structure

## Overview

This guide outlines a comprehensive approach to restructuring the `relmed_analysis_behaviour` repository to support multiple manuscripts, tutorials, and collaborators while maintaining shared core functionality.

## Current Challenges

- Code duplication across analyses
- Difficulty managing multiple manuscripts from the same codebase
- Risk of breaking changes affecting multiple projects
- Unclear ownership and collaboration boundaries

## Proposed Solution: Hybrid Single-Repo Approach

We recommend starting with a structured single repository that can evolve into multiple repositories as projects mature.

### Key Principles

1. **Shared Core, Isolated Projects**: Common functions in `core/`, project-specific work in `projects/`
2. **Backwards Compatibility**: Prevent breaking changes through versioning and configuration
3. **Clear Ownership**: Each project has clear maintainers and boundaries
4. **Evolution Path**: Easy migration to separate repos when manuscripts are ready for publication

## New Repository Structure

```
relmed_analysis_behaviour/
├── core/                           # Shared core functions
│   ├── data_handling/ # Open and load data, preprocessing
│   ├── model_utils/ # Functions to work with models
│   ├── models/ # Model library, each model follows specific template for naming function signatures
│   ├── analysis_utils/ # Non model-related functions
│   ├── plot_utils/ # Plotting functions
│   ├── config/
│   │   └── defaults.jl
│   └── CHANGELOG.md               # Track breaking changes
├── projects/                      # Individual project folders
│   ├── manuscript_1_pilot_analysis/
│   │   ├── config.jl             # Project-specific configuration. e.g.: plotting specification for journal
│   │   ├── notebooks/
│   │   ├── scripts/
│   │   ├── figures/
│   │   ├── results/
│   │   ├── tests/                # Integration tests
│   │   └── README.md
│   ├── manuscript_2_eeg_analysis/
│   │   ├── config.jl
│   │   ├── notebooks/
│   │   ├── scripts/
│   │   ├── figures/
│   │   ├── results/
│   │   ├── tests/
│   │   └── README.md
│   ├── tutorial_1_modeling/
│   │   ├── notebooks/
│   │   ├── figures/
│   │   └── README.md
│   └── shared_analyses/           # Cross-project analyses
│       ├── task_validation/
│       ├── control_task/
│       └── vigour_analysis/
├── data/                         # Centralized data storage
│   ├── raw/
│   ├── processed/
│   ├── external/
│   └── sequences/                # Experimental sequences
├── archive/                      # Completed/published projects
├── templates/                    # Project templates for new work
├── environment/                  # Docker and environment setup
├── docs/                        # Documentation
│   ├── data_dictionaries/
│   ├── analysis_protocols/
│   └── collaboration_guide.md
└── tests/                       # Repository-wide tests
    └── integration_tests.jl
```

## Minimal core
We must keep core functions lean and general. If in doubt, don't add to core.

## Anti-Breaking-Change Strategies

### 0. Use issues, issue-branches, and PRs
Workflow should always be:
1. Create an issue.
2. Create a branch for the issue.
3. Work on branch.
4. Create tests along with your work.
5. Use a PR to merge into main.

If you need to change core functionality, then the following steps should be added before merging:
1. Make sure tests pass for all projects.
2. Ask for review by another human + Copilot.

### 1. Function Namespacing
The purpose of this is to prevent naming clashes between modules.
**Before (Global Functions - Prone to Conflicts):**
```julia
# plotting_utils.jl
function create_plot(data, type="scatter")
    # implementation
end

# In scripts
include("../plotting_utils.jl")
plot = create_plot(my_data)  # Global namespace
```

**After (Modular Approach):**
```julia
# core/plotting/plotting_utils.jl
module PlottingUtils
    export create_plot, save_figure
    
    function create_plot(data; plot_type="scatter", theme="default")
        # implementation
    end
end

# In project scripts
using ..PlottingUtils: create_plot
plot = create_plot(my_data, plot_type="scatter")
```

### 2. Configuration-Driven Behavior

Rather than rely on function arguments, with changing signatures everytime you add an argument, pass a config object. This way we avoid breaking older uses of a function.

**Global Configuration:**
```julia
# core/config/defaults.jl
const GLOBAL_CONFIG = Dict(
    :data => Dict(
        :cache_enabled => true,
        :default_format => "dataframe",
        :timeout_seconds => 30
    ),
    :plotting => Dict(
        :theme => "default",
        :dpi => 300,
        :color_palette => "viridis"
    ),
    :analysis => Dict(
        :significance_level => 0.05,
        :correction_method => "bonferroni"
    )
)
```

**Project-Specific Overrides:**
```julia
# projects/manuscript_1/config.jl
const PROJECT_CONFIG = merge_configs(GLOBAL_CONFIG, Dict(
    :data => Dict(
        :default_format => "raw",
        :custom_filters => ["exclude_pilot_subjects"]
    ),
    :plotting => Dict(
        :theme => "publication",
        :color_palette => "custom_lab_colors"
    )
))
```

**Configuration-Aware Functions:**
```julia
function fetch_data(source; config=nothing, overrides...)
    cfg = merge_configs(
        GLOBAL_CONFIG[:data],
        get_project_config()[:data],
        config,
        Dict(overrides...)
    )
    
    format = cfg[:default_format]
    cache = cfg[:cache_enabled]
    # ... implementation uses config values
end
```

### 3. Backwards Compatibility

```julia
# Maintain old function signatures with deprecation warnings
function old_function_name(args...)
    @warn "old_function_name is deprecated, use new_function_name instead" maxlog=1
    return new_function_name(args...)
end
```

### 4. Version Management
We will use semantic versioning for each core module separately.

```julia
# Track versions in modules
module DataUtils
    const VERSION = v"1.2.0"
    # ... functions
end
```

## Collaboration Workflow

### Starting New Projects

1. Copy template structure from `templates/project_template/`
2. Create project-specific configuration in `config.jl`
3. Set up project README with goals, timeline, and collaborators
4. Create initial notebook/script structure

### Updating Core Functions

1. **Create Feature Branch**: `git checkout -b core/update-function-name`
2. **Implement with Backwards Compatibility**: Add deprecation warnings for breaking changes
3. **Test Against Projects**: Run integration tests
4. **Document Changes**: Update `core/CHANGELOG.md`
5. **Get Approval**: Tag relevant project maintainers in PR
6. **Grace Period**: Wait 1-2 weeks before removing deprecated functions

### Project-Specific Work

- Full autonomy within project folders
- Import shared functions from `core/`
- Document project-specific requirements in README
- Set up project-specific tests

## Migration Strategy

### Phase 1: Core Refactoring (Week 1-2)
- [ ] Create new directory structure
- [ ] Move utility files to `core/` with module organization
- [ ] Update import statements in existing scripts
- [ ] Test that existing functionality works

### Phase 2: Project Organization (Week 3-4)
- [ ] Create project folders for current analyses
- [ ] Move existing analysis directories into appropriate projects
- [ ] Create project-specific configuration files
- [ ] Write project READMEs

### Phase 3: Data Reorganization (Week 5)
- [ ] Move sequence files to `data/sequences/`
- [ ] Organize other data files appropriately
- [ ] Update data loading paths in scripts

### Phase 4: Documentation & Guidelines (Week 6)
- [ ] Create collaboration guidelines
- [ ] Set up project templates
- [ ] Document the new structure
- [ ] Train collaborators on new workflow

## Long-term Evolution

### Publication-Ready Projects

Once manuscripts are ready for submission:

1. **Extract to Separate Repository**: Use `git subtree split` to create clean repo
2. **Archive in Main Repo**: Move to `archive/` folder
3. **Maintain Independence**: Published repos become self-contained

### Benefits of This Approach

- **Development Phase**: Easy collaboration and experimentation
- **Publication Phase**: Clean, focused repositories for each paper
- **Long-term**: Stable published work, evolving development environment

## Communication Protocol

### Core Changes
- All breaking changes require GitHub issue
- Tag relevant collaborators in PRs
- Provide migration examples in CHANGELOG
- Core maintainer helps with project migrations

### Project Updates
- Regular updates in project READMEs
- Cross-project coordination via shared analyses
- Quarterly review of core function usage

## Getting Started

1. **Read this guide completely**
2. **Review your current project needs**
3. **Identify which analyses belong in which projects**
4. **Start with Phase 1 migration**
5. **Set up your project-specific configuration**
6. **Begin using the new structure for new analyses**

## Questions and Support

- **General questions**: Open issue with `question` label
- **Core function requests**: Open issue with `core-enhancement` label  
- **Project-specific help**: Tag project maintainers in issue
- **Breaking changes**: Open issue with `breaking-change` label

---
