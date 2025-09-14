# Core Module

This folder contains shared core functions and utilities used across multiple projects. Only general-purpose, reusable code should be placed here. Keep the core minimal and well-documented.

Any change to functions in these folders should respect the following rules:
- All breaking changes require GitHub issue
- Tag relevant collaborators in PRs
- Provide migration examples in CHANGELOG
- Deprecate functions before removing them

## Files

- **model_utils.jl**: Utility functions for computational modeling, including parameter handling, model fitting helpers, and data processing functions commonly used across different modeling approaches.

