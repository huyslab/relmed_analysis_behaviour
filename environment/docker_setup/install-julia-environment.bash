#!/bin/bash
set -exuo pipefail
# Must run as non-root (jovyan). Assumes:
#   JULIA_DEPOT_PATH=/opt/julia
#   JULIA_PROJECT=/home/jovyan/environment
#   Julia is installed and on PATH

# Install tools INTO THE SAME PROJECT (avoid a separate/global env)

# Instantiate + precompile your pinned environment

# Ensure the IJulia kernel launches with this project
julia --project="${JULIA_PROJECT}" -e '
using IJulia;
installkernel("Julia", env=Dict("JULIA_PROJECT"=>ENV["JULIA_PROJECT"]))
'

# Move the kernelspec out of $HOME so it survives uid changes
mv "${HOME}/.local/share/jupyter/kernels/julia"* "${CONDA_DIR}/share/jupyter/kernels/"
chmod -R go+rx "${CONDA_DIR}/share/jupyter"
rm -rf "${HOME}/.local"


