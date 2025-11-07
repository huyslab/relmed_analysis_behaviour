# Based on the Dockerfile for juypter/datascience 2024-05-04. 
# Edited to fix Julia version to 1.10.3, fix Julia package versions
# using Project.toml and Manifest.toml, and fix R package versions
# by using a dated CRAN repo, and installing R packages only via R.
# Additionally, CMDSTAN is installed.
# JULIA_CPU_TARGET setup before precompile see 
# https://github.com/jupyter/docker-stacks/blob/main/images/minimal-notebook/setup-scripts/setup-julia-packages.bash

ARG REGISTRY=quay.io
ARG OWNER=jupyter
ARG BASE_CONTAINER=$REGISTRY/$OWNER/scipy-notebook:2024-04-29 
FROM $BASE_CONTAINER

# Fix: https://github.com/hadolint/hadolint/wiki/DL4006
# Fix: https://github.com/koalaman/shellcheck/wiki/SC3014
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER root

# R pre-requisites
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    fonts-dejavu \
    gfortran \
    gcc \
    libgit2-dev \
    pkg-config \
    build-essential \
    libxml2-dev \
    libssl-dev \
    tmux \
    vim && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install R
RUN mamba install --yes \
     'r-base=4.4.3' 'r-essentials=4.4' && \
     mamba clean --all -f -y && \
     fix-permissions "${CONDA_DIR}" && \
     fix-permissions "/home/${NB_USER}"

# Verify Rscript is installed
RUN which Rscript && Rscript --version

# Single shared depot + your project path
ENV JULIA_DEPOT_PATH=/opt/julia \
    JULIA_PKGDIR=/opt/julia \
    JULIA_PROJECT=/home/${NB_USER}/environment

# Setup Julia
COPY docker_setup/install-julia.bash /opt/setup-scripts/install-julia.bash
RUN chmod +x /opt/setup-scripts/install-julia.bash
RUN /opt/setup-scripts/install-julia.bash

# Create depot and give it to jovyan
RUN mkdir -p /opt/julia && chown -R ${NB_USER}:${NB_GID} /opt/julia

# Switch to runtime user for all Julia package work
USER ${NB_UID}

# Setup packages
RUN mkdir -p environment
COPY --chown=jovyan Manifest.toml Project.toml "/home/${NB_USER}/environment/"
RUN if [ "$(uname -m)" = "x86_64" ]; then \
        export JULIA_CPU_TARGET="generic;sandybridge,-xsaveopt,clone_all;haswell,-rdrnd,base(1);x86-64-v4,-rdrnd,base(1)"; \
    elif [ "$(uname -m)" = "aarch64" ]; then \
        export JULIA_CPU_TARGET="generic;cortex-a57;thunderx2t99;carmel,clone_all;apple-m1,base(3);neoverse-512tvb,base(3)"; \
    fi && \
    julia --project="${JULIA_PROJECT}" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile();'

# Setup IJulia kernel
USER root
RUN julia --project="${JULIA_PROJECT}" -e 'using IJulia; installkernel("Julia", env=Dict("JULIA_PROJECT"=>ENV["JULIA_PROJECT"]))'

# Move the kernelspec out of $HOME so it survives uid changes
RUN mv "${HOME}/.local/share/jupyter/kernels/julia"* "${CONDA_DIR}/share/jupyter/kernels/" && \
    chmod -R go+rx "${CONDA_DIR}/share/jupyter" && \
    rm -rf "${HOME}/.local"

# Fix permissions
RUN fix-permissions "${JULIA_PKGDIR}" "${CONDA_DIR}/share/jupyter" && \
    fix-permissions "/home/${NB_USER}"


# Set CRAN date
RUN echo 'options(repos = c(CRAN = "https://packagemanager.rstudio.com/cran/2024-09-16"), download.file.method = "libcurl")' >> $(Rscript -e 'cat(R.home())')/etc/Rprofile.site

# lme4 pre-requisites
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    cmake && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Run script installing R packages
COPY docker_setup/install_r.R install_r.R
RUN ["Rscript", "install_r.R"]

# CMDSTAN pre-requisites
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    clang && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Run script installing cmdstan
COPY docker_setup/install_cmdstan.R install_cmdstan.R
# Add build-time argument for architecture
ARG TARGETARCH

# Conditionally run CmdStan install only for ARM64
RUN if [ "$TARGETARCH" = "arm64" ]; then \
      echo "Building for ARM64 – installing CmdStan..."; \
      Rscript install_cmdstan.R; \
    else \
      echo "Skipping CmdStan install for $TARGETARCH build."; \
    fi

# Fix permissions if the folder exists
RUN if [ -d "/opt/cmdstan" ]; then \
      fix-permissions "/opt/cmdstan"; \
    fi
RUN fix-permissions "/home/${NB_USER}"

# Set CMDSTAN home
ENV CMDSTAN=/opt/cmdstan/cmdstan-2.37.0

USER ${NB_UID}