# Based on the Dockerfile for juypter/datascience 2024-05-04. 
# Edited to fix Julia version to 1.10.3, fix Julia package versions
# using Project.toml and Manifest.toml, and fix R package versions
# by using a dated CRAN repo, and installing R packages only via R.
# Additionally, CMDSTAN is installed

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
     'r-base' && \
     mamba clean --all -f -y && \
     fix-permissions "${CONDA_DIR}" && \
     fix-permissions "/home/${NB_USER}"

# install Julia packages in /opt/julia instead of ${HOME}
ENV JULIA_DEPOT_PATH=/opt/julia \
    JULIA_PKGDIR=/opt/julia

# Setup Julia
COPY docker_setup/install-julia.bash /opt/setup-scripts/install-julia.bash
RUN chmod +x /opt/setup-scripts/install-julia.bash
RUN /opt/setup-scripts/install-julia.bash

# USER ${NB_UID}

# Setup packages
COPY --chown=jovyan Manifest.toml Project.toml "/home/${NB_USER}" 
COPY docker_setup/install-julia-environment.bash /opt/setup-scripts/install-julia-environment.bash
RUN chmod +x /opt/setup-scripts/install-julia-environment.bash
RUN /opt/setup-scripts/install-julia-environment.bash && \
    fix-permissions "/home/${NB_USER}"

# # Set CRAN date
RUN echo 'options(repos = c(CRAN = "https://packagemanager.rstudio.com/cran/2024-05-03"), download.file.method = "libcurl")' >> $(Rscript -e 'cat(R.home())')/etc/Rprofile.site

# Run script installing R packages
COPY docker_setup/install_r.R install_r.R
RUN ["Rscript", "install_r.R"]


# CMDSTAN pre-requisites
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    clang &&\
    apt-get clean && rm -rf /var/lib/apt/lists/*


# Run script installing cmdstand
COPY docker_setup/install_cmdstan.R install_cmdstan.R
RUN ["Rscript", "install_cmdstan.R"] && \
    fix-permissions "/home/${NB_USER}"

# Set CMDSTAN home
ENV CMDSTAN_HOME=/home/jovyan/.cmdstanr/cmdstan-2.34.1

