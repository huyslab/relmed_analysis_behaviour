#!/bin/bash
set -exuo pipefail
# Requirements:
# - Run as root
# - JULIA_PKGDIR=/opt/julia is set (same as JULIA_DEPOT_PATH)

JULIA_VERSION="${JULIA_VERSION:-1.10.10}"

JULIA_ARCH=$(uname -m)
JULIA_SHORT_ARCH="${JULIA_ARCH}"
if [ "${JULIA_SHORT_ARCH}" == "x86_64" ]; then
    JULIA_SHORT_ARCH="x64"
fi

JULIA_INSTALLER="julia-${JULIA_VERSION}-linux-${JULIA_ARCH}.tar.gz"
JULIA_MAJOR_MINOR=$(echo "${JULIA_VERSION}" | cut -d. -f 1,2)

cd /tmp
mkdir "/opt/julia-${JULIA_VERSION}"
curl --progress-bar --location --output "${JULIA_INSTALLER}" \
  "https://julialang-s3.julialang.org/bin/linux/${JULIA_SHORT_ARCH}/${JULIA_MAJOR_MINOR}/${JULIA_INSTALLER}"
tar xzf "${JULIA_INSTALLER}" -C "/opt/julia-${JULIA_VERSION}" --strip-components=1
rm "${JULIA_INSTALLER}"

ln -fs /opt/julia-*/bin/julia /usr/local/bin/julia

mkdir -p /etc/julia
echo "push!(Libdl.DL_LOAD_PATH, \"${CONDA_DIR}/lib\")" >> /etc/julia/juliarc.jl

# Create the shared depot and give it to jovyan
mkdir -p "${JULIA_PKGDIR}"
chown -R "${NB_USER}:${NB_GID}" "${JULIA_PKGDIR}"
fix-permissions "${JULIA_PKGDIR}"