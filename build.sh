#!/usr/bin/env bash

set -e

# Move to this script's directory.
CDPATH= cd -- "$(dirname -- "$0")"

case $1 in
  plain|debug|debugoptimized|release|minsize)
    BUILDTYPE=$1
    shift
    ;;
  *)
    BUILDTYPE=release
    ;;
esac

BUILDDIR=build/${BUILDTYPE}

MESON=$(PATH="${PATH}:${HOME}/.local/bin" command -v meson || :)
MESON=${MESON:?"Could not find meson. Is it installed and in PATH?"}

# b_lto_threads=0 → GCC picks nproc for LTRANS partitions, avoiding the
# "using serial compilation of N LTRANS jobs" warning.  Caller can override
# by passing -Db_lto_threads=N in "$@".
if [ -f "${BUILDDIR}/build.ninja" ]
then
  "${MESON}" configure "${BUILDDIR}" -Dbuildtype="${BUILDTYPE}" -Dprefix="${INSTALL_PREFIX:-/usr/local}" -Db_lto_threads=0 "$@"
else
  "${MESON}" setup "${BUILDDIR}" --buildtype "${BUILDTYPE}" --prefix "${INSTALL_PREFIX:-/usr/local}" -Db_lto_threads=0 "$@"
fi

"${MESON}" compile -C "${BUILDDIR}"

if [ -n "${INSTALL_PREFIX}" ]
then
  "${MESON}" install -C "${BUILDDIR}"
fi
