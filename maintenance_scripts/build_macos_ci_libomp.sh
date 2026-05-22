#!/usr/bin/env bash
# Build a libomp.dylib whose minimum macOS version matches the wheel tag.
# Homebrew libomp on modern CI runners often requires macOS 14+, which breaks
# delocate when wheels target older macOS releases.
set -euo pipefail

usage() {
  echo "Usage: $0 <arch> <deployment_target> [install_prefix]" >&2
  exit 1
}

ARCH="${1:-}"
DEPLOY_TARGET="${2:-}"
INSTALL_PREFIX="${3:-${GITHUB_WORKSPACE:-$(pwd)}/.local-libomp}"
LLVM_OPENMP_VERSION="${LLVM_OPENMP_VERSION:-18.1.8}"

if [[ -z "$ARCH" || -z "$DEPLOY_TARGET" ]]; then
  usage
fi

for tool in cmake clang clang++ git; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "[ERROR] Required tool not found: $tool" >&2
    exit 1
  fi
done

LIBOMP="$INSTALL_PREFIX/lib/libomp.dylib"

_libomp_arch_ok() {
  file "$LIBOMP" | grep -Eiq "$ARCH"
}

_libomp_deploy_ok() {
  local minos=""
  minos=$(otool -l "$LIBOMP" | awk '
    /cmd LC_BUILD_VERSION/ { in_build=1; next }
    in_build && /minos/ { print $2; exit }
  ')
  if [[ -z "$minos" ]]; then
    minos=$(otool -l "$LIBOMP" | awk '
      /cmd LC_VERSION_MIN_MACOSX/ { in_min=1; next }
      in_min && /version/ { print $2; exit }
    ')
  fi
  [[ "$(printf '%s' "$minos" | tr -d '[:space:]')" == "$DEPLOY_TARGET" ]]
}

_libomp_symbol_ok() {
  # LLVM-built libomp exports kmpc symbols with nm -g; avoid brittle exact names.
  nm -g "$LIBOMP" 2>/dev/null | grep -q 'kmpc_dispatch_deinit'
}

if [[ -f "$LIBOMP" ]] && _libomp_arch_ok && _libomp_deploy_ok && _libomp_symbol_ok; then
  echo "[OK] Reusing libomp at $INSTALL_PREFIX (arch=$ARCH, min macOS=$DEPLOY_TARGET)"
  exit 0
fi

echo "Building LLVM OpenMP ${LLVM_OPENMP_VERSION} for ${ARCH} (min macOS ${DEPLOY_TARGET})..."

WORK="$(mktemp -d)"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

LLVM_SRC="$WORK/llvm-project"
for attempt in 1 2 3; do
  if git clone --depth 1 --branch "llvmorg-${LLVM_OPENMP_VERSION}" --filter=blob:none --sparse \
    https://github.com/llvm/llvm-project.git "$LLVM_SRC"; then
    break
  fi
  if [ "$attempt" -eq 3 ]; then
    echo "[ERROR] Failed to clone llvm-project after 3 attempts" >&2
    exit 1
  fi
  echo "git clone failed, retrying in ${attempt}0s..." >&2
  rm -rf "$LLVM_SRC"
  sleep $((attempt * 10))
done

(
  cd "$LLVM_SRC"
  git sparse-checkout set openmp cmake
)

OPENMP_SRC="$LLVM_SRC/openmp"
for required_path in \
  "$OPENMP_SRC/CMakeLists.txt" \
  "$LLVM_SRC/cmake/Modules/ExtendPath.cmake" \
  "$LLVM_SRC/cmake/Modules/LLVMCheckCompilerLinkerFlag.cmake"; do
  if [[ ! -f "$required_path" ]]; then
    echo "[ERROR] Missing LLVM source component: $required_path" >&2
    exit 1
  fi
done

BUILD="$WORK/build"
mkdir -p "$BUILD" "$INSTALL_PREFIX"

cmake -S "$OPENMP_SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
  -DCMAKE_OSX_ARCHITECTURES="$ARCH" \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="$DEPLOY_TARGET" \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLIBOMP_ENABLE_SHARED=ON \
  -DLIBOMP_INSTALL_ALIASES=OFF

cmake --build "$BUILD" --target install -j"$(sysctl -n hw.ncpu 2>/dev/null || echo 2)"

if ! _libomp_arch_ok || ! _libomp_deploy_ok || ! _libomp_symbol_ok; then
  echo "[ERROR] Built libomp failed verification:" >&2
  _libomp_arch_ok || echo "  - architecture mismatch (expected ${ARCH})" >&2
  _libomp_deploy_ok || echo "  - deployment target mismatch (expected ${DEPLOY_TARGET})" >&2
  _libomp_symbol_ok || echo "  - missing OpenMP runtime symbols" >&2
  file "$LIBOMP" >&2 || true
  otool -l "$LIBOMP" | awk '/LC_BUILD_VERSION|LC_VERSION_MIN_MACOSX|minos|version/' >&2 || true
  nm -g "$LIBOMP" 2>/dev/null | grep kmpc | head -5 >&2 || true
  exit 1
fi

echo "[OK] Installed libomp to $INSTALL_PREFIX"
file "$LIBOMP"
otool -l "$LIBOMP" | awk '/LC_BUILD_VERSION|LC_VERSION_MIN_MACOSX|minos|version/'
