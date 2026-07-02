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
  # Verify libomp exports standard OpenMP runtime symbols AND __kmpc_dispatch_deinit.
  #
  # __kmpc_dispatch_deinit is an Intel-private OpenMP symbol removed from the LLVM
  # runtime in LLVM 13, but Apple Clang's OpenMP codegen still emits calls to it for
  # loops with dynamic/guided scheduling.  Without it, dlopen fails at wheel-install
  # time with "symbol not found in flat namespace '___kmpc_dispatch_deinit'".
  # We inject a no-op stub at build time (see below) so the bundled libomp always
  # satisfies this symbol — matching what Apple Clang's codegen expects.
  #
  # NOTE: dump to a temp file before grepping to avoid SIGPIPE under set -o pipefail:
  # grep -E exits on first match, which sends SIGPIPE to nm; pipefail promotes that
  # to a non-zero exit even though the symbols were found.
  local _sym_tmp
  _sym_tmp="$(mktemp)"
  # Use -gU so we require an exported definition (T), not merely a mention in nm output.
  nm -gU "$LIBOMP" 2>/dev/null > "$_sym_tmp" || true
  # Both conditions must hold:
  #   1. Standard kmpc/omp symbols present (libomp is functional)
  #   2. __kmpc_dispatch_deinit exported (stub was injected successfully)
  local _has_kmpc _has_deinit
  grep -Eq '__kmpc_|omp_get_max_threads' "$_sym_tmp" && _has_kmpc=1 || _has_kmpc=0
  grep -q '__kmpc_dispatch_deinit'       "$_sym_tmp" && _has_deinit=1 || _has_deinit=0
  rm -f "$_sym_tmp"
  [[ "$_has_kmpc" -eq 1 && "$_has_deinit" -eq 1 ]]
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

# Inject a no-op __kmpc_dispatch_deinit stub into the LLVM openmp runtime build.
#
# Background: Apple Clang's OpenMP codegen emits a call to __kmpc_dispatch_deinit
# at the end of every loop that uses schedule(dynamic) or schedule(guided).  This
# was an Intel-private runtime ABI extension; the LLVM OpenMP runtime REMOVED the
# function in LLVM 13 (it became a no-op internally and was later elided).  Building
# libomp from LLVM 13+ therefore produces a dylib that lacks the symbol, causing
# dlopen to fail at wheel-install time:
#   "symbol not found in flat namespace '___kmpc_dispatch_deinit'"
#
# Fix: append a cmake snippet to the END of runtime/src/CMakeLists.txt.  At that
# point the `omp` target is already defined, so target_sources() works correctly.
# The stub compiles to a single tiny object (< 1 kB) with a no-op implementation,
# matching what the original function did once LLVM decided to phase it out.
RUNTIME_CMAKELISTS="$OPENMP_SRC/runtime/src/CMakeLists.txt"
if [[ ! -f "$RUNTIME_CMAKELISTS" ]]; then
  echo "[ERROR] Expected LLVM openmp runtime CMakeLists.txt not found: $RUNTIME_CMAKELISTS" >&2
  exit 1
fi

cat >> "$RUNTIME_CMAKELISTS" << 'CMAKE_STUB_EOF'

# ---------------------------------------------------------------------------
# Sequenzo CI: compatibility stub for __kmpc_dispatch_deinit
# Removed from LLVM 13+ runtime but Apple Clang still emits calls to it.
# ---------------------------------------------------------------------------
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/kmp_dispatch_deinit_stub.cpp"
  "/* no-op stub: __kmpc_dispatch_deinit removed in LLVM 13 but still\n"
  "   emitted by Apple Clang for dynamic/guided OpenMP schedule loops. */\n"
  "extern \"C\" __attribute__((visibility(\"default\")))\n"
  "void __kmpc_dispatch_deinit(void*, int) {}\n"
)
target_sources(omp PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/kmp_dispatch_deinit_stub.cpp")
# ---------------------------------------------------------------------------
CMAKE_STUB_EOF

echo "[INFO] Injected __kmpc_dispatch_deinit stub into LLVM openmp CMakeLists.txt"

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
  nm -g "$LIBOMP" 2>/dev/null | { grep -E '__kmpc_|omp_get_max_threads' || true; } | head -5 >&2 || true
  exit 1
fi

echo "[OK] Installed libomp to $INSTALL_PREFIX"
file "$LIBOMP"
otool -l "$LIBOMP" | awk '/LC_BUILD_VERSION|LC_VERSION_MIN_MACOSX|minos|version/'