# Agent Result: Fix NVIDIA/cccl#7782

## Root Cause

CUB algorithms accept a user-provided `cudaStream_t` argument but use the
**current CUDA device** (via `cudaGetDevice`) for queries like PTX version,
SM count, and available shared memory. If the stream belongs to a *different*
device than the current one, those queries return values for the wrong device,
which can cause e.g. a kernel to be launched with more dynamic SMEM than the
stream's device actually supports.

## Change Made

### `cub/cub/util_device.cuh`
Added `cub::detail::assert_current_device_matches_stream(cudaStream_t stream)`:
- Only active in **host code** (guarded by `NV_IF_TARGET(NV_IS_HOST, ...)`).
- No-op for the special pseudo-streams `nullptr`, `cudaStreamLegacy`, and
  `cudaStreamPerThread`, since those are always tied to the current device.
- No-op on CUDA < 12.8 (where `cudaStreamGetDevice` is unavailable); the
  guard is `#if _CCCL_CTK_AT_LEAST(12, 8)`.
- Uses `_CCCL_ASSERT` to report a violation (active when
  `CCCL_ENABLE_HOST_ASSERTIONS` is defined, i.e. in debug builds).
- Silently skips if either `cudaStreamGetDevice` or `cudaGetDevice` fails to
  keep the check non-fatal for unexpected edge cases.

### `cub/cub/detail/launcher/cuda_runtime.cuh`
Added a fully qualified call to `::cub::detail::assert_current_device_matches_stream(stream)` in
`cub::detail::TripleChevronFactory::operator()`. This is the single entry point used by
all CUB dispatch paths when building a kernel launcher, so every algorithm that
accepts a stream gets the check without touching each dispatch file individually.

### `cub/test/catch2_test_util_device.cu`
Added two new test cases (guarded by `#if _CCCL_CTK_AT_LEAST(12, 8)`):
1. **Matching-stream test** - verifies that the function does not assert for all
   three special streams and for a stream created on the current device.
2. **Multi-device test** (`TEST_LAUNCH == 0` only) - on systems with >= 2 GPUs,
   verifies that `cudaStreamGetDevice` returns the expected device ordinal for a
   stream created on device 0. The mismatch path itself triggers
   `_CCCL_ASSERT` which aborts the process in debug builds and is therefore not
   exercised directly.

## Testing

The new tests are in `cub/test/catch2_test_util_device.cu`:
- `assert_current_device_matches_stream passes for matching stream` - passes on
  any system with at least one CUDA GPU and CUDA >= 12.8.
- `assert_current_device_matches_stream identifies stream device` - requires
  >= 2 CUDA GPUs; automatically skipped on single-GPU systems.

Tests run with the existing CUB test harness (CMake + CTest).

## Lint

The codebase uses C++/CUDA (`.cuh`/`.cu` files). The changes follow the
existing code style of the repository (2-space indentation for nested
preprocessor directives, `_CCCL_ASSERT` for assertions, `NV_IF_TARGET` for
host/device dispatch, fully qualified free-function calls per project convention).

## Competing PR

PR #9119 (`thom-gg:validate-device-and-stream-matches`) addresses the same issue
by inserting calls in every dispatch `.cuh` file (385 additions across 30+ files).
The approach here is more minimal: a single call in
`cub::detail::TripleChevronFactory::operator()` covers all dispatch paths. The
CTK version guard was also corrected to `_CCCL_CTK_AT_LEAST(12, 8)` since
`cudaStreamGetDevice` was introduced in CUDA 12.8 (not 12.3).
