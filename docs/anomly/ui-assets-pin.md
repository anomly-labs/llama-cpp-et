<!-- Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe. -->
# The web UI archive is pinned

`llama-server`'s web UI is not built in CI; a prebuilt archive is downloaded from the upstream
bucket `ggml-org/llama-ui`. Upstream picks the archive by git build number and falls back to
`latest`. In this fork the build number never names an upstream archive, so builds used to take
`latest`, and broke on 22 Nov 2026 when the newest upstream UI stopped matching the asset list
`tools/ui/embed.cpp` in this tree expects (`missing required asset(s): bundle[hash].js ...`).

What this fork does instead:

- `LLAMA_UI_PINNED_VERSION` (default `b10036`, the upstream build this fork is based on) is the
  archive that is downloaded. Override with the `HF_UI_VERSION` environment variable.
- `LLAMA_UI_ALLOW_LATEST` (default `OFF`) controls whether `latest` is tried at all.
- If the embed step rejects downloaded assets, the build continues without an embedded UI and
  prints a warning. A UI mismatch can no longer fail a build.
- The weekly schedules of the `CI (cross)` and `CI (msys)` workflows are removed; they run on
  demand and on push.

**When rebasing onto a newer upstream:** set `LLAMA_UI_PINNED_VERSION` in `tools/ui/CMakeLists.txt`
to the new base build (the `bNNNNN` in the upstream release tag) and check that
`https://huggingface.co/buckets/ggml-org/llama-ui/resolve/bNNNNN/dist.tar.gz.sha256` exists.

## Windows Clang builds and the runner's STL

The hosted Windows runner updates Visual Studio's STL and its preinstalled Clang separately. On
22 Nov 2026 the STL began to require Clang 22 while the runner's Clang was older, and every
Windows Clang job failed with `error STL1000: Unexpected compiler version`. The two toolchain
files `cmake/x64-windows-llvm.cmake` and `cmake/arm64-windows-llvm.cmake` now define
`_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH`, so a runner image change of this kind no longer
fails the build.
