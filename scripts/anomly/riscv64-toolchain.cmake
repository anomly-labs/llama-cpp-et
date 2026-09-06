# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
# Ubuntu riscv64 cross toolchain (gcc-riscv64-linux-gnu), static binaries for qemu-riscv64.
# -DRISCV_MARCH=rv64gc (scalar) or rv64gcv (vector, GGML_RVV); the exact profile must match either way.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)
set(CMAKE_C_COMPILER   riscv64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER riscv64-linux-gnu-g++)
if(NOT DEFINED RISCV_MARCH)
  set(RISCV_MARCH rv64gc)
endif()
set(CMAKE_C_FLAGS_INIT   "-march=${RISCV_MARCH} -mabi=lp64d")
set(CMAKE_CXX_FLAGS_INIT "-march=${RISCV_MARCH} -mabi=lp64d")
set(CMAKE_EXE_LINKER_FLAGS_INIT "-static")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
