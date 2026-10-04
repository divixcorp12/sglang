cmake_minimum_required(VERSION 3.24)
project(exl3_cpu_bench LANGUAGES CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "This benchmark requires Linux CPU affinity support")
endif()
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR
   NOT CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 15 OR
   CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 16)
  message(FATAL_ERROR "Use the validated GCC 15 compiler")
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

# Only ATen/c10, never torch_python. Avoid TorchConfig's CUDA discovery: this
# executable runs no CUDA code and needs no Python interpreter at configure/run.
set(EXL3_TORCH_ROOT "" CACHE PATH "LibTorch or installed torch directory (include/ and lib/)")
find_path(ATEN_INCLUDE ATen/ATen.h PATHS "${EXL3_TORCH_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
find_library(TORCH_CPU torch_cpu PATHS "${EXL3_TORCH_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(C10 c10 PATHS "${EXL3_TORCH_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
set(EXL3_CXX11_ABI 1 CACHE STRING "Match the consumer's _GLIBCXX_USE_CXX11_ABI (0 or 1)")
if(NOT EXL3_CXX11_ABI MATCHES "^[01]$")
  message(FATAL_ERROR "EXL3_CXX11_ABI must be 0 or 1")
endif()
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=${EXL3_CXX11_ABI})

set(EXPERT_STREAM_LANES 8 CACHE STRING "lanes per demand record (1..32)")
if(NOT EXPERT_STREAM_LANES MATCHES "^[0-9]+$" OR EXPERT_STREAM_LANES LESS 1 OR EXPERT_STREAM_LANES GREATER 32)
  message(FATAL_ERROR "EXPERT_STREAM_LANES must be 1..32")
endif()
add_compile_definitions(SGLANG_EXPERT_STREAM_LANES=${EXPERT_STREAM_LANES})

option(EXL3_FETCH_BENCHMARK "Download pinned Google Benchmark when not installed" ON)
find_package(benchmark 1.9 QUIET CONFIG)
if(NOT benchmark_FOUND)
  if(NOT EXL3_FETCH_BENCHMARK)
    message(FATAL_ERROR "Install Google Benchmark or enable EXL3_FETCH_BENCHMARK")
  endif()
  include(FetchContent)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(google_benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG eddb0241389718a23a42db6af5f0164b6e0139af) # v1.9.4
  FetchContent_MakeAvailable(google_benchmark)
endif()

get_filename_component(QUANT "${CMAKE_CURRENT_SOURCE_DIR}/../../../exl3" ABSOLUTE)
if(NOT EXISTS "${QUANT}/optimized/kernel.cpp")
  message(FATAL_ERROR "Cannot locate the CPU kernels: ${QUANT}")
endif()
# The vendored baseline contains no Python calls; replace its unnecessary
# binding header in the build copy, leaving its arithmetic/threading unchanged.
file(READ "${QUANT}/moe_mul1.cpp" BASELINE)
string(REPLACE "#include <torch/extension.h>" "#include <ATen/ATen.h>" BASELINE "${BASELINE}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${QUANT}/moe_mul1.cpp")
file(CONFIGURE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/baseline.cpp" CONTENT "${BASELINE}" @ONLY)

foreach(BACKEND baseline optimized)
  if(BACKEND STREQUAL "baseline")
    set(KERNEL "${CMAKE_CURRENT_BINARY_DIR}/baseline.cpp")
    set(DEFAULT_WORKERS 12)
  else()
    set(KERNEL "${QUANT}/optimized/kernel.cpp")
    set(DEFAULT_WORKERS 16)
  endif()
  add_executable(exl3_cpu_${BACKEND} src/cpu_forward.cpp src/fixture.cpp "${KERNEL}")
  # ${QUANT}: the baseline copy includes optimized/cpu_experts_cabi.h, which in place resolves beside the source.
  target_include_directories(exl3_cpu_${BACKEND} PRIVATE "${QUANT}/optimized" "${QUANT}")
  target_include_directories(exl3_cpu_${BACKEND} SYSTEM PRIVATE "${ATEN_INCLUDE}" "${ATEN_INCLUDE}/torch/csrc/api/include")
  target_compile_definitions(exl3_cpu_${BACKEND} PRIVATE
    EXL3_MOE_CPU_ACT_RESIDUAL=1 EXL3_MOE_CPU_ACT_BLOCK=128
    EXL3_BENCH_BACKEND="${BACKEND}" EXL3_BENCH_DEFAULT_WORKERS=${DEFAULT_WORKERS})
  # Kernel flags match the selected study. Keep the harness's NaN/error checks
  # outside fast-math; kernel and harness are distinct translation units.
  set_source_files_properties("${KERNEL}" PROPERTIES COMPILE_OPTIONS "-Ofast;-march=native")
  target_compile_options(exl3_cpu_${BACKEND} PRIVATE -O3 -g)
  target_link_libraries(exl3_cpu_${BACKEND} PRIVATE benchmark::benchmark OpenMP::OpenMP_CXX Threads::Threads "${TORCH_CPU}" "${C10}")
  set_target_properties(exl3_cpu_${BACKEND} PROPERTIES BUILD_RPATH "${EXL3_TORCH_ROOT}/lib")
endforeach()

# The full-stack bench (docs/superpowers/specs/2026-10-01-expert-stream-full-stack-bench-design.md). The expert-stream
# host headers need tvm-ffi's headers (reader_base.h) and liburing; nothing loads CUDA (the copy engine's host
# backend). Built only when EXL3_TVM_FFI_ROOT names the tvm_ffi package, so the CPU-forward bench configures as before.
set(EXL3_TVM_FFI_ROOT "" CACHE PATH "The tvm_ffi Python package directory (include/ and lib/), for the full-stack bench")
if(EXL3_TVM_FFI_ROOT)
  find_path(TVM_FFI_INCLUDE tvm/ffi/function.h PATHS "${EXL3_TVM_FFI_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
  find_path(DLPACK_INCLUDE dlpack/dlpack.h PATHS "${EXL3_TVM_FFI_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
  find_library(TVM_FFI tvm_ffi PATHS "${EXL3_TVM_FFI_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
  find_library(URING uring REQUIRED)
  get_filename_component(MOE_CSRC "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
  if(NOT EXISTS "${MOE_CSRC}/expert_stream/host/ram_thread.h")
    message(FATAL_ERROR "Cannot locate the expert-stream host headers: ${MOE_CSRC}")
  endif()
  set(FULL_STACK_SOURCES src/full_stack.cpp src/self_test.cpp src/placement.cpp src/device_sim.cpp src/row_images.cpp
    src/stack_fixture.cpp src/fixture.cpp "${QUANT}/optimized/kernel.cpp")
  foreach(VARIANT prod instr)
    set(TARGET exl3_full_stack_${VARIANT})
    if(VARIANT STREQUAL "instr")
      set(INSTR 1)
    else()
      set(INSTR 0)
    endif()
    add_executable(${TARGET} ${FULL_STACK_SOURCES})
    target_include_directories(${TARGET} PRIVATE "${QUANT}/optimized" "${MOE_CSRC}")
    # tvm-ffi's dlpack first: the host headers need its DLPack, not any copy under the torch include tree.
    target_include_directories(${TARGET} SYSTEM PRIVATE "${TVM_FFI_INCLUDE}" "${DLPACK_INCLUDE}"
      "${ATEN_INCLUDE}" "${ATEN_INCLUDE}/torch/csrc/api/include")
    target_compile_definitions(${TARGET} PRIVATE
      EXL3_MOE_CPU_ACT_RESIDUAL=1 EXL3_MOE_CPU_ACT_BLOCK=128
      EXL3_FULL_STACK_INSTR=${INSTR} EXL3_BENCH_BACKEND="full-stack-${VARIANT}")
    target_compile_options(${TARGET} PRIVATE -O3 -g)
    target_link_libraries(${TARGET} PRIVATE benchmark::benchmark OpenMP::OpenMP_CXX Threads::Threads
      "${TORCH_CPU}" "${C10}" "${TVM_FFI}" "${URING}")
    set_target_properties(${TARGET} PROPERTIES BUILD_RPATH "${EXL3_TORCH_ROOT}/lib;${EXL3_TVM_FFI_ROOT}/lib")
  endforeach()
else()
  message(STATUS "Full-stack bench skipped: set EXL3_TVM_FFI_ROOT to build exl3_full_stack_{prod,instr}")
endif()
