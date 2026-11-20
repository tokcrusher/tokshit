cmake_minimum_required(VERSION 3.20)

project(tokshit VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(TOKSHIT_NEON "Enable NEON kernels on aarch64" ON)
option(TOKSHIT_DOTPROD "Compile dotprod variants" ON)
option(TOKSHIT_I8MM "Compile i8mm (SMMLA) variants" ON)
option(TOKSHIT_BUILD_TOOLS "Build benchmark tools" ON)
option(TOKSHIT_BUILD_TESTS "Build unit tests" ON)
option(TOKSHIT_NO_EXCEPTIONS "Build with -fno-exceptions -fno-rtti" ON)

include(cmake/CompilerWarnings.cmake)

set(TOKSHIT_CORE_SOURCES
  src/quant.cpp
  src/mmap.cpp
  src/gguf.cpp
  src/dequant.cpp
  src/compute/kernels_scalar.cpp
  src/compute/kernels_dispatch.cpp
  src/metrics.cpp
  src/pool.cpp
  src/ahb.cpp
  src/tokenizer.cpp
  src/sampling.cpp
  src/model_load.cpp
  src/model_hot.cpp
  src/model_cache.cpp
  src/model_layer.cpp
  src/model_moe.cpp
  src/model_token.cpp
  src/model_math.cpp
  src/model_batch.cpp
  src/kv.cpp
  src/compute/kernels_batch.cpp
  src/compute/attn_batch.cpp
  src/spec_loop.cpp
  src/expert_io.cpp
  src/reasoning.cpp
  src/tui_flags.cpp
  src/tui_status.cpp
  src/tui_command.cpp
)

if(TOKSHIT_NEON AND TOKSHIT_DOTPROD AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
  list(APPEND TOKSHIT_CORE_SOURCES src/compute/kernels_neon_dotprod.cpp)
  set(TOKSHIT_HAVE_NEON_DOTPROD ON)
endif()

if(TOKSHIT_NEON AND TOKSHIT_DOTPROD AND TOKSHIT_I8MM AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
  list(APPEND TOKSHIT_CORE_SOURCES src/compute/kernels_neon_i8mm.cpp)
  set(TOKSHIT_HAVE_NEON_I8MM ON)
endif()

add_library(tokshit_core STATIC ${TOKSHIT_CORE_SOURCES})
add_library(tokshit::core ALIAS tokshit_core)

target_include_directories(tokshit_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_features(tokshit_core PUBLIC cxx_std_20)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
  target_compile_options(tokshit_core PRIVATE -march=armv8.2-a+dotprod+fp16+lse)
endif()

tokshit_set_warnings(tokshit_core)

# AHardwareBuffer is reached through dlopen("libandroid.so"), not -landroid.
# A -landroid binary carries a LIBANDROID vN version requirement that Termux's
# unversioned libandroid.so stub cannot satisfy, so the executable dies in the
# linker before main with "cannot find libandroid.so from verneed". dlopen
# skips both the link-time and the version check.
if(ANDROID)
  target_link_libraries(tokshit_core PUBLIC ${CMAKE_DL_LIBS})
endif()

if(TOKSHIT_HAVE_NEON_DOTPROD)
  target_compile_definitions(tokshit_core PRIVATE TOKSHIT_HAVE_NEON_DOTPROD=1)
  set_source_files_properties(src/compute/kernels_neon_dotprod.cpp
    PROPERTIES COMPILE_OPTIONS "-march=armv8.2-a+dotprod+fp16+lse")
endif()

if(TOKSHIT_HAVE_NEON_I8MM)
  target_compile_definitions(tokshit_core PRIVATE TOKSHIT_HAVE_NEON_I8MM=1)
  set_source_files_properties(src/compute/kernels_neon_i8mm.cpp
    PROPERTIES COMPILE_OPTIONS "-march=armv8.2-a+dotprod+i8mm+fp16+lse")
endif()

if(NOT MSVC)
  target_compile_options(tokshit_core PRIVATE -fno-math-errno)
endif()

if(TOKSHIT_NO_EXCEPTIONS)
  if(MSVC)
    target_compile_options(tokshit_core PRIVATE /EHs-c- /GR-)
  else()
    target_compile_options(tokshit_core PRIVATE -fno-exceptions -fno-rtti)
  endif()
endif()

find_package(Threads REQUIRED)
target_link_libraries(tokshit_core PUBLIC Threads::Threads)

add_executable(tokshit
  src/main.cpp
  src/cli_common.cpp
  src/cli_flags.cpp
  src/cli_decode.cpp
  src/cli_run.cpp
  src/cli_chat.cpp
)
target_link_libraries(tokshit PRIVATE tokshit_core)
tokshit_set_warnings(tokshit)

# тесты живут вне публичного репозитория: включаются, только если каталог
# tests/ присутствует в checkout (свежий клон собирается без них)
# tests live outside the public repository: built only when the tests/
# directory is present in the checkout (a fresh clone builds without them)
if(TOKSHIT_BUILD_TESTS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/main.cpp)
  enable_testing()

  add_executable(tokshit_tests
    tests/main.cpp
    tests/test_quant.cpp
    tests/test_result.cpp
    tests/test_gguf.cpp
    tests/test_mmap.cpp
    tests/test_dequant.cpp
    tests/test_kernels.cpp
    tests/test_cache.cpp
    tests/test_ahb.cpp
    tests/test_expert_io.cpp
    tests/test_freq_profile.cpp
    tests/test_pool.cpp
    tests/test_metrics.cpp
    tests/test_sampling.cpp
    tests/test_tokenizer.cpp
    tests/test_kv.cpp
    tests/test_attn.cpp
    tests/test_batch_kernels.cpp
    tests/test_reasoning.cpp
    tests/test_tui.cpp
  )
  target_link_libraries(tokshit_tests PRIVATE tokshit_core)
  target_include_directories(tokshit_tests SYSTEM PRIVATE tests)
  tokshit_set_warnings(tokshit_tests)

  add_test(NAME tokshit_tests COMMAND tokshit_tests)
endif()

if(TOKSHIT_BUILD_TOOLS)
  add_executable(qstat src/tools/qstat.cpp)
  target_link_libraries(qstat PRIVATE tokshit_core)
  tokshit_set_warnings(qstat)

  add_executable(bwbench src/tools/bwbench.cpp)
  target_link_libraries(bwbench PRIVATE tokshit_core)
  tokshit_set_warnings(bwbench)

  add_executable(kbench src/tools/kbench.cpp)
  target_link_libraries(kbench PRIVATE tokshit_core)
  tokshit_set_warnings(kbench)

  add_executable(pbench src/tools/pbench.cpp)
  target_link_libraries(pbench PRIVATE tokshit_core)
  tokshit_set_warnings(pbench)

  add_executable(vcheck src/tools/vcheck.cpp)
  target_link_libraries(vcheck PRIVATE tokshit_core)
  tokshit_set_warnings(vcheck)

  add_executable(ahbtest src/tools/ahbtest.cpp)
  target_link_libraries(ahbtest PRIVATE tokshit_core)
  tokshit_set_warnings(ahbtest)

  add_executable(densebench src/tools/densebench.cpp)
  target_link_libraries(densebench PRIVATE tokshit_core)
  tokshit_set_warnings(densebench)
endif()
