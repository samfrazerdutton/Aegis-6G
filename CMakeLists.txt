cmake_minimum_required(VERSION 3.24)
project(Aegis6G LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

# --- transport layer: host-side, zero external dependencies ---
add_library(aegis_transport STATIC
  src/keccak.cpp
  src/mlkem_ntt.cpp
  src/mlkem_sample.cpp
  src/mlkem768.cpp
  src/aead.cpp
  src/session.cpp
  src/transport.cpp
)
target_include_directories(aegis_transport PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

find_package(Threads REQUIRED)
enable_testing()
foreach(t keccak mlkem_ntt mlkem_sample mlkem768 mlkem_kat aead session transport)
  add_executable(test_${t} tests/test_${t}.cpp)
  target_link_libraries(test_${t} PRIVATE aegis_transport Threads::Threads)
  add_test(NAME ${t} COMMAND test_${t} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endforeach()

add_executable(bench_transport bench/bench_transport.cpp)
target_link_libraries(bench_transport PRIVATE aegis_transport Threads::Threads)

add_library(aegis_ml STATIC src/mnist.cpp src/mlp.cpp)
target_include_directories(aegis_ml PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(train_mlp tools/train_mlp.cpp)
target_link_libraries(train_mlp PRIVATE aegis_ml)

add_executable(test_mlp_packed tests/test_mlp_packed.cpp)
target_link_libraries(test_mlp_packed PRIVATE aegis_ml)
add_test(NAME mlp_packed COMMAND test_mlp_packed
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# ---- GPU-Resident-Library (built separately, see tools/build_gpufhe.sh) ----
set(GPUFHE_LIB ${CMAKE_CURRENT_SOURCE_DIR}/build-gpufhe/libgpufhe.a)
if(EXISTS ${GPUFHE_LIB})
  find_package(CUDAToolkit REQUIRED)
  add_library(gpufhe STATIC IMPORTED)
  set_target_properties(gpufhe PROPERTIES
    IMPORTED_LOCATION ${GPUFHE_LIB}
    INTERFACE_INCLUDE_DIRECTORIES
      ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gpufhe/include)

  # Must be a CUDA target: gpufhe is built with CUDA_SEPARABLE_COMPILATION,
  # so its device code is relocatable and needs a device-link step. A plain
  # C++ executable linking this .a fails with unresolved device symbols.
  add_library(aegis_keyser STATIC src/keyser.cpp)
  target_include_directories(aegis_keyser PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(aegis_keyser PUBLIC gpufhe)

  add_executable(test_keyser tests/test_keyser.cpp)
  set_target_properties(test_keyser PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON
    LINKER_LANGUAGE CUDA)
  target_link_libraries(test_keyser PRIVATE aegis_keyser gpufhe CUDA::cudart)
  add_test(NAME keyser COMMAND test_keyser
           WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(aegis_demo tools/aegis_demo.cu)
  set_target_properties(aegis_demo PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_link_libraries(aegis_demo PRIVATE gpufhe aegis_ml aegis_keyser
                        aegis_transport CUDA::cudart Threads::Threads)

  add_executable(infer_encrypted tools/infer_encrypted.cu)
  set_target_properties(infer_encrypted PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_link_libraries(infer_encrypted PRIVATE gpufhe aegis_ml aegis_keyser CUDA::cudart)
else()
  message(STATUS "libgpufhe.a not found -- run tools/build_gpufhe.sh "
                 "to enable encrypted inference targets")
endif()
