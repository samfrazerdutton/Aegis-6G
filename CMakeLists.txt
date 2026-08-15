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
