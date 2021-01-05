# Using non-default comiler; ideally avoid setting this
set(CMAKE_C_COMPILER gcc-10)
set(CMAKE_CXX_COMPILER g++-10)

project(bayes_filter)

cmake_minimum_required(VERSION 3.5)

find_package(Eigen3 REQUIRED)
find_package(Threads REQUIRED)

add_library(bayes_filter)

target_sources(bayes_filter
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kalman.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/extended_kalman.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/unscented_kalman.cpp)

target_include_directories(bayes_filter
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(bayes_filter
  PUBLIC
    Eigen3::Eigen
    Threads::Threads)

set_target_properties(bayes_filter
  PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
    CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")

# Custom function to build multiple executables with the same properties and links
function(custom_add_executable SOURCE_FILE)
  get_filename_component(executable_file ${SOURCE_FILE} NAME_WE)
  add_executable(${executable_file} ${SOURCE_FILE})
  set_target_properties(${executable_file}
    PROPERTIES
      CXX_STANDARD 20
      CXX_STANDARD_REQUIRED YES
      CXX_EXTENSIONS NO
      CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Werror")
  target_link_libraries(${executable_file}
    PRIVATE
      bayes_filter)
endfunction()

custom_add_executable(${CMAKE_CURRENT_SOURCE_DIR}/main/main.cpp)

custom_add_executable(${CMAKE_CURRENT_SOURCE_DIR}/examples/simple_pendulum/sp_extended_kalman.cpp)
custom_add_executable(${CMAKE_CURRENT_SOURCE_DIR}/examples/simple_pendulum/sp_kalman.cpp)
custom_add_executable(${CMAKE_CURRENT_SOURCE_DIR}/examples/simple_pendulum/sp_unscented_kalman.cpp)

custom_add_executable(${CMAKE_CURRENT_SOURCE_DIR}/examples/spring_mass_damper/smd_extended_kalman.cpp)
custom_add_executable(${CMAKE_CURRENT_SOURCE_DIR}/examples/spring_mass_damper/smd_kalman.cpp)
custom_add_executable(${CMAKE_CURRENT_SOURCE_DIR}/examples/spring_mass_damper/smd_unscented_kalman.cpp)
