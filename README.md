# Kalman Filter #

### This repository contains implementation of various types of Kalman filters. ###

It contains implementations for
1. Kalman Filter
2. Extended Kalman Filter
3. Unscented Kalman Filter

It also includes examples for each of the above for a
1. spring mass damper system

#### To build and run ####
     git clone https://github.com/vss2sn/kalman_filter.git  
     cd path_planning  
     mkdir build  
     cd build  
     cmake .. && make -j4  
     ./main  

#### To use and older compiler ####

To remove the requirement for C++20 and use an older compiler:
1. Comment the following lines from the CMAkeLists.txt:
  * `set(CMAKE_C_COMPILER gcc-10)`
  * `set(CMAKE_CXX_COMPILER g++-10)`
2. Change the `CXX_STANDARD 20` option in the CMakeLists.txt (`set_target_properties`) to `CXX_STANDARD 17`
3. Comment out the `PredictSTL` and `UpdateSTL` functions in `unscented_kalman.hpp` and `unscented_kalman.cpp`
4. Comment out the `#include <ranges>` in `unscented_kalman.cpp`

#### TODO ####
1. Add example for pendulum
2. Add documentation
3. Add references
4. Add explanatory notes for each step of the filter
5. Consider setting the initial state for the filters
6. Option for dimension correctness checking (esp. for control input)
