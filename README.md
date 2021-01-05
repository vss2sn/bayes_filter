# Bayes Filter #

### This repository contains implementations of various types of Bayes filters. ###

It contains implementations for
1. Kalman Filter
2. Extended Kalman Filter
3. Unscented Kalman Filter

It also includes the following examples:
1. Spring mass damper system (kf, efk, ukf)
2. Simple pendulum (kf, ekf, ukf)

The kalman filter example in the non-linear simple pendulum example has been added to demonstrate the performance of the kalman filter when the measurement to state space conversion, the process matrix and the control matrix are non-linear and approximations/assumptions are made to help it replicate the actual process (in this case, the small angle approximation)

#### To build and run ####
     git clone https://github.com/vss2sn/bayes_filter.git  
     cd bayes_filter  
     mkdir build  
     cd build  
     cmake .. && make -j  
     ./main # or the examples' executables

#### To run without C++20 ####
Please checkout the branch `c++17` and use that/rebase on it

#### TODO ####
1. Add documentation
2. Consider setting the initial state for the filters
3. Option for dimension correctness checking (esp. for control input)

#### References ####
1. https://github.com/tysik/kalman_filters/
2. https://github.com/PrieureDeSion/kalmanfilter-cpp/
