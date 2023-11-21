cmake_minimum_required(VERSION 3.0)
project(MahalanobisExample)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 11)

# Add the path to Eigen library here
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Add executable
add_executable(mahalanobis mahalanobis_dist.cpp) 
add_executable(kf kf.cpp) 
