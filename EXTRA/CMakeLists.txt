cmake_minimum_required(VERSION 3.16)
project(ml_labs_extras LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)

add_executable(lesso regression/lesso.cpp)

target_link_libraries(lesso PRIVATE Eigen3::Eigen)
