# Minimum version of CMake
cmake_minimum_required(VERSION 3.12)

# Set the project name
project(MIP)

set(Python3_EXECUTABLE "${CMAKE_SOURCE_DIR}/mip/bin/python")

# Find Python3 interpreter and development environment
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Add a custom target to run the main Python script
add_custom_target(RunMain ALL
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/src/main.py
    COMMENT "Running main Python script"
)

# Add a custom target to run tests
add_custom_target(RunTests ALL
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/test/grad_test.py
    COMMENT "Running Python tests"
)
