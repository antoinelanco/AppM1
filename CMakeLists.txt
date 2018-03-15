cmake_minimum_required(VERSION 3.5)
project(AppM1 C CXX)

set(CMAKE_CXX_STANDARD 11)

file(GLOB_RECURSE SOURCE_FILES "src/*.h" "src/*.cpp")

add_executable(AppM1 ${SOURCE_FILES})