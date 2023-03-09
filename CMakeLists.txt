cmake_minimum_required(VERSION 3.10)
include(FindPackageHandleStandardArgs)

project(Cartoon_Classifier)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_SOURCE_DIR}")
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE RelWithDebInfo)
add_compile_options(-O0)
add_definitions(-w)

find_package(OpenCV REQUIRED)

file(GLOB SOURCES "src/*.cpp")

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
)

link_libraries(
    ${OpenCV_LIBS}
)

add_executable(detect ${SOURCES})