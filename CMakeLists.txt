cmake_minimum_required(VERSION 2.6)
project(CS5780TermProject)

# Set modules path
set(CMAKE_MODULE_PATH ${CMAKE_ROOT}/Modules
            ${CS5780TermProject_SOURCE_DIR}/cmake)

# Set version
set(CS5780_Term_Project_VERSION_MAJOR 1)
set(CS5780_Term_Project_VERSION_MINOR 0)

find_package(OpenCV REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    )

link_directories(
    ${OpenCV_LIBRARY_DIRS}
    )

subdirs(src)
