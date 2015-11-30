cmake_minimum_required(VERSION 2.8)  
project(Track)
#OPENCV
FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS} )
link_directories(${OPENCV_LIBRARY_DIRS})
add_definitions(${OPENCV_DEFINITIONS})

set(src track.cpp main.cpp)
add_executable(track ${src})
TARGET_LINK_LIBRARIES(track ${OpenCV_LIBS})
