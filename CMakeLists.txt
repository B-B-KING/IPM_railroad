cmake_minimum_required(VERSION 2.8)

project(railspur-c)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS}/include)

add_executable(main main.cpp src/Binary.cpp)

target_link_libraries(main ${OpenCV_LIBS})