cmake_minimum_required(VERSION 3.5)

#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)

project( fast1 )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} ./ )

option(BUILD_DEVELOP "Building all the things" OFF)

if(BUILD_DEVELOP)
    add_executable( fast1       ocv_fast2.cpp )
    add_executable( slow1       ocv_slow.cpp )
    add_executable( slower      ocv_slower.cpp )
    add_executable( param       ocv_parameterized.cpp )
    add_executable( hill        hill.cpp )
    add_executable( pt          ptest.cpp )
    add_executable( pw          pworm.cpp )
    add_executable( bz          bz.cpp )
    add_executable( geometry    geometry.cpp )
    target_link_libraries( fast1  ${OpenCV_LIBS} )
    target_link_libraries( slow1  ${OpenCV_LIBS} )
    target_link_libraries( slower ${OpenCV_LIBS} )
    target_link_libraries( param  ${OpenCV_LIBS} )
    target_link_libraries( hill   ${OpenCV_LIBS} )
    target_link_libraries( pt     ${OpenCV_LIBS} )
    target_link_libraries( pw     ${OpenCV_LIBS} )
    target_link_libraries( bz     ${OpenCV_LIBS} )
    # target_link_libraries( geometry     ${OpenCV_LIBS} )
endif()


find_package( OpenMP REQUIRED )

add_executable( slowglass    hill.cpp PerlinNoise.hpp)
target_link_libraries( slowglass  ${OpenCV_LIBS} OpenMP::OpenMP_CXX )
