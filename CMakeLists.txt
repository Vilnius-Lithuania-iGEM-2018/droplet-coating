cmake_minimum_required(VERSION 3.7)
project(droplet-coating)

find_package( OpenCV REQUIRED )

MESSAGE( STATUS "CMAKE_SYSTEM_NAME: " ${CMAKE_SYSTEM_NAME} )
set(CMAKE_CXX_STANDARD 14)

add_executable(droplet-coating main.cpp)

target_link_libraries(droplet-coating ${OpenCV_LIBS})