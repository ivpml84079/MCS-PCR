cmake_minimum_required(VERSION 3.16)
project(MCS-PCR LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -DNDEBUG")
set(CMAKE_BUILD_TYPE Release)

find_package(PCL REQUIRED COMPONENTS common io filters features kdtree octree search registration)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)

add_executable(test_dataset
    examples/test_dataset.cpp
    source/Registration.cpp
    source/Graph.cpp
    source/Hough.cpp
    source/LineDetection.cpp
    source/LoopRegistration2D.cpp
    source/Projection.cpp
    source/Utils.cpp
    source/ZEstimation.cpp
)

target_include_directories(test_dataset PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PCL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(test_dataset PRIVATE
    ${PCL_LIBRARIES}
    Eigen3::Eigen
    yaml-cpp
)

target_compile_definitions(test_dataset PRIVATE
    ${PCL_DEFINITIONS}
    NDEBUG
)

add_executable(example
    examples/example.cpp
    source/Registration.cpp
    source/Graph.cpp
    source/Hough.cpp
    source/LineDetection.cpp
    source/LoopRegistration2D.cpp
    source/Projection.cpp
    source/Utils.cpp
    source/ZEstimation.cpp
)

target_include_directories(example PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PCL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(example PRIVATE
    ${PCL_LIBRARIES}
    Eigen3::Eigen
    yaml-cpp
)

target_compile_definitions(example PRIVATE
    EIGEN_MALLOC_ALREADY_ALIGNED=1
)

target_compile_definitions(example PRIVATE
    ${PCL_DEFINITIONS}
    NDEBUG
)