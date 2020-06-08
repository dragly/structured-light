cmake_minimum_required(VERSION 3.10)

project(StructuredLight)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ZLIB REQUIRED)
find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)
find_package(Threads)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(StructuredLight src/main.cpp)

target_include_directories(
    StructuredLight
    PRIVATE
    ${HALIDE_DIR}/include
    ${HALIDE_DIR}/tools
    ${PNG_INCLUDE_DIR}
    ${JPEG_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(
    StructuredLight
    PRIVATE
    ${PNG_LIBRARY}
    ${JPEG_LIBRARY}
    ${CMAKE_THREAD_LIBS_INIT}
    dl
    -L${HALIDE_DIR}/bin/
    Halide
    Eigen3::Eigen
)
