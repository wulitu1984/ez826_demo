cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(demo-resnet LANGUAGES CXX)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")

set(CMAKE_C_COMPILER ${PROJECT_SOURCE_DIR}/../gcc-linaro-7.5.0-2019.12-x86_64_arm-linux-gnueabihf/bin/arm-linux-gnueabihf-gcc)
set(CMAKE_CXX_COMPILER ${PROJECT_SOURCE_DIR}/../gcc-linaro-7.5.0-2019.12-x86_64_arm-linux-gnueabihf/bin/arm-linux-gnueabihf-c++)
add_compile_options(-Wall -march=armv7-a -marm -mfpu=neon)
add_compile_options(-Wno-psabi)

include_directories(${PROJECT_SOURCE_DIR}/../3rd-party/)
include_directories(${PROJECT_SOURCE_DIR}/src/)
link_directories(${PROJECT_SOURCE_DIR}/../3rd-party/hcnn2/)
set(PLATFORM_RT_LIB ax log openvx openvx-nn nn soft-nn hcnn2)

set(SRC 
	${PROJECT_SOURCE_DIR}/src/yolopost.cpp
	${PROJECT_SOURCE_DIR}/src/main.cpp)

add_executable(demo-resnet ${SRC})
target_link_libraries(demo-resnet
	${PLATFORM_RT_LIB}
)
