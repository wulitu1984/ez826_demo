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
link_directories(${PROJECT_SOURCE_DIR}/../3rd-party/opencv2/lib)

if( PLATFORM STREQUAL "82x")
link_directories(${PROJECT_SOURCE_DIR}/../3rd-party/hcnn2/lib82x)
set(PLATFORM_RT_LIB ax log openvx openvx-nn nn soft-nn hcnn2)
elseif( PLATFORM STREQUAL "51x")
link_directories(${PROJECT_SOURCE_DIR}/../3rd-party/hcnn2/lib51x)
set(PLATFORM_RT_LIB lombo_malloc log openvx openvx-nn nn soft-nn hcnn2)
endif()

set(OpenCV_LIBS opencv_core 
	opencv_highgui 
	opencv_imgproc
	opencv_imgcodecs
	opencv_videoio
	opencv_video
	opencv_calib3d
	opencv_features2d
	opencv_flann
	opencv_ml
	)

set(SRC ${PROJECT_SOURCE_DIR}/src/main.cpp)
set(SRC_P ${PROJECT_SOURCE_DIR}/src/main_power.cpp)

add_executable(demo-resnet ${SRC})
target_link_libraries(demo-resnet
	${PLATFORM_RT_LIB}
	${OpenCV_LIBS}
)

add_executable(demo-resnet-power ${SRC_P})
target_link_libraries(demo-resnet-power
	${PLATFORM_RT_LIB}
)

