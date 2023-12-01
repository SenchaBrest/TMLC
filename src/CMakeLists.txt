cmake_minimum_required(VERSION 3.25)
project(src)

set(CMAKE_CXX_STANDARD 17)

include_directories(../onnxruntime/include)
set(ONNX_RUNTIME_LIBRARY_PATH ../onnxruntime/lib/libonnxruntime.a)

add_library(ONNX_RUNTIME_LIBRARY STATIC IMPORTED)
set_target_properties(ONNX_RUNTIME_LIBRARY PROPERTIES IMPORTED_LOCATION ${ONNX_RUNTIME_LIBRARY_PATH})

add_executable(src tglang.cpp main.cpp)
target_link_libraries(src PUBLIC ONNX_RUNTIME_LIBRARY)

add_library(tglang SHARED tglang.cpp)
target_include_directories(tglang PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(tglang PUBLIC ONNX_RUNTIME_LIBRARY)

