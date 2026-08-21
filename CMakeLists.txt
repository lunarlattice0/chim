cmake_minimum_required(VERSION 3.16)
project(chim CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Verify the model file exists in the project root
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/model.gguf")
    message(FATAL_ERROR "model.gguf not found in project root: ${CMAKE_SOURCE_DIR}")
endif()

# Add llama.cpp as a subdirectory with specific options
set(GGML_BLAS ON)
set(LLAMA_USE_SYSTEM_GGML OFF)
set(LLAMA_BUILD_TESTS OFF)
set(GGML_BUILD_EXAMPLES OFF)
set(GGML_BUILD_TESTS OFF)
set(GGML_LTO ON)
set(GGML_BLAS_VENDOR OpenBLAS)
set(BUILD_SHARED_LIBS OFF)
set(GGML_VULKAN OFF)

# Add the llama.cpp subdirectory
add_subdirectory(subprojects/llama.cpp)

add_compile_options(-march=native)

# Create the executable
add_executable(chim
    src/main.cpp
    src/llm/llm.cpp
)

target_include_directories(chim PRIVATE src/)

# Also need to link rz_ghidra libraries manually.
target_link_libraries(chim PRIVATE
    llama
)

# Link libtcc (TinyCC)
find_library(TCC_LIBRARY NAMES tcc REQUIRED)
target_link_libraries(chim PRIVATE ${TCC_LIBRARY})
