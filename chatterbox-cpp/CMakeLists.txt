cmake_minimum_required(VERSION 3.14)
project(tts-cpp LANGUAGES C CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(TTS_CPP_BUILD_LIBRARY     "tts-cpp: build the static tts-cpp library"                 ON)
option(TTS_CPP_INSTALL           "tts-cpp: generate install rules"                           ON)

if (NOT TARGET ggml)
    add_subdirectory(ggml)
endif()

find_package(OpenMP)

set(TTS_CPP_LIB_SOURCES
    src/main.cpp
    src/gpt2_bpe.cpp
    src/chatterbox_tts.cpp
    src/chatterbox_engine.cpp
    src/voice_features.cpp
    src/mel_extract_stft.cpp
    src/voice_encoder.cpp
    src/campplus.cpp
    src/s3tokenizer.cpp
    src/t3_mtl.cpp
)

add_library(mtl_tokenizer STATIC src/mtl_tokenizer.cpp)
target_include_directories(mtl_tokenizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
set_target_properties(mtl_tokenizer PROPERTIES POSITION_INDEPENDENT_CODE ON)

if (TTS_CPP_BUILD_LIBRARY)
    add_library(tts-cpp STATIC ${TTS_CPP_LIB_SOURCES})
    add_library(tts-cpp::tts-cpp ALIAS tts-cpp)

    target_include_directories(tts-cpp
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
    )
    target_compile_features(tts-cpp PUBLIC cxx_std_17)
    target_link_libraries(tts-cpp PUBLIC ggml)
    target_link_libraries(tts-cpp PRIVATE mtl_tokenizer)

    if (OpenMP_CXX_FOUND)
        target_link_libraries(tts-cpp PRIVATE OpenMP::OpenMP_CXX)
    endif()

    if (GGML_CUDA)
        target_compile_definitions(tts-cpp PRIVATE GGML_USE_CUDA)
    endif()
    if (GGML_METAL)
        target_compile_definitions(tts-cpp PRIVATE GGML_USE_METAL)
    endif()
    if (GGML_VULKAN)
        target_compile_definitions(tts-cpp PRIVATE GGML_USE_VULKAN)
    endif()
    if (GGML_OPENCL)
        target_compile_definitions(tts-cpp PRIVATE GGML_USE_OPENCL)
    endif()

    set_target_properties(tts-cpp PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        EXPORT_NAME tts-cpp
    )
endif()

if (TTS_CPP_INSTALL)
    install(TARGETS tts-cpp
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
    install(DIRECTORY include/tts-cpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h"
    )
endif()
