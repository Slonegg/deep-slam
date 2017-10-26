cmake_minimum_required(VERSION 3.1.2)
project(deep_slam)

set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE)
set(CMAKE_DEBUG_POSTFIX d)
set(CONAN_SYSTEM_INCLUDES ON)

if (NOT ${PROJECT_NAME_UPPER}_DONT_GRAB_DEPENDENCIES)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/conan.cmake)
    conan_cmake_run(REQUIRES CONANFILE conanfile.py
                    BASIC_SETUP CMAKE_TARGETS
                    BUILD missing)
    # setup output directory if it is not set by parent project
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR}/bin)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR}/bin)
endif ()

# some global properties
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# list options
message(STATUS "${PROJECT_NAME} configuration:")

option(${PROJECT_NAME}_BUILD_SHARED "Build shared version of ${PROJECT_NAME}" ON)
message(STATUS "  ${PROJECT_NAME}_BUILD_SHARED=${${PROJECT_NAME}_BUILD_SHARED}")

option(${PROJECT_NAME}_BUILD_DEMOS "Build ${PROJECT_NAME} demos" OFF)
message(STATUS "  ${PROJECT_NAME}_BUILD_DEMOS=${${PROJECT_NAME}_BUILD_DEMOS}")

option(${PROJECT_NAME}_BUILD_TOOLS "Build ${PROJECT_NAME} tools" ON)
message(STATUS "  ${PROJECT_NAME}_BUILD_TOOLS=${${PROJECT_NAME}_BUILD_TOOLS}")

option(${PROJECT_NAME}_BUILD_TESTS "Build ${PROJECT_NAME} tests" ON)
message(STATUS "  ${PROJECT_NAME}_BUILD_TESTS=${${PROJECT_NAME}_BUILD_TESTS}")

# global configuration
include_directories(
    include
)

# global options
set(CMAKE_CXX_STANDARD 14)
if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
    set(CMAKE_C_FLAGS_RELEASE "/MD /Zi /O2 /D NDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "/MD /Zi /O2 /D NDEBUG")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "/DEBUG /INCREMENTAL:NO")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "/DEBUG /INCREMENTAL:NO")
    add_definitions(-D_SCL_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_WARNINGS -D_WIN32_WINNT=0x600)
endif()
if (CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-fPIC)
    # GCC warnings: http://stackoverflow.com/a/3376483
    add_definitions(-Wall -Wextra -Wfloat-equal -Wundef -Wshadow -Wpointer-arith -Wcast-align
                    -Wstrict-overflow=5 -Wwrite-strings -Wconversion -Wunreachable-code)
    # to make package relocatable
    set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib:$ORIGIN/")
endif()
add_definitions(-DGLOG_NO_ABBREVIATED_SEVERITIES)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(src)
if(${PROJECT_NAME}_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
if(${PROJECT_NAME}_BUILD_DEMOS)
    add_subdirectory(demos)
endif()
if(${PROJECT_NAME}_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

# copy data from bin directory
if(WIN32)
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin
            DESTINATION .
            FILES_MATCHING PATTERN "*.dll" PATTERN "*.ini")

else()
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin
            DESTINATION .
            FILES_MATCHING PATTERN "*.so" PATTERN "*.so.*" PATTERN "*.ini")
endif()
