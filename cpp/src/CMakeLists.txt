set(INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/deep_slam)
set(HEADERS
    ${INCLUDE_DIR}/example_header.h
)
set(SOURCES
    example_source.cpp
)

# arrange sources
source_group(headers FILES ${HEADERS})
source_group(sources FILES ${SOURCES})

# add target
add_library(${PROJECT_NAME} SHARED
    ${SOURCES}
    ${HEADERS}
)

# link libraries
if (CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

install(TARGETS ${PROJECT_NAME}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
install(FILES ${HEADERS}
        DESTINATION include/deep_slam COMPONENT development)
