function(add_test_target TEST_NAME TEST_SOURCE)
    get_filename_component(TEST_TARGET ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_TARGET} ${TEST_SOURCE})
    target_link_libraries(${TEST_TARGET}
        CONAN_PKG::gtest
    )
    set_target_properties(${TEST_TARGET} PROPERTIES FOLDER test)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_TARGET} WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
endfunction()

add_test_target("example" test_example.cpp)