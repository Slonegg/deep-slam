add_executable(example_tool
    example_tool.cpp
)
set_target_properties(example_tool PROPERTIES FOLDER tools)
target_link_libraries(example_tool ${PROJECT_NAME})
install(TARGETS example_tool
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

