#SETUP PROJECT
set (PROJECT python_smurff)
message("Configuring " ${PROJECT} "...")
project (${PROJECT})

include_directories(${NUMPY_INCLUDE_DIR})
include_directories(../../lib/smurff-cpp)
include_directories(include)

# With CMake, a clean separation can be made between the source tree and the
# build tree.  When all source is compiled, as with pure C/C++, the source is
# no-longer needed in the build tree.  However, with pure *.py source, the
# source is processed directly.  To handle this, we reproduce the availability
# of the source files in the build tree.
add_custom_target( ReplicatePythonSourceTree ALL ${CMAKE_COMMAND} -P
  ${SCRIPT_DIR}/ReplicatePythonSourceTree.cmake
  ${CMAKE_CURRENT_BINARY_DIR}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )

add_subdirectory( smurff )

configure_file(setup.py.in ${CMAKE_CURRENT_BINARY_DIR}/setup.py)


# run python setup.py on install
install(CODE "MESSAGE(WARNING \"Python wrapper needs ot be installed manually. \\
Please run '${PYTHON_EXECUTABLE} setup.py install' in ${CMAKE_CURRENT_BINARY_DIR}\")")

# install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} setup.py install WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})")
