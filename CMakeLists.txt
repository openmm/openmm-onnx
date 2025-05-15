#---------------------------------------------------
# OpenMM ONNX Plugin
#----------------------------------------------------

CMAKE_MINIMUM_REQUIRED(VERSION 3.5)

# We need to know where OpenMM is installed so we can access the headers and libraries.
SET(OPENMM_DIR "/usr/local/openmm" CACHE PATH "Where OpenMM is installed")
INCLUDE_DIRECTORIES("${OPENMM_DIR}/include")
LINK_DIRECTORIES("${OPENMM_DIR}/lib" "${OPENMM_DIR}/lib/plugins")

# We also need to know where ONNX Runtime is installed.
FIND_LIBRARY(ONNX_LIBRARY_PATH onnxruntime)
FIND_FILE(ONNX_HEADER_PATH "onnxruntime/core/session/onnxruntime_cxx_api.h")
CMAKE_PATH(GET ONNX_LIBRARY_PATH PARENT_PATH ONNX_LIBRARY_DIR)
CMAKE_PATH(GET ONNX_HEADER_PATH PARENT_PATH ONNX_INCLUDE_DIR)
INCLUDE_DIRECTORIES("${ONNX_INCLUDE_DIR}")
LINK_DIRECTORIES("${ONNX_LIBRARY_DIR}")

# Specify the C++ version we are building for.
SET (CMAKE_CXX_STANDARD 14)

# Set flags for linking on mac
IF(APPLE)
    SET(CMAKE_INSTALL_NAME_DIR "@rpath")
    SET(EXTRA_COMPILE_FLAGS "-stdlib=libc++")
ENDIF(APPLE)

# Select where to install
IF(${CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT})
    IF(WIN32)
        SET(CMAKE_INSTALL_PREFIX "$ENV{ProgramFiles}/OpenMM" CACHE PATH "Where to install the plugin" FORCE)
    ELSE(WIN32)
        SET(CMAKE_INSTALL_PREFIX "/usr/local/openmm" CACHE PATH "Where to install the plugin" FORCE)
    ENDIF(WIN32)
ENDIF(${CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT})

# The source is organized into subdirectories, but we handle them all from
# this CMakeLists file rather than letting CMake visit them as SUBDIRS.
SET(ONNX_PLUGIN_SOURCE_SUBDIRS openmmapi serialization)

# Set the library name
SET(ONNX_LIBRARY_NAME OpenMMONNX)
SET(SHARED_ONNX_TARGET ${ONNX_LIBRARY_NAME})

# These are all the places to search for header files which are to be part of the API.
SET(API_INCLUDE_DIRS "openmmapi/include" "openmmapi/include/internal")

# Locate header files.
SET(API_INCLUDE_FILES)
FOREACH(dir ${API_INCLUDE_DIRS})
    FILE(GLOB fullpaths ${dir}/*.h)
    SET(API_INCLUDE_FILES ${API_INCLUDE_FILES} ${fullpaths})
ENDFOREACH(dir)

# Collect up source files
SET(SOURCE_FILES) # empty
SET(SOURCE_INCLUDE_FILES)
FOREACH(subdir ${ONNX_PLUGIN_SOURCE_SUBDIRS})
    FILE(GLOB src_files  ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.cpp)
    FILE(GLOB incl_files ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.h)
    SET(SOURCE_FILES         ${SOURCE_FILES}         ${src_files})   #append
    SET(SOURCE_INCLUDE_FILES ${SOURCE_INCLUDE_FILES} ${incl_files})

    ## Make sure we find these locally before looking in OpenMM/include if
    ## OpenMM was previously installed there.
    INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/include)
ENDFOREACH(subdir)

# Create the library.

ADD_LIBRARY(${SHARED_ONNX_TARGET} SHARED ${SOURCE_FILES} ${SOURCE_INCLUDE_FILES} ${API_INCLUDE_FILES})
SET_TARGET_PROPERTIES(${SHARED_ONNX_TARGET}
    PROPERTIES COMPILE_FLAGS "-DONNX_BUILDING_SHARED_LIBRARY ${EXTRA_COMPILE_FLAGS}"
    LINK_FLAGS "${EXTRA_COMPILE_FLAGS}")
TARGET_LINK_LIBRARIES(${SHARED_ONNX_TARGET} OpenMM onnxruntime)
INSTALL_TARGETS(/lib RUNTIME_DIRECTORY /lib ${SHARED_ONNX_TARGET})

# install headers
FILE(GLOB API_ONLY_INCLUDE_FILES "openmmapi/include/*.h")
INSTALL (FILES ${API_ONLY_INCLUDE_FILES} DESTINATION include)
FILE(GLOB API_ONLY_INCLUDE_FILES_INTERNAL "openmmapi/include/internal/*.h")
INSTALL (FILES ${API_ONLY_INCLUDE_FILES_INTERNAL} DESTINATION include/internal)

# Enable testing

ENABLE_TESTING()
ADD_SUBDIRECTORY(tests)
ADD_SUBDIRECTORY(serialization/tests)

# Copy test files to the build directory.

file(GLOB_RECURSE TEST_FILES RELATIVE "${CMAKE_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/tests/*.onnx"
)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
set(COPIED_TEST_FILES)
foreach(TEST_FILE ${TEST_FILES})
    set(infile "${CMAKE_SOURCE_DIR}/${TEST_FILE}")
    set(outfile "${CMAKE_BINARY_DIR}/${TEST_FILE}")
    add_custom_command(
        OUTPUT "${outfile}"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${infile}" "${outfile}"
        DEPENDS "${infile}"
        COMMENT "CMake-copying file ${infile} to ${outfile}")
    set(COPIED_TEST_FILES ${COPIED_TEST_FILES} "${outfile}")
endforeach()
add_custom_target(CopyTestFiles ALL DEPENDS ${COPIED_TEST_FILES})

# Build the Python API

FIND_PROGRAM(PYTHON_EXECUTABLE python)
FIND_PROGRAM(PIP_EXECUTABLE pip)
FIND_PROGRAM(SWIG_EXECUTABLE swig)
IF(PYTHON_EXECUTABLE AND SWIG_EXECUTABLE)
    SET(ONNX_BUILD_PYTHON_WRAPPERS ON CACHE BOOL "Build wrappers for Python")
ELSE(PYTHON_EXECUTABLE AND SWIG_EXECUTABLE)
    SET(ONNX_BUILD_PYTHON_WRAPPERS OFF CACHE BOOL "Build wrappers for Python")
ENDIF(PYTHON_EXECUTABLE AND SWIG_EXECUTABLE)
IF(ONNX_BUILD_PYTHON_WRAPPERS)
    ADD_SUBDIRECTORY(python)
ENDIF(ONNX_BUILD_PYTHON_WRAPPERS)
