#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "clFFT" for configuration "Release"
set_property(TARGET clFFT APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clFFT PROPERTIES
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "/Library/Developer/CommandLineTools/SDKs/MacOSX11.0.sdk/System/Library/Frameworks/OpenCL.framework"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclFFT.2.12.2.dylib"
  IMPORTED_SONAME_RELEASE "libclFFT.2.dylib"
  )

list(APPEND _IMPORT_CHECK_TARGETS clFFT )
list(APPEND _IMPORT_CHECK_FILES_FOR_clFFT "${_IMPORT_PREFIX}/lib/libclFFT.2.12.2.dylib" )

# Import target "StatTimer" for configuration "Release"
set_property(TARGET StatTimer APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(StatTimer PROPERTIES
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "/Library/Developer/CommandLineTools/SDKs/MacOSX11.0.sdk/System/Library/Frameworks/OpenCL.framework"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libStatTimer.2.12.2.dylib"
  IMPORTED_SONAME_RELEASE "libStatTimer.2.dylib"
  )

list(APPEND _IMPORT_CHECK_TARGETS StatTimer )
list(APPEND _IMPORT_CHECK_FILES_FOR_StatTimer "${_IMPORT_PREFIX}/lib/libStatTimer.2.12.2.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
