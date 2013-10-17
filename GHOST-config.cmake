include("${CMAKE_CURRENT_LIST_DIR}/GHOST-targets.cmake")
get_filename_component(GHOST_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/../../include/ghost" ABSOLUTE)

set(GHOST_LIBRARIES hwloc)
set(GHOST_LIBDIR "${CMAKE_CURRENT_LIST_DIR}")
