include(LibFindMacros)

libfind_pkg_check_modules(HWLOC_PKGCONF hwloc)
find_path(HWLOC_INCLUDE_DIR NAMES hwloc.h PATHS ${HWLOC_PKGCONF_INCLUDE_DIRS})
set(HWLOC_PROCESS_INCLUDES HWLOC_INCLUDE_DIR)
libfind_process(HWLOC)
