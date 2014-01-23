include(LibFindMacros)

libfind_pkg_check_modules(LIBSCI_PKGCONF sci)
find_path(LIBSCI_INCLUDE_DIR NAMES cblas.h PATHS ${LIBSCI_PKGCONF_INCLUDE_DIRS})
set(LIBSCI_PROCESS_INCLUDES LIBSCI_INCLUDE_DIR)
libfind_process(LIBSCI)
