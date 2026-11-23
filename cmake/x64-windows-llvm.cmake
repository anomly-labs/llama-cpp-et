set( CMAKE_SYSTEM_NAME Windows )
set( CMAKE_SYSTEM_PROCESSOR x86_64 )

set( CMAKE_C_COMPILER    clang )
set( CMAKE_CXX_COMPILER  clang++ )

# The hosted Windows runner's MSVC STL can be newer than its preinstalled Clang
# and then refuses to compile (error STL1000). Accept the pairing rather than
# fail every Windows build each time the runner image moves.
set( CMAKE_CXX_FLAGS_INIT "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH" )
