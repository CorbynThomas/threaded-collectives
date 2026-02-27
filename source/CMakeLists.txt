target_sources(thread_collectives PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/local_collectives.cpp" 
                                  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/thread_functions.cpp")

target_include_directories(thread_collectives PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

find_package(MPI REQUIRED)
target_link_libraries(thread_collectives PRIVATE MPI::MPI_CXX)

find_package(OpenMP REQUIRED)
target_link_libraries(thread_collectives PRIVATE OpenMP::OpenMP_CXX)