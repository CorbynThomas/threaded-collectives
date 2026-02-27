add_executable(monte_carlo "thread_monte_carlo.cpp")

find_package(MPI REQUIRED)
target_link_libraries(monte_carlo PRIVATE MPI::MPI_CXX)

find_package(OpenMP REQUIRED)
target_link_libraries(monte_carlo PRIVATE OpenMP::OpenMP_CXX)

target_link_libraries(monte_carlo PRIVATE thread_collectives)