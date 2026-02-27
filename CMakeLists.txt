cmake_minimum_required(VERSION 3.13)
project(threadcollectives LANGUAGES C CXX)


# library name, SHARED or STATIC, files to include
add_library(thread_collectives SHARED "")
add_subdirectory("source")

add_subdirectory("benchmarks")