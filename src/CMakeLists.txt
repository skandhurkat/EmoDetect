add_executable(readImage readImage.cpp)
add_executable(extractMoments extractMoments.cpp)
target_link_libraries(readImage ${OpenCV_LIBS})
target_link_libraries(extractMoments ${OpenCV_LIBS})
