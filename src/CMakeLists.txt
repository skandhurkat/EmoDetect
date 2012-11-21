add_executable(readImage readImage.cpp)
<<<<<<< HEAD
add_executable(extractMoments extractMoments.cpp)
add_executable(extractHoG extractHoG.cpp phog.cpp)
=======
add_executable(extractMoments FeatureExtractors/extractMoments.cpp 
    extractMomentsMain.cpp)
>>>>>>> master
target_link_libraries(readImage ${OpenCV_LIBS})
target_link_libraries(extractMoments ${OpenCV_LIBS})
target_link_libraries(extractHoG ${OpenCV_LIBS})
