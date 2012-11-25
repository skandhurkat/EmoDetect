subdirs(FeatureExtractors)
link_directories(FeatureExtractors)
add_executable(readImage readImage.cpp)
add_executable(extractMomentsMain
    LearningAlgorithms/learningAlgorithms.cpp
    extractMomentsMain.cpp)
add_executable(extractPHoGMain FeatureExtractors/extractPHoG.cpp
    LearningAlgorithms/svm.cpp LearningAlgorithms/rt.cpp
    LearningAlgorithms/ann.cpp
    extractPHoGMain.cpp)
add_executable(extractHaarMain FeatureExtractors/extractHaar.cpp 
    LearningAlgorithms/svm.cpp LearningAlgorithms/rt.cpp
    LearningAlgorithms/ann.cpp
    extractHaarMain.cpp)
add_executable(extractGaborMain FeatureExtractors/extractGabor.cpp 
    LearningAlgorithms/svm.cpp LearningAlgorithms/rt.cpp
    LearningAlgorithms/ann.cpp
    extractGaborMain.cpp)
target_link_libraries(readImage ${OpenCV_LIBS})
target_link_libraries(extractMomentsMain ${OpenCV_LIBS}
    featureExtractors)
target_link_libraries(extractHaarMain ${OpenCV_LIBS})
target_link_libraries(extractGaborMain ${OpenCV_LIBS})
target_link_libraries(extractPHoGMain ${OpenCV_LIBS})
