subdirs(FeatureExtractors)
link_directories(${PROJECT_SOURCE_DIR}/src/FeatureExtractors)
add_executable(readImage readImage.cpp)
add_executable(trainAndValidate
    LearningAlgorithms/learningAlgorithms.cpp
    trainAndValidate.cpp)
add_executable(trainAndValidateAllAlgos
    LearningAlgorithms/learningAlgorithms.cpp
    trainAndValidateAllAlgos.cpp)
add_executable(trainAndCrossValidateSVM
    LearningAlgorithms/learningAlgorithms.cpp
    trainAndCrossValidateSVM.cpp)
add_executable(extractMomentsMain
    LearningAlgorithms/learningAlgorithms.cpp
    extractMomentsMain.cpp)
add_executable(extractPHoGMain
    LearningAlgorithms/learningAlgorithms.cpp
    extractPHoGMain.cpp)
add_executable(extractHaarMain
    LearningAlgorithms/learningAlgorithms.cpp
    extractHaarMain.cpp)
add_executable(extractGaborMain FeatureExtractors/extractGabor.cpp 
    LearningAlgorithms/learningAlgorithms.cpp
    extractGaborMain.cpp)
target_link_libraries(readImage ${OpenCV_LIBS})
target_link_libraries(trainAndValidate ${OpenCV_LIBS}
    featureExtractors)
target_link_libraries(trainAndValidateAllAlgos ${OpenCV_LIBS}
    featureExtractors)
target_link_libraries(trainAndCrossValidateSVM ${OpenCV_LIBS}
    featureExtractors)
target_link_libraries(extractMomentsMain ${OpenCV_LIBS}
    featureExtractors)
target_link_libraries(extractHaarMain ${OpenCV_LIBS}
    featureExtractors)
target_link_libraries(extractGaborMain ${OpenCV_LIBS}
    featureExtractors)
target_link_libraries(extractPHoGMain ${OpenCV_LIBS}
    featureExtractors)
install(TARGETS trainAndValidate DESTINATION bin)
