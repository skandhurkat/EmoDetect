subdirs(FeatureExtractors)
link_directories(${PROJECT_SOURCE_DIR}/src/FeatureExtractors)
add_executable(readImage readImage.cpp)
add_executable(train
    LearningAlgorithms/learningAlgorithms.cpp
    train.cpp)
add_executable(test
    LearningAlgorithms/learningAlgorithms.cpp
    test.cpp)
add_executable(demo
    LearningAlgorithms/learningAlgorithms.cpp
    demo.cpp)
add_executable(validateHuman
    validateHuman.cpp)
add_executable(trainAndValidate
    LearningAlgorithms/learningAlgorithms.cpp
    trainAndValidate.cpp)
add_executable(trainAndValidateAllAlgos
    LearningAlgorithms/learningAlgorithms.cpp
    trainAndValidateAllAlgos.cpp)
add_executable(trainAndTestAllAlgos
    LearningAlgorithms/learningAlgorithms.cpp
    trainAndTestAllAlgos.cpp)
add_executable(trainAndCrossValidateSVM
    LearningAlgorithms/learningAlgorithms.cpp
    trainAndCrossValidateSVM.cpp)
add_executable(trainAndCrossValidateRT
    LearningAlgorithms/learningAlgorithms.cpp
    trainAndCrossValidateRT.cpp)
add_executable(t-TestRT
		LearningAlgorithms/learningAlgorithms.cpp
		t-TestRT.cpp)
add_executable(t-TestSVM
		LearningAlgorithms/learningAlgorithms.cpp
		t-TestSVM.cpp)
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
target_link_libraries(train ${OpenCV_LIBS}
    featureExtractors)
target_link_libraries(test ${OpenCV_LIBS}
    featureExtractors)
target_link_libraries(demo ${OpenCV_LIBS}
    featureExtractors)
target_link_libraries(validateHuman ${OpenCV_LIBS})
target_link_libraries(trainAndValidate ${OpenCV_LIBS}
    featureExtractors)
target_link_libraries(trainAndValidateAllAlgos ${OpenCV_LIBS}
    featureExtractors)
target_link_libraries(trainAndTestAllAlgos ${OpenCV_LIBS}
    featureExtractors)
target_link_libraries(trainAndCrossValidateSVM ${OpenCV_LIBS}
    featureExtractors)
target_link_libraries(trainAndCrossValidateRT ${OpenCV_LIBS}
    featureExtractors)
target_link_libraries(t-TestRT ${OpenCV_LIBS} featureExtractors)
target_link_libraries(t-TestSVM ${OpenCV_LIBS} featureExtractors)
target_link_libraries(extractMomentsMain ${OpenCV_LIBS}
    featureExtractors)
target_link_libraries(extractHaarMain ${OpenCV_LIBS}
    featureExtractors)
target_link_libraries(extractGaborMain ${OpenCV_LIBS}
    featureExtractors)
target_link_libraries(extractPHoGMain ${OpenCV_LIBS}
    featureExtractors)
install(TARGETS train test demo DESTINATION bin)
