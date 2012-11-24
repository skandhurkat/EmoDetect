add_library(featureExtractors STATIC
    extractFeatures.cpp
    extractGabor.cpp
    extractHaar.cpp
    extractMoments.cpp
    extractPHoG.cpp)

target_link_libraries(featureExtractors ${OpenCV_LIBS})
