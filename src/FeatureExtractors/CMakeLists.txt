add_library(featureExtractors SHARED
    extractFeatures.cpp
    extractGabor.cpp
    extractHaar.cpp
    extractMoments.cpp
    extractPHoG.cpp)

target_link_libraries(featureExtractors ${OpenCV_LIBS})

install(TARGETS featureExtractors
    RUNTIME DESTINATION bin COMPONENT libraries
    LIBRARY DESTINATION lib COMPONENT libraries
    ARCHIVE DESTINATION lib/static COMPONENT libraries)
