EmoDetect
=========

A project that enables computers to identify human emotion.

All files copyright 2012 Rishabh Animesh, Skand Hurkat, Abhinandan
Majumdar, Aayush Saxena.

Licence
-------

EmoDetect is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public Licence as published by the Free
Software Foundation, either version 3 of the software, or (at your option)
any later version.

EmoDetect is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public Licence for
more details.

You should have received a copy of the GNU General Public Licence along
with EmoDetect. If not, see <http://www.gnu.org/licenses>

About
-----

We've trained classifiers on over 10,000 images from the [MUG Facial
Expression Database](http://mug.ee.auth.gr/fed); the trained classifiers
are available in the data/xml folder.

### Files ###

* demo                    : Binary for Project Demonstration
* extractGaborMain        : Extracts Gabor features and dumps them to a file
* extractPHoGMain         : Extracts HoG features and dumps them to a file
* extractHaarMain         : Extracts Haar features and dumps them to a file
* extractMomentsMain      : Extracts Mom features and dumps them to a file
* readImage               : Read image from a file and display it
                            (Hello World equivalent for OpenCV)
* test                    : Test a specific extractor-learner combination on
                            a set of test images
* train                   : Train from a database of images using a specifi 
                            extractor-learner combination and save the 
                            classifier to a file
* trainAndCrossValidateRT : Train and Cross Validate for Random Trees 
                            (RT) and produce the optimum max_depth
* trainAndCrossValidateSVM: Train and Cross Validate for Support 
                            Vector Machines (SVM) and produce the optimum 
                            max_depth
* trainAndTestAllAlgos    : Train and Test without cross validation 
                            using the trained classifer (default)
* trainAndValidate        : Train using a saved classifier and measure 
                            the test error on a test set
* validateHuman           : Executable shows 50 random images and 
                            asks the user to guess their right emotion

Run any binary with --help option to see the command line interface for
the binary
    
Compiling
---------

* Package [OpenCV](http://www.opencv.org) (>2.4) installed with the WITH_QT flag set
* [CMAKE](http://www.cmake.org) (>2.6) is required for compilation

CMake will generate the build files for your favourite environment. The
code works on Linux, Windows and Mac.
