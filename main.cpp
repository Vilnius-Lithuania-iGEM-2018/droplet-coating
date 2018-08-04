//
// Created by Lukas Praninskas on 04/08/2018.
//

#include <iostream>
#include <cv.hpp>

int main(int argc, char **argv) {
    using namespace std;

    cv::namedWindow("Droplet Coating");

    if(strcmp(argv[1], "-vid") == 0) {
        //input data coming from a video
        //cv::processVideo(argv[2]);
    }

    return 0;
}