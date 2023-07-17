#ifndef BINARY_H
#define BINARY_H
// #pragma once

#include<opencv2/opencv.hpp>

int calc_mean(cv::Mat im1, cv::Mat im2);

void binary(cv::Mat gray, double mean);


#endif//*_binary.h_*