#include"Binary.h"
#include<opencv2/opencv.hpp>


int calc_mean(cv::Mat im1, cv::Mat im2)
{
	cv::Scalar mean1, mean2;
	cv::Scalar stddev1, stddev2;

	cv::meanStdDev(im1, mean1, stddev1);
	cv::meanStdDev(im2, mean2, stddev2);
	double mean_p1 = mean1.val[0];
	double mean_p2 = mean2.val[0];

	double mean = (mean_p1 + mean_p2) / 2;
	return std::sqrt(mean * 40);
}


void binary(cv::Mat gray, double mean)
{
	for(int i = 0; i < gray.cols; i++)
	{
		for(int j = 0; j < gray.rows; j++)
		{
			uchar intensity =gray.at<uchar>(j, i);
			if( 4 * mean > intensity > mean)
			{
				gray.at<uchar>(j, i) = 0;
			}
			else{
				gray.at<uchar>(j, i) = 255;
			}

		}
	}
}