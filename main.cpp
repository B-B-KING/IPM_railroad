#include<opencv2/core/core.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/core/utility.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<iostream>
#include<time.h>
#include<eigen3/Eigen/Core>


#include "Binary.h"


using namespace std;
// using namespace cv;


int calc_mean(cv::Mat im1, cv::Mat im2);
void binary(cv::Mat gary, double mean);
void calchist(cv::Mat img, cv::Mat histImg);

int main(int argc, char** argv)
{
	
	clock_t start, end;
	// cv::namedWindow("src", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("dst", cv::WINDOW_AUTOSIZE);
	// cv::namedWindow("lsd", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("copy", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("f45", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("f135", cv::WINDOW_AUTOSIZE);
	cv::VideoCapture cap;

	cv::Ptr<cv::LineSegmentDetector> lsd = cv::createLineSegmentDetector(cv::LSD_REFINE_STD);

	cap.open(string(argv[1]));
	cv::Mat frame, dst;
	
	vector<cv::Vec4f> vecLines;
	// Eigen::Matrix3d kernel;
	cv::Mat kernel_black = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3, 3));
	cv::Mat kernel_open = cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(2, 3));
	cv::Mat kernel_media = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	cout << "kernel_black:" << kernel_black <<endl;
	cout << "kernel_open:" << kernel_open <<endl;
	
	cv::Mat kernel45 = (cv::Mat_<int>(3, 3) << -2, -1, 0,-1, 0, 1, 0, 1, 2);
	cv::Mat kernel135 = (cv::Mat_<int>(3,3) << 0, 1, 2, -1, 0, 1, -2, -1, 0);
	cout << "kernel45&135:" << endl << kernel45 << endl << kernel135 <<endl;

	// float h_ranges[] = {0, 180},s_ranges[] = {0, 256};
	// const float* ranges[] = {h_ranges, s_ranges};
	// int histsize[] = {30, 32}, ch[] = {0, 1, 2};
	cv::VideoWriter writer("", CV_FOURCC('M', 'J', 'P', 'G'), 30, Size(640, 480))
	for(;;)
	{

		start = clock();
		cap >> frame;
		cv::Mat copy(frame), copy1, gray;
		cv::Mat f45, f135;
		// cv::resize(copy, copy, cv::Size(640, 480));
		// cv::pyrDown(frame, frame);
		// cv::Mat mask = cv::Mat::zeros(copy.rows, copy.cols, CV_8UC3);
		if(copy.empty()) break;
		cv::cvtColor(copy, gray,cv::COLOR_BGR2GRAY);
		
		cv::Mat hist;
		// calchist(gray, hist);

		// cv::calcHist(&copy, 1, ch, cv::noArray(), hist, 2, histsize, ranges, true);
		// cv::equalizeHist(copy, copy1);

		cv::medianBlur(gray, copy1, 3);
		cv::adaptiveThreshold(copy1, dst, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 3, 10);
		cv::filter2D(gray, f45, CV_8U, kernel45);
		cv::filter2D(gray, f135, CV_8U, kernel135);
		double mean = calc_mean(f45, f135);
		binary(gray, mean);
		cout << mean << endl;


		// cv::morphologyEx(dst, dst, cv::MORPH_CLOSE, kernel_black, Point(-1, -1), 1);
		// cv::morphologyEx(dst, dst, cv::MORPH_OPEN, kernel_open);
		//"xingtaixue chuli "
		
		// cv::Canny(copy, dst, 50, 100);
		// lsd->detect(dst, vecLines);
		// lsd->drawSegments(mask, vecLines);
		//"calc LSD line"


		cv::imshow("src", frame);
		// cv::imshow("copy1", copy1);
		cv::imshow("copy", gray);
		cv::imshow("dst", dst);
		cv::imshow("f45", f45);
		cv::imshow("f135", f135);
		// cv::imshow("hist", hist);
		end = clock();

		cout << "run time:" << (double)(end - start) * 1000 /CLOCKS_PER_SEC << "ms" << endl;
		// // cout << "clock : " << CLOCKS_PER_SEC << endl;
		// cout << "frame shape :" << frame.size << endl;
		if(cv::waitKey(1) >= 0) break;
	}

	return 0;
}





void calchist(cv::Mat img, cv::Mat histImg)
{
	int hbins = 60,sbins = 64;
	int histSize[] = {hbins,sbins};
	//hue varies from 0 to 179
	float hranges[] = {0,180};
	//saturation varies from 0 to 255
	float sranges[] = {0,255};
	const float *ranges[] = {hranges,sranges};
	//two channels 0th,1th
	int channels[] = {0,1};
	cv::MatND hist;
	//compute h-s histogram
	calcHist(&img,1,channels,cv::Mat(),hist,2,histSize,ranges);
	//get the max value
	double maxVal = .0;
	cv::minMaxLoc(hist,0,&maxVal,0,0);
	int scale = 8;
	//show the histogram on the image
	histImg = cv::Mat::zeros(sbins*scale,hbins*scale,CV_8UC3);
	for (int h = 0;h < hbins;h++)
	{
		for (int s = 0;s<sbins;s++)
		{
			float binVal = hist.at<float>(h,s);
			int intensity = cvRound(binVal*0.9*255/maxVal);
			cv::rectangle(histImg, cv::Point(h*scale,s*scale),cv::Point((h+1)*scale-1,(s+1)*scale-1),cv::Scalar::all(intensity),CV_FILLED);
		}
	}

}


// /*********************************************
//              内容：计算H-S 直方图分布      
//              时间：2013 5.27
// 	     作者：恋上蛋炒面      
// *********************************************/
// #include <opencv2/core/core.hpp>
// #include <opencv2/highgui/highgui.hpp>
// #include <opencv2/imgproc/imgproc.hpp>
// #include<iostream>
// using namespace cv;
 
// int main(int argc, char** argv)
// {
// 	Mat source = imread(std::string(argv[1]));
// 	namedWindow("Source");
// 	imshow("Source",source);
// 	Mat hsv;
// 	cvtColor(source,hsv,CV_BGR2HSV);
// 	//Quantize the hue to 60 levels
// 	//and the saturation to 64 levels
// 	int hbins = 60,sbins = 64;
// 	int histSize[] = {hbins,sbins};
// 	//hue varies from 0 to 179
// 	float hranges[] = {0,180};
// 	//saturation varies from 0 to 255
// 	float sranges[] = {0,255};
// 	const float *ranges[] = {hranges,sranges};
// 	//two channels 0th,1th
// 	int channels[] = {0,1};
// 	MatND hist;
// 	//compute h-s histogram
// 	calcHist(&hsv,1,channels,Mat(),hist,2,histSize,ranges);
// 	//get the max value
// 	double maxVal = .0;
// 	minMaxLoc(hist,0,&maxVal,0,0);
// 	int scale = 8;
// 	//show the histogram on the image
// 	Mat histImg = Mat::zeros(sbins*scale,hbins*scale,CV_8UC3);
// 	for (int h = 0;h < hbins;h++)
// 	{
// 		for (int s = 0;s<sbins;s++)
// 		{
// 			float binVal = hist.at<float>(h,s);
// 			int intensity = cvRound(binVal*0.9*255/maxVal);
// 			rectangle(histImg,Point(h*scale,s*scale),Point((h+1)*scale-1,(s+1)*scale-1),Scalar::all(intensity),CV_FILLED);
// 		}
// 	}
 
// 	namedWindow("H-S Histogram");
// 	imshow("H-S Histogram",histImg);
// 	// imwrite("hshistogram.jpg",histImg);
// 	waitKey(0);
// }