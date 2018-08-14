#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>

using namespace cv;
using namespace std;

Mat integralImage(Mat Im)
{
	Mat Image;
	Im.convertTo(Image, CV_64FC1);
	int VerticalSize = Image.rows;
	int HorizontalSize = Image.cols;

	Mat IntegralImage = Mat::zeros(Size(HorizontalSize, VerticalSize), CV_64FC1);

	//int nr = Image.rows; // number of rows
	//int nc = Image.cols * Image.channels(); // total number of elements per line

	//first, compute sums along the horizontal direction
	for (int i = 0; i<VerticalSize; i++)
	{
		IntegralImage.at<double>(i, 0) = (double)Image.at<double>(i, 0);

		for (int j = 1; j<HorizontalSize; j++)
		{
			double PreviousSum = (double)IntegralImage.at<double>(i, j - 1);
			double CurrentValue = (double)Image.at<double>(i, j);
			IntegralImage.at<double>(i, j) = double(PreviousSum+CurrentValue);
			//if(j == 300)
			//{
                //cout<<PreviousSum<<endl;
			//}
		}
	}
	//second, compute sums along the vertical direction
	for (int i = 1; i<VerticalSize; i++)
	{
		for (int j = 0; j<HorizontalSize; j++)
		{
			double PreviousSum = (double)IntegralImage.at<double>(i - 1, j);
			double CurrentValue = (double)IntegralImage.at<double>(i, j);
			IntegralImage.at<double>(i, j) = double(PreviousSum + CurrentValue);
		}
	}
	return IntegralImage;   	//返回的Mat都是CV_64FC1型
}
