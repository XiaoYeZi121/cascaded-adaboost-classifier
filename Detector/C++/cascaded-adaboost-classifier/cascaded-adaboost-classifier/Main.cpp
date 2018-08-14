#define _CRT_SECURE_NO_DEPRECATE

#include "opencv2/opencv.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "getFiles.h"
#include "glob.h"
#include "integralImage.h"
#include "BoostedCommittee.h"

#include <math.h>
#include <iostream>
#include "time.h"

#include <fstream>

#include <unistd.h> //use access function in Linux
#include <sys/types.h> //use mkdir function in Linux
#include <sys/stat.h>   //use mkdir function in Linux

using namespace std;
using namespace cv;

int round_double(double number)
{
	return (number > 0.0) ? floor(number + 0.5) : ceil(number - 0.5);
}

#define max(a,b) ((a) >= (b) ? (a) : (b))
#define min(a,b) ((a) <= (b) ? (a) : (b))

int main()
{
	Size MaxObjectSize = Size(512, 512);
	Size MinObjectSize = Size(128, 128);
	double ScaleFactor = 1.1;

	string RootPath = "";

	string ClassifierPath = RootPath + "dt/*";
	vector<string> ClassifierNames;
	ClassifierNames = getFiles(ClassifierPath);

	vector<CBoostedCommittee> Classifiers(ClassifierNames.size());
	vector<double> Thresholds(ClassifierNames.size());

	string ThresholdFilePath = RootPath + "dt2";

	for (int i = 0; i < ClassifierNames.size(); i++)
	{
		//读取分类器
		string classifier_name = ClassifierNames[i];
		CBoostedCommittee Temp;
		FILE *fid = fopen((char*)classifier_name.data(), "r");
		Temp.LoadFromFile(fid);
		Classifiers[i] = Temp;

		//读取threshold
		int silly = classifier_name.find_last_of("/");
		string silly2 = classifier_name.substr(silly + 1);
		int silly3 = silly2.find_last_of(".");
		string silly4 = silly2.substr(0, silly3);
		string threshold_file_name = ThresholdFilePath + "/" + silly4 + "_Threshold.txt";
		FILE *fid2 = fopen((char*)threshold_file_name.data(), "r");
		double threshold;
		fscanf(fid2, "%lf", &threshold); //f是浮点数的格式符，它所定义的是浮点型数据;lf是double型数据的格式符号。
		Thresholds[i] = threshold;
	}

	vector<int> FeatureSelected;

	for (int i = 0; i < Classifiers.size(); i++)
	{
		CBoostedCommittee StrongClassifier = Classifiers[i];
		std::vector <CSPHypothesis> m_vHypotheses = StrongClassifier.get_m_vHypotheses();
		for (int j = 0; j < m_vHypotheses.size(); j++)
		{
			std::vector <int> m_vDims = m_vHypotheses[j].get_m_vDims();
			for (int k = 0; k < m_vDims.size(); k++)
			{
				int feature_dim = m_vDims[k] + 1;
				FeatureSelected.push_back(feature_dim);
			}
		}
	}

	//将FeatureSelected排序，并消除重复的元素
	sort(FeatureSelected.begin(), FeatureSelected.end());
	vector<int>::iterator iter = unique(FeatureSelected.begin(), FeatureSelected.end());
	FeatureSelected.erase(iter, FeatureSelected.end());

	vector< vector<int> > FeatureLocation;
	for (int i = 0; i < FeatureSelected.size(); i++)
	{
		int dim = FeatureSelected[i];
		int BinInd = (dim % 9);
		//cout << dim << ' ' << BinInd << endl;
		if (BinInd == 0)
		{
			BinInd = 9;
		}
		int BlockInd;
		if (BinInd == 9)
		{
			BlockInd = floor(dim / 9);    //dim能被9整除，则为当前块的最后一个bin
		    //cout << dim << ' ' << BlockInd << endl;
		}
		else
		{
			BlockInd = floor(dim / 9) + 1; //dim不能被9整除，则是在下一个block里
										   //cout << dim << ' ' << BlockInd << endl;
		}
		int xInd = BlockInd % 15;
		//cout << BlockInd << ' ' << xInd << endl;
		if (xInd == 0)
		{
			xInd = 15;
		}
		int yInd;
		if (xInd == 15)
		{
			yInd = BlockInd / 15;             //BlockInd能被15整除，则为当前这一行的最后一个block
											  //cout << BlockInd << ' ' << yInd << endl;
		}
		else
		{
			yInd = floor(BlockInd / 15) + 1;  //BlockInd不能被15整除，则是在下一行里
											  //cout << BlockInd << ' ' << yInd << endl;
		}
		vector<int> SingleFeatureLocation(5);
		SingleFeatureLocation[0] = BinInd;
		SingleFeatureLocation[1] = BlockInd;
		SingleFeatureLocation[2] = xInd;
		SingleFeatureLocation[3] = yInd;
		SingleFeatureLocation[4] = 1;
		FeatureLocation.push_back(SingleFeatureLocation);
	}

	/****************************************************************************/

	string ImagePath = RootPath + "TestImages/*";
	vector<string> ImageNames;
	ImageNames = getFiles(ImagePath);

	for (int NumImage = 0; NumImage < ImageNames.size(); NumImage++)
	{
		cout << endl << "processing image" << NumImage + 1 << endl;

		Mat ImageRaw = imread(ImageNames[NumImage], 0);
		Mat Image;
		ImageRaw.convertTo(Image, CV_64FC1);
		Size ImageSize = Size(Image.cols, Image.rows);

		double factor = 1.0;
		Size OriginalWindowSize = MinObjectSize;
		int nwin_y = 15;
		int nwin_x = 15;

		//Mat Mask = Mat::zeros(Image.size(), CV_8UC1);
		//threshold(ImageRaw, Mask, 20, 1, THRESH_BINARY);
		//Mat MaskIntegral = integralImage(Mask); //integralImage返回的Mat都是CV_64FC1型，因此MaskIntegral是CV_64FC1型

		vector<Rect> BoxesFinal;
		vector<double> scores;

		//vector<Mat> HOG_of_BoxesFinal;

		int NumOfObjects = 1;
		while (1)
		{
			int a = clock(); //计时开始
			Size WindowSize = Size(floor(OriginalWindowSize.width*factor + 0.5), floor(OriginalWindowSize.height*factor + 0.5));
			Size ScaledImageSize = Size(floor(ImageSize.width / factor + 0.5), floor(ImageSize.height / factor + 0.5));
			if (ScaledImageSize.width < OriginalWindowSize.width || ScaledImageSize.height < OriginalWindowSize.height)
			{
				break;
			}
			Size ProcessingRectSize = Size(ScaledImageSize.width - OriginalWindowSize.width + 1, ScaledImageSize.height - OriginalWindowSize.height + 1); //在本次循环中检测框会遍历到的图像区域
			if (ProcessingRectSize.width <= 0 || ProcessingRectSize.height <= 0)
			{
				break;
			}
			if (WindowSize.width > MaxObjectSize.width || WindowSize.height > MaxObjectSize.height)
			{
				break;
			}
			if (WindowSize.width < MinObjectSize.width || WindowSize.height < MinObjectSize.height)
			{
				continue;
			}
			Mat ScaledImage;
			resize(Image, ScaledImage, ScaledImageSize);

			//计算梯度
			int XStep = 8;
			int YStep = 8;

			Mat Grad_xr;
			Mat Grad_yu;

			Mat hx = Mat::zeros(1, 3, CV_64FC1);
			hx.at<double>(0, 0) = -1;
			hx.at<double>(0, 1) = 0;
			hx.at<double>(0, 2) = 1;
			Mat hy = Mat::zeros(3, 1, CV_64FC1);
			hy.at<double>(0, 0) = 1;
			hy.at<double>(1, 0) = 0;
			hy.at<double>(2, 0) = -1;
			filter2D(ScaledImage, Grad_xr, -1, hx, Point(-1, -1), 0.0, BORDER_DEFAULT); //when ddepth = -1, the output image will have the same depth as the source.
			filter2D(ScaledImage, Grad_yu, -1, hy, Point(-1, -1), 0.0, BORDER_DEFAULT);

			Mat GradMagnitTemp = Grad_xr.mul(Grad_xr) + Grad_yu.mul(Grad_yu);
			Mat GradMagnit;
			cv::sqrt(GradMagnitTemp, GradMagnit);

			Mat GradIntegral = integralImage(GradMagnit);

			Mat GradAngles = Mat::zeros(Size(GradMagnit.cols, GradMagnit.rows), CV_64FC1);
			const double eps = 1e-6;
			for (int i = 0; i < GradMagnit.rows; i++)
			{
				for (int j = 0; j < GradMagnit.cols; j++)
				{
					double dx = Grad_xr.at<double>(i, j);
					double dy = Grad_yu.at<double>(i, j);
					double result = atan(dy / (dx + eps));
					GradAngles.at<double>(i, j) = result;
				}
			}

			Mat QAngle = Mat::zeros(Size(GradAngles.cols, GradAngles.rows), CV_32SC1);
			const double pi = 3.14159265358979323846;
			for (int i = 0; i < GradAngles.rows; i++)
			{
				for (int j = 0; j < GradAngles.cols; j++)
				{
					double ang = GradAngles.at<double>(i, j);
					int bin_index = static_cast<int>(floor((ang - (-pi / 2)) / (pi / 9) + 1)); //[-pi / 2, -pi / 2 + pi / 9), [-pi / 2 + pi / 9, -pi / 2 + 2 * pi / 9), ··· bin_index的取值范围为1到10， 当且仅当ang = pi / 2时bin_index = 10
					if (bin_index == 10)
					{
						bin_index = 9;
					}
					QAngle.at<int>(i, j) = bin_index;
				}
			}

			vector<Mat> Hist(9);
			for (int index = 1; index <= 9; index++)
			{
				Mat hist = Mat::zeros(ScaledImageSize, CV_64FC1);
				for (int i = 0; i < QAngle.rows; i++)
				{
					for (int j = 0; j < QAngle.cols; j++)
					{
						if (QAngle.at<int>(i, j) == index) //QAngle中元素取值的范围是1到9
						{
							hist.at<double>(i, j) = GradMagnit.at<double>(i, j);
						}
					}
				}
				Hist[index - 1] = hist;
			}

			vector<Mat> HistIntegral(9);
			for (int index = 1; index <= 9; index++)
			{
				Mat hist_integral = integralImage(Hist[index - 1]);
				HistIntegral[index - 1] = hist_integral;
			}

			//开始进行窗口扫描
			for (int y = 1; y < ProcessingRectSize.height; y = y + YStep)
			{
				for (int x = 1; x < ProcessingRectSize.width; x = x + XStep)
				{
					//double temp = MaskIntegral.at<double>(floor(y*factor)-1, floor(x*factor)-1)
					//- MaskIntegral.at<double>(floor(y*factor)-1, floor((x + OriginalWindowSize.width - 1)*factor)-1)
					//- MaskIntegral.at<double>(floor((y + OriginalWindowSize.height - 1)*factor)-1, floor(x*factor)-1)
					//+ MaskIntegral.at<double>(floor((y + OriginalWindowSize.height - 1)*factor)-1, floor((x + OriginalWindowSize.width - 1)*factor)-1);

					//if (temp > OriginalWindowSize.width * OriginalWindowSize.height * factor * factor * 0.5)
					if (1)
					{
						/**************************** HOG Feature *******************************/
						Mat H1 = Mat::zeros(nwin_x*nwin_y * 9, 1, CV_64FC1);
						if (!H1.isContinuous())
						{
							cout << "Error, The features are not stored continiously in the memory !" << endl;
							cout << "This error is happened when detecting image" << NumImage + 1 << endl;
						}
						for (int k = 0; k < FeatureSelected.size(); k++)
						{
							int BinInd = FeatureLocation[k][0];
							int BlockInd = FeatureLocation[k][1];
							int xInd = FeatureLocation[k][2];
							int yInd = FeatureLocation[k][3];

							double H_Norm = GradIntegral.at<double>(y - 1 + (yInd - 1) * 8, x - 1 + (xInd - 1) * 8) //block 的 strid 为 8， 大小为16，注意x 和y 要首先-1因为C++中的计数从0开始
								- GradIntegral.at<double>(y - 1 + (yInd - 1) * 8, x - 1 + (xInd - 1) * 8 + 16 - 1)
								- GradIntegral.at<double>(y - 1 + (yInd - 1) * 8 + 16 - 1, x - 1 + (xInd - 1) * 8)
								+ GradIntegral.at<double>(y - 1 + (yInd - 1) * 8 + 16 - 1, x - 1 + (xInd - 1) * 8 + 16 - 1);
							double H_Temp = HistIntegral[BinInd - 1].at<double>(y - 1 + (yInd - 1) * 8, x - 1 + (xInd - 1) * 8)
								- HistIntegral[BinInd - 1].at<double>(y - 1 + (yInd - 1) * 8, x - 1 + (xInd - 1) * 8 + 16 - 1)
								- HistIntegral[BinInd - 1].at<double>(y - 1 + (yInd - 1) * 8 + 16 - 1, x - 1 + (xInd - 1) * 8)
								+ HistIntegral[BinInd - 1].at<double>(y - 1 + (yInd - 1) * 8 + 16 - 1, x - 1 + (xInd - 1) * 8 + 16 - 1);
							H1.at<double>((BlockInd - 1) * 9 + BinInd - 1) = static_cast<double>(H_Temp) / (H_Norm + eps);
						}

						bool PredictResult = true;

						double H1_Temp[2025];     //2025 = nwin_x*nwin_y*9
						double* p;
						for (int k = 0; k < H1.rows; k++)
						{
							p = H1.ptr<double>(k);//获取行指针
							H1_Temp[k] = p[0];
						}

						for (int k = 0; k < Classifiers.size(); k++)
						{
							CBoostedCommittee StrongClassifier = Classifiers[k];
							double Confidence = StrongClassifier.Predict(H1_Temp);
							//cout<<Confidence<<endl;
							if ((Confidence - Thresholds[k]) <0)
							{
								PredictResult = false;
								break;
							}
						}

						if (PredictResult == true)
						{
							Rect box;
							box.x = x*factor;
							box.y = y*factor;
							box.width = WindowSize.width;
							box.height = WindowSize.height;
							BoxesFinal.push_back(box);

							////compute the full HOG feature of the hard sample
							//Mat H1_Full = Mat::zeros(nwin_x*nwin_y * 9, 1, CV_64FC1);     //2025 = nwin_x*nwin_y*9

							//int counter = 1;
							//for (int n = 0; n<nwin_y; n++)
							//{
							//	for (int m = 0; m<nwin_x; m++)
							//	{
							//		Rect block;
							//		block.x = m * 8;
							//		block.y = n * 8;
							//		block.width = 16;
							//		block.height = 16;
							//		double H_Norm = GradIntegral.at<double>(y - 1 + block.y, x - 1 + block.x) //block 的 strid 为 8， 大小为16，注意x 和y 要首先-1因为C++中的计数从0开始
							//			- GradIntegral.at<double>(y - 1 + block.y, x - 1 + block.x + 16 - 1)
							//			- GradIntegral.at<double>(y - 1 + block.y + 16 - 1, x - 1 + block.x)
							//			+ GradIntegral.at<double>(y - 1 + block.y + 16 - 1, x - 1 + block.x + 16 - 1);
							//		for (int bin = 0; bin<9; bin++)
							//		{
							//			double H_Temp = HistIntegral[bin].at<double>(y - 1 + block.y, x - 1 + block.x)
							//				- HistIntegral[bin].at<double>(y - 1 + block.y, x - 1 + block.x + 16 - 1)
							//				- HistIntegral[bin].at<double>(y - 1 + block.y + 16 - 1, x - 1 + block.x)
							//				+ HistIntegral[bin].at<double>(y - 1 + block.y + 16 - 1, x - 1 + block.x + 16 - 1);
							//			double shoot = static_cast<double>(H_Temp) / (H_Norm + eps);
							//			H1_Full.at<double>((counter - 1) * 9 + bin, 0) = shoot;
							//		}
							//		counter = counter + 1;
							//	}
							//}
							//HOG_of_BoxesFinal.push_back(H1_Full);


							//double H1_Full_Temp[2025];     //2025 = nwin_x*nwin_y*9
							//double* p2;
							//for (int k = 0; k < H1_Full.rows; k++)
							//{
							//	p2 = H1_Full.ptr<double>(k);//获取行指针
							//	H1_Full_Temp[k] = p2[0];
							//}

							NumOfObjects = NumOfObjects + 1;
						}
					}
				}
			}
			factor = factor*ScaleFactor;
			int b = clock();
			int c = b - a;//算出来的单位是毫秒
			cout << "Time Spent" << c << endl;
		}

		/********************** NMS *************************/
		int numBoxes = BoxesFinal.size();
		vector<CvPoint> BoxesFinal_Points(numBoxes);
		vector<CvPoint> BoxesFinal_OppositePoints(numBoxes);

		for (int monkey = 0; monkey<BoxesFinal.size(); monkey++)
		{
			double a1 = max(BoxesFinal[monkey].y, 1);
			double a2 = max(BoxesFinal[monkey].x, 1);
			double a3 = min(a1 + BoxesFinal[monkey].height - 1, ImageRaw.rows - 1);
			double a4 = min(a2 + BoxesFinal[monkey].width - 1, ImageRaw.cols - 1);

			CvPoint p1 = Point(a2, a1);
			CvPoint p2 = Point(a4, a3);
			BoxesFinal_Points[monkey] = p1;
			BoxesFinal_OppositePoints[monkey] = p2;
		}

		float overlapThreshold = 0.5;
		vector<int> is_suppressed(numBoxes);
		vector<int> Nums(numBoxes);
		nonMaximumSuppression(numBoxes, BoxesFinal_Points, BoxesFinal_OppositePoints, overlapThreshold, is_suppressed, Nums);

		/***************************************************/
		Mat ImageDisplay = ImageRaw.clone();
		cvtColor(ImageDisplay, ImageDisplay, CV_GRAY2BGR);

		int silly = ImageNames[NumImage].find_last_of("/");
		string silly2 = ImageNames[NumImage].substr(silly + 1);
		string DetectionResultFullPath = RootPath + "Result1" + "/" + silly2;
		int silly3 = silly2.find_last_of(".");
		string silly4 = silly2.substr(0, silly3);

		for (int monkey = 0; monkey<BoxesFinal.size(); monkey++)
		{
			double a1 = max(BoxesFinal[monkey].y, 1);
			double a2 = max(BoxesFinal[monkey].x, 1);
			double a3 = min(a1 + BoxesFinal[monkey].height - 1, ImageRaw.rows - 1);
			double a4 = min(a2 + BoxesFinal[monkey].width - 1, ImageRaw.cols - 1);

			if (is_suppressed[monkey] == 0 && Nums[monkey] >= 0)
			{
				rectangle(ImageDisplay, BoxesFinal_Points[monkey], BoxesFinal_OppositePoints[monkey], Scalar(0, 0, 255), 3, 8, 0);
			}
		}

		imwrite(DetectionResultFullPath, ImageDisplay);
	}


	return 0;
}

