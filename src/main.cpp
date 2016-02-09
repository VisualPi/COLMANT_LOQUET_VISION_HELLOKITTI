// TP_NOTE_1.cpp : définit le point d'entrée pour l'application console.
//

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/calib3d.hpp>

#include <iostream>
#include <iomanip>

void findMatchings(cv::Mat&, cv::Mat&, std::vector<cv::Point2f>&, std::vector<cv::Point2f>&);
void showMatchings(cv::Mat, cv::Mat, const  std::vector<cv::Point2f>&, const  std::vector<cv::Point2f>&);
void rectify(cv::Mat&, cv::Mat&, std::vector<cv::Point2f>&, std::vector<cv::Point2f>&, cv::Mat&, cv::Mat&);
cv::Mat computeDisparity(cv::Mat&, cv::Mat&);

int main(int argc, char** argv)
{
	/*if (argc < 3)
	{
		std::cerr << "Required arguments: left.jpg right.jpg" << std::endl;
		return 1;
	}*/
	std::vector<cv::Mat> images;
	std::stringstream path("");
	//path << "..\\..\\images\\2011_09_26_drive_0052_sync\\image_03\\data\\";
	for (int i = 0; i < 78; ++i)
	{
		path << "..\\..\\images\\2011_09_26_drive_0052_sync\\image_03\\data\\"
			<< std::setfill('0') << std::setw(10) << i
			<< ".png";
		cv::Mat image1 = cv::imread(path.str(), CV_LOAD_IMAGE_GRAYSCALE);
		images.push_back(image1);
		path = std::stringstream();
	}

	for (auto it = images.begin(); it != images.end()-1; ++it)
	{
		cv::imshow("img", *it);

		cv::Mat image1 = ( *it );
		cv::Mat image2 = ( *it ) + 1;
		std::vector<cv::Point2f> points1;
		std::vector<cv::Point2f> points2;
		findMatchings(image1, image2, points1, points2);
		findMatchings(image2, image1, points2, points1);
		showMatchings(image1, image2, points1, points2);
		cv::Mat rectified1(image1.size(), image1.type());
		cv::Mat rectified2(image2.size(), image2.type());
		rectify(image1, image2, points1, points2, rectified1, rectified2);
		cv::imshow("rectified L", rectified1);
		cv::imshow("rectified R", rectified2);
		cv::Mat disparity = computeDisparity(rectified1, rectified2);
		cv::imshow("disparity", disparity);
		cv::waitKey();
	}

	//cv::Mat image1 = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	//cv::Mat image2 = cv::imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
	//std::vector<cv::Point2f> points1;
	//std::vector<cv::Point2f> points2;
	//findMatchings(image1, image2, points1, points2);
	//findMatchings(image2, image1, points2, points1);
	//showMatchings(image1, image2, points1, points2);
	//cv::Mat rectified1(image1.size(), image1.type());
	//cv::Mat rectified2(image2.size(), image2.type());
	//rectify(image1, image2, points1, points2, rectified1, rectified2);
	//cv::imshow("rectified L", rectified1);
	//cv::imshow("rectified R", rectified2);
	////cv::waitKey();
	//cv::Mat disparity = computeDisparity(rectified1, rectified2);
	//cv::imshow("disparity", disparity);
	//cv::waitKey();
	return 0;
}

void findMatchings(cv::Mat& img1, cv::Mat& img2, std::vector<cv::Point2f>& pts1, std::vector<cv::Point2f>& pts2)
{
	int maxCorners = 500;
	double qualityLevel = 0.01;
	double minDistance = 10;
	std::vector<cv::Point2f> tmpA;
	std::vector<cv::Point2f> tmpB;
	std::vector<uchar> status;
	std::vector<float> errors;

	cv::goodFeaturesToTrack(img1, tmpA, maxCorners, qualityLevel, minDistance);
	cv::calcOpticalFlowPyrLK(img1, img2, tmpA, tmpB, status, errors);
	for (int i = 0; i < maxCorners; ++i)
	{
		if (static_cast<int>( status[i] ) != 0)
		{
			pts1.push_back(tmpA[i]);
			pts2.push_back(tmpB[i]);
		}
	}
}
void showMatchings(cv::Mat img1, cv::Mat img2, const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2)
{
	//cv::Mat tmp1, tmp2;
	//cv::cvtColor(img1, tmp1, CV_GRAY2BGR);
	//cv::cvtColor(img2, tmp2, CV_GRAY2BGR);
	//for (int i = 0; i < pts1.size(); ++i)
	//{
	//	cv::line(tmp1, pts1[i], pts2[i], cv::Scalar(150, 122, 46));
	//	cv::line(tmp2, pts1[i], pts2[i], cv::Scalar(150, 122, 46));
	//}
	//cv::imshow("Display Matching L", tmp1);
	//cv::imshow("Display Matching R", tmp2);

	cv::Size s1 = img1.size();
	cv::Size s2 = img2.size();
	cv::Mat im3(s1.height, s1.width + s2.width, img1.type());
	cv::Mat left(im3, cv::Rect(0, 0, s1.width, s1.height));
	img1.copyTo(left);
	cv::Mat right(im3, cv::Rect(s1.width, 0, s2.width, s2.height));
	img2.copyTo(right);
	cv::Mat dispImg;
	im3.copyTo(dispImg);
	cv::cvtColor(im3, dispImg, CV_GRAY2BGR);

	for (int i = 0; i < pts1.size(); ++i)
		cv::line(dispImg, pts1[i], cv::Point2f(pts2[i].x + img1.size().width, pts2[i].y), cv::Scalar(132, 98, 6));
	cv::imshow("Display Matching LR", dispImg);
}

void rectify(cv::Mat& img1, cv::Mat& img2, std::vector<cv::Point2f>& pts1, std::vector<cv::Point2f>& pts2, cv::Mat& rectified1, cv::Mat& rectified2)
{
	cv::Mat F = cv::findFundamentalMat(pts1, pts2);
	cv::Mat Hl;
	cv::Mat Hr;
	cv::stereoRectifyUncalibrated(pts1, pts2, F, img1.size(), Hl, Hr);
	cv::warpPerspective(img1, rectified1, Hl, img1.size());
	cv::warpPerspective(img2, rectified2, Hr, img2.size());
}
cv::Mat computeDisparity(cv::Mat& rectified1, cv::Mat& rectified2)
{
	cv::Mat disp(rectified1.size(), CV_16SC1);
	cv::Mat disp2(rectified1.size(), CV_8UC1);
	cv::Ptr<cv::StereoBM> sbm = cv::StereoBM::create();
	sbm->compute(rectified1, rectified2, disp);
	double minVal, maxVal;
	cv::minMaxLoc(disp, &minVal, &maxVal);
	disp.convertTo(disp2, CV_8UC1, 255 / ( maxVal - minVal ));
	return disp2;

}
