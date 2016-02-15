// TP_NOTE_1.cpp : définit le point d'entrée pour l'application console.
//

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/calib3d.hpp>

#include <Plate.hpp>

#include <iostream>
#include <iomanip>

#define DISPARITY 0
#define HARD_CHECK 0  //used for images datas from kitty website

#if HARD_CHECK == 1
#define MOY_MAX 26000
#define MOY_MIN 1000
#else
#define	MOY_MAX 26000
#define	MOY_MIN 3000
#endif

#pragma region BOOLEAN DEFINITIONS
bool showSteps = false; //show all steps of the analyse
bool showResult = false;//show the result in the Detection function (useful for HARD_CHECK "debugging")
bool saveRegions = false;//not used for now
#pragma endregion booleans

#pragma region functions definitions
void findMatchings(cv::Mat&, cv::Mat&, std::vector<cv::Point2f>&, std::vector<cv::Point2f>&);
void showMatchings(cv::Mat, cv::Mat, const  std::vector<cv::Point2f>&, const  std::vector<cv::Point2f>&);
void rectify(cv::Mat&, cv::Mat&, std::vector<cv::Point2f>&, std::vector<cv::Point2f>&, cv::Mat&, cv::Mat&);
cv::Mat computeDisparity(cv::Mat&, cv::Mat&);

std::vector<Plate> Detection(cv::Mat& input_image, const std::string& windowName = "", const cv::Size& windowPos = cv::Size(-1, -1));
std::vector<cv::RotatedRect> Detection2(cv::Mat& input_image, const std::string& windowName = "", const cv::Size& windowPos = cv::Size(-1, -1));
void CharactersDetection(const Plate& plate);
#pragma endregion definition des fonctions

#pragma region variables definitions
std::string window_name = "Hello Kitty !";
int cpt = 0;
std::vector<cv::Mat> results;
#pragma endregion declaration des variables

int main(int argc, char** argv)
{
	cv::namedWindow(window_name, CV_WINDOW_AUTOSIZE);
#if DISPARITY == 1
	std::vector<cv::Mat> images;
	std::stringstream path("");
	for (int i = 0; i < 78; ++i) //parce qu'il y a 77 images dans le dossier, a voir si on fait pas un count des *.png ou si on passe un nombre d'image a analyser
	{
		path << "..\\..\\images\\2011_09_26_drive_0052_sync\\image_02\\data\\"
			<< std::setfill('0') << std::setw(10) << i
			<< ".png";
		cv::Mat image1 = cv::imread(path.str(), CV_LOAD_IMAGE_GRAYSCALE);
		images.push_back(image1);
		path = std::stringstream();
	}

	std::vector<cv::Mat> images2;
	for (int i = 0; i < 78; ++i) //parce qu'il y a 77 images dans le dossier, a voir si on fait pas un count des *.png ou si on passe un nombre d'image a analyser
	{
		path << "..\\..\\images\\2011_09_26_drive_0052_sync\\image_03\\data\\"
			<< std::setfill('0') << std::setw(10) << i
			<< ".png";
		cv::Mat image1 = cv::imread(path.str(), /*CV_LOAD_IMAGE_COLOR*/CV_LOAD_IMAGE_GRAYSCALE);
		images2.push_back(image1);
		path = std::stringstream();
	}

	for (std::vector<cv::Mat>::const_iterator it = images.begin(); it != images.end(); ++it)
	{
		cv::Mat image1 = ( *it );
		std::vector<cv::Point2f> points1;
		std::vector<cv::Point2f> points2;
		findMatchings(image1, images2[cpt], points1, points2);
		findMatchings(images2[cpt], image1, points2, points1);
		cv::Mat disparity = computeDisparity(image1, images2[cpt]);
		cv::imshow(window_name, disparity);
		cv::waitKey();
		cpt++;
	}
#else
	std::vector<cv::Mat> images;
	std::stringstream path("");
#if HARD_CHECK == 1
	int nbImg = 78;//78
	for (int i = 0; i < nbImg; ++i) //parce qu'il y a 77 images dans le dossier, a voir si on fait pas un count des *.png ou si on passe un nombre d'image a analyser
	{
		path << "..\\..\\images\\2011_09_26_drive_0052_sync\\image_02\\data\\"
			<< std::setfill('0') << std::setw(10) << i
			<< ".png";
#else
	int nbImg = 12;
	for (int i = 0; i < nbImg; ++i) //parce qu'il y a 77 images dans le dossier, a voir si on fait pas un count des *.png ou si on passe un nombre d'image a analyser
	{
		path << "..\\..\\images\\test\\"
			<< std::setfill('0') << std::setw(10) << i
			<< ".jpg";
#endif
		cv::Mat image1 = cv::imread(path.str(), CV_LOAD_IMAGE_COLOR);
		images.push_back(image1);
		path = std::stringstream();
	}
	for (std::vector<cv::Mat>::const_iterator it = images.begin(); it != images.end(); ++it)
	{
		cv::Mat image1 = ( *it );
		//cv::Mat image1 = cv::imread("D:\\ESGI\\5A\\opencv\\mastering_opencv\\trunk\\Chapter5_NumberPlateRecognition\\test\\9773BNB.jpg", CV_LOAD_IMAGE_COLOR);

#if HARD_CHECK == 1
		//Si hard check est actif, on va découper l'image en sous images et traiter chacunes de ces sous images
		int nbCaseX = 5;
		int nbCaseY = 3;
		int lenX = image1.size().width / nbCaseX;
		int lenY = image1.size().height / nbCaseY;
		int factRes = 3;
		for (int x = 0; x < nbCaseX; ++x)
		{
			for (int y = 0; y < nbCaseY; ++y)
			{
				cv::Mat cropImg = image1(cv::Rect(( x * image1.size().width ) / nbCaseX, ( y * image1.size().height ) / nbCaseY, lenX, lenY));
				cv::Mat res;
				cv::resize(cropImg, res, cv::Size(cropImg.size().width * factRes, cropImg.size().height * factRes));
				std::string n = std::to_string(x) + "," + std::to_string(y);
				std::vector<cv::RotatedRect> plates = Detection2(res, n, cv::Size(( ( x * image1.size().width ) / nbCaseX ) * factRes, ( ( y * image1.size().height ) / nbCaseY ) * factRes));
				for (std::vector<cv::RotatedRect>::const_iterator pit = plates.begin(); pit != plates.end(); ++pit)
				{
					cv::Point2f rect_points[4]; ( *pit ).points(rect_points);
					for (int j = 0; j < 4; j++)
					{
						cv::Point2f pt = rect_points[j] / factRes;
						cv::Point2f pt2 = rect_points[( j + 1 ) % 4] / factRes;
						pt.x += ( x * image1.size().width ) / nbCaseX;
						pt.y += ( y * image1.size().height ) / nbCaseY;
						pt2.x += ( x * image1.size().width ) / nbCaseX;
						pt2.y += ( y * image1.size().height ) / nbCaseY;

						line(image1, pt, pt2, cv::Scalar(0, 0, 255), 1, 8);
					}
				}
				if (x < nbCaseX - 1 && y < nbCaseY - 1)//ajout du sous découpage découpage "centrale"
				{
					int newX = ( ( ( x + 1 ) * image1.size().width ) / nbCaseX ) - ( lenX / 2 );
					int newY = ( ( ( y + 1 ) * image1.size().height ) / nbCaseY ) - ( lenY / 2 );
					cv::Mat cropImg = image1(cv::Rect(newX, newY, lenX, lenY));
					cv::Mat res2;
					cv::resize(cropImg, res2, cv::Size(cropImg.size().width * factRes, cropImg.size().height * factRes));
					std::string n = std::to_string(x) + "," + std::to_string(y) + "bis";
					std::vector<cv::RotatedRect> plates = Detection2(res2, n, cv::Size(newX * factRes, newY * factRes));
					for (std::vector<cv::RotatedRect>::const_iterator pit = plates.begin(); pit != plates.end(); ++pit)
					{
						cv::Point2f rect_points[4]; ( *pit ).points(rect_points);
						for (int j = 0; j < 4; j++)
						{
							cv::Point2f pt = rect_points[j] / factRes;
							cv::Point2f pt2 = rect_points[( j + 1 ) % 4] / factRes;
							pt.x += ( x * image1.size().width ) / nbCaseX;
							pt.y += ( y * image1.size().height ) / nbCaseY;
							pt2.x += ( x * image1.size().width ) / nbCaseX;
							pt2.y += ( y * image1.size().height ) / nbCaseY;
							line(image1, pt, pt2, cv::Scalar(0, 0, 255), 1, 8);
						}
					}
				}
			}
		}
		cv::imshow(window_name, image1);
		cv::waitKey();

#else
		std::vector<Plate>& plates = Detection(image1);
		cv::Mat result;
		image1.copyTo(result);
		for (std::vector<Plate>::const_iterator pit = plates.begin(); pit != plates.end(); ++pit)
		{
			cv::Point2f rect_points[4]; ( *pit ).GetRect().points(rect_points);
			for (int j = 0; j < 4; j++)
				line(result, rect_points[j], rect_points[( j + 1 ) % 4], cv::Scalar(255, 100, 0), 5, 8);
		}
		cv::imshow(window_name, result);
		cv::waitKey();
		for (std::vector<Plate>::const_iterator pit = plates.begin(); pit != plates.end(); ++pit)
		{
			//cv::imshow("plate", ( *pit ).GetImage());
			//cv::waitKey();
			CharactersDetection(*pit);
		}
#endif
	}
#endif
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
	cv::imshow(window_name, dispImg);
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

//detections plaques
bool Validate(std::vector<cv::Point>& cont)
{
	cv::RotatedRect rect = cv::minAreaRect(cv::Mat(cont));
	bool output = false;
	int width = rect.boundingRect().width;
	int height = rect.boundingRect().height;
	if (( width > height ) && ( width / height > 2 ))
	{
		int moy = height * width;
		//std::cout << "moy : " << moy << std::endl;
		if (( moy <= MOY_MAX ) && ( moy >= MOY_MIN ))
			output = true;
	}
	return output;
}
std::vector<Plate> Detection(cv::Mat& input_image, const std::string& windowName, const cv::Size& windowPos)
{
	cv::Mat gray;
	cv::cvtColor(input_image, gray, cv::COLOR_BGR2GRAY);
	if (showSteps)
	{
		cv::imshow("test", gray);
		cv::waitKey();
	}
	cv::Mat blur;
	cv::GaussianBlur(gray, blur, cv::Size(5, 5), 0);
	if (showSteps)
	{
		cv::imshow("test", blur);
		cv::waitKey();
	}
	cv::Mat sobel;
	cv::Sobel(blur, sobel, CV_8U, 1, 0, 3);
	if (showSteps)
	{
		cv::imshow("test", sobel);
		cv::waitKey();
	}
	cv::Mat tres;
	threshold(sobel, tres, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
	if (showSteps)
	{
		cv::imshow("test", tres);
		cv::waitKey();
	}
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(23, 2));
	cv::Mat closing;
	cv::morphologyEx(tres, closing, cv::MORPH_CLOSE, element);
	if (showSteps)
	{
		cv::imshow("test", closing);
		cv::waitKey();
	}
	std::vector<std::vector<cv::Point>> contours;
	std::vector<Plate> plates;
	cv::findContours(closing, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	std::vector<std::vector<cv::Point> >::iterator itc = contours.begin();
	cv::Mat result;
	input_image.copyTo(result);
	while (itc != contours.end())
	{
		if (Validate(*itc))
		{
			cv::RotatedRect rect = cv::minAreaRect(cv::Mat(*itc));
			cv::Mat pMat = input_image(rect.boundingRect());
			plates.push_back(Plate(pMat, rect));
			++itc;
		}
		else
			itc = contours.erase(itc);
	}
	if (showResult)
	{
		for (std::vector<Plate>::const_iterator rc = plates.begin(); rc != plates.end(); ++rc)
		{
			cv::Point2f rect_points[4]; ( *rc ).GetRect().points(rect_points);
			for (int j = 0; j < 4; j++)
				line(result, rect_points[j], rect_points[( j + 1 ) % 4], cv::Scalar(0, 0, 255), 1, 8);
		}
		if (windowName != "" && windowPos.width != -1 && windowPos.height != -1)
		{
			imshow(windowName, result);
			cv::moveWindow(windowName, windowPos.width / 2, windowPos.height / 2);
		}
		else
		{
			imshow(window_name, result);
			cv::waitKey();
		}
	}
	return plates;
}
std::vector<cv::RotatedRect> Detection2(cv::Mat& input_image, const std::string& windowName, const cv::Size& windowPos)
{
	cv::Mat gray;
	cv::cvtColor(input_image, gray, cv::COLOR_BGR2GRAY);
	if (showSteps)
	{
		cv::imshow("test", gray);
		cv::waitKey();
	}
	cv::Mat blur;
	cv::GaussianBlur(gray, blur, cv::Size(5, 5), 0);
	if (showSteps)
	{
		cv::imshow("test", blur);
		cv::waitKey();
	}
	cv::Mat sobel;
	cv::Sobel(blur, sobel, CV_8U, 1, 0, 3);
	if (showSteps)
	{
		cv::imshow("test", sobel);
		cv::waitKey();
	}
	cv::Mat tres;
	threshold(sobel, tres, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
	if (showSteps)
	{
		cv::imshow("test", tres);
		cv::waitKey();
	}
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(23, 2));
	cv::Mat closing;
	cv::morphologyEx(tres, closing, cv::MORPH_CLOSE, element);
	if (showSteps)
	{
		cv::imshow("test", closing);
		cv::waitKey();
	}
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::RotatedRect> plates;
	cv::findContours(closing, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	std::vector<std::vector<cv::Point> >::iterator itc = contours.begin();
	cv::Mat result;
	input_image.copyTo(result);
	while (itc != contours.end())
	{
		if (Validate(*itc))
		{
			cv::RotatedRect rect = cv::minAreaRect(cv::Mat(*itc));
			plates.push_back(rect);
			++itc;
		}
		else
			itc = contours.erase(itc);
	}
	if (showResult)
	{
		for (std::vector<cv::RotatedRect>::const_iterator rc = plates.begin(); rc != plates.end(); ++rc)
		{
			cv::Point2f rect_points[4]; ( *rc ).points(rect_points);
			for (int j = 0; j < 4; j++)
				line(result, rect_points[j], rect_points[( j + 1 ) % 4], cv::Scalar(0, 0, 255), 1, 8);
		}
		if (windowName != "" && windowPos.width != -1 && windowPos.height != -1)
		{
			imshow(windowName, result);
			cv::moveWindow(windowName, windowPos.width / 2, windowPos.height / 2);
		}
		else
		{
			imshow(window_name, result);
			cv::waitKey();
		}
	}
	return plates;
}

void CharactersDetection(const Plate& plate)
{
	cv::Mat input = plate.GetImage();
	cv::Mat gray;
	cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
	//vector<CharSegment> output;
	//Threshold input image
	cv::Mat img_threshold;
	threshold(gray, img_threshold, 60, 255, CV_THRESH_BINARY_INV);

	imshow("Threshold plate", img_threshold);
	cv::waitKey();

	cv::Mat img_contours;
	img_threshold.copyTo(img_contours);
	//Find contours of possibles characters
	std::vector< std::vector< cv::Point> > contours;
	findContours(img_contours,
				 contours, // a vector of contours
				 CV_RETR_EXTERNAL, // retrieve the external contours
				 CV_CHAIN_APPROX_NONE); // all pixels of each contours

										// Draw blue contours on a white image
	cv::Mat result;
	img_threshold.copyTo(result);
	cvtColor(result, result, CV_GRAY2RGB);
	cv::drawContours(result, contours,
					 -1, // draw all contours
					 cv::Scalar(255, 0, 0), // in blue
					 1); // with a thickness of 1

						 //Start to iterate to each contour founded
	std::vector<std::vector<cv::Point> >::iterator itc = contours.begin();

	//Remove patch that are no inside limits of aspect ratio and area.    
	while (itc != contours.end())
	{

		//Create bounding rect of object
		cv::Rect mr = cv::boundingRect(cv::Mat(*itc));
		rectangle(result, mr, cv::Scalar(0, 255, 0));
		//Crop image
		cv::Mat auxRoi(img_threshold, mr);
		/*if (verifySizes(auxRoi))
		{
			auxRoi = preprocessChar(auxRoi);
			output.push_back(CharSegment(auxRoi, mr));
			rectangle(result, mr, cv::Scalar(0, 125, 255));
		}*/
		++itc;
	}
	//std::cout << "Num chars: " << output.size() << "\n";



	imshow("SEgmented Chars", result);
	cv::waitKey();
	//return output;
}