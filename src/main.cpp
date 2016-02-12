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
#define HARD_CHECK 1


#pragma region functions definitions
void findMatchings(cv::Mat&, cv::Mat&, std::vector<cv::Point2f>&, std::vector<cv::Point2f>&);
void showMatchings(cv::Mat, cv::Mat, const  std::vector<cv::Point2f>&, const  std::vector<cv::Point2f>&);
void rectify(cv::Mat&, cv::Mat&, std::vector<cv::Point2f>&, std::vector<cv::Point2f>&, cv::Mat&, cv::Mat&);
cv::Mat computeDisparity(cv::Mat&, cv::Mat&);
std::vector<Plate> segment(cv::Mat input);

void Detection(cv::Mat input_image);
#pragma endregion definition des fonctions

#pragma region variables definitions
std::string window_name = "Hello Kitty !";
int cpt = 0;
std::vector<cv::Mat> results;

bool showSteps = true;
bool showContours = true;
bool saveRegions = true;
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
		//showMatchings(( *it ), images2[cpt], points1, points2);
		cv::Mat disparity = computeDisparity(image1, images2[cpt]);
		cv::imshow(window_name, disparity);
		cv::waitKey();
		cpt++;
	}
#else
	std::vector<cv::Mat> images;
	std::stringstream path("");
	for (int i = 0; i < 78; ++i) //parce qu'il y a 77 images dans le dossier, a voir si on fait pas un count des *.png ou si on passe un nombre d'image a analyser
	{
		path << "..\\..\\images\\2011_09_26_drive_0052_sync\\image_02\\data\\"
			<< std::setfill('0') << std::setw(10) << i
			<< ".png";
		cv::Mat image1 = cv::imread(path.str(), CV_LOAD_IMAGE_COLOR);
		images.push_back(image1);
		path = std::stringstream();
	}
	for (std::vector<cv::Mat>::const_iterator it = images.begin(); it != images.end(); ++it)
	{
		cv::Mat image1 = (*it);
		//cv::Mat image1 = cv::imread("C:\\Users\\jonathanL\\Documents\\ocv\\trunk\\3028BYS.JPG", CV_LOAD_IMAGE_COLOR);
#if HARD_CHECK == 1
		int nbCaseX = 6;
		int nbCaseY = 4;
		int lenX = image1.size().width / nbCaseX;
		int lenY = image1.size().height / nbCaseY;
		int factRes = 2;
		for (int x = 0; x < nbCaseX; ++x)
		{
			for (int y = 0; y < nbCaseY; ++y)
			{
				cv::Mat cropImg = image1(cv::Rect((x * image1.size().width) / nbCaseX, (y * image1.size().height) / nbCaseY, lenX , lenY));
				cv::Mat res;
				cv::resize(cropImg, res, cv::Size(cropImg.size().width * factRes, cropImg.size().height * factRes));
				std::string n = std::to_string(x) + "," + std::to_string(y);
				cv::imshow(n, res);
				cv::moveWindow(n, ((x * image1.size().width) / nbCaseX) * factRes, ((y * image1.size().height) / nbCaseY) * factRes);
				//Detection(res);
			}
		}
		cv::waitKey();
		
#else
		Detection(image1);

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
		if (static_cast<int>(status[i]) != 0)
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
	disp.convertTo(disp2, CV_8UC1, 255 / (maxVal - minVal));
	return disp2;

}





//detections plaques

bool verifySizes(cv::RotatedRect mr)
{
	float error = 0.4f;
	//Spain car plate size: 52x11 aspect 4,7272
	float aspect = 4.7272f;
	//Set a min and max area. All other patchs are discarded
	int min = 5 * static_cast<int>(aspect) * 15; // minimum area
	int max = 50 * static_cast<int>(aspect) * 125; // maximum area
	//Get only patchs that match to a respect ratio.
	float rmin = aspect - aspect*error;
	float rmax = aspect + aspect*error;

	int area = static_cast<int>(mr.size.height) * static_cast<int>(mr.size.width);
	float r = (float) mr.size.width / (float) mr.size.height;
	if (r < 1)
		r = (float) mr.size.height / (float) mr.size.width;

	if ((area < min || area > max) || (r < rmin || r > rmax))
	{
		return false;
	}
	else
	{
		return true;
	}

}

cv::Mat histeq(cv::Mat in)
{
	cv::Mat out(in.size(), in.type());
	if (in.channels() == 3)
	{
		cv::Mat hsv;
		std::vector<cv::Mat> hsvSplit;
		cvtColor(in, hsv, CV_BGR2HSV);
		cv::split(hsv, hsvSplit);
		cv::equalizeHist(hsvSplit[2], hsvSplit[2]);
		merge(hsvSplit, hsv);
		cvtColor(hsv, out, CV_HSV2BGR);
	}
	else if (in.channels() == 1)
	{
		equalizeHist(in, out);
	}

	return out;

}

std::vector<Plate> segment(cv::Mat input)
{
	std::vector<Plate> output;

	//convert image to gray
	cv::Mat img_gray;
	cvtColor(input, img_gray, CV_BGR2GRAY);
	cv::blur(img_gray, img_gray, cv::Size(5, 5));

	//Find vertical lines. Car plates have high density of vertical lines
	cv::Mat img_sobel;
	cv::Sobel(img_gray, img_sobel, CV_8U, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
	if (showSteps)
	{
		imshow(window_name, img_sobel);
		cv::waitKey();
	}


	//threshold image
	cv::Mat img_threshold;
	threshold(img_sobel, img_threshold, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
	if (showSteps)
	{
		cv::imshow(window_name, img_threshold);
		cv::waitKey();
	}

	//Morphplogic operation close
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 3)); //17,3
	cv::morphologyEx(img_threshold, img_threshold, CV_MOP_CLOSE, element);
	if (showSteps)
	{
		imshow(window_name, img_threshold);
		cv::waitKey();
	}

	//Find contours of possibles plates
	std::vector< std::vector< cv::Point> > contours;
	cv::findContours(img_threshold,
					 contours, // a vector of contours
					 CV_RETR_EXTERNAL, // retrieve the external contours
					 CV_CHAIN_APPROX_NONE); // all pixels of each contours

	//Start to iterate to each contour founded
	std::vector<std::vector<cv::Point> >::iterator itc = contours.begin();
	std::vector<cv::RotatedRect> rects;

	//Remove patch that are no inside limits of aspect ratio and area.    
	while (itc != contours.end())
	{
		//Create bounding rect of object
		cv::RotatedRect mr = cv::minAreaRect(cv::Mat(*itc));
		if (!verifySizes(mr))
		{
			itc = contours.erase(itc);
		}
		else
		{
			++itc;
			rects.push_back(mr);
		}
	}

	// Draw blue contours on a white image
	cv::Mat result;
	input.copyTo(result);
	cv::drawContours(result, contours,
					 -1, // draw all contours
					 cv::Scalar(255, 0, 0), // in blue
					 1); // with a thickness of 1

	for (int i = 0; i < rects.size(); i++)
	{

		//For better rect cropping for each posible box
		//Make floodfill algorithm because the plate has white background
		//And then we can retrieve more clearly the contour box
		circle(result, rects[i].center, 3, cv::Scalar(0, 255, 0), -1);
		//get the min size between width and height
		float minSize = (rects[i].size.width < rects[i].size.height) ? rects[i].size.width : rects[i].size.height;
		minSize = minSize - minSize * 0.5f;
		//initialize rand and get 5 points around center for floodfill algorithm
		srand(time(NULL));
		//Initialize floodfill parameters and variables
		cv::Mat mask;
		mask.create(input.rows + 2, input.cols + 2, CV_8UC1);
		mask = cv::Scalar::all(0);
		int loDiff = 30;
		int upDiff = 30;
		int connectivity = 4;
		int newMaskVal = 255;
		int NumSeeds = 10;
		cv::Rect ccomp;
		int flags = connectivity + (newMaskVal << 8) + CV_FLOODFILL_FIXED_RANGE + CV_FLOODFILL_MASK_ONLY;
		for (int j = 0; j < NumSeeds; j++)
		{
			cv::Point seed;
			seed.x = static_cast<int>(rects[i].center.x) + rand() % (int) static_cast<int>(minSize) -(static_cast<int>(minSize) / 2);
			seed.y = static_cast<int>(rects[i].center.y) + rand() % (int) static_cast<int>(minSize) -(static_cast<int>(minSize) / 2);
			circle(result, seed, 1, cv::Scalar(0, 255, 255), -1);
			if (seed.y < 0) seed.y = 0;
			int area = floodFill(input, mask, seed, cv::Scalar(255, 0, 0), &ccomp, cv::Scalar(loDiff, loDiff, loDiff), cv::Scalar(upDiff, upDiff, upDiff), flags);
		}
		if (showSteps)
		{
			imshow(window_name, mask);
			cv::waitKey();
		}

		//Check new floodfill mask match for a correct patch.
		//Get all points detected for get Minimal rotated Rect
		std::vector<cv::Point> pointsInterest;
		cv::Mat_<uchar>::iterator itMask = mask.begin<uchar>();
		cv::Mat_<uchar>::iterator end = mask.end<uchar>();
		for (; itMask != end; ++itMask)
			if (*itMask == 255)
				pointsInterest.push_back(itMask.pos());

		cv::RotatedRect minRect = minAreaRect(pointsInterest);

		if (verifySizes(minRect))
		{
			// rotated rectangle drawing 
			cv::Point2f rect_points[4]; minRect.points(rect_points);
			for (int j = 0; j < 4; j++)
				line(result, rect_points[j], rect_points[(j + 1) % 4], cv::Scalar(0, 0, 255), 1, 8);

			//Get rotation matrix
			float r = (float) minRect.size.width / (float) minRect.size.height;
			float angle = minRect.angle;
			if (r < 1)
				angle = 90 + angle;
			cv::Mat rotmat = getRotationMatrix2D(minRect.center, angle, 1);

			//Create and rotate image
			cv::Mat img_rotated;
			warpAffine(input, img_rotated, rotmat, input.size(), CV_INTER_CUBIC);

			//Crop image
			cv::Size rect_size = minRect.size;
			if (r < 1)
				cv::swap(rect_size.width, rect_size.height);
			cv::Mat img_crop;
			getRectSubPix(img_rotated, rect_size, minRect.center, img_crop);

			cv::Mat resultResized;
			resultResized.create(33, 144, CV_8UC3);
			resize(img_crop, resultResized, resultResized.size(), 0, 0, cv::INTER_CUBIC);
			//Equalize croped image
			cv::Mat grayResult;
			cv::cvtColor(resultResized, grayResult, CV_BGR2GRAY);
			cv::blur(grayResult, grayResult, cv::Size(3, 3));
			grayResult = histeq(grayResult);
			if (saveRegions)
			{
				std::stringstream ss(std::stringstream::in | std::stringstream::out);
				ss << "tmp/" << "saved" /*filename */ << "_" << i << ".jpg";
				imwrite(ss.str(), grayResult);
			}
			output.push_back(Plate(grayResult, minRect.boundingRect()));
		}
	}
	if (showContours)
		imshow(window_name, result);
	return output;
}


void Detection(cv::Mat input_image)
{
	cv::Mat img_gray;
	cvtColor(input_image, img_gray, CV_BGR2GRAY);

	cv::Mat otsu;
	cv::threshold(img_gray, otsu, 0, 255, cv::THRESH_OTSU);
	cv::imshow(window_name, otsu);
	cv::waitKey();

	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(20, 5)); //longueur/hauteur

	cv::Mat top_hat;
	cv::morphologyEx(otsu, top_hat, CV_MOP_TOPHAT, element);
	cv::imshow(window_name, top_hat);
	cv::waitKey();

	/*for (int i = 1; i < 31; i = i + 2)
	{
	cv::Mat dst;
	cv::blur(top_hat, dst, cv::Size(i, i), cv::Point(-1, -1));
	std::string name = "blur " + std::to_string(i);
	cv::imshow(window_name, dst);
	cv::waitKey();
	}
	for (int i = 1; i < 31; i = i + 2)
	{
	cv::Mat gaus;
	cv::GaussianBlur(top_hat, gaus, cv::Size(i, i), 0, 0);
	std::string name = "gauss " + std::to_string(i);
	cv::imshow(window_name, gaus);
	cv::waitKey();
	}*/

	/*for (int i = 1; i < 31; i = i + 2)
	{
	cv::Mat median;
	cv::medianBlur(top_hat, median, i);
	std::string name = "median " + std::to_string(i);
	cv::imshow(window_name, median);
	cv::waitKey();
	}*/
	cv::Mat median;
	cv::medianBlur(top_hat, median, 31);
	cv::imshow(window_name, median);
	cv::waitKey();

	/*for (int i = 1; i < 31; i = i + 2)
	{
	cv::Mat bilateral;
	bilateralFilter(top_hat, bilateral, i, i * 2, i / 2);
	std::string name = "bilateral " + std::to_string(i);
	cv::imshow(window_name, bilateral);
	cv::waitKey();
	}*/

	cv::Mat dilation;
	cv::morphologyEx(top_hat, dilation, CV_MOP_DILATE, element); //Fonctionne peut etre mieux avec le top_hat !
	cv::imshow(window_name, dilation);
	cv::waitKey();

	cv::Mat dilation2;
	cv::morphologyEx(median, dilation2, CV_MOP_DILATE, element);
	cv::imshow(window_name, dilation2);
	cv::waitKey();

	std::vector< std::vector< cv::Point> > contours;
	cv::findContours(dilation, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE); //utilisation de la dilatation top_hat 

	std::vector<cv::RotatedRect> rects;
	std::vector<std::vector<cv::Point> >::iterator itc = contours.begin();
	while (itc != contours.end())
	{
		cv::RotatedRect mr = cv::minAreaRect(cv::Mat(*itc));
		std::cout << mr.size << std::endl;

		if (mr.size.width * mr.size.height < 2500)
			itc = contours.erase(itc);
		else
			++itc;

		/*if (!verifySizes(mr))
		{
		itc = contours.erase(itc);
		}
		else
		{
		++itc;
		rects.push_back(mr);
		}*/

		/*float area = fabs(cv::contourArea(*itc));
		float bbArea = mr.size.width * mr.size.height;
		float ratio = area / bbArea;

		if ((ratio < 0.45) || (bbArea < 400))
		{
		itc = contours.erase(itc);
		}
		else
		{
		++itc;
		rects.push_back(mr);
		}*/
	}

	cv::Mat result;
	input_image.copyTo(result);
	cv::drawContours(result, contours,
					 -1, // draw all contours
					 cv::Scalar(255, 0, 0), // in blue
					 1); // with a thickness of 1

	imshow(window_name, result);
	cv::waitKey();

}