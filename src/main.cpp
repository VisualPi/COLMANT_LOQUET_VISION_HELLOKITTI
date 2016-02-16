// TP_NOTE_1.cpp : définit le point d'entrée pour l'application console.
//

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/calib3d.hpp>

#include "opencv2/ml.hpp"

#include <Plate.hpp>
//#include <OCR.h>

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

std::vector<char> CharsAlphabet = { 'A','B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' };

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
void CharactersDetection(Plate& plate);

cv::Mat features(cv::Mat in, int sizeData);
cv::Mat preprocessChar(cv::Mat in);
int classify(cv::Mat f);
#pragma endregion definition des fonctions

#pragma region variables definitions
std::string window_name = "Hello Kitty !";
int cpt = 0;
#pragma endregion declaration des variables

int main(int argc, char** argv)
{
	cv::namedWindow(window_name, CV_WINDOW_AUTOSIZE);
	std::vector<cv::Mat> images;
	std::vector<cv::Mat> alphabet;
#if DISPARITY == 1
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
	for (int i = 0; i < CharsAlphabet.size(); ++i)
	{
		path << "..\\..\\images\\letters\\"
			<< CharsAlphabet[i] << ".png";
		cv::Mat number = cv::imread(path.str(), CV_LOAD_IMAGE_COLOR);

		cv::Mat gray;
		cv::cvtColor(number, gray, cv::COLOR_BGR2GRAY);
		cv::Mat img_threshold;
		threshold(gray, img_threshold, 60, 255, CV_THRESH_BINARY_INV);
		alphabet.push_back(img_threshold);
		path = std::stringstream();
	}

	for (std::vector<cv::Mat>::const_iterator it = images.begin(); it != images.end(); ++it)
	{
		cv::Mat image1 = ( *it );
		//cv::Mat image1 = cv::imread("D:\\ESGI\\5A\\opencv\\mastering_opencv\\trunk\\Chapter5_NumberPlateRecognition\\test\\9773BNB.jpg", CV_LOAD_IMAGE_COLOR);

#if HARD_CHECK == 1
		//Si hard check est actif, on va découper l'image en sous images et traiter chacunes de ces sous images
		int nbCaseX = 2;
		int nbCaseY = 2;
		int lenX = image1.size().width / nbCaseX;
		int lenY = image1.size().height / nbCaseY;
		int factRes = 1;
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
		for (std::vector<Plate>::iterator pit = plates.begin(); pit != plates.end(); ++pit)
		{
			//cv::imshow("plate", ( *pit ).GetImage());
			//cv::waitKey();
			CharactersDetection(*pit);
			std::cout << "number of chars detected = " << ( *pit ).GetChars().size() << std::endl;
			
			for (int i = 0; i < ( *pit ).GetChars().size(); i++)
			{
				//Preprocess each char for all images have same sizes
				//cv::imshow("plate", ( *pit ).GetChars()[i].m_char>3);


				cv::Mat ch = preprocessChar(( *pit ).GetChars()[i].m_char);
				cv::resize(ch, ch, cv::Size(ch.size().width * 10, ch.size().height * 10));
				//cv::imshow("char", ch);
				cv::Mat canny1;
				Canny(ch>10, canny1, 100, 100 * 2, 3);
				std::vector<std::vector<cv::Point> > contours1;
				std::vector<cv::Vec4i> hierarchy;
				//findContours(canny1,
				//			 contours1, // a vector of contours
				//			 CV_RETR_EXTERNAL, // retrieve the external contours
				//			 CV_CHAIN_APPROX_SIMPLE); // all pixels of each contours
				findContours(canny1, contours1, hierarchy,
							 CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
				int indexAlph = -1;
				double min = std::numeric_limits<double>::max();
				for (int j = 0; j < alphabet.size(); ++j)
				{
 					cv::Mat alph;
					cv::resize(alphabet[j], alph, cv::Size(ch.size().width, ch.size().height));
					
					
					//cv::imshow("number", alph);
					//cv::waitKey();

					cv::Mat canny2;
					Canny(alph, canny2, 50, 50 * 2, 3);
					std::vector<std::vector<cv::Point> > contours2;
					std::vector<cv::Vec4i> hierarchy2;
					//findContours(canny2,
					//			 contours2, // a vector of contours
					//			 CV_RETR_EXTERNAL, // retrieve the external contours
					//			 CV_CHAIN_APPROX_SIMPLE); // all pixels of each contours
					findContours(canny2, contours2, hierarchy2,
								 CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);


					cv::waitKey();
					double d,d2,d3;
					int idx = 0;
					int ind_min;
					double ret;
					
					//Matching example contours[0] with contours of the photo contours2[idx].
					//Comparing output of matchShapes function, the lower is better. 
					for (; idx >= 0; idx = hierarchy2[idx][0])
					{
						double ret = matchShapes(contours1[0], contours2[idx], CV_CONTOURS_MATCH_I1, 0.0);
						std::cout << "for caractere " << CharsAlphabet[j] << " with hierarchy -> d = " << ret << std::endl;
						//std::cout << "for k = " << k << " and l = " << l << " -> d2 = " << d2 << std::endl;
						//std::cout << "for k = " << k << " and l = " << l << " -> d3 = " << d3 << std::endl;
						if (ret < min && ret > 0.0)
						{
							min = ret;
							ind_min = idx;
							indexAlph = j;
						}
					}


					try
					{
						//d = cv::matchShapes(keep1, keep2, 1, 0.0);
						//d = cv::matchShapes(ch, alph, 1, 0.0);
						for (int k = 0; k < contours1.size(); k++)
						{
							for (int l = 0; l < contours2.size(); l++)
							{
								d = cv::matchShapes(contours1[k], contours2[l], CV_CONTOURS_MATCH_I1, 0.0);
								d2 = cv::matchShapes(contours1[k], contours2[l], CV_CONTOURS_MATCH_I2, 0.0);
								d3 = cv::matchShapes(contours1[k], contours2[l], CV_CONTOURS_MATCH_I3, 0.0);
								//std::cout << "for k = " << k << " and l = " << l << " -> d = " << d << std::endl;
								//std::cout << "for k = " << k << " and l = " << l << " -> d2 = " << d2 << std::endl;
								//std::cout << "for k = " << k << " and l = " << l << " -> d3 = " << d3 << std::endl;
								//if(d < 0.015 || d2 < 0.015 || d3 < 0.015)
									//std::cout << "it seems to match with character : " << CharsAlphabet[j] << " !" << std::endl;
								//else
									//std::cout << "it doesn't match !" << std::endl;
							}
						}
						/*d = cv::matchShapes(canny1, canny2, CV_CONTOURS_MATCH_I1, 0.0);
						d2 = cv::matchShapes(canny1, canny2, CV_CONTOURS_MATCH_I2, 0.0);
						d3 = cv::matchShapes(canny1, canny2, CV_CONTOURS_MATCH_I3, 0.0);
						if (d < 0.015 || d2 < 0.015 || d3 < 0.015)
							std::cout << d << ", " << d2 << ", " << d3 << " it seems to match with character : " << CharsAlphabet[j] << " !" << std::endl;*/
						
						std::cout << "-------------------------------------------------" << std::endl;
						/*if (d < 0.15)
							std::cout << d << "is less than 0.15 so it seems to match !" << std::endl;
						else
							std::cout << d << "is greater than 0.15 so it doesn't match !" << std::endl;*/
					}
					catch (cv::Exception e)
					{
						std::cout << e.what() << std::endl;
					}
					cv::imshow("contours", canny1);
					cv::imshow("contours2", canny2);
				}
				if(indexAlph != -1)
					std::cout << std::endl << "the best match is : " << CharsAlphabet[indexAlph] << std::endl;
 				cv::waitKey();
				/*if (saveSegments)
				{
					stringstream ss(stringstream::in | stringstream::out);
					ss << "tmpChars/" << filename << "_" << i << ".jpg";
					imwrite(ss.str(), ch);
				}*/
				//For each segment Extract Features
				//cv::Mat f = features(ch, 15);
				//For each segment feature Classify
				//int character = classify(f);
				//input->chars.push_back(strCharacters[character]);
				//input->charsPos.push_back(segments[i].pos);
			}
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
//fonction de validationd des plaques (rect) potentielles
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
//for test images !
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
            cv::Rect r = rect.boundingRect();
            if (r.x + r.width >= input_image.size().width) {
                r.width -= (input_image.size().width - (r.x + r.width));
                r.height -= (input_image.size().height - (r.y + r.height));
            }
            if (r.x < 0)
                r.x = 0;
            if (r.y < 0)
                r.y = 0;

            rect.boundingRect() = r;
			cv::Mat pMat = input_image(r);
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
//For kitti images !
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
			cv::Rect r = rect.boundingRect();
            if (r.x + r.width >= input_image.size().width) 
            {
                r.width -= (input_image.size().width - (r.x + r.width));
                r.height -= (input_image.size().height - (r.y + r.height));
            }
            if (r.x < 0)
                r.x = 0;
            if (r.y < 0)
                r.y = 0;
            
            rect.boundingRect() = r;
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


cv::Mat preprocessChar(cv::Mat in)
{
	//Remap image
	int h = in.rows;
	int w = in.cols;
	cv::Mat transformMat = cv::Mat::eye(2, 3, CV_32F);
	int m = std::max(w, h);
	transformMat.at<float>(0, 2) = m / 2 - w / 2;
	transformMat.at<float>(1, 2) = m / 2 - h / 2;

	cv::Mat warpImage(m, m, in.type());
	warpAffine(in, warpImage, transformMat, warpImage.size(), CV_INTER_LINEAR, IPL_BORDER_CONSTANT, cv::Scalar(0));

	cv::Mat out;
	resize(warpImage, out, cv::Size(20, 20));

	//cv::imshow("char", warpImage);
	//cv::waitKey();
	return out;
}
bool ValidateChar(cv::Mat in)
{
	float aspect = 45.0f / 77.0f;
	float charAspect = (float) in.cols / (float) in.rows;
	float error = 0.35;
	float minHeight = 15;
	float maxHeight = 28;
	//We have a different aspect ratio for number 1, and it can be ~0.2
	float minAspect = 0.2;
	float maxAspect = aspect + aspect*error;
	//area of pixels
	float area = cv::countNonZero(in);
	//bb area
	float bbArea = in.cols*in.rows;
	//% of pixel in area
	float percPixels = area / bbArea;

	//std::cout << "Aspect: " << aspect << " [" << minAspect << "," << maxAspect << "] " << "Area " << percPixels << " Char aspect " << charAspect << " Height char " << in.rows << "\n";
	if (percPixels < 0.8 && charAspect > minAspect && charAspect < maxAspect && in.rows >= minHeight && in.rows < maxHeight)
		return true;
	else
		return false;

}
void CharactersDetection(Plate& plate)
{
	cv::Mat input = plate.GetImage();
	cv::Mat gray;
	cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
	cv::imshow("test", gray);
	cv::waitKey();
	//vector<CharSegment> output;
	//Threshold input image
	cv::Mat img_threshold;
	threshold(gray, img_threshold, 60, 255, CV_THRESH_BINARY_INV);

	if (showSteps)
	{
		imshow("Threshold plate", img_threshold);
		cv::waitKey();
	}
	

	cv::Mat img_contours;
	img_threshold.copyTo(img_contours);
	//Find contours of possibles characters
	std::vector< std::vector< cv::Point> > contours;
	findContours(img_contours,
				 contours, // a vector of contours
				 CV_RETR_EXTERNAL, // retrieve the external contours
				 CV_CHAIN_APPROX_NONE); // all pixels of each contours

	if (showSteps)
	{
		imshow("contours", img_contours);
		cv::waitKey();
	}
	
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

	if (showSteps)
	{
		cv::imshow("result", result);
		cv::waitKey();

	}

	//Remove patch that are no inside limits of aspect ratio and area.    
	while (itc != contours.end())
	{

		//Create bounding rect of object
		cv::Rect mr = cv::boundingRect(cv::Mat(*itc));
		rectangle(result, mr, cv::Scalar(0, 255, 0));
		//Crop image
		cv::Mat auxRoi(img_threshold, mr);


		if (ValidateChar(auxRoi))
		{
			mr.x -= 2.5;
			mr.y -= 2.5;
			mr.width += 5;
			mr.height += 5;
			auxRoi = preprocessChar(cv::Mat(img_threshold, mr));
			plate.AddChars(auxRoi, mr);
			rectangle(result, mr, cv::Scalar(0, 125, 255));
			cv::imshow("result", result);
			cv::waitKey();
		}

		++itc;
	}
	//std::cout << "Num chars: " << output.size() << "\n";
	//return output;
}