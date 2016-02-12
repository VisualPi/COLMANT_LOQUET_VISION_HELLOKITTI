#pragma once
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/calib3d.hpp>
class Plate
{
public:
	Plate() {}
	~Plate() {}
	Plate(const cv::Mat& img, const cv::Rect& pos): m_plateImg(img), m_position(pos){}
	std::string				str()
	{
		std::string result = "";
		//Order numbers
		std::vector<int> orderIndex;
		std::vector<int> xpositions;
		for (int i = 0; i < m_charsPos.size(); i++)
		{
			orderIndex.push_back(i);
			xpositions.push_back(m_charsPos[i].x);
		}
		/*float*/int min = xpositions[0];
		int minIdx = 0;
		for (int i = 0; i < xpositions.size(); i++)
		{
			min = xpositions[i];
			minIdx = i;
			for (int j = i; j < xpositions.size(); j++)
			{
				if (xpositions[j] < min)
				{
					min = xpositions[j];
					minIdx = j;
				}
			}
			int aux_i = orderIndex[i];
			int aux_min = orderIndex[minIdx];
			orderIndex[i] = aux_min;
			orderIndex[minIdx] = aux_i;

			/*float*/int aux_xi = xpositions[i];
			/*float*/int aux_xmin = xpositions[minIdx];
			xpositions[i] = aux_xmin;
			xpositions[minIdx] = aux_xi;
		}
		for (int i = 0; i < orderIndex.size(); i++)
		{
			result = result + m_chars[orderIndex[i]];
		}
		return result;
	}

	cv::Rect&				GetPosition() { return m_position; }
	cv::Mat					GetImg() const { return m_plateImg; }
	std::vector<char>&		GetChars() { return m_chars; }
	std::vector<cv::Rect>&	GetCharsPos() { return m_charsPos; }

private:
	cv::Rect				m_position;
	cv::Mat					m_plateImg;
	std::vector<char>		m_chars;
	std::vector<cv::Rect>	m_charsPos;
};