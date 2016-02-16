#pragma once
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/calib3d.hpp>

struct Chars
{
	Chars(const cv::Mat& char_img, const cv::Rect& char_rect): m_char(char_img), m_rect(char_rect) {}
	cv::Mat m_char;
	cv::Rect m_rect;
};

class Plate
{
public:
	Plate(): m_image(cv::Mat()), m_rect(cv::RotatedRect()) {}
	Plate(cv::Mat& img, cv::RotatedRect& rect) : m_image(img), m_rect(rect), m_chars(std::vector<Chars>()) {}
	Plate(const Plate& plate): m_image(plate.m_image), m_rect(plate.m_rect) {}
	~Plate() {}

	const cv::Mat& GetImage() const { return m_image; }
	const cv::RotatedRect& GetRect() const { return m_rect; }
	const std::vector<Chars>& GetChars() const { return m_chars; }
	void AddChars(const Chars& c) { m_chars.push_back(c); }
	void AddChars(const cv::Mat& m, const cv::Rect& r) { m_chars.push_back(Chars(m, r)); }

	void SetImage(const cv::Mat& img) { m_image = img; }
	void SetRect(const cv::RotatedRect& rect) { m_rect = rect; }
private:
	cv::Mat				m_image;
	cv::RotatedRect		m_rect;
	std::vector<Chars>	m_chars;


};