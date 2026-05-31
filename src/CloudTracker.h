#ifndef CLOUDTRACKER_H
#define CLOUDTRACKER_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class CloudTracker {
public:
  CloudTracker();
  ~CloudTracker();

  // Loads two satellite images from given paths
  bool loadImages(const std::string &path1, const std::string &path2);

  // Processes the images: thresholding, masking, optical flow, and wind speed
  // estimation timeDifference is in hours (or any arbitrary unit depending on
  // satellite pass frequency)
  void process(double timeDifference);

  // Display the results in OpenCV windows
  void showResults();

private:
  cv::Mat m_img1_color, m_img2_color;
  cv::Mat m_img1_gray, m_img2_gray;
  cv::Mat m_cloud_mask1, m_cloud_mask2;
  cv::Mat m_highlighted_clouds;
  cv::Mat m_map_overlay; // Stores static country borders and text
  cv::Mat m_flow;

  cv::Mat m_wind_speed_map;
  cv::Mat m_flow_visualization;

  double m_cloud_coverage_percentage;
  double m_avg_wind_speed;

  // Internal helper functions
  cv::Mat preprocessAndMask(const cv::Mat &src_gray, cv::Mat &out_mask);
  void extractOverlays();
  void computeOpticalFlow();
  void estimateWindSpeed(double timeDifference);
  void drawMotionVectors(int step, const cv::Scalar &color);
  void drawLegend(cv::Mat &image);
};

#endif // CLOUDTRACKER_H
