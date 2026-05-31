#include "CloudTracker.h"
#include <iostream>

CloudTracker::CloudTracker()
    : m_cloud_coverage_percentage(0.0), m_avg_wind_speed(0.0) {}

CloudTracker::~CloudTracker() {}

bool CloudTracker::loadImages(const std::string &path1,
                              const std::string &path2) {
  m_img1_color = cv::imread(path1);
  m_img2_color = cv::imread(path2);

  if (m_img1_color.empty() || m_img2_color.empty()) {
    std::cerr << "Error: Could not load one or both images." << std::endl;
    return false;
  }

  // Ensure images are the same size
  if (m_img1_color.size() != m_img2_color.size()) {
    cv::resize(m_img2_color, m_img2_color, m_img1_color.size());
  }

  // Convert to grayscale
  cv::cvtColor(m_img1_color, m_img1_gray, cv::COLOR_BGR2GRAY);
  cv::cvtColor(m_img2_color, m_img2_gray, cv::COLOR_BGR2GRAY);

  return true;
}

cv::Mat CloudTracker::preprocessAndMask(const cv::Mat &src_gray,
                                        cv::Mat &out_mask) {
  cv::Mat blurred;
  // Apply Gaussian Blur to reduce noise
  cv::GaussianBlur(src_gray, blurred, cv::Size(5, 5), 0);

  // Perform thresholding to identify cloud regions.
  // In infrared satellite imagery, both land and ocean surfaces are warm
  // (dark/black), while clouds are cold (gray to white). A threshold around 50
  // captures most cloud cover without picking up background noise.
  cv::threshold(blurred, out_mask, 50, 255, cv::THRESH_BINARY);

  return blurred;
}

void CloudTracker::extractOverlays() {
  // --- Step 1: Find static bright pixels (present in both frames, unchanged)
  // ---
  cv::Mat diff;
  cv::absdiff(m_img1_gray, m_img2_gray, diff);

  // Static pixels have very low difference between frames
  cv::Mat static_mask;
  cv::threshold(diff, static_mask, 8, 255, cv::THRESH_BINARY_INV);

  // Bright white pixels (text and borders are rendered as white overlays)
  cv::Mat bright_mask;
  cv::threshold(m_img1_gray, bright_mask, 170, 255, cv::THRESH_BINARY);

  // Candidate overlay = static AND bright
  cv::Mat candidate;
  cv::bitwise_and(static_mask, bright_mask, candidate);

  // --- Step 2: Use Connected Component Analysis to separate text from clouds
  // --- Text characters are small, thin components. Clouds are large blobs.
  cv::Mat labels, stats, centroids;
  int num_labels =
      cv::connectedComponentsWithStats(candidate, labels, stats, centroids);

  m_map_overlay = cv::Mat::zeros(candidate.size(), CV_8UC1);

  for (int i = 1; i < num_labels; i++) {
    int area = stats.at<int>(i, cv::CC_STAT_AREA);
    int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
    int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);
    int left = stats.at<int>(i, cv::CC_STAT_LEFT);
    int top = stats.at<int>(i, cv::CC_STAT_TOP);

    // Text characters are typically small (area < a few thousand pixels)
    // and have moderate aspect ratios. Clouds are massive blobs.
    // Borders are thin lines (width >> height or height >> width, with small
    // area relative to bounding box)
    double bbox_area = static_cast<double>(width) * height;
    double fill_ratio = (bbox_area > 0) ? (area / bbox_area) : 1.0;

    bool is_text = false;

    // Small components are likely text characters
    if (area < 5000) {
      is_text = true;
    }
    // Thin lines (borders): very elongated and sparse fill
    else if ((width > 10 * height || height > 10 * width) && fill_ratio < 0.3) {
      is_text = true;
    }
    // Medium components with low fill ratio (grouped text labels)
    else if (area < 20000 && fill_ratio < 0.4) {
      is_text = true;
    }

    if (is_text) {
      // Mark this component in the overlay mask
      cv::Mat component_mask = (labels == i);
      component_mask.convertTo(component_mask, CV_8UC1, 255);
      cv::bitwise_or(m_map_overlay, component_mask, m_map_overlay);
    }
  }

  // --- Step 3: Dilate to cover anti-aliased edges around text ---
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
  cv::Mat dilated_overlay;
  cv::dilate(m_map_overlay, dilated_overlay, kernel);

  // --- Step 4: Remove overlay pixels from cloud masks ---
  cv::Mat not_overlay;
  cv::bitwise_not(dilated_overlay, not_overlay);

  cv::bitwise_and(m_cloud_mask1, not_overlay, m_cloud_mask1);
  cv::bitwise_and(m_cloud_mask2, not_overlay, m_cloud_mask2);
}

void CloudTracker::computeOpticalFlow() {
  // We only want to track flow in cloud regions.
  // Calculate Dense Optical Flow using Farneback method on the grayscale images
  // Note: We use the raw grayscale (or blurred) images for flow, as thresholded
  // masks might lack the texture needed for good optical flow tracking.

  cv::calcOpticalFlowFarneback(m_img1_gray, m_img2_gray, m_flow, 0.5, 3, 15, 3,
                               5, 1.2, 0);
}

void CloudTracker::estimateWindSpeed(double timeDifference) {
  // Avoid division by zero
  if (timeDifference <= 0)
    timeDifference = 1.0;

  m_wind_speed_map = cv::Mat::zeros(m_flow.size(), CV_32FC1);
  double total_speed = 0.0;
  int cloud_pixel_count = 0;

  for (int y = 0; y < m_flow.rows; ++y) {
    for (int x = 0; x < m_flow.cols; ++x) {
      // Only consider pixels that are clouds in the first image
      if (m_cloud_mask1.at<uchar>(y, x) > 0) {
        const cv::Point2f &flow_at_xy = m_flow.at<cv::Point2f>(y, x);
        // Compute cloud displacement (magnitude of the vector)
        double displacement = cv::norm(flow_at_xy);

        // Wind Speed = Cloud Displacement / Time Difference
        double speed = displacement / timeDifference;

        m_wind_speed_map.at<float>(y, x) = static_cast<float>(speed);
        total_speed += speed;
        cloud_pixel_count++;
      }
    }
  }

  if (cloud_pixel_count > 0) {
    m_avg_wind_speed = total_speed / cloud_pixel_count;
  } else {
    m_avg_wind_speed = 0.0;
  }
}

void CloudTracker::drawMotionVectors(int step, const cv::Scalar &color) {
  // Clone original image to draw vectors
  m_flow_visualization = m_img1_color.clone();

  for (int y = 0; y < m_flow.rows; y += step) {
    for (int x = 0; x < m_flow.cols; x += step) {
      // Only draw vectors where clouds exist
      if (m_cloud_mask1.at<uchar>(y, x) > 0) {
        const cv::Point2f &flow_at_xy = m_flow.at<cv::Point2f>(y, x);
        cv::Point2f pt1(static_cast<float>(x), static_cast<float>(y));
        cv::Point2f pt2(pt1.x + flow_at_xy.x, pt1.y + flow_at_xy.y);

        // Draw arrow to represent wind direction
        cv::arrowedLine(m_flow_visualization, pt1, pt2, color, 1, cv::LINE_AA,
                        0, 0.3);
      }
    }
  }
}

void CloudTracker::process(double timeDifference) {
  if (m_img1_gray.empty() || m_img2_gray.empty())
    return;

  // 1 & 2 done in loadImages
  // 3, 4, 5. Blur and Mask
  preprocessAndMask(m_img1_gray, m_cloud_mask1);
  preprocessAndMask(m_img2_gray, m_cloud_mask2);

  // Extract static borders and text and remove them from cloud masks
  extractOverlays();

  // 6. Calculate cloud coverage percentage
  int total_pixels = m_cloud_mask1.rows * m_cloud_mask1.cols;
  int cloud_pixels = cv::countNonZero(m_cloud_mask1);
  m_cloud_coverage_percentage =
      (static_cast<double>(cloud_pixels) / total_pixels) * 100.0;

  // Create highlighted clouds image (will be colored by wind speed in
  // showResults)
  m_highlighted_clouds = m_img1_color.clone();

  // 7, 8. Track cloud movement and compute displacement
  computeOpticalFlow();

  // 9. Estimate wind speed
  estimateWindSpeed(timeDifference);

  // Prepare motion vectors visualization
  drawMotionVectors(16, cv::Scalar(0, 0, 255)); // Red arrows every 16 pixels
}

void CloudTracker::drawLegend(cv::Mat &image) {
  // Draw a wind speed color legend at the center bottom
  int legend_width = 200;
  int legend_height = 20;
  int margin = 20;

  int start_x = (image.cols - legend_width) / 2;
  int start_y = image.rows - legend_height - margin - 30; // leave room for text

  if (start_x < 0 || start_y < 0)
    return; // Image too small

  // Create a color gradient map
  cv::Mat gradient(1, legend_width, CV_8UC1);
  for (int i = 0; i < legend_width; i++) {
    gradient.at<uchar>(0, i) = static_cast<uchar>((i * 255) / legend_width);
  }
  cv::Mat color_gradient;
  cv::applyColorMap(gradient, color_gradient, cv::COLORMAP_JET);
  cv::resize(color_gradient, color_gradient,
             cv::Size(legend_width, legend_height));

  // Copy onto image
  cv::Rect rois(start_x, start_y, legend_width, legend_height);
  color_gradient.copyTo(image(rois));

  // Draw outline
  cv::rectangle(image, rois, cv::Scalar(255, 255, 255), 1);

  // Draw text (Low, Med, High)
  std::string text_min = "Low";
  std::string text_mid = "Med";
  std::string text_max = "High";

  cv::putText(image, text_min, cv::Point(start_x, start_y + legend_height + 20),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1,
              cv::LINE_AA);
  // Draw "Med" shadow for contrast if needed, or just white
  cv::putText(
      image, text_mid,
      cv::Point(start_x + legend_width / 2 - 15, start_y + legend_height + 20),
      cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
  cv::putText(
      image, text_max,
      cv::Point(start_x + legend_width - 35, start_y + legend_height + 20),
      cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
  cv::putText(image, "Wind Speed",
              cv::Point(start_x + legend_width / 2 - 40, start_y - 10),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1,
              cv::LINE_AA);
}

void CloudTracker::showResults() {
  if (m_img1_color.empty())
    return;

  // Display Wind Speed Map (Normalize for visualization)
  cv::Mat wind_speed_vis;
  cv::normalize(m_wind_speed_map, wind_speed_vis, 0, 255, cv::NORM_MINMAX,
                CV_8UC1);

  // Invert visualization so low values map to Red and high values to Blue
  // to match physical expectations where storm eye (small optical flow) has
  // high wind speeds.
  wind_speed_vis = 255 - wind_speed_vis;

  cv::applyColorMap(wind_speed_vis, wind_speed_vis, cv::COLORMAP_JET);

  // Apply mask so we only see speed on clouds
  cv::Mat masked_speed_vis =
      cv::Mat::zeros(wind_speed_vis.size(), wind_speed_vis.type());
  wind_speed_vis.copyTo(masked_speed_vis, m_cloud_mask1);

  // Make highlights colored by wind speed instead of cloud density
  wind_speed_vis.copyTo(m_highlighted_clouds, m_cloud_mask1);

  // Add legend
  drawLegend(m_highlighted_clouds);

  // Overlay static text and borders in Yellow on Original + Motion vector views
  m_img1_color.setTo(cv::Scalar(0, 255, 255), m_map_overlay);
  m_flow_visualization.setTo(cv::Scalar(0, 255, 255), m_map_overlay);

  // Overlay country names in Yellow on Highlighted Clouds for contrast over the
  // heatmap
  m_highlighted_clouds.setTo(cv::Scalar(0, 255, 255), m_map_overlay);

  // 10. Display highlighted clouds, motion vectors, wind speed, and statistics
  cv::imshow("1. Original Image 1", m_img1_color);
  cv::imshow("2. Highlighted Clouds", m_highlighted_clouds);
  cv::imshow("3. Wind Direction Vectors", m_flow_visualization);

  // Print statistics
  std::cout << "--- Statistics ---" << std::endl;
  std::cout << "Cloud Coverage: " << m_cloud_coverage_percentage << " %"
            << std::endl;
  std::cout << "Average Wind Speed: " << m_avg_wind_speed << " units/time"
            << std::endl;
  std::cout << "------------------" << std::endl;
  std::cout << "Press any key in the image windows to exit..." << std::endl;

  cv::waitKey(0);
}
