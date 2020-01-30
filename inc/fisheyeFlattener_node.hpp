#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/cudaimgproc.hpp>
#include <camera_model/camera_models/CameraFactory.h>
#include "cv_bridge/cv_bridge.h"
#include <experimental/filesystem>
#include <opencv2/cudawarping.hpp>

std::vector<cv::Mat> generateAllUndistMap(camera_model::CameraPtr p_cam,
                                          Eigen::Vector3d rotation,
                                          const unsigned &imgWidth,
                                          const double &fov //degree
);

cv::Mat genOneUndistMap(
    camera_model::CameraPtr p_cam,
    Eigen::Quaterniond rotation,
    const unsigned &imgWidth,
    const unsigned &imgHeight,
    const double &f);