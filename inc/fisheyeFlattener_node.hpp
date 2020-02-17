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
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

namespace fisheye_flattener_pkg
{
class FisheyeFlattener : public nodelet::Nodelet
{
public:
    FisheyeFlattener() {}

private:
    std::string inputTopic;
    std::string outputTopicPrefix;
    std::string camFilePath;
    camera_model::CameraPtr cam;
    Eigen::Vector3d cameraRotation;
    int imgWidth = 0;
    double fov = 0; //in degree
    std::vector<cv::Mat> undistMaps;
    std::vector<cv::cuda::GpuMat> undistMapsGPUX;
    std::vector<cv::cuda::GpuMat> undistMapsGPUY;
    std::vector<ros::Publisher> img_pub;
    bool enable_cuda = false;


    virtual void onInit();

    std::vector<cv::Mat> generateAllUndistMap(
        camera_model::CameraPtr p_cam,
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

    void imgCB(const sensor_msgs::Image::ConstPtr &msg);
};

} // namespace FisheyeFlattener
