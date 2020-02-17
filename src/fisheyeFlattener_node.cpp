#include "fisheyeFlattener_node.hpp"
#include <pluginlib/class_list_macros.h>
#include <omp.h>

#define DEG_TO_RAD (M_PI / 180.0)

namespace fisheye_flattener_pkg
{
void FisheyeFlattener::onInit()
{
    ros::NodeHandle nh = this->getPrivateNodeHandle();

    // obtain camera intrinsics
    nh.param<std::string>("cam_file", camFilePath, "cam.yaml");
    NODELET_INFO("Camere file: %s", camFilePath.c_str());
    if (!std::experimental::filesystem::exists(camFilePath))
    {
        ROS_ERROR("Camera file does not exist.");
        return;
    }
    cam = camera_model::CameraFactory::instance()
              ->generateCameraFromYamlFile(camFilePath);

    // remapping parameters
    nh.param<bool>("use_gpu", enable_cuda, false);
    nh.param<double>("rotationVectorX", cameraRotation.x(), 0);
    nh.param<double>("rotationVectorY", cameraRotation.y(), 0);
    nh.param<double>("rotationVectorZ", cameraRotation.z(), 0);
    nh.param<double>("fov", fov, 90);
    nh.param<int>("imgWidth", imgWidth, 500);
    if (imgWidth <= 0)
    {
        NODELET_ERROR("Resolution must be non-negative");
        return;
    }
    if (fov < 0)
    {
        NODELET_ERROR("FOV must be non-negative");
        return;
    }

    nh.param<std::string>("inputTopic", inputTopic, "img");
    nh.param<std::string>("outputTopicPrefix", outputTopicPrefix, "flatImg");

    undistMaps = generateAllUndistMap(cam, cameraRotation, imgWidth, fov);
    if (enable_cuda) {
        for (auto mat : undistMaps) {
            cv::Mat xy[2];
            cv::split(mat, xy);
            undistMapsGPUX.push_back(cv::cuda::GpuMat(xy[0]));
            undistMapsGPUY.push_back(cv::cuda::GpuMat(xy[1]));
        }
    }


    NODELET_INFO("Unsidtortion maps generated.");
    for (int i = 0; i < undistMaps.size(); i++)
    {
        img_pub.push_back(nh.advertise<sensor_msgs::Image>(outputTopicPrefix + "_" + std::to_string(i), 3));
    }
    ros::Subscriber img_sub = nh.subscribe(inputTopic, 3, &FisheyeFlattener::imgCB, this);
    ros::spin();
    return;
}

void FisheyeFlattener::imgCB(const sensor_msgs::Image::ConstPtr &msg)
{
    cv_bridge::CvImagePtr cv_ptr;
    cv::cuda::GpuMat img_cuda;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg);
        if (enable_cuda) {
            img_cuda.upload(cv_ptr->image);
        }
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    for (int i = 0; i < undistMaps.size(); i++)
    {
        cv_bridge::CvImage outImg;
        outImg.header = msg->header;
        outImg.encoding = msg->encoding;
        if (enable_cuda) {
            cv::cuda::GpuMat output;
            cv::cuda::remap(img_cuda, output, undistMapsGPUX[i], undistMapsGPUY[i], cv::INTER_LINEAR);
            output.download(outImg.image);
        } else {
            cv::remap(cv_ptr->image, outImg.image, undistMaps[i], cv::Mat(), cv::INTER_LINEAR);
        }

        img_pub[i].publish(outImg);
    }
};

std::vector<cv::Mat> FisheyeFlattener::generateAllUndistMap(
    camera_model::CameraPtr p_cam,
    Eigen::Vector3d rotation,
    const unsigned &imgWidth,
    const double &fov //degree
)
{
    ROS_INFO("Generating undistortion maps:");
    double sideVerticalFOV = (fov - 180) * DEG_TO_RAD;
    if (sideVerticalFOV < 0)
        sideVerticalFOV = 0;
    double centerFOV = fov * DEG_TO_RAD - sideVerticalFOV * 2;
    // calculate focal length of fake pinhole cameras (pixel size = 1 unit)
    double f_center = (double)imgWidth / 2 / tan(centerFOV / 2);
    double f_side = (double)imgWidth / 2;

    ROS_INFO("Center FOV: %f_center", centerFOV);
    // int sideImgHeight = sideVerticalFOV / centerFOV * imgWidth;
    int sideImgHeight = 2 * f_side * tan(sideVerticalFOV/2);
    ROS_INFO("Side image height: %d", sideImgHeight);
    std::vector<cv::Mat> maps(5, cv::Mat());

    // test points
    Eigen::Vector3d testPoints[] = {
        Eigen::Vector3d(0, 0, 1),
        Eigen::Vector3d(1, 0, 1),
        Eigen::Vector3d(0, 1, 1),
        Eigen::Vector3d(1, 1, 1),
    };
    for (int i = 0; i < sizeof(testPoints) / sizeof(Eigen::Vector3d); i++)
    {
        Eigen::Vector2d temp;
        p_cam->spaceToPlane(testPoints[i], temp);
        ROS_INFO("Test point %d : (%.2f,%.2f,%.2f) projected to (%.2f,%.2f)", i,
                 testPoints[i][0], testPoints[i][1], testPoints[i][2],
                 temp[0], temp[1]);
    }

    // center pinhole camera orientation
    std::vector<Eigen::Quaterniond> outputDir;
    outputDir.reserve(5);
    outputDir[0] = Eigen::AngleAxis<double>(rotation.norm(), rotation.normalized()).inverse();
    //turn to y
    outputDir[1] = outputDir[0] * Eigen::AngleAxis<double>(-M_PI / 2, Eigen::Vector3d(1, 0, 0));
    for (unsigned i = 2; i < 5; i++)
    { //turn right
        outputDir[i] = outputDir[i - 1] * Eigen::AngleAxis<double>(M_PI / 2, Eigen::Vector3d(0, 1, 0));
    }
    // calculate focal length of fake pinhole cameras (pixel size = 1 unit)
    ROS_INFO("Pinhole cameras focal length: %f_center", f_center);

    const int numOutput = (sideImgHeight > 0) ? 5 : 1;
#pragma omp parallel for
    for (unsigned i = 0; i < numOutput; i++)
    {
        maps[i] = genOneUndistMap(p_cam, outputDir[i],
                                  imgWidth,
                                  (i == 0) ? imgWidth : sideImgHeight,
                                  (i == 0) ? f_center : f_side);
    }
    return maps;
}

/**
 * @brief 
 * 
 * @param p_cam 
 * @param rotation rotational offset from normal
 * @param imgWidth 
 * @param imgHeight 
 * @param f_center focal length in pin hole camera camera_mode (pixels are 1 unit sized)
 * @return CV_32FC2 mapping matrix 
 */
cv::Mat FisheyeFlattener::genOneUndistMap(
    camera_model::CameraPtr p_cam,
    Eigen::Quaterniond rotation,
    const unsigned &imgWidth,
    const unsigned &imgHeight,
    const double &f_center)
{
    cv::Mat map = cv::Mat(imgHeight, imgWidth, CV_32FC2);
    ROS_INFO("Generating map of size (%d,%d)", map.size[0], map.size[1]);
    ROS_INFO("Perspective facing (%.2f,%.2f,%.2f)",
             (rotation * Eigen::Vector3d(0, 0, 1))[0],
             (rotation * Eigen::Vector3d(0, 0, 1))[1],
             (rotation * Eigen::Vector3d(0, 0, 1))[2]);
    for (int x = 0; x < imgWidth; x++)
        for (int y = 0; y < imgHeight; y++)
        {
            Eigen::Vector3d objPoint =
                rotation *
                Eigen::Vector3d(
                    ((double)x - (double)imgWidth / 2),
                    ((double)y - (double)imgHeight / 2),
                    f_center);
            Eigen::Vector2d imgPoint;
            p_cam->spaceToPlane(objPoint, imgPoint);
            map.at<cv::Vec2f>(cv::Point(x, y)) = cv::Vec2f(imgPoint.x(), imgPoint.y());
        }

    ROS_INFO("Upper corners: (%.2f, %.2f), (%.2f, %.2f)",
             map.at<cv::Vec2f>(cv::Point(0, 0))[0],
             map.at<cv::Vec2f>(cv::Point(0, 0))[1],
             map.at<cv::Vec2f>(cv::Point(imgWidth - 1, 0))[0],
             map.at<cv::Vec2f>(cv::Point(imgWidth - 1, 0))[1]);

    // Eigen::Vector3d objPoint =
    //     rotation *
    //     Eigen::Vector3d(
    //         ((double)0 - (double)imgWidth / 2),
    //         ((double)0 - (double)imgHeight / 2),
    //         f_center);
    // std::cout << objPoint << std::endl;

    // objPoint =
    //     rotation *
    //     Eigen::Vector3d(
    //         ((double)imgWidth / 2),
    //         ((double)0 - (double)imgHeight / 2),
    //         f_center);
    // std::cout << objPoint << std::endl;

    return map;
}

PLUGINLIB_EXPORT_CLASS(fisheye_flattener_pkg::FisheyeFlattener, nodelet::Nodelet);
} // namespace FisheyeFlattener
