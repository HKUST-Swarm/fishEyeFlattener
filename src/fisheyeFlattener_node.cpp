#include "fisheyeFlattener_node.hpp"

#define DEG_TO_RAD (M_PI / 180.0)

namespace globalVar
{
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
} // namespace globalVar

void imgCB(const sensor_msgs::Image::ConstPtr &msg);

int main(int argc, char **argv)
{
    using namespace globalVar;

    ros::init(argc, argv, "fisheyeFlattener_node");
    ros::NodeHandle nh("~");

    // obtain camera intrinsics
    nh.param<std::string>("cam_file", camFilePath, "cam.yaml");
    ROS_INFO(camFilePath.c_str());
    if (!std::experimental::filesystem::exists(camFilePath))
    {
        ROS_ERROR("Camera file does not exist.");
        return 1;
    }
    cam = camera_model::CameraFactory::instance()
              ->generateCameraFromYamlFile(camFilePath);

    // remapping parameters
    nh.param<bool>("use_gpu", globalVar::enable_cuda, false);
    nh.param<double>("rotationVectorX", cameraRotation.x(), 0);
    nh.param<double>("rotationVectorY", cameraRotation.y(), 0);
    nh.param<double>("rotationVectorZ", cameraRotation.z(), 0);
    nh.param<double>("fov", fov, 90);
    nh.param<int>("imgWidth", imgWidth, 500);
    if (imgWidth <= 0)
    {
        ROS_ERROR("Resolution must be non-negative");
        return 1;
    }
    if (fov < 0)
    {
        ROS_ERROR("FOV must be non-negative");
        return 1;
    }

    nh.param<std::string>("inputTopic", inputTopic, "img");
    nh.param<std::string>("outputTopicPrefix", outputTopicPrefix, "flatImg");

    undistMaps = generateAllUndistMap(cam, cameraRotation, imgWidth, fov);
    if (globalVar::enable_cuda) {
        for (auto mat : undistMaps) {
            cv::Mat xy[2];
            cv::split(mat, xy);
            undistMapsGPUX.push_back(cv::cuda::GpuMat(xy[0]));
            undistMapsGPUY.push_back(cv::cuda::GpuMat(xy[1]));
        }
    }


    for (int i = 0; i < undistMaps.size(); i++)
    {
        img_pub.push_back(nh.advertise<sensor_msgs::Image>(outputTopicPrefix + "_" + std::to_string(i), 3));
    }
    ros::Subscriber img_sub = nh.subscribe(inputTopic, 3, imgCB);
    ros::spin();
    return 0;
}

void imgCB(const sensor_msgs::Image::ConstPtr &msg)
{
    cv_bridge::CvImagePtr cv_ptr;
    cv::cuda::GpuMat img_cuda;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg);
        if (globalVar::enable_cuda) {
            img_cuda.upload(cv_ptr->image);
        }
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    for (int i = 0; i < globalVar::undistMaps.size(); i++)
    {
        cv_bridge::CvImage outImg;
        outImg.header = msg->header;
        outImg.encoding = msg->encoding;
        if (globalVar::enable_cuda) {
            cv::cuda::GpuMat output;
            cv::cuda::remap(img_cuda, output, globalVar::undistMapsGPUX[i], globalVar::undistMapsGPUY[i], cv::INTER_LINEAR);
            output.download(outImg.image);
        } else {
            cv::remap(cv_ptr->image, outImg.image, globalVar::undistMaps[i], cv::Mat(), cv::INTER_LINEAR);
        }

        globalVar::img_pub[i].publish(outImg);
    }
};

std::vector<cv::Mat> generateAllUndistMap(
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
    std::vector<cv::Mat> maps;
    maps.reserve(5);

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
    // auto t = Eigen::AngleAxis<double>(rotation.norm(), rotation.normalized()).inverse();
    auto t = Eigen::AngleAxisd(rotation.z() / 180 * M_PI, Eigen::Vector3d::UnitZ()) * 
        Eigen::AngleAxisd(rotation.y() / 180 * M_PI, Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(rotation.x() / 180 * M_PI, Eigen::Vector3d::UnitX());
    // .inverse();


    ROS_INFO("Pinhole cameras focal length: center %f side %f", f_center, f_side);
    maps.push_back(genOneUndistMap(p_cam, t, imgWidth, imgWidth, f_center));

    if (sideImgHeight > 0)
    {
        //facing y
        t = t * Eigen::AngleAxis<double>(-M_PI / 2, Eigen::Vector3d(1, 0, 0));
        maps.push_back(genOneUndistMap(p_cam, t, imgWidth, sideImgHeight, f_side));

        //turn right/left?
        t = t * Eigen::AngleAxis<double>(M_PI / 2, Eigen::Vector3d(0, 1, 0));
        maps.push_back(genOneUndistMap(p_cam, t, imgWidth, sideImgHeight, f_side));
        t = t * Eigen::AngleAxis<double>(M_PI / 2, Eigen::Vector3d(0, 1, 0));
        maps.push_back(genOneUndistMap(p_cam, t, imgWidth, sideImgHeight, f_side));
        t = t * Eigen::AngleAxis<double>(M_PI / 2, Eigen::Vector3d(0, 1, 0));
        maps.push_back(genOneUndistMap(p_cam, t, imgWidth, sideImgHeight, f_side));
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
cv::Mat genOneUndistMap(
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

    Eigen::Vector3d objPoint =
        rotation *
        Eigen::Vector3d(
            ((double)0 - (double)imgWidth / 2),
            ((double)0 - (double)imgHeight / 2),
            f_center);
    std::cout << objPoint << std::endl;

    objPoint =
        rotation *
        Eigen::Vector3d(
            ((double)imgWidth / 2),
            ((double)0 - (double)imgHeight / 2),
            f_center);
    std::cout << objPoint << std::endl;

    return map;
}