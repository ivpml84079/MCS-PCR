#include "Registration.h"

#include <chrono>
#include <iomanip>
#include <filesystem>

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cout << "Usage: ./example <source PLY file> <target PLY file> <GT file> <downsample resolution>" << std::endl;
        return -1;
    }

    WHTTC::Param param;
    param.resolution = std::stod(std::string(argv[4]));
    WHTTC whttc(param);

    // 構建完整檔案路徑
    std::string src_file_path = argv[1];
    std::string tgt_file_path = argv[2];
    std::string gt_file_path = argv[3];

    std::cout << "Loading point clouds ..." << std::endl;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src_org(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tgt(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tgt_org(new pcl::PointCloud<pcl::PointXYZ>);

    if (pcl::io::loadPLYFile<pcl::PointXYZ>(src_file_path, *cloud_src_org) == -1 ||
        pcl::io::loadPLYFile<pcl::PointXYZ>(tgt_file_path, *cloud_tgt_org) == -1) {
        std::cerr << "loadPLYFile fail." << std::endl;
        return -1;
    }

    std::cout << "Downsampling point clouds ..." << std::endl;
    voxelBasedDownsample(cloud_src_org, cloud_src, param.resolution);
    voxelBasedDownsample(cloud_tgt_org, cloud_tgt, param.resolution);

    std::cout << "Loading ground truth ..." << std::endl;
    Eigen::Matrix4d T_gt = loadGroundTruthTransformation(gt_file_path);
    Eigen::Matrix3d R_gt = T_gt.block<3, 3>(0, 0);
    Eigen::Vector3d t_gt = T_gt.block<3, 1>(0, 3);

    std::cout << "Ground Truth Transformation:" << std::endl;
    std::cout << T_gt << std::endl;

    if (whttc.regis(cloud_src, cloud_tgt, 2)) {
        Eigen::Matrix4d T_est = whttc.getEstTransformation();

        std::cout << std::endl << "Estimated Transformation:" << std::endl;
        std::cout << T_est << std::endl;

        Eigen::Matrix3d R_est = T_est.block<3, 3>(0, 0);
        Eigen::Vector3d t_est = T_est.block<3, 1>(0, 3);

        double rot_error = calculateRotationError(R_gt, R_est);
        double trans_error = calculateTranslationError(t_gt, t_est);

        std::cout << "Registration result:" << std::endl;
        std::cout << "RE: " << rot_error << " (deg)" << std::endl;
        std::cout << "TE: " << trans_error << " m" << std::endl;
        std::cout << "Runtime: " << whttc.getTime() << " s" << std::endl << std::endl;
    }
    else {
        std::cout << "Fail." << std::endl << std::endl;
    }

    return 0;
}
