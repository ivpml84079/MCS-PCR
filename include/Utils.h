#pragma once

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/pcl_config.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/octree/octree.h>
#include <pcl/common/centroid.h>
#include <pcl/common/pca.h>

#include <yaml-cpp/yaml.h>


// Point cloud pair information structure
struct PointCloudPair {
    std::string source;
    std::string target;
    std::string transformation_file;
};

// Dataset configuration structure
struct DatasetConfig {
    std::string dataset_name;
    std::string description;
    std::string root;
    std::string groundtruth;
    std::string raw_data;
    std::vector<PointCloudPair> pairs;
};



DatasetConfig loadYamlConfig(const std::string& yaml_file, bool verbose = false);

int voxelBasedDownsample(
    pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_in,
    pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_down,
    float downsample_size, bool verbose = false
);

Eigen::Matrix4d loadGroundTruthTransformation(const std::string& transformation_file);

double calculateRotationError(const Eigen::Matrix3d& R_gt, const Eigen::Matrix3d& R_est);

double calculateTranslationError(const Eigen::Vector3d& t_gt, const Eigen::Vector3d& t_est);

Eigen::Matrix4d integrate(const Eigen::Matrix2d& rotation_2d, const Eigen::Vector2d& translation_2d, double dz);

//std::pair<Eigen::Matrix3d, Eigen::Vector3d> disintegrate(const Eigen::Matrix4d& T) {
//    return std::make_pair(T.block<3, 3>(0, 0), T.block<3, 1>(0, 3));
//}
