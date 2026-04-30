#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Dense>
#include <unordered_map>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <iostream>

#include "Projection.h"

class ZEstimator {
public:
    struct Correspondence {
        Eigen::Vector3i source_key;
        Eigen::Vector3i target_key;
        double distance = -1.0;
        Eigen::Vector3f source_center;
        Eigen::Vector3f target_center;
        Eigen::Vector3f source_normal;
        Eigen::Vector3f target_normal;
    };

    std::mt19937 rng;
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    double resolution;
    double voxel_size;

    ZEstimator(double res_param) : rng(42), resolution(res_param), voxel_size(2 * res_param) {}

    double calculateNormalAngle(const Eigen::Vector3f& n1, const Eigen::Vector3f& n2);

    double calculatePlaneDistance(const Eigen::Vector3f& center1, const Eigen::Vector3f& normal1,
        const Eigen::Vector3f& center2, const Eigen::Vector3f& normal2);

    void transformVoxel(VoxelData& voxel, const Eigen::Matrix2d& rotation,
        const Eigen::Vector2d& translation);

    void buildTargetKDTree(const std::unordered_map<Eigen::Vector3i, VoxelData, Vector3iHash>& target_voxels,
        pcl::PointCloud<pcl::PointXYZ>::Ptr& target_centers,
        std::vector<Eigen::Vector3i>& target_keys);

    double evaluate(const std::vector<double>& dzs,
        pcl::PointCloud<pcl::PointXYZ>::Ptr src, pcl::PointCloud<pcl::PointXYZ>::Ptr tgt,
        const Eigen::Matrix2d& rotation, const Eigen::Vector2d& translation);

    std::unordered_map<Eigen::Vector3i, VoxelData, Vector3iHash> filterVoxelsForZEstimation(const std::unordered_map<Eigen::Vector3i, VoxelData, Vector3iHash>& voxels);

    double estimate(pcl::PointCloud<pcl::PointXYZ>::Ptr src, pcl::PointCloud<pcl::PointXYZ>::Ptr tgt,
        const std::unordered_map<Eigen::Vector3i, VoxelData, Vector3iHash>& source_voxels,
        const std::unordered_map<Eigen::Vector3i, VoxelData, Vector3iHash>& target_voxels,
        const Eigen::Matrix2d& rotation,
        const Eigen::Vector2d& translation);

    double evaluate(const std::vector<double>& dzs,
        const std::unordered_map<Eigen::Vector3i, VoxelData, Vector3iHash>& src_voxels,
        const std::unordered_map<Eigen::Vector3i, VoxelData, Vector3iHash>& tgt_voxels,
        const Eigen::Matrix2d& rotation, const Eigen::Vector2d& translation);
};
