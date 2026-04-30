#pragma once

#include <iostream>
#include <vector>
#include <chrono>
#include <unordered_map>

#include <pcl/pcl_config.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/octree/octree.h>
#include <pcl/common/centroid.h>
#include <pcl/common/pca.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <omp.h>

struct VoxelData {
    pcl::PointCloud<pcl::PointXYZ>::Ptr points;
    Eigen::Vector3f center;
    Eigen::Vector3f normal;
    Eigen::Vector3f eigenvalues;
    Eigen::Matrix3f eigenvectors;

    VoxelData() : points(new pcl::PointCloud<pcl::PointXYZ>) { }
};

struct Vector3iHash {
    std::size_t operator()(const Eigen::Vector3i& v) const {
        std::size_t seed = 0;
        seed ^= std::hash<int>()(v[0]) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int>()(v[1]) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int>()(v[2]) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

class IFVRProcessor {
public:
    float voxel_size_;
    int min_points_per_voxel_;
    std::unordered_map<Eigen::Vector3i, VoxelData, Vector3iHash> voxels_;

    IFVRProcessor(double res_param): voxel_size_(2 * res_param), min_points_per_voxel_(3) { }

    bool process(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, int verbose);

    bool voxelizePointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, int verbose);

    bool computeNormalForVoxel(VoxelData& voxel);

    pcl::PointCloud<pcl::PointXYZ>::Ptr getFeasiblePoints(
        const Eigen::Vector3f& space_normal = Eigen::Vector3f(0.0f, 0.0f, 1.0f),  // x-y space
        float max_angle_degrees = 10.0f
    );

    const std::unordered_map<Eigen::Vector3i, VoxelData, Vector3iHash> getVoxelsForZEstimation();

    const std::unordered_map<Eigen::Vector3i, VoxelData, Vector3iHash>& getVoxels() const { return voxels_; }
};

class PointsPixelator {
public:
    struct PixelInfo {
        bool value;                    // 二元值（true = 1, false = 0）
        size_t count;                  // 像素中的點數
        pcl::IndicesPtr point_indices;  // 點雲中的索引，使用 PCL 的索引指標
        Eigen::Vector3f centroid;      // 質心座標 (x, y, z)
        int line_seg_idx;

        PixelInfo() :
            value(false), count(0), line_seg_idx(-1),
            point_indices(new std::vector<int>()), centroid(Eigen::Vector3f::Zero()) {
        }
    };

    using PixelInfoPtr = std::shared_ptr<PixelInfo>;

    struct ProjectionImage {
        std::vector<std::vector<bool>> binaryImage;  // 二元影像 valid_image
        std::vector<std::vector<PixelInfoPtr>> pixelInfo;  // 像素詳細資訊，使用 shared_ptr 節省記憶體
        int width;                 // 影像寬度
        int height;                // 影像高度
        float pixelSize;           // 像素大小
        size_t thresholdCount;     // 閾值
        Eigen::Vector2f minPoint;  // 點雲的最小XY座標
        pcl::PointCloud<pcl::PointXY>::Ptr points2D;

        ProjectionImage() :
            width(-1), height(-1), pixelSize(0.0f), thresholdCount(1),
            minPoint(Eigen::Vector2f::Zero()), points2D(new pcl::PointCloud<pcl::PointXY>()) {
        }
    } image_map;

    float pixel_size;           // 像素大小
    size_t thresholdCount_;     // 閾值

    PointsPixelator(double res_param, size_t thresh_count = 3): pixel_size(res_param), thresholdCount_(thresh_count) { }

    bool pixelize(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, int verbose = false);

    ProjectionImage getProjectionImage();
};