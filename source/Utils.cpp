#include "Utils.h"

int voxelBasedDownsample(
    pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_in,
    pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_down,
    float downsample_size, bool verbose
) {
    if (cloud_in->empty()) {
        std::cerr << "Error: Input cloud is empty!" << std::endl;
        return -1;
    }

    pcl::PointCloud<pcl::PointXYZ> cloud_sub;
    pcl::PointCloud<pcl::PointXYZ> cloud_out;

    float leafsize = downsample_size * (std::pow(static_cast<int64_t>(std::numeric_limits<int32_t>::max()) - 1, 1.0 / 3.0) - 1);

    // Build up octree
    pcl::octree::OctreePointCloud<pcl::PointXYZ> oct(leafsize);
    oct.setInputCloud(cloud_in);
    oct.defineBoundingBox();
    oct.addPointsFromInputCloud();

    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setLeafSize(downsample_size, downsample_size, downsample_size);
    vg.setInputCloud(cloud_in);

    size_t num_leaf = oct.getLeafCount();

    cloud_down->clear();

    pcl::octree::OctreePointCloud<pcl::PointXYZ>::LeafNodeIterator it = oct.leaf_depth_begin();
    pcl::octree::OctreePointCloud<pcl::PointXYZ>::LeafNodeIterator it_e = oct.leaf_depth_end();

    for (size_t i = 0; i < num_leaf; ++i, ++it) {
        pcl::IndicesPtr ids = std::make_shared<std::vector<int>>();
        pcl::octree::OctreePointCloud<pcl::PointXYZ>::LeafNode* node = static_cast<pcl::octree::OctreePointCloud<pcl::PointXYZ>::LeafNode*>(*it);

        if (node && node->getContainerPtr()) {
            node->getContainerPtr()->getPointIndices(*ids);

            if (!ids->empty()) {
                vg.setIndices(ids);
                cloud_sub.clear();
                vg.filter(cloud_sub);

                cloud_out.insert(cloud_out.end(), cloud_sub.begin(), cloud_sub.end());
            }
        }
    }

    *cloud_down = cloud_out;

    if (verbose) {
        std::cout << "Downsampled cloud size: " << cloud_down->size() << std::endl;
    }

    return static_cast<int>(cloud_down->size());
}

DatasetConfig loadYamlConfig(const std::string& yaml_file, bool verbose) {
    DatasetConfig config;

    try {
        YAML::Node yaml_config = YAML::LoadFile(yaml_file);

        config.dataset_name = yaml_config["dataset_name"].as<std::string>();
        config.description = yaml_config["description"].as<std::string>();
        config.root = yaml_config["root"].as<std::string>();
        config.groundtruth = yaml_config["groundtruth"].as<std::string>();
        config.raw_data = yaml_config["raw_data"].as<std::string>();

        // 讀取點雲配對
        if (yaml_config["pairs"]) {
            for (const auto& pair_node : yaml_config["pairs"]) {
                PointCloudPair pair;
                pair.source = pair_node["source"].as<std::string>();
                pair.target = pair_node["target"].as<std::string>();
                pair.transformation_file = pair_node["transformation_file"].as<std::string>();
                config.pairs.push_back(pair);
            }
        }

        if (verbose) {
            std::cout << "YAML configuration file loaded successfully: " << yaml_file << std::endl;
            std::cout << "Dataset name: " << config.dataset_name << std::endl;
            std::cout << "Number of pairs in the dataset: " << config.pairs.size() << std::endl;
        }
    }
    catch (const YAML::Exception& e) {
        std::cerr << "YAML parsing error: " << e.what() << std::endl;
        throw;
    }

    return config;
}

// Read the ground truth transformation matrix
Eigen::Matrix4d loadGroundTruthTransformation(const std::string& transformation_file) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();

    std::ifstream file(transformation_file);
    if (!file.is_open()) {
        std::cerr << "Unable to open transformation file: " << transformation_file << std::endl;
        return T;
    }

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            file >> T(i, j);
        }
    }
    file.close();

    return T;
}

// Calculate rotation error (degrees)
double calculateRotationError(const Eigen::Matrix3d& R_gt, const Eigen::Matrix3d& R_est) {
    Eigen::Matrix3d R_error = R_gt.transpose() * R_est;
    double trace = R_error.trace();
    double angle_rad = std::acos(std::max(-1.0, std::min(1.0, (trace - 1.0) / 2.0)));
    return angle_rad * 180.0 / M_PI; 
}

// Calculate the translation error (Euclidean distance)
double calculateTranslationError(const Eigen::Vector3d& t_gt, const Eigen::Vector3d& t_est) {
    return (t_gt - t_est).norm();
}

Eigen::Matrix4d integrate(const Eigen::Matrix2d& rotation_2d, const Eigen::Vector2d& translation_2d, double dz) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<2, 2>(0, 0) = rotation_2d;
    T.block<2, 1>(0, 3) = translation_2d;
    T(2, 3) = dz;
    return T;
}
