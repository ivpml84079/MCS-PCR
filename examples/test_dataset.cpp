#include "Registration.h"

#include <chrono>
#include <iomanip>
#include <filesystem>

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cout << "Usage: ./test_dataset <dataset name> <downsample resolution> <SR rotation threshold> <SR translation threshold>" << std::endl;
        return -1;
    }

    // Load YAML config file
    DatasetConfig config;
    try {
        config = loadYamlConfig("./configs/" + std::string(argv[1]) + ".yaml");  //argv[1]
    }
    catch (const std::exception& e) {
        std::cerr << "Loading YAML config file fail: " << e.what() << std::endl;
        return -1;
    }

    std::vector<double> rotation_errors;
    std::vector<double> translation_errors;
    std::vector<bool> success;

    // Create result files
    std::string result_folder = "./reg_results/";
    std::filesystem::create_directories(result_folder + std::string(argv[1]));

    std::string results_file_name = result_folder + std::string(argv[1]) + "/registration_results.txt";
    std::ofstream results_file(results_file_name);
    results_file << "Pair_ID, Source, Target, Rotation_Error(deg), Translation_Error(m), Time(ms)" << std::endl;
    std::string trans_file_name = result_folder + std::string(argv[1]) + "/est_transforms.txt";
    std::ofstream trans_file(trans_file_name);
    trans_file.close();

    double rot_e_thres = std::stod(argv[3]);
    double trans_e_thres = std::stod(argv[4]);

    // Set parameter
    WHTTC::Param param;
    param.resolution = std::stod(std::string(argv[2]));
    WHTTC whttc(param);

    // Processing each point cloud pair
    for (size_t pair_idx = 0; pair_idx < config.pairs.size(); ++pair_idx) {
        const auto& pair = config.pairs[pair_idx];

        std::cout << "\n=== Processing point cloud pair " << pair_idx + 1 << "/" << config.pairs.size()
            << ": " << pair.source << " -> " << pair.target << " ===" << std::endl;

        // Complete file path
        std::string src_file_path = config.root + config.raw_data + pair.source;
        std::string tgt_file_path = config.root + config.raw_data + pair.target;
        std::string gt_file_path = config.root + config.groundtruth + pair.transformation_file;

        // Point cloud objects
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src_org(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tgt(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tgt_org(new pcl::PointCloud<pcl::PointXYZ>);

        // Load point cloud files
        std::cout << "Loading point clouds ..." << std::endl;
        if (pcl::io::loadPLYFile<pcl::PointXYZ>(src_file_path, *cloud_src_org) == -1) {
            std::cerr << "Load source point cloud fail: " << src_file_path << std::endl;
            continue;
        }

        if (pcl::io::loadPLYFile<pcl::PointXYZ>(tgt_file_path, *cloud_tgt_org) == -1) {
            std::cerr << "Load target point cloud fail: " << tgt_file_path << std::endl;
            continue;
        }

        // Downsample point cloud
        std::cout << "Downsampling point clouds ..." << std::endl;
        voxelBasedDownsample(cloud_src_org, cloud_src, param.resolution);
        voxelBasedDownsample(cloud_tgt_org, cloud_tgt, param.resolution);
        std::cout << "Source point cloud size: " << cloud_src->points.size() << " points" << std::endl;
        std::cout << "Target point cloud size: " << cloud_tgt->points.size() << " points" << std::endl;

        // Load ground truth file
        std::cout << "Loading ground truth ..." << std::endl;
        Eigen::Matrix4d T_gt = loadGroundTruthTransformation(gt_file_path);
        Eigen::Matrix3d R_gt = T_gt.block<3, 3>(0, 0);
        Eigen::Vector3d t_gt = T_gt.block<3, 1>(0, 3);

        std::cout << "Ground Truth Transformation:" << std::endl;
        std::cout << T_gt << std::endl;

        std::cout << "Executing our algorithm ..." << std::endl;
        try {
            if (whttc.regis(cloud_src, cloud_tgt, 0)) {
                Eigen::Matrix4d T_est = whttc.getEstTransformation();

                std::cout << "\nEstimated Transformation:" << std::endl;
                std::cout << T_est << std::endl;

                // Evaluate RE, TE and SR
                Eigen::Matrix3d R_est = T_est.block<3, 3>(0, 0);
                Eigen::Vector3d t_est = T_est.block<3, 1>(0, 3);

                double rot_error = calculateRotationError(R_gt, R_est);
                double trans_error = calculateTranslationError(t_gt, t_est);

                rotation_errors.push_back(rot_error);
                translation_errors.push_back(trans_error);
                success.push_back(rot_error <= rot_e_thres && trans_error <= trans_e_thres);

                std::cout << "\n=== Registration result ===" << std::endl;
                std::cout << "Rotation error: " << rot_error << " (deg)" << std::endl;
                std::cout << "Translation error: " << trans_error << " m" << std::endl;
                std::cout << "Runtime: " << whttc.getTime() << " s" << std::endl;

                trans_file.open(trans_file_name, std::ios::app);
                trans_file << std::setprecision(12) << T_est << std::endl;
                trans_file.close();

                results_file << pair_idx + 1 << ", " << pair.source << ", " << pair.target << ", "
                    << rot_error << ", " << trans_error << ", " << whttc.getTime() << std::endl;
            }
            else {
                Eigen::Matrix4d T_est;
                T_est << -1, -1, -1, -1,
                         -1, -1, -1, -1,
                         -1, -1, -1, -1,
                         -1, -1, -1, -1;
                std::cout << "\nFail." << std::endl;

                Eigen::Matrix3d R_est = T_est.block<3, 3>(0, 0);
                Eigen::Vector3d t_est = T_est.block<3, 1>(0, 3);

                double rot_error = -1;
                double trans_error = -1;

                rotation_errors.push_back(rot_error);
                translation_errors.push_back(trans_error);
                success.push_back(false);

                std::cout << "\n=== Registration result ===" << std::endl;
                std::cout << "Rotation error: " << rot_error << " (deg)" << std::endl;
                std::cout << "Translation error: " << trans_error << " m" << std::endl;
                std::cout << "Runtime: " << whttc.getTime() << " s" << std::endl;

                trans_file.open(trans_file_name, std::ios::app);
                trans_file << std::setprecision(12) << T_est << std::endl;
                trans_file.close();

                results_file << pair_idx + 1 << ", " << pair.source << ", " << pair.target << ", "
                    << rot_error << ", " << trans_error << ", " << -1 << std::endl;
            }
        }
        catch (const std::exception& e) {
            std::cerr << "An error occurred while processing point cloud pair:" << e.what() << std::endl;
            continue;
        }
    }

    // Final results
    if (!success.empty()) {
        double avg_rot_error = 0;
        double avg_trans_error = 0;
        for (int i = 0; i < success.size(); ++i) {
            if (success[i]) {
                avg_rot_error += rotation_errors[i];
                avg_trans_error += translation_errors[i];
            }
        }
        int total_suc_num = std::accumulate(success.begin(), success.end(), 0);
        avg_rot_error /= total_suc_num;
        avg_trans_error /= total_suc_num;

        std::cout << "\n=== Overall results ===" << std::endl;
        std::cout << "Number of successfully registration pairs: " << total_suc_num << "/" << config.pairs.size() << std::endl;
        std::cout << "Average rotation error: " << avg_rot_error << " (deg)" << std::endl;
        std::cout << "Average translation error: " << avg_trans_error << " m" << std::endl;
    }

    results_file.close();
    std::cout << "The results have been saved to:" << std::endl;
    std::cout << results_file_name << std::endl;
    std::cout << trans_file_name << std::endl << std::endl;

    return 0;

}
