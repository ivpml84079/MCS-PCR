#include "Registration.h"

#include <omp.h>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <format>

int main(int argc, char** argv) {
    omp_set_num_threads(omp_get_max_threads());
    std::cout << "使用 " << omp_get_max_threads() << " 個執行緒" << std::endl;

    // 載入 YAML 配置
    DatasetConfig config;
    try {
        config = loadYamlConfig(".\\configs\\" + std::string(argv[1]) + ".yaml");  //argv[1]
    }
    catch (const std::exception& e) {
        std::cerr << "載入配置檔失敗: " << e.what() << std::endl;
        return -1;
    }

    // 誤差統計
    std::vector<double> rotation_errors;
    std::vector<double> translation_errors;
    std::vector<bool> success;

    // 創建結果輸出檔案
    std::ostringstream oss;
    oss << argv[3] << "_" << argv[4] << "_" << argv[5] << "_" << argv[6] << "\\";
    std::string result_folder = ".\\reg_results\\" + oss.str();
    std::filesystem::create_directories(result_folder + std::string(argv[1]));

    std::ofstream results_file(result_folder + std::string(argv[1]) + "\\registration_results.txt");
    results_file << "Pair_ID, Source, Target, Rotation_Error(deg), Translation_Error(m), Time(ms)" << std::endl;
    std::string trans_file_name = result_folder + std::string(argv[1]) + "\\est_transforms.txt";
    std::ofstream trans_file(trans_file_name);
    trans_file.close();

    WHTTC::Param param;
    param.resolution = std::stod(std::string(argv[2]));
    param.max_line_num = std::stoi(std::string(argv[3]));
    param.angle_tolerance = std::stod(std::string(argv[4]));
    param.fac_epsilon = std::stod(std::string(argv[5]));
    param.fac_tau = std::stod(std::string(argv[6]));
    WHTTC whttc(param);

    // 處理每個點雲配對
    for (size_t pair_idx = 0; pair_idx < config.pairs.size(); ++pair_idx) {
        const auto& pair = config.pairs[pair_idx];

        std::cout << "\n=== 處理點雲配對 " << pair_idx + 1 << "/" << config.pairs.size()
            << ": " << pair.source << " -> " << pair.target << " ===" << std::endl;


        // 構建完整檔案路徑
        std::string src_file_path = config.root + config.raw_data + pair.source;
        std::string tgt_file_path = config.root + config.raw_data + pair.target;
        std::string gt_file_path = config.root + config.groundtruth + pair.transformation_file;

        // 創建點雲物件
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src_org(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tgt(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tgt_org(new pcl::PointCloud<pcl::PointXYZ>);

        // 讀取點雲檔案
        if (pcl::io::loadPLYFile<pcl::PointXYZ>(src_file_path, *cloud_src_org) == -1) {
            std::cerr << "無法讀取 source 檔案: " << src_file_path << std::endl;
            continue;
        }

        if (pcl::io::loadPLYFile<pcl::PointXYZ>(tgt_file_path, *cloud_tgt_org) == -1) {
            std::cerr << "無法讀取 target 檔案: " << tgt_file_path << std::endl;
            continue;
        }

        // 讀取 ground truth
        Eigen::Matrix4d T_gt = loadGroundTruthTransformation(gt_file_path);  //    Eigen::Matrix4d::Identity();
        Eigen::Matrix3d R_gt = T_gt.block<3, 3>(0, 0);
        Eigen::Vector3d t_gt = T_gt.block<3, 1>(0, 3);

        std::cout << "Ground Truth Transformation:" << std::endl;
        std::cout << T_gt << std::endl;

        // 點雲下採樣
        voxelBasedDownsample(cloud_src_org, cloud_src, param.resolution);
        voxelBasedDownsample(cloud_tgt_org, cloud_tgt, param.resolution);

        std::cout << "Source 點雲大小: " << cloud_src->points.size() << " 點" << std::endl;
        std::cout << "Target 點雲大小: " << cloud_tgt->points.size() << " 點" << std::endl;

        try {
            if (whttc.regis(cloud_src, cloud_tgt, 2)) {
                // 構建完整的 transformation 矩陣
                Eigen::Matrix4d T_est = whttc.getEstTransformation();

                std::cout << "\nEstimated Transformation:" << std::endl;
                std::cout << T_est << std::endl;

                // 計算誤差
                Eigen::Matrix3d R_est = T_est.block<3, 3>(0, 0);
                Eigen::Vector3d t_est = T_est.block<3, 1>(0, 3);

                double rot_error = calculateRotationError(R_gt, R_est);
                double trans_error = calculateTranslationError(t_gt, t_est);

                rotation_errors.push_back(rot_error);
                translation_errors.push_back(trans_error);
                success.push_back(rot_error <= 5 && trans_error <= 2);

                std::cout << "\n=== 配準結果 ===" << std::endl;
                std::cout << "旋轉誤差: " << rot_error << " 度" << std::endl;
                std::cout << "平移誤差: " << trans_error << " 公尺" << std::endl;
                std::cout << "處理時間: " << whttc.getTime() << " s" << std::endl;

                trans_file.open(trans_file_name, std::ios::app);
                trans_file << std::setprecision(12) << T_est << std::endl;
                trans_file.close();

                // 寫入結果檔案
                results_file << pair_idx + 1 << ", " << pair.source << ", " << pair.target << ", "
                    << rot_error << ", " << trans_error << ", " << whttc.getTime() << std::endl;
            }
            else {
                Eigen::Matrix4d T_est;
                T_est << -1, -1, -1, -1,
                         -1, -1, -1, -1,
                         -1, -1, -1, -1,
                         -1, -1, -1, -1;
                std::cout << "\nEstimated Transformation:" << std::endl;
                std::cout << T_est << std::endl;

                // 計算誤差
                Eigen::Matrix3d R_est = T_est.block<3, 3>(0, 0);
                Eigen::Vector3d t_est = T_est.block<3, 1>(0, 3);

                double rot_error = -1;
                double trans_error = -1;

                rotation_errors.push_back(rot_error);
                translation_errors.push_back(trans_error);
                success.push_back(false);

                std::cout << "\n=== 配準結果 ===" << std::endl;
                std::cout << "旋轉誤差: " << rot_error << " 度" << std::endl;
                std::cout << "平移誤差: " << trans_error << " 公尺" << std::endl;
                std::cout << "處理時間: " << -1 << " s" << std::endl;

                trans_file.open(trans_file_name, std::ios::app);
                trans_file << std::setprecision(12) << T_est << std::endl;
                trans_file.close();

                // 寫入結果檔案
                results_file << pair_idx + 1 << ", " << pair.source << ", " << pair.target << ", "
                    << rot_error << ", " << trans_error << ", " << -1 << std::endl;
            }
        }
        catch (const std::exception& e) {
            std::cerr << "處理點雲配對時發生錯誤: " << e.what() << std::endl;
            continue;
        }
    }

    // 計算統計結果
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

        std::cout << "\n=== 整體統計結果 ===" << std::endl;
        std::cout << "成功處理的配對數: " << total_suc_num << "/" << config.pairs.size() << std::endl;
        std::cout << "平均旋轉誤差: " << avg_rot_error << " 度" << std::endl;
        std::cout << "平均平移誤差: " << avg_trans_error << " 公尺" << std::endl;
    }

    results_file.close();
    std::cout << "結果已保存至 registration_results.txt" << std::endl;

    return 0;


    /*
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src_org(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tgt(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tgt_org(new pcl::PointCloud<pcl::PointXYZ>);

    std::string src_file_path = "E:\\Dataset\\5-Park\\1-RawPointCloud\\1.ply";
    std::string tgt_file_path = "E:\\Dataset\\5-Park\\1-RawPointCloud\\2.ply";

    // 讀取點雲檔案
    if (pcl::io::loadPLYFile<pcl::PointXYZ>(src_file_path, *cloud_src_org) == -1) {
        std::cerr << "無法讀取 source 檔案: " << src_file_path << std::endl;
    }

    if (pcl::io::loadPLYFile<pcl::PointXYZ>(tgt_file_path, *cloud_tgt_org) == -1) {
        std::cerr << "無法讀取 target 檔案: " << tgt_file_path << std::endl;
    }

    WHTTC::Param param;
    float voxel_size_down = param.resolution * 0.5;

    voxelBasedDownsample(cloud_src_org, cloud_src, voxel_size_down);
    voxelBasedDownsample(cloud_tgt_org, cloud_tgt, voxel_size_down);

    WHTTC whttc;
    whttc.regis(cloud_src, cloud_tgt, 2);
    std::cout << whttc.getEstTransformation() << std::endl;
    std::cout << "Total runtime: " << whttc.getTime() << " s" << std::endl;
    */
}