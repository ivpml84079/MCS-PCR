#include "Projection.h"

bool IFVRProcessor::process(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, int verbose) {
    voxels_.clear();

    auto start = std::chrono::high_resolution_clock::now();

    // Divide the point cloud into voxels and initialize voxels_
    auto voxelize_bgn = std::chrono::high_resolution_clock::now();
    voxelizePointCloud(cloud, verbose);
    auto voxelize_end = std::chrono::high_resolution_clock::now();
    auto voxelize_duration = std::chrono::duration_cast<std::chrono::milliseconds>(voxelize_end - voxelize_bgn);

    // Using OpenMP parallel processing voxels
    auto valid_bgn = std::chrono::high_resolution_clock::now();
    std::vector<std::pair<Eigen::Vector3i, VoxelData>> valid_voxels;
    valid_voxels.reserve(voxels_.size());

#pragma omp parallel for
    for (auto& [voxel_idx, voxel_data] : voxels_) {
        if (voxel_data.points->size() < min_points_per_voxel_) {
            continue;
        }

        // Calculate normal vector and eigenvalue
        if (!computeNormalForVoxel(voxel_data)) {
            continue;
        }

        // Check if the voxel conforms to the plane.
        float min_eigenvalue = voxel_data.eigenvalues[2];  // Minimum eigenvalue

        if (30 * voxel_data.eigenvalues[2] < voxel_data.eigenvalues[1] && voxel_data.eigenvalues[0] < 6 * voxel_data.eigenvalues[1]) {
#pragma omp critical
            {
                valid_voxels.push_back(std::make_pair(voxel_idx, voxel_data));
            }
        }
    }

    // Update voxels_ to a valid voxels_
    voxels_.clear();
    for (const auto& [voxel_idx, voxel_data] : valid_voxels) {
        voxels_[voxel_idx] = voxel_data;
    }
    auto valid_end = std::chrono::high_resolution_clock::now();
    auto valid_duration = std::chrono::duration_cast<std::chrono::milliseconds>(valid_end - valid_bgn);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    if (verbose > 1) {
        std::cout << "IFVR processing time: " << duration.count() << " ms" << std::endl;
        std::cout << "\tVoxelizing time: " << voxelize_duration.count() << " ms" << std::endl;
        std::cout << "\tChecking for infeasible voxels time: " << valid_duration.count() << " ms" << std::endl;
    }

    return true;
}

bool IFVRProcessor::voxelizePointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, int verbose) {
    std::vector<std::unordered_map<Eigen::Vector3i, VoxelData, Vector3iHash>> thread_voxels(omp_get_max_threads());

#pragma omp parallel for
    for (int i = 0; i < cloud->points.size(); ++i) {
        const auto& point = cloud->points[i];
        int thread_id = omp_get_thread_num();

        Eigen::Vector3i voxel_idx(
            static_cast<int>(std::floor(point.x / voxel_size_)),
            static_cast<int>(std::floor(point.y / voxel_size_)),
            static_cast<int>(std::floor(point.z / voxel_size_))
        );

        thread_voxels[thread_id][voxel_idx].points->push_back(point);
    }
    for (const auto& thread_map : thread_voxels) {
        for (const auto& [voxel_idx, voxel_data] : thread_map) {
            if (voxel_data.points->size() > 0) {
                voxels_[voxel_idx].points->insert(
                    voxels_[voxel_idx].points->end(),
                    voxel_data.points->begin(),
                    voxel_data.points->end()
                );
            }
        }
    }

    if (verbose > 1) {
        std::cout << "Voxelization complete, totaling " << voxels_.size() << " voxels" << std::endl;
    }
}


bool IFVRProcessor::computeNormalForVoxel(VoxelData& voxel) {
    if (voxel.points->size() < 3) {
        return false;
    }

    // Calculate the center point
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*voxel.points, centroid);
    voxel.center = centroid.head<3>();

    // Use PCA to calculate the eigenvalues ​​and eigenvectors of the covariance matrix.
    pcl::PCA<pcl::PointXYZ> pca;
    pca.setInputCloud(voxel.points);

    // Obtaining eigenvalues ​​and eigenvectors
    voxel.eigenvalues = pca.getEigenValues();
    voxel.eigenvectors = pca.getEigenVectors();

    // The normal vector is the eigenvector corresponding to the smallest eigenvalue.
    voxel.normal = voxel.eigenvectors.col(2);

    // Ensure that the normal vectors point in the same direction (e.g., upwards).
    if (voxel.normal[2] < 0) {
        voxel.normal = -voxel.normal;
    }

    return true;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr IFVRProcessor::getFeasiblePoints(
    const Eigen::Vector3f& space_normal,  // x-y space
    float max_angle_degrees
) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    Eigen::Vector3f normalized_space_normal = space_normal.normalized();

    for (const auto& [voxel_idx, voxel_data] : voxels_) {
        float dot_product = voxel_data.normal.dot(normalized_space_normal);
        dot_product = std::min(std::max(dot_product, -1.0f), 1.0f);
        float normal_angle = std::acos(std::abs(dot_product));

        float plane_angle = (M_PI / 2.0) - normal_angle;
        float plane_angle_degrees = plane_angle * (180.0 / M_PI);

        if (plane_angle_degrees < max_angle_degrees) {
            *filtered_cloud += *voxel_data.points;
        }
    }

    return filtered_cloud;
}

const std::unordered_map<Eigen::Vector3i, VoxelData, Vector3iHash> IFVRProcessor::getVoxelsForZEstimation() {
    std::unordered_map<Eigen::Vector3i, VoxelData, Vector3iHash> filtered_voxels;

    // 將參考平面法向量標準化
    Eigen::Vector3f normalized_space_normal = Eigen::Vector3f(0.0f, 0.0f, 1.0f);

    // 計算保留的點數和voxel數
    int retained_voxel_count = 0;
    int retained_point_count = 0;

    for (const auto& [voxel_idx, voxel_data] : voxels_) {
        // 計算法向量與參考平面法向量的夾角
        float dot_product = std::abs(voxel_data.normal.dot(normalized_space_normal));

        // 確保dot_product在[-1, 1]範圍內，避免浮點誤差
        dot_product = std::min(std::max(dot_product, -1.0f), 1.0f);

        float angle = std::acos(dot_product);
        float angle_degrees = angle * (180.0f / M_PI);
        
        if (angle_degrees < 30.f) {
            filtered_voxels[voxel_idx] = voxel_data;
        }
    }

    return filtered_voxels;
}



int XH(const std::vector<std::vector<bool>>& image, int x, int y) {
    int sum = 0;
    // Neighborhood numbering: Starting from the east, clockwise
    int dx[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
    int dy[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };

    for (int i = 0; i < 4; i++) {
        int x1 = x + dx[2 * i];
        int y1 = y + dy[2 * i];
        int x2 = x + dx[2 * i + 1];
        int y2 = y + dy[2 * i + 1];
        int x3 = x + dx[(2 * i + 2) & 0b111];
        int y3 = y + dy[(2 * i + 2) & 0b111];
        int b_i = !image[x1][y1] && (image[x2][y2] || image[x3][y3]);
        sum += b_i;
    }
    return sum;
}

int n1(const std::vector<std::vector<bool>>& image, int x, int y) {
    int sum = 0;
    int dx[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
    int dy[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };

    for (int k = 0; k < 4; k++) {
        int x1 = x + dx[2 * k];
        int y1 = y + dy[2 * k];
        int x2 = x + dx[2 * k + 1];
        int y2 = y + dy[2 * k + 1];
        sum += (image[x1][y1] || image[x2][y2]);
    }
    return sum;
}

int n2(const std::vector<std::vector<bool>>& image, int x, int y) {
    int sum = 0;
    int dx[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
    int dy[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };

    for (int k = 0; k < 4; k++) {
        int x1 = x + dx[2 * k + 1];
        int y1 = y + dy[2 * k + 1];
        int x2 = x + dx[(2 * k + 2) & 0b111];
        int y2 = y + dy[(2 * k + 2) & 0b111];
        sum += (image[x1][y1] || image[x2][y2]);
    }
    return sum;
}

bool checkG3(const std::vector<std::vector<bool>>& image, int x, int y, bool first) {
    if (first) {
        return ((image[x - 1][y + 1] || image[x - 1][y] || !image[x + 1][y + 1]) && image[x][y + 1]) == 0;
    }
    else {
        return ((image[x + 1][y - 1] || image[x + 1][y] || !image[x - 1][y - 1]) && image[x][y - 1]) == 0;
    }
}

std::vector<std::pair<int, int>> thinIteration(const std::vector<std::vector<bool>>& image, bool first) {
    int rows = image.size();
    int cols = image[0].size();
    std::vector<std::pair<int, int>> pixelsToRemove;

    for (int x = 1; x < rows - 1; x++) {
        for (int y = 1; y < cols - 1; y++) {
            if (image[x][y]) {
                int xh = XH(image, x, y);
                int n1_p = n1(image, x, y);
                int n2_p = n2(image, x, y);
                int min_n_p = std::min(n1_p, n2_p);
                if (xh == 1 && min_n_p >= 2 && min_n_p <= 3 && checkG3(image, x, y, first)) {
                    pixelsToRemove.emplace_back(x, y);
                }
            }
        }
    }
    return pixelsToRemove;
}

std::vector<std::vector<bool>> thinning(const std::vector<std::vector<bool>>& input) {
    std::vector<std::vector<bool>> image = input;
    bool change = true;

    while (change) {
        change = false;
        auto pixelsToRemove1 = thinIteration(image, true);
        for (const auto& p : pixelsToRemove1) {
            image[p.first][p.second] = false;
        }
        change |= !pixelsToRemove1.empty();

        auto pixelsToRemove2 = thinIteration(image, false);
        for (const auto& p : pixelsToRemove2) {
            image[p.first][p.second] = false;
        }
        change |= !pixelsToRemove2.empty();
    }
    return image;
}

bool PointsPixelator::pixelize(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, int verbose) {
    image_map = ProjectionImage();
    image_map.pixelSize = pixel_size;
    image_map.thresholdCount = thresholdCount_;

    if (cloud->empty()) {
        std::cerr << "Warning: Input point cloud is empty." << std::endl;
        return false;
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Find the boundary of the point cloud in the x-y space.
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float max_x = -std::numeric_limits<float>::max();
    float max_y = -std::numeric_limits<float>::max();

    for (const auto& point : cloud->points) {
        min_x = std::min(min_x, point.x);
        min_y = std::min(min_y, point.y);
        max_x = std::max(max_x, point.x);
        max_y = std::max(max_y, point.y);
    }

    image_map.minPoint = Eigen::Vector2f(min_x, min_y);

    // Calculate image size
    image_map.width = static_cast<int>(std::ceil((max_x - min_x) / pixel_size));
    image_map.height = static_cast<int>(std::ceil((max_y - min_y) / pixel_size));

    // Initialize the grid datas
    image_map.binaryImage.resize(image_map.height, std::vector<bool>(image_map.width, false));
    image_map.pixelInfo.resize(image_map.height, std::vector<PixelInfoPtr>(image_map.width, nullptr));

    std::vector<std::vector<Eigen::Vector3f>> pixel_sums(
        image_map.height, std::vector<Eigen::Vector3f>(image_map.width, Eigen::Vector3f::Zero()));

    std::vector<std::vector<size_t>> pixel_counts(
        image_map.height, std::vector<size_t>(image_map.width, 0));

    // Assign points to the corresponding grid.
    for (int i = 0; i < cloud->points.size(); ++i) {
        const pcl::PointXYZ& point = cloud->points[i];

        // Calculate the grid coordinates of the points
        int gridX = static_cast<int>((point.x - min_x) / pixel_size);
        int gridY = static_cast<int>((point.y - min_y) / pixel_size);

        // Ensure the point is within the image area
        if (gridX >= 0 && gridX < image_map.width && gridY >= 0 && gridY < image_map.height) {
            // When a pixel with a certain value is encountered for the first time, a PixelInfo object is created
            if (pixel_counts[gridY][gridX] == 0) {
                image_map.pixelInfo[gridY][gridX] = std::make_shared<PixelInfo>();
            }

            pixel_counts[gridY][gridX]++;
            image_map.pixelInfo[gridY][gridX]->point_indices->push_back(i);

            pixel_sums[gridY][gridX] += Eigen::Vector3f(point.x, point.y, point.z);
        }
    }

    for (int y = 0; y < image_map.height; ++y) {
        for (int x = 0; x < image_map.width; ++x) {
            size_t count = pixel_counts[y][x];

            if (count > 0) {
                auto& pixelPtr = image_map.pixelInfo[y][x];
                pixelPtr->count = count;

                if (count >= thresholdCount_) {
                    pixelPtr->value = true;
                    image_map.binaryImage[y][x] = true;
                }

                pixelPtr->centroid = pixel_sums[y][x] / static_cast<float>(count);

                image_map.points2D->emplace_back(pixelPtr->centroid.x(), pixelPtr->centroid.y());
            }
        }
    }

    image_map.binaryImage = thinning(image_map.binaryImage);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (verbose > 1) {
        std::cout << "Pixelizing time: " << duration.count() << " ms" << std::endl;
    }

    return true;
}

typename PointsPixelator::ProjectionImage PointsPixelator::getProjectionImage() {
    return image_map;
}
