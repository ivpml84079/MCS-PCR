#include "ZEstimation.h"
#include <pcl/io/pcd_io.h>


#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

struct Segment {
    int l;              // inclusive
    int r;              // inclusive
    double mean;         // mean of gaps in this segment
};

struct ChangePointResult {
    std::vector<double> sorted_values;
    std::vector<double> gaps;
    std::vector<int> change_points;              // index in gaps
    std::vector<int> cut_indices;                // cut between sorted[i] and sorted[i+1]
    std::vector<std::vector<double>> clusters;
    std::vector<Segment> segments;
};

class SortedChangePointClustering {
public:
    static ChangePointResult run(const std::vector<double>& data,
        int min_seg_len = 2,
        double gain_threshold = 0.0f,
        double large_gap_factor = 3.0f) {
        ChangePointResult result;

        if (data.empty()) {
            return result;
        }

        // 1) sort
        result.sorted_values = data;
        std::sort(result.sorted_values.begin(), result.sorted_values.end());

        if (result.sorted_values.size() == 1) {
            result.clusters.push_back(result.sorted_values);
            return result;
        }

        // 2) compute adjacent gaps
        result.gaps.resize(result.sorted_values.size() - 1);
        for (size_t i = 0; i + 1 < result.sorted_values.size(); ++i) {
            result.gaps[i] = result.sorted_values[i + 1] - result.sorted_values[i];
        }

        // 3) prefix sums for SSE computation
        PrefixStat ps(result.gaps);

        // 4) binary segmentation on gaps
        std::vector<int> cps;
        binarySegmentation(result.gaps, ps, 0, (int)result.gaps.size() - 1,
            min_seg_len, gain_threshold, cps);

        std::sort(cps.begin(), cps.end());
        result.change_points = cps;

        // 5) build segments from change points
        result.segments = buildSegments(result.gaps, cps);

        // 6) identify large-gap segments => convert to cut points
        //    We use median gap as baseline, and a segment is "between-cluster"
        //    if its mean is much larger than the global median.
        double med_gap = median(result.gaps);
        if (med_gap <= 0.0f) {
            med_gap = 1e-6f;
        }

        std::vector<int> cuts;
        for (const auto& seg : result.segments) {
            if (seg.mean >= large_gap_factor * med_gap) {
                // this segment indicates a "large-gap region"
                // each gap index k means cut between sorted[k] and sorted[k+1]
                for (int k = seg.l; k <= seg.r; ++k) {
                    cuts.push_back(k);
                }
            }
        }

        // 避免過度切割：如果一整段大 gap 很長，只取其中最大的 gap 當切點
        result.cut_indices = compressCutsByLargestGapPerSegment(result.gaps, cuts);

        // 7) build clusters
        result.clusters = buildClusters(result.sorted_values, result.cut_indices);

        return result;
    }

private:
    struct PrefixStat {
        std::vector<double> prefix_sum;
        std::vector<double> prefix_sq;

        explicit PrefixStat(const std::vector<double>& x) {
            int n = (int)x.size();
            prefix_sum.assign(n + 1, 0.0);
            prefix_sq.assign(n + 1, 0.0);
            for (int i = 0; i < n; ++i) {
                prefix_sum[i + 1] = prefix_sum[i] + x[i];
                prefix_sq[i + 1] = prefix_sq[i] + (double)x[i] * x[i];
            }
        }

        double sum(int l, int r) const {
            return prefix_sum[r + 1] - prefix_sum[l];
        }

        double sqSum(int l, int r) const {
            return prefix_sq[r + 1] - prefix_sq[l];
        }

        int len(int l, int r) const {
            return r - l + 1;
        }

        double mean(int l, int r) const {
            return sum(l, r) / len(l, r);
        }

        double sse(int l, int r) const {
            double s = sum(l, r);
            double ss = sqSum(l, r);
            int n = len(l, r);
            return ss - s * s / n;
        }
    };

    static void binarySegmentation(const std::vector<double>& x,
        const PrefixStat& ps,
        int l, int r,
        int min_seg_len,
        double gain_threshold,
        std::vector<int>& cps) {
        int n = r - l + 1;
        if (n < 2 * min_seg_len) {
            return;
        }

        double parent_cost = ps.sse(l, r);

        int best_cp = -1;
        double best_gain = 0.0;

        for (int cp = l + min_seg_len - 1; cp <= r - min_seg_len; ++cp) {
            double left_cost = ps.sse(l, cp);
            double right_cost = ps.sse(cp + 1, r);
            double gain = parent_cost - (left_cost + right_cost);

            if (gain > best_gain) {
                best_gain = gain;
                best_cp = cp;
            }
        }

        if (best_cp != -1 && best_gain > gain_threshold) {
            cps.push_back(best_cp);
            binarySegmentation(x, ps, l, best_cp, min_seg_len, gain_threshold, cps);
            binarySegmentation(x, ps, best_cp + 1, r, min_seg_len, gain_threshold, cps);
        }
    }

    static std::vector<Segment> buildSegments(const std::vector<double>& gaps,
        const std::vector<int>& cps) {
        std::vector<Segment> segs;
        if (gaps.empty()) return segs;

        std::vector<int> sorted_cps = cps;
        std::sort(sorted_cps.begin(), sorted_cps.end());

        int start = 0;
        for (int cp : sorted_cps) {
            segs.push_back({ start, cp, mean(gaps, start, cp) });
            start = cp + 1;
        }
        segs.push_back({ start, (int)gaps.size() - 1, mean(gaps, start, (int)gaps.size() - 1) });
        return segs;
    }

    static double mean(const std::vector<double>& v, int l, int r) {
        double s = 0.0;
        for (int i = l; i <= r; ++i) s += v[i];
        return static_cast<double>(s / (r - l + 1));
    }

    static double median(std::vector<double> v) {
        if (v.empty()) return 0.0f;
        std::sort(v.begin(), v.end());
        size_t n = v.size();
        if (n % 2 == 1) return v[n / 2];
        return 0.5f * (v[n / 2 - 1] + v[n / 2]);
    }

    static std::vector<int> compressCutsByLargestGapPerSegment(const std::vector<double>& gaps,
        const std::vector<int>& rawCuts) {
        if (rawCuts.empty()) return {};

        std::vector<int> cuts = rawCuts;
        std::sort(cuts.begin(), cuts.end());

        std::vector<int> compressed;
        int group_start = cuts[0];
        int prev = cuts[0];

        for (size_t i = 1; i < cuts.size(); ++i) {
            if (cuts[i] == prev + 1) {
                prev = cuts[i];
            }
            else {
                compressed.push_back(argmaxGap(gaps, group_start, prev));
                group_start = cuts[i];
                prev = cuts[i];
            }
        }
        compressed.push_back(argmaxGap(gaps, group_start, prev));
        return compressed;
    }

    static int argmaxGap(const std::vector<double>& gaps, int l, int r) {
        int best = l;
        for (int i = l + 1; i <= r; ++i) {
            if (gaps[i] > gaps[best]) best = i;
        }
        return best;
    }

    static std::vector<std::vector<double>> buildClusters(const std::vector<double>& sorted_values,
        const std::vector<int>& cut_indices) {
        std::vector<std::vector<double>> clusters;
        if (sorted_values.empty()) return clusters;

        if (cut_indices.empty()) {
            clusters.push_back(sorted_values);
            return clusters;
        }

        std::vector<int> cuts = cut_indices;
        std::sort(cuts.begin(), cuts.end());

        int start = 0;
        for (int cut : cuts) {
            clusters.emplace_back(sorted_values.begin() + start,
                sorted_values.begin() + cut + 1);
            start = cut + 1;
        }
        clusters.emplace_back(sorted_values.begin() + start, sorted_values.end());
        return clusters;
    }
};



class MeanShift1D {
private:
    double bandwidth_;
    std::vector<double> seeds_;
    bool bin_seeding_;
    int min_bin_freq_;
    bool cluster_all_;
    int max_iter_;
    double stop_threshold_;

    // Results
    std::vector<double> cluster_centers_;
    std::vector<int> labels_;
    int n_iter_;

    // Sorted data for efficient range queries
    std::vector<double> sorted_data_;
    std::vector<int> sorted_indices_;

public:
    MeanShift1D(double bandwidth = -1.0,
        const std::vector<double>& seeds = {},
        bool bin_seeding = false,
        int min_bin_freq = 1,
        bool cluster_all = true,
        int max_iter = 300)
        : bandwidth_(bandwidth),
        seeds_(seeds),
        bin_seeding_(bin_seeding),
        min_bin_freq_(min_bin_freq),
        cluster_all_(cluster_all),
        max_iter_(max_iter),
        stop_threshold_(1e-3),
        n_iter_(0) {
    }

    // Prepare sorted data for efficient range queries
    void prepare_sorted_data(const std::vector<double>& data) {
        sorted_indices_.resize(data.size());
        std::iota(sorted_indices_.begin(), sorted_indices_.end(), 0);

        // Sort indices by data values
        std::sort(sorted_indices_.begin(), sorted_indices_.end(),
            [&data](int i, int j) { return data[i] < data[j]; });

        sorted_data_.resize(data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            sorted_data_[i] = data[sorted_indices_[i]];
        }
    }

    // Find all points within bandwidth using binary search (O(log n + k))
    std::vector<int> find_neighbors_1d(double center, double bandwidth) const {
        double lower_bound = center - bandwidth;
        double upper_bound = center + bandwidth;

        // Binary search for range [lower_bound, upper_bound]
        auto lower_it = std::lower_bound(sorted_data_.begin(), sorted_data_.end(), lower_bound);
        auto upper_it = std::upper_bound(sorted_data_.begin(), sorted_data_.end(), upper_bound);

        std::vector<int> neighbors;
        for (auto it = lower_it; it != upper_it; ++it) {
            int sorted_idx = std::distance(sorted_data_.begin(), it);
            neighbors.push_back(sorted_indices_[sorted_idx]);
        }

        return neighbors;
    }

    // Find K nearest neighbors using binary search
    std::vector<int> find_k_neighbors_1d(double center, int k) const {
        if (sorted_data_.empty()) return {};

        // Find closest point using binary search
        auto it = std::lower_bound(sorted_data_.begin(), sorted_data_.end(), center);
        int center_idx = std::distance(sorted_data_.begin(), it);

        // Adjust if we're at the boundary
        if (center_idx > 0 && (center_idx >= static_cast<int>(sorted_data_.size()) ||
            std::abs(sorted_data_[center_idx - 1] - center) < std::abs(sorted_data_[center_idx] - center))) {
            center_idx--;
        }
        if (center_idx >= static_cast<int>(sorted_data_.size())) {
            center_idx = static_cast<int>(sorted_data_.size()) - 1;
        }

        std::vector<std::pair<double, int>> candidates;

        // Expand around the center point
        int left = center_idx, right = center_idx;
        while (candidates.size() < static_cast<size_t>(k) &&
            (left >= 0 || right < static_cast<int>(sorted_data_.size()))) {

            bool take_left = false;
            if (left >= 0 && right < static_cast<int>(sorted_data_.size())) {
                take_left = std::abs(sorted_data_[left] - center) <= std::abs(sorted_data_[right] - center);
            }
            else if (left >= 0) {
                take_left = true;
            }

            if (take_left && left >= 0) {
                candidates.emplace_back(std::abs(sorted_data_[left] - center), sorted_indices_[left]);
                left--;
            }
            else if (right < static_cast<int>(sorted_data_.size())) {
                candidates.emplace_back(std::abs(sorted_data_[right] - center), sorted_indices_[right]);
                right++;
            }
        }

        // Sort by distance and take k closest
        std::sort(candidates.begin(), candidates.end());
        std::vector<int> neighbors;
        for (size_t i = 0; i < std::min(static_cast<size_t>(k), candidates.size()); ++i) {
            neighbors.push_back(candidates[i].second);
        }

        return neighbors;
    }

    // Calculate bandwidth using quantile of distances
    double estimate_bandwidth(const std::vector<double>& data,
        double quantile = 0.3,
        int n_samples = -1) const {
        std::vector<double> sample_data = data;

        // Subsample if requested
        if (n_samples > 0 && n_samples < static_cast<int>(data.size())) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::shuffle(sample_data.begin(), sample_data.end(), gen);
            sample_data.resize(n_samples);
        }

        std::vector<double> sorted_sample = sample_data;
        std::sort(sorted_sample.begin(), sorted_sample.end());

        std::vector<double> distances;
        int n_neighbors = std::max(1, static_cast<int>(sorted_sample.size() * quantile));

        for (size_t i = 0; i < sorted_sample.size(); ++i) {
            // For 1D data, the k-th nearest neighbor distance is simply
            // the distance to the k-th element in sorted order
            int left_idx = std::max(0, static_cast<int>(i) - n_neighbors);
            int right_idx = std::min(static_cast<int>(sorted_sample.size()) - 1,
                static_cast<int>(i) + n_neighbors);

            double max_dist = std::max(std::abs(sorted_sample[i] - sorted_sample[left_idx]),
                std::abs(sorted_sample[i] - sorted_sample[right_idx]));
            distances.push_back(max_dist);
        }

        if (distances.empty()) return 1.0;

        return std::accumulate(distances.begin(), distances.end(), 0.0) / distances.size();
    }

    // Generate bin seeds for faster initialization
    std::vector<double> get_bin_seeds(const std::vector<double>& data, double bin_size) const {
        if (bin_size <= 0) return data;

        std::map<int, int> bin_counts;

        // Bin the data points
        for (double val : data) {
            int bin_idx = static_cast<int>(std::round(val / bin_size));
            bin_counts[bin_idx]++;
        }

        // Select bins with sufficient frequency
        std::vector<double> bin_seeds;
        for (const auto& bin_pair : bin_counts) {
            if (bin_pair.second >= min_bin_freq_) {
                bin_seeds.push_back(bin_pair.first * bin_size);
            }
        }

        if (bin_seeds.empty() || bin_seeds.size() == data.size()) {
            return data;  // Fall back to using all data points
        }

        return bin_seeds;
    }

    // Single seed mean shift iteration
    std::pair<double, int> mean_shift_single_seed(double seed,
        const std::vector<double>& data,
        double bandwidth,
        int max_iter) const {
        double stop_thresh = stop_threshold_ * bandwidth;
        int iterations = 0;

        while (iterations < max_iter) {
            // Find points within bandwidth
            std::vector<int> neighbors = find_neighbors_1d(seed, bandwidth);

            if (neighbors.empty()) {
                break;
            }

            // Calculate new mean
            double old_seed = seed;
            double sum = 0.0;

            for (int idx : neighbors) {
                sum += data[idx];
            }
            seed = sum / neighbors.size();

            // Check convergence
            if (std::abs(seed - old_seed) <= stop_thresh) {
                break;
            }

            iterations++;
        }

        return std::make_pair(seed, iterations);
    }

    // Remove duplicate cluster centers
    std::vector<double> remove_duplicates(
        const std::vector<std::pair<double, int>>& centers_with_counts,
        double bandwidth) const {

        // Sort by count (descending), then by position for deterministic results
        std::vector<std::pair<double, int>> sorted_centers = centers_with_counts;
        std::sort(sorted_centers.begin(), sorted_centers.end(),
            [](const auto& a, const auto& b) {
                if (a.second != b.second) return a.second > b.second;
                return a.first < b.first;  // deterministic ordering
            });

        std::vector<bool> unique(sorted_centers.size(), true);

        for (size_t i = 0; i < sorted_centers.size(); ++i) {
            if (!unique[i]) continue;

            for (size_t j = i + 1; j < sorted_centers.size(); ++j) {
                if (unique[j] &&
                    std::abs(sorted_centers[i].first - sorted_centers[j].first) < bandwidth) {
                    unique[j] = false;
                }
            }
        }
        std::sort(sorted_centers.begin(), sorted_centers.end(),
            [](const auto& a, const auto& b) {
                if (a.second != b.second) return a.second > b.second;
                return a.first < b.first;  // deterministic ordering
            });
        std::vector<double> final_centers;
        for (size_t i = 0; i < sorted_centers.size(); ++i) {
            if (unique[i]) {
                final_centers.push_back(sorted_centers[i].first);
            }
        }

        // Sort final centers for consistent output
        // std::sort(final_centers.begin(), final_centers.end());
        return final_centers;
    }

    // Find closest cluster center for each point
    std::vector<int> assign_labels(const std::vector<double>& data,
        const std::vector<double>& centers,
        double bandwidth) const {
        std::vector<int> labels(data.size(), -1);

        if (centers.empty()) return labels;

        for (size_t i = 0; i < data.size(); ++i) {
            double min_distance = std::numeric_limits<double>::max();
            int closest_center = -1;

            for (size_t j = 0; j < centers.size(); ++j) {
                double distance = std::abs(data[i] - centers[j]);
                if (distance < min_distance) {
                    min_distance = distance;
                    closest_center = static_cast<int>(j);
                }
            }

            if (cluster_all_ || min_distance <= bandwidth) {
                labels[i] = closest_center;
            }
        }

        return labels;
    }

    // Main fit function
    void fit(const std::vector<double>& data) {
        if (data.empty()) {
            throw std::invalid_argument("Data cannot be empty");
        }

        // Prepare sorted data for efficient range queries
        prepare_sorted_data(data);

        double bandwidth = bandwidth_;
        if (bandwidth <= 0) {
            bandwidth = estimate_bandwidth(data);
            std::cout << "Estimated bandwidth: " << bandwidth << std::endl;
        }

        // Initialize seeds
        std::vector<double> seeds = seeds_;
        if (seeds.empty()) {
            if (bin_seeding_) {
                seeds = get_bin_seeds(data, bandwidth);
                std::cout << "Using " << seeds.size() << " bin seeds" << std::endl;
            }
            else {
                seeds = data;
                std::cout << "Using all " << seeds.size() << " data points as seeds" << std::endl;
            }
        }

        // Perform mean shift for each seed
        std::vector<std::pair<double, int>> centers_with_counts;
        int max_iterations = 0;

        for (double seed : seeds) {
            auto result = mean_shift_single_seed(seed, data, bandwidth, max_iter_);
            double center = result.first;
            int iterations = result.second;

            max_iterations = std::max(max_iterations, iterations);

            // Count points within bandwidth of this center
            std::vector<int> neighbors = find_neighbors_1d(center, bandwidth);
            if (!neighbors.empty()) {
                centers_with_counts.emplace_back(center, neighbors.size());
            }
        }

        n_iter_ = max_iterations;

        if (centers_with_counts.empty()) {
            throw std::runtime_error("No centers found. Try increasing bandwidth or changing seeds.");
        }

        // Remove duplicate centers
        cluster_centers_ = remove_duplicates(centers_with_counts, bandwidth);

        // Assign labels
        labels_ = assign_labels(data, cluster_centers_, bandwidth);
    }

    // Predict labels for new data
    std::vector<int> predict(const std::vector<double>& data) const {
        if (cluster_centers_.empty()) {
            throw std::runtime_error("Model not fitted. Call fit() first.");
        }

        return assign_labels(data, cluster_centers_, bandwidth_);
    }

    // Getters
    const std::vector<double>& get_cluster_centers() const {
        return cluster_centers_;
    }

    const std::vector<int>& get_labels() const {
        return labels_;
    }

    int get_n_iter() const {
        return n_iter_;
    }

    // Utility function to print results
    void print_results() const {
        std::cout << "Number of clusters: " << cluster_centers_.size() << std::endl;
        std::cout << "Max iterations: " << n_iter_ << std::endl;

        std::cout << "Cluster centers: [";
        for (size_t i = 0; i < cluster_centers_.size(); ++i) {
            std::cout << cluster_centers_[i];
            if (i < cluster_centers_.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
};


double ZEstimator::calculateNormalAngle(const Eigen::Vector3f& n1, const Eigen::Vector3f& n2) {
    double dot_product = n1.normalized().dot(n2.normalized());
    dot_product = std::max(-1.0, std::min(1.0, static_cast<double>(dot_product)));
    return std::acos(dot_product) * 180.0 / M_PI;
}

double ZEstimator::calculatePlaneDistance(const Eigen::Vector3f& center1, const Eigen::Vector3f& normal1, const Eigen::Vector3f& center2, const Eigen::Vector3f& normal2) {
    Eigen::Vector3f avg_xy_point;
    avg_xy_point.x() = (center1.x() + center2.x()) / 2.0f;
    avg_xy_point.y() = (center1.y() + center2.y()) / 2.0f;
    avg_xy_point.z() = 0.0f; // z設為0，稍後會調整

    // 計算這個點到兩個平面的z座標
    // 平面方程: normal · (point - center) = 0
    // 求解z: z = center.z() - (normal.x()*(x-center.x()) + normal.y()*(y-center.y())) / normal.z()

    float z1, z2;
    if (std::abs(normal1.z()) > 1e-6) {
        z1 = center1.z() - (normal1.x() * (avg_xy_point.x() - center1.x()) +
            normal1.y() * (avg_xy_point.y() - center1.y())) / normal1.z();
    }
    else {
        z1 = center1.z();
    }

    if (std::abs(normal2.z()) > 1e-6) {
        z2 = center2.z() - (normal2.x() * (avg_xy_point.x() - center2.x()) +
            normal2.y() * (avg_xy_point.y() - center2.y())) / normal2.z();
    }
    else {
        z2 = center2.z();
    }

    return z2 - z1;
}

void ZEstimator::transformVoxel(VoxelData& voxel, const Eigen::Matrix2d& rotation, const Eigen::Vector2d& translation) {
    Eigen::Vector2d center_2d(voxel.center.x(), voxel.center.y());
    Eigen::Vector2d transformed_center_2d = rotation * center_2d + translation;
    voxel.center.x() = transformed_center_2d.x();
    voxel.center.y() = transformed_center_2d.y();

    // 轉換normal（只考慮xy分量的旋轉）
    Eigen::Vector2d normal_2d(voxel.normal.x(), voxel.normal.y());
    Eigen::Vector2d transformed_normal_2d = rotation * normal_2d;
    voxel.normal.x() = transformed_normal_2d.x();
    voxel.normal.y() = transformed_normal_2d.y();
    voxel.normal.normalize();
}

void ZEstimator::buildTargetKDTree(const std::unordered_map<Eigen::Vector3i, VoxelData, Vector3iHash>& target_voxels, pcl::PointCloud<pcl::PointXYZ>::Ptr& target_centers, std::vector<Eigen::Vector3i>& target_keys) {
    target_centers.reset(new pcl::PointCloud<pcl::PointXYZ>());
    target_keys.clear();

    for (const auto& [key, voxel] : target_voxels) {
        pcl::PointXYZ point;
        point.x = voxel.center.x();
        point.y = voxel.center.y();
        point.z = 0;//voxel.center.z();
        target_centers->push_back(point);
        target_keys.push_back(key);
    }

    kdtree.setInputCloud(target_centers);
}

double ZEstimator::evaluate(const std::vector<double>& dzs, pcl::PointCloud<pcl::PointXYZ>::Ptr src, pcl::PointCloud<pcl::PointXYZ>::Ptr tgt, const Eigen::Matrix2d& rotation, const Eigen::Vector2d& translation) {
    if (dzs.empty()) return 0.0;
    if (dzs.size() == 1) return dzs[0];

    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    T.block<2, 2>(0, 0) = rotation.cast<float>();
    T.block<2, 1>(0, 3) = translation.cast<float>();

    pcl::PointCloud<pcl::PointXYZ>::Ptr tr_src_org(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*src, *tr_src_org, T);
    pcl::PointCloud<pcl::PointXYZ>::Ptr tr_src(new pcl::PointCloud<pcl::PointXYZ>);

    kdtree.setInputCloud(tgt);

    double resolution_sq = resolution * resolution;
    double dz = 0.0;
    int max_count = 0;
    for (int i = 0; i < 2; i++) {
        *tr_src = *tr_src_org;
        for (auto& point : tr_src->points) {
            point.z += dzs[i];
        }

        int count = 0;
        for (int src_i = 0; src_i < tr_src->size(); src_i++) {
            std::vector<int> pointIdxNKNSearch(1);
            std::vector<float> pointNKNSquaredDistance(1);

            if (kdtree.nearestKSearch(tr_src->at(src_i), 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {

                if (pointNKNSquaredDistance[0] < resolution_sq) {
                    count++;
                }
            }
        }

        if (count > max_count) {
            dz = dzs[i];
            max_count = count;
        }
    }

    return dz;
}

std::unordered_map<Eigen::Vector3i, VoxelData, Vector3iHash> ZEstimator::filterVoxelsForZEstimation(const std::unordered_map<Eigen::Vector3i, VoxelData, Vector3iHash>& voxels) {
    std::unordered_map<Eigen::Vector3i, VoxelData, Vector3iHash> filtered_voxels;

    // 將參考平面法向量標準化
    Eigen::Vector3f normalized_space_normal = Eigen::Vector3f(0.0f, 0.0f, 1.0f);

    // 計算保留的點數和voxel數
    int retained_voxel_count = 0;
    int retained_point_count = 0;

    for (const auto& [voxel_idx, voxel_data] : voxels) {
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

double ZEstimator::estimate(
    pcl::PointCloud<pcl::PointXYZ>::Ptr src, pcl::PointCloud<pcl::PointXYZ>::Ptr tgt, 
    const std::unordered_map<Eigen::Vector3i, VoxelData, Vector3iHash>& source_voxels, 
    const std::unordered_map<Eigen::Vector3i, VoxelData, Vector3iHash>& target_voxels, 
    const Eigen::Matrix2d& rotation, const Eigen::Vector2d& translation
) {

    std::unordered_map<Eigen::Vector3i, VoxelData, Vector3iHash> source_voxels_for_z = filterVoxelsForZEstimation(source_voxels);
    std::unordered_map<Eigen::Vector3i, VoxelData, Vector3iHash> target_voxels_for_z = filterVoxelsForZEstimation(target_voxels);
    std::cout << "src ground vx size: " << source_voxels_for_z.size() << std::endl;
    std::cout << "tgt ground vx size: " << target_voxels_for_z.size() << std::endl;

    // 1. 轉換source voxels
    std::unordered_map<Eigen::Vector3i, VoxelData, Vector3iHash> transformed_source = source_voxels_for_z;
    for (auto& [key, voxel] : transformed_source) {
        transformVoxel(voxel, rotation, translation);
    }

    // 2. 建立target voxels的KDTree
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_centers;
    std::vector<Eigen::Vector3i> target_keys;
    buildTargetKDTree(target_voxels_for_z, target_centers, target_keys);

    // 3. 建立對應關係表
    std::vector<Correspondence> correspondences;

    for (const auto& [source_key, source_voxel] : transformed_source) {
        // 使用KDTree進行半徑搜尋
        pcl::PointXYZ search_point;
        search_point.x = source_voxel.center.x();
        search_point.y = source_voxel.center.y();
        search_point.z = 0;

        std::vector<int> point_indices;
        std::vector<float> point_distances;

        // 進行半徑搜尋
        if (kdtree.radiusSearch(search_point, voxel_size, point_indices, point_distances) > 0) {
            // 將結果按距離排序（通常KDTree已經排序了，但為了確保）
            std::vector<std::pair<float, int>> distance_index_pairs;
            for (size_t i = 0; i < point_indices.size(); ++i) {
                distance_index_pairs.push_back({ point_distances[i], point_indices[i] });
            }
            std::sort(distance_index_pairs.begin(), distance_index_pairs.end());

            // 檢查法向量角度並建立對應關係
            for (const auto& [dist, idx] : distance_index_pairs) {
                const Eigen::Vector3i& target_key = target_keys[idx];
                const VoxelData& target_voxel = target_voxels_for_z.at(target_key);
                double angle = calculateNormalAngle(source_voxel.normal, target_voxel.normal);

                if (angle < 3.0) { // 角度差小於3度
                    Correspondence corr;
                    corr.source_key = source_key;
                    corr.target_key = target_key;
                    corr.source_center = source_voxel.center;
                    corr.target_center = target_voxel.center;
                    corr.source_normal = source_voxel.normal;
                    corr.target_normal = target_voxel.normal;
                    corr.distance = calculatePlaneDistance(source_voxel.center, source_voxel.normal,
                        target_voxel.center, target_voxel.normal);
                    correspondences.push_back(corr);
                    break; // 只取最近的一個符合條件的
                }
            }
        }
    }

    std::cout << "找到 " << correspondences.size() << " 個對應關係" << std::endl;

    // 4. 隨機採樣和距離標準差檢查
    std::vector<double> valid_distances;
    std::uniform_int_distribution<> dis(0, correspondences.size() - 1);

    for (int iter = 0; iter < 1000 && valid_distances.size() < 100; ++iter) {
        if (correspondences.size() < 5) break;

        // 隨機選擇5個對應關係
        std::vector<int> selected_indices;
        std::unordered_set<int> used_indices;

        while (selected_indices.size() < 5 && selected_indices.size() < correspondences.size()) {
            int idx = dis(rng);
            selected_indices.push_back(idx);
            //if (used_indices.find(idx) == used_indices.end()) {
            //    selected_indices.push_back(idx);
            //    used_indices.insert(idx);
            //}
        }

        // 計算這5個距離的標準差
        std::vector<double> sample_distances;
        for (int idx : selected_indices) {
            sample_distances.push_back(correspondences[idx].distance);
        }

        // 計算平均值
        double mean = 0.0;
        for (double d : sample_distances) {
            mean += d;
        }
        mean /= sample_distances.size();

        // 計算標準差
        double variance = 0.0;
        for (double d : sample_distances) {
            variance += (d - mean) * (d - mean);
        }
        variance /= sample_distances.size();
        double std_dev = std::sqrt(variance);

        // 如果標準差小於pr，存儲這些距離
        if (std_dev < resolution) {
            double m = 0.0;
            for (double d : sample_distances) {
                m += d;
            }
            valid_distances.push_back(m / 5);
            //std::cout << m / 5 << " ";
        }
    }

    std::cout << std::endl << "收集到 " << valid_distances.size() << " 個有效距離" << std::endl;

    //MeanShift1D mean_shift(resolution / 4, {}, true);
    //mean_shift.fit(valid_distances);
    //mean_shift.print_results();
    //evaluate(mean_shift.get_cluster_centers(), src, tgt, rotation, translation);

    auto result = SortedChangePointClustering::run(
        valid_distances,
        5,      // min_seg_len
        0.01,   // gain_threshold
        2.5     // large_gap_factor
    );

    std::vector<double> cluster_centers(result.clusters.size());
    std::cout << "cluster_centers num: " << result.clusters.size() << "[";
    for (int i = 0; i < result.clusters.size(); i++) {
        cluster_centers[i] = std::accumulate(result.clusters[i].begin(), result.clusters[i].end(), 0.) / result.clusters[i].size();
        std::cout << cluster_centers[i] << " ";
    }

    return evaluate(cluster_centers, source_voxels, target_voxels, rotation, translation);
    //return evaluate(mean_shift.get_cluster_centers(), source_voxels, target_voxels, rotation, translation);
    //return evaluate(mean_shift.get_cluster_centers(), src, tgt, rotation, translation);
    //mean_shift.get_cluster_centers();//cluster_centers;
}

double ZEstimator::evaluate(
    const std::vector<double>& dzs, 
    const std::unordered_map<Eigen::Vector3i, VoxelData, Vector3iHash>& src_voxels, 
    const std::unordered_map<Eigen::Vector3i, VoxelData, Vector3iHash>& tgt_voxels,
    const Eigen::Matrix2d& rotation, const Eigen::Vector2d& translation
) {

    if (dzs.empty()) return 0.0;
    if (dzs.size() == 1) return dzs[0];

    std::unordered_map<Eigen::Vector3i, VoxelData, Vector3iHash> tfm_src_voxels = src_voxels;
    for (auto& [key, voxel] : tfm_src_voxels) {
        transformVoxel(voxel, rotation, translation);
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr tgt_vxl_centroid(new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<Eigen::Vector3f> tgt_vxl_normal(tgt_voxels.size());
    tgt_vxl_centroid->points.resize(tgt_voxels.size());
    tgt_vxl_centroid->width = static_cast<uint32_t>(tgt_voxels.size());
    tgt_vxl_centroid->height = 1;
    tgt_vxl_centroid->is_dense = true;

    int vxl_i = 0;
    for (const auto& [key, voxel] : tgt_voxels) {
        tgt_vxl_centroid->points[vxl_i].getVector3fMap() = voxel.center;
        tgt_vxl_normal[vxl_i] = voxel.normal;
        vxl_i++;
    }
    pcl::io::savePCDFile(".\\results\\centroid_tgt.pcd", *tgt_vxl_centroid, true);


    pcl::PointCloud<pcl::PointXYZ>::Ptr src_vxl_centroid(new pcl::PointCloud<pcl::PointXYZ>);
    src_vxl_centroid->points.resize(tfm_src_voxels.size());
    src_vxl_centroid->width = static_cast<uint32_t>(tfm_src_voxels.size());
    src_vxl_centroid->height = 1;
    src_vxl_centroid->is_dense = true;

    vxl_i = 0;
    for (const auto& [key, voxel] : tfm_src_voxels) {
        src_vxl_centroid->points[vxl_i].x = voxel.center.x();
        src_vxl_centroid->points[vxl_i].y = voxel.center.y();
        src_vxl_centroid->points[vxl_i].z = voxel.center.z();
        vxl_i++;
    }
    pcl::io::savePCDFile(".\\results\\centroid_src.pcd", *src_vxl_centroid, true);

    kdtree.setInputCloud(tgt_vxl_centroid);

    double dz = 0.0;
    int max_count = 0;
    for (int i = 0; i < dzs.size(); i++) {
        int count = 0;
        for (const auto &[key, voxel] : tfm_src_voxels) {
            std::vector<int> pointIdxNKNSearch;
            std::vector<float> pointNKNSquaredDistance;

            pcl::PointXYZ centroid;
            centroid.getVector3fMap() = voxel.center;
            centroid.z += dzs[i];

            if (kdtree.radiusSearch(centroid, voxel_size, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
                //std::cout << "find.." << std::endl;
                for (int neb_i = 0; neb_i < pointIdxNKNSearch.size(); ++neb_i) {
                    if (calculateNormalAngle(voxel.normal, tgt_vxl_normal[pointIdxNKNSearch[neb_i]]) < 3.0) {
                        count++;
                        break;
                    }
                }
            }
        }

        if (count > max_count) {
            dz = dzs[i];
            max_count = count;
        }
    }

    return dz;
}




