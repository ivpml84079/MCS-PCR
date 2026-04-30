#include "LoopRegistration2D.h"

std::pair<Eigen::Matrix2d, Eigen::Vector2d> procrustes(const std::vector<Eigen::Vector2d>& X,
    const std::vector<Eigen::Vector2d>& Y,
    bool scaling = false,
    bool reflection = true) {

    int n = X.size();

    if (n != Y.size() || n == 0) {
        return std::pair<Eigen::Matrix2d, Eigen::Vector2d>();
    }

    // 轉換為 Eigen 矩陣
    Eigen::MatrixXd matX(n, 2);
    Eigen::MatrixXd matY(n, 2);

    for (int i = 0; i < n; i++) {
        matX.row(i) = X[i];
        matY.row(i) = Y[i];
    }

    // 計算質心
    Eigen::Vector2d muX = matX.colwise().mean();
    Eigen::Vector2d muY = matY.colwise().mean();

    // 中心化
    Eigen::MatrixXd X0 = matX.rowwise() - muX.transpose();
    Eigen::MatrixXd Y0 = matY.rowwise() - muY.transpose();

    // 計算平方和
    double ssqX = X0.squaredNorm();
    double ssqY = Y0.squaredNorm();

    // 檢查是否所有點都相同
    if (ssqX < 1e-10 || ssqY < 1e-10) {
        return std::make_pair(Eigen::Matrix2d::Identity(), muX);
    }

    // 標準化
    double normX = std::sqrt(ssqX);
    double normY = std::sqrt(ssqY);

    X0 /= normX;
    Y0 /= normY;

    // 計算最優旋轉矩陣
    Eigen::MatrixXd A = X0.transpose() * Y0;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

    //std::cout << "A: " << std::endl;
    //std::cout << A << std::endl;


    Eigen::Matrix2d T = svd.matrixV() * svd.matrixU().transpose();

    // 處理反射
    if (!reflection && T.determinant() < 0) {
        Eigen::Matrix2d V = svd.matrixV();
        V.col(1) *= -1;
        T = V * svd.matrixU().transpose();
    }

    // 計算跡
    double traceTA = (T * A).trace();

    // 計算縮放因子
    double b;
    if (scaling) {
        b = traceTA * normX / normY;
    }
    else {
        b = 1.0;
    }

    // 計算平移
    Eigen::RowVector2d muY_row = muY.transpose();
    Eigen::Vector2d c = muX - b * (muY_row * T).transpose();

    //result.T = T.transpose();  // 注意：需要轉置以匹配 MATLAB 的約定
    //result.b = b;
    //result.c = c;

    return std::make_pair(T.transpose(), c);
}

pcl::PointCloud<pcl::PointXY>::Ptr apply2DTransformation(
    const pcl::PointCloud<pcl::PointXY>::Ptr& points,
    const Eigen::Matrix2d& R,
    const Eigen::Vector2d& t
) {
    pcl::PointCloud<pcl::PointXY>::Ptr transformedPoints(new pcl::PointCloud<pcl::PointXY>);
    transformedPoints->resize(points->size());

    for (int i = 0; i < points->size(); i++) {
        Eigen::Vector2d point(points->at(i).x, points->at(i).y);
        Eigen::Vector2d transformedPoint = R * point + t;

        transformedPoints->at(i).x = transformedPoint.x();
        transformedPoints->at(i).y = transformedPoint.y();
    }

    return transformedPoints;
}

std::tuple<double, int, double>
evaluateRegistration(const pcl::PointCloud<pcl::PointXY>::Ptr& srcPoints,
    const pcl::PointCloud<pcl::PointXY>::Ptr& tgtPoints, double param_s) {

    const double distanceThreshold = param_s * 0.5;

    if (srcPoints->empty() || tgtPoints->empty()) {
        return std::make_tuple(0.0, 0, 0.0);
    }

    // 構建KD樹
    pcl::KdTreeFLANN<pcl::PointXY> kdtree;
    kdtree.setInputCloud(tgtPoints);

    int matchCount = 0;
    double totalDistance = 0.0;
    std::vector<double> matchDistances;

    // 對源點雲中的每個點找到最近的目標點
    for (int i = 0; i < srcPoints->size(); i++) {
        std::vector<int> pointIdxNKNSearch(1);
        std::vector<float> pointNKNSquaredDistance(1);

        if (kdtree.nearestKSearch(srcPoints->at(i), 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
            double distance = std::sqrt(pointNKNSquaredDistance[0]);

            if (distance < distanceThreshold) {
                matchCount++;
                matchDistances.push_back(distance);
                totalDistance += distance;
            }
        }
    }

    // 計算重疊比率
    double overlapRatio = static_cast<double>(matchCount) / srcPoints->size();

    // 計算得分
    double avgMatchDistance = matchDistances.empty() ?
        std::numeric_limits<double>::infinity() : totalDistance / matchDistances.size();

    double distanceScore = 1.0 / (1.0 + avgMatchDistance);
    double score = matchCount * overlapRatio * distanceScore;

    return std::make_tuple(score, matchCount, overlapRatio);
}

bool LoopBased2DRegister::regis(
	const pcl::PointCloud<pcl::PointXY>::Ptr points_2d_src, 
	const pcl::PointCloud<pcl::PointXY>::Ptr points_2d_tgt, 
	const TLIGraph& graph_src, const TLIGraph& graph_tgt, const std::vector<TLIGraph::TCSubgraph>& tcsubgraphs
) {
	std::vector<Eigen::Matrix2d> cand_2d_rotations;
	std::vector<Eigen::Vector2d> cand_2d_translations;

	for (int sub_g_i = 0; sub_g_i < tcsubgraphs.size(); ++sub_g_i) {
		TLIGraph::TCSubgraph subgraph = tcsubgraphs[sub_g_i];

		TCLoopFinder tcl_finder(subgraph.distance_similarity);
		std::vector<TCLoop> tcloops = tcl_finder.findTCLoops();

		for (auto &tcl : tcloops) {
			std::vector<Eigen::Vector2d> coor_src, coor_tgt;
			for (auto node_pair_ind : tcl.nodes) {
				auto [node_ind_src, node_ind_tgt] = subgraph.node_pairs[node_pair_ind];
                //std::cout << "loop ids: " << node_ind_src << " " << node_ind_tgt << std::endl;
				TLIGraph::Node node_src = graph_src.nodes[node_ind_src];
				TLIGraph::Node node_tgt = graph_tgt.nodes[node_ind_tgt];
				coor_src.emplace_back(node_src.x, node_src.y);
				coor_tgt.emplace_back(node_tgt.x, node_tgt.y);
			}
            //std::cout << std::endl << std::endl;

            auto [rotation_2d, translation_2d] = procrustes(coor_tgt, coor_src, false, false);
            cand_2d_rotations.emplace_back(rotation_2d);
            cand_2d_translations.emplace_back(translation_2d);
		}
	}

    int best_metch_count = -1;
    Eigen::Matrix2d best_rotation;
    Eigen::Vector2d best_translation;

    for (int i = 0; i < cand_2d_rotations.size(); ++i) {
        pcl::PointCloud<pcl::PointXY>::Ptr trans_2d_src = apply2DTransformation(points_2d_src, cand_2d_rotations[i], cand_2d_translations[i]);
        auto [score, match_count, overlap_ratio] = evaluateRegistration(trans_2d_src, points_2d_tgt, resolution);
        if (best_metch_count < match_count) {
            rotation_2d = cand_2d_rotations[i];
            translation_2d = cand_2d_translations[i];
            best_metch_count = match_count;
        }
    }

    if (best_metch_count == -1) return false;

    //std::cout << rotation_2d << std::endl;
    //std::cout << translation_2d << std::endl;

    return true;
}
