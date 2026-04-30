#pragma once

#include <pcl/types.h>
#include <pcl/pcl_config.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include "Graph.h"

class LoopBased2DRegister {
public:
	Eigen::Matrix2d rotation_2d;
	Eigen::Vector2d translation_2d;
	double resolution;

    LoopBased2DRegister(double res_param) : resolution(res_param) {}

	bool regis(const pcl::PointCloud<pcl::PointXY>::Ptr points_2d_src, 
		const pcl::PointCloud<pcl::PointXY>::Ptr points_2d_tgt, 
        const TLIGraph &graph_src, const TLIGraph& graph_tgt, 
		const std::vector<TLIGraph::TCSubgraph> &tcsubgraphs);

    const Eigen::Matrix2d& getRotation2D() const { return rotation_2d; }
    const Eigen::Vector2d& getTranslation2D() const { return translation_2d; }

    struct TCLoop {
        std::vector<int> nodes;
        double score;

        TCLoop(const std::vector<int>& n, double w) : nodes(n), score(w) {}

        bool operator<(const TCLoop& other) const {
            return score < other.score; // 用於max heap
        }
    };

    class TCLoopFinder {
    private:
        const Eigen::MatrixXd& graph;
        int n_vertices;

        // 計算clique的總權重
        double calculateLoopScore(const std::vector<int>& clique) const {
            double total_weight = 0.0;
            for (size_t i = 0; i < clique.size(); ++i) {
                for (size_t j = i + 1; j < clique.size(); ++j) {
                    total_weight += graph(clique[i], clique[j]);
                }
            }
            return total_weight;
        }

        // 使用回溯法找出所有k-clique
        void findTCLoopsRec(std::vector<int>& current_loop, int start,
            std::priority_queue<TCLoop>& remained_loops, int tcl_num) {
            if (current_loop.size() == 3) {
                double score = calculateLoopScore(current_loop);
                remained_loops.push(TCLoop(current_loop, score));

                // 如果超過N個，移除權重最小的
                if (remained_loops.size() > tcl_num) {
                    remained_loops.pop();
                }
                return;
            }

            for (int v = start; v < n_vertices; ++v) {
                // 檢查v是否與current中所有節點相連
                bool connected = true;
                for (int u : current_loop) {
                    if (graph(u, v) <= 0.0) {
                        connected = false;
                        break;
                    }
                }

                if (connected) {
                    current_loop.push_back(v);
                    findTCLoopsRec(current_loop, v + 1, remained_loops, tcl_num);
                    current_loop.pop_back();
                }
            }
        }

    public:
        TCLoopFinder(const Eigen::MatrixXd& g) : graph(g), n_vertices(g.rows()) {}

        // 找出 Topological Closed Loops
        std::vector<TCLoop> findTCLoops() {
            if (n_vertices < 3) {
                return std::vector<TCLoop>();
            }

            std::priority_queue<TCLoop> remained_loops;
            std::vector<int> current_loop;

            findTCLoopsRec(current_loop, 0, remained_loops, 3);

            // 將結果從heap轉換為vector（由大到小排序）
            std::vector<TCLoop> result;
            while (!remained_loops.empty()) {
                result.push_back(remained_loops.top());
                remained_loops.pop();
            }

            // 反轉以得到由大到小的順序
            std::reverse(result.begin(), result.end());

            return result;
        }
    };


	
};