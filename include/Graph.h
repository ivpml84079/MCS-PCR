#pragma once

#include <algorithm>
#include <numeric>
#include <set>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include <queue>
#include <unordered_set>
#include <Eigen/Dense>
#include <pcl/types.h>


class TLIGraph {
public:
	struct Edge {
		int u;             // identifier of one end node
		int v;             // identifier of the other end node
		double length = 0.0;   // Euclidean distance between the two nodes
		int line = -1;         // index of the line this edge belongs to
		using Ptr = std::shared_ptr<Edge>;
	};

	struct Node {
		int id = -1;
		std::array<int, 2> lines{ {-1, -1} };        // indices of the two lines
		double x = 0.0;
		double y = 0.0;
		double angle = 0.0;                        // acute angle between the lines in degrees
		bool is_right_angle = false;               // whether angle is approximately 90¢X
		std::array<std::shared_ptr<Edge>, 4> edges; // pointers to incident edges (cw order)
	};

	

	struct Seed {
		int srcId;
		int tgtId;
		double score;
	};

	//struct MatchNodes {
	//	std::vector<std::pair<int, int>> pairs;
	//};

	struct NodePairWithFromEdge {
		int srcNode, tgtNode;
		int srcFrom, tgtFrom;
	};

	struct NodePairWithOri {
		int srcNode, tgtNode;
		int ori;
		bool operator==(const NodePairWithOri& other) const {
			return srcNode == other.srcNode && tgtNode == other.tgtNode && ori == other.ori;
		}

		struct Hash {
			std::size_t operator()(const NodePairWithOri& pair) const noexcept {
				size_t hash_val = 0LLU;
				hash_val |= static_cast<long long int>(pair.srcNode) << 18;
				hash_val |= static_cast<long long int>(pair.tgtNode) << 4;
				hash_val |= static_cast<long long int>(pair.ori);
				return hash_val;
			}
		};
	};

	struct TCSubgraph {
		std::vector<std::pair<int, int>> node_pairs;
		Eigen::MatrixXd distance_similarity;
	};

	std::vector<Node> nodes;

	double resolution;
	double angle_tolerance;
	double redius;
	double right_angle_eps;
	double dist_tolerance;

	TLIGraph(double res_param, double ang_param, double fac_epsilon, double graph_redius): resolution(res_param), angle_tolerance(ang_param), dist_tolerance(fac_epsilon * res_param), redius(graph_redius), right_angle_eps(1) {}

	bool build(const std::vector<std::tuple<double, double, double>>& lines);

	typename std::vector<TLIGraph::TCSubgraph> match(const TLIGraph& graph, int topK);


public:
	static double computeL2Distance(const Node& n1, const Node& n2);

	static bool computeIntersection(const std::tuple<double, double, double>& line1,
		const std::tuple<double, double, double>& line2,
		double& out_x, double& out_y);

	static double computeAcuteAngleDeg(const std::tuple<double, double, double>& line1,
		const std::tuple<double, double, double>& line2);

	static std::pair<std::array<double, 2>, std::array<double, 2>>
		computeLineDirections(const std::tuple<double, double, double>& line);

// --------------------------------------------------------------------------------
	static int getForwardNodeIndex(const Node& currNode, Edge::Ptr currEdge);

	static int getFromEdgeIndex(const Node& currNode, Edge::Ptr prevEdge);

	bool isSimilarLength(double a, double b);

	void matchAlongQueue(
		std::queue<NodePairWithFromEdge>& q,
		const std::vector<Node>& graph_src,
		const std::vector<Node>& graph_tgt,
		std::vector<bool>& srcVisited,
		std::vector<bool>& tgtVisited,
		std::unordered_set<NodePairWithOri, NodePairWithOri::Hash>& isMapped,
		std::vector<std::pair<int, int>>& match_result
	);

	std::vector<TCSubgraph> filterSubgraphs(
		const std::vector<Node>& graph_src,
		const std::vector<Node>& graph_tgt,
		std::vector<std::vector<std::pair<int, int>>> subgraphs_match_pairs, int topK);
};