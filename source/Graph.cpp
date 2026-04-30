#include "Graph.h"



double TLIGraph::computeL2Distance(const Node& n1, const Node& n2) {
    double dx = n1.x - n2.x;
    double dy = n1.y - n2.y;
    return std::sqrt(dx * dx + dy * dy);
}

bool TLIGraph::computeIntersection(const std::tuple<double, double, double>& line1, const std::tuple<double, double, double>& line2, double& out_x, double& out_y) {
    const double a1 = std::get<0>(line1);
    const double b1 = std::get<1>(line1);
    const double c1 = std::get<2>(line1);
    const double a2 = std::get<0>(line2);
    const double b2 = std::get<1>(line2);
    const double c2 = std::get<2>(line2);
    const double det = a1 * b2 - a2 * b1;
    // If determinant is zero, the lines are parallel or coincident.
    const double eps = 1e-12;
    if (std::fabs(det) < eps) {
        return false;
    }
    // Solve for x and y using Cramer's rule.  Note that the
    // equations are written as a x + b y + c = 0, so we need to
    // rearrange to a x + b y = -c.
    // det != 0
    out_x = (b1 * (c2)-b2 * (c1)) / det;
    out_y = ((c1)*a2 - (c2)*a1) / det;
    return true;
}

double TLIGraph::computeAcuteAngleDeg(const std::tuple<double, double, double>& line1, const std::tuple<double, double, double>& line2) {
    const double a1 = std::get<0>(line1);
    const double b1 = std::get<1>(line1);
    // direction vector along line1
    const double dx1 = b1;
    const double dy1 = -a1;
    const double a2 = std::get<0>(line2);
    const double b2 = std::get<1>(line2);
    const double dx2 = b2;
    const double dy2 = -a2;
    // dot and cross products
    const double dot = dx1 * dx2 + dy1 * dy2;
    const double mag1 = std::sqrt(dx1 * dx1 + dy1 * dy1);
    const double mag2 = std::sqrt(dx2 * dx2 + dy2 * dy2);
    double cosTheta = 0.0;
    if (mag1 > 0 && mag2 > 0) {
        cosTheta = dot / (mag1 * mag2);
        // ensure numeric stability
        if (cosTheta > 1.0) cosTheta = 1.0;
        if (cosTheta < -1.0) cosTheta = -1.0;
    }
    // Take absolute value of cosine to get acute angle (<=90°)
    double theta = std::acos(std::fabs(cosTheta));
    return theta * 180.0 / M_PI;
}

std::pair<std::array<double, 2>, std::array<double, 2>> TLIGraph::computeLineDirections(const std::tuple<double, double, double>& line) {
    const double a = std::get<0>(line);
    const double b = std::get<1>(line);
    // A direction vector perpendicular to the normal (a,b) is (b,-a).
    const double dx = b;
    const double dy = -a;
    // We return the vector and its negative.
    return { {dx, dy}, {-dx, -dy} };
}



int TLIGraph::getForwardNodeIndex(const Node& currNode, Edge::Ptr currEdge) {
    return currNode.id == currEdge->u ? currEdge->v : currEdge->u;
}

int TLIGraph::getFromEdgeIndex(const Node& currNode, Edge::Ptr prevEdge) {
    int edgeIdx = 0;
    for (auto edge : currNode.edges) {
        if (edge != nullptr && edge == prevEdge) return edgeIdx;
        edgeIdx++;
    }
    return -1;
}

bool TLIGraph::isSimilarLength(double a, double b) {
    return std::abs(a - b) <= dist_tolerance;
}

void TLIGraph::matchAlongQueue(std::queue<NodePairWithFromEdge>& q, const std::vector<Node>& graph_src, const std::vector<Node>& graph_tgt, std::vector<bool>& srcVisited, std::vector<bool>& tgtVisited, std::unordered_set<NodePairWithOri, NodePairWithOri::Hash>& isMapped, std::vector<std::pair<int, int>>& match_result) {
    while (!q.empty()) {
        NodePairWithFromEdge nodePair = q.front();
        q.pop();

        int srcNodeId = nodePair.srcNode;
        int tgtNodeId = nodePair.tgtNode;
        int srcAlignedEdge = nodePair.srcFrom;
        int tgtAlignedEdge = nodePair.tgtFrom;

        for (int oriEdgeIdx = 1; oriEdgeIdx < 4; ++oriEdgeIdx) {
            int prevSrcNodeId = srcNodeId;
            int prevTgtNodeId = tgtNodeId;
            Edge::Ptr currSrcEdge = graph_src[prevSrcNodeId].edges[(oriEdgeIdx + srcAlignedEdge) % 4];
            Edge::Ptr currTgtEdge = graph_tgt[prevTgtNodeId].edges[(oriEdgeIdx + tgtAlignedEdge) % 4];
            if (currSrcEdge == nullptr || currTgtEdge == nullptr) continue;
            double totalSrcEdgeLen = currSrcEdge->length;
            double totalTgtEdgeLen = currTgtEdge->length;

            while (true) {
                const Node& prevSrcNode = graph_src[prevSrcNodeId];
                const Node& prevTgtNode = graph_tgt[prevTgtNodeId];
                const Node& currSrcNode = graph_src[getForwardNodeIndex(prevSrcNode, currSrcEdge)];
                const Node& currTgtNode = graph_tgt[getForwardNodeIndex(prevTgtNode, currTgtEdge)];

                int currSrcFromEdgeId = getFromEdgeIndex(currSrcNode, currSrcEdge);
                int currTgtFromEdgeId = getFromEdgeIndex(currTgtNode, currTgtEdge);

                if (isSimilarLength(totalSrcEdgeLen, totalTgtEdgeLen)
                    && (std::abs(currSrcNode.angle - currTgtNode.angle) <= angle_tolerance)
                    && ((currSrcFromEdgeId % 2 == currTgtFromEdgeId % 2) || (currSrcNode.is_right_angle && currTgtNode.is_right_angle))
                    ) {

                    if (tgtVisited[currTgtNode.id]) break;
                    srcVisited[currSrcNode.id] = true;
                    tgtVisited[currTgtNode.id] = true;

                    NodePairWithFromEdge nodeP;
                    nodeP.srcNode = currSrcNode.id;
                    nodeP.tgtNode = currTgtNode.id;
                    nodeP.srcFrom = currSrcFromEdgeId;
                    nodeP.tgtFrom = currTgtFromEdgeId;
                    q.push(nodeP);
                    match_result.emplace_back<std::pair<int, int>>({ currSrcNode.id, currTgtNode.id });
                    NodePairWithOri pairWithOri;
                    pairWithOri.srcNode = currSrcNode.id;
                    pairWithOri.tgtNode = currTgtNode.id;
                    int shift = (((currTgtFromEdgeId - currSrcFromEdgeId) % 4) + 4) % 4;
                    pairWithOri.ori = shift;
                    isMapped.insert(pairWithOri);

                    break;
                }
                else if (totalSrcEdgeLen < totalTgtEdgeLen) {
                    int fromEdgeId = currSrcFromEdgeId;
                    int forwardEdgeId = (fromEdgeId + 2) % 4;
                    currSrcEdge = currSrcNode.edges[forwardEdgeId];
                    if (currSrcEdge == nullptr) break;
                    prevSrcNodeId = currSrcNode.id;
                    //totalSrcEdgeLen += currSrcEdge->length;
                    totalSrcEdgeLen = computeL2Distance(
                        graph_src[getForwardNodeIndex(graph_src[prevSrcNodeId], currSrcEdge)],
                        graph_src[nodePair.srcNode]
                    );
                }
                else {
                    int fromEdgeId = currTgtFromEdgeId;
                    int forwardEdgeId = (fromEdgeId + 2) % 4;
                    currTgtEdge = currTgtNode.edges[forwardEdgeId];
                    if (currTgtEdge == nullptr) break;
                    prevTgtNodeId = currTgtNode.id;
                    //totalTgtEdgeLen += currTgtEdge->length;
                    totalSrcEdgeLen = computeL2Distance(
                        graph_tgt[getForwardNodeIndex(graph_tgt[prevTgtNodeId], currTgtEdge)],
                        graph_tgt[nodePair.tgtNode]
                    );
                }
            }
        }
    }
}

Eigen::MatrixXd buildDistanceMatrix(const std::vector<TLIGraph::Node>& graph) {
    std::size_t n = graph.size();
    Eigen::MatrixXd distanceMatrix(n, n);

    for (std::size_t i = 0; i < n; ++i) {
        distanceMatrix(i, i) = 0.0;

        for (std::size_t j = i + 1; j < n; ++j) {
            double dist = TLIGraph::computeL2Distance(graph[i], graph[j]);
            distanceMatrix(i, j) = dist;
            distanceMatrix(j, i) = dist;
        }
    }

    return distanceMatrix;
}



Eigen::MatrixXd computeDistanceSimilarity(
    const Eigen::MatrixXd& distance_matrix_src,
    const Eigen::MatrixXd& distance_matrix_tgt,
    const std::vector<std::pair<int, int>>& matches,
    const double res_param) {

    std::size_t m = matches.size();
    Eigen::MatrixXd distance_similarity(m, m);
    double sq = 2.0 * res_param * res_param;

    for (std::size_t i = 0; i < m; ++i) {
        distance_similarity(i, i) = -1.0;

        for (std::size_t j = i + 1; j < m; ++j) {
            int src_i = matches[i].first;
            int src_j = matches[j].first;

            int tgt_i = matches[i].second;
            int tgt_j = matches[j].second;

            double dist_src = distance_matrix_src(src_i, src_j);
            double dist_tgt = distance_matrix_tgt(tgt_i, tgt_j);
            double diff = std::abs(dist_src - dist_tgt);

            if (diff > res_param) {
                distance_similarity(i, j) = -1.0;
                distance_similarity(j, i) = -1.0;
            }
            else {
                double sim = std::exp(-(diff * diff) / sq);
                distance_similarity(i, j) = sim;
                distance_similarity(j, i) = sim;
            }
        }
    }

    return distance_similarity;
}

double calculateTotalWeight(const Eigen::MatrixXd &distance_similarity, const std::vector<int>& pair_inds) {
    double weight = 0.0;
    for (size_t i = 0; i < pair_inds.size(); ++i) {
        for (size_t j = i + 1; j < pair_inds.size(); ++j) {
            weight += distance_similarity(pair_inds[i], pair_inds[j]);
        }
    }
    return weight;
}

void bronKerbosch(std::set<int>& R, std::set<int>& P, std::set<int>& X,
    double &total_weight, std::vector<int> &pair_inds, const Eigen::MatrixXd &distance_similarity)
{
    // 如果 P 和 X 都為空，R 就是一個最大團
    if (P.empty() && X.empty()) {
        std::vector<int> curr_pairs(R.begin(), R.end());
        double weight = calculateTotalWeight(distance_similarity, curr_pairs);

        if (weight > total_weight) {
            pair_inds = curr_pairs;
            total_weight = weight;
        }
        return;
    }

    // 選擇 pivot（從 P ∪ X 中選擇度數最大的節點）
    int pivot = -1;
    int max_degree = -1;

    std::set<int> union_set = P;
    union_set.insert(X.begin(), X.end());

    for (int u : union_set) {
        int degree = 0;
        for (int v : P) {
            if ((distance_similarity(u, v) >= 0)) degree++;
        }
        if (degree > max_degree) {
            max_degree = degree;
            pivot = u;
        }
    }

    // 計算 P \ N(pivot)
    std::set<int> P_minus_N;
    if (pivot != -1) {
        for (int v : P) {
            if (!(distance_similarity(pivot, v) >= 0)) {
                P_minus_N.insert(v);
            }
        }
    }
    else {
        P_minus_N = P;
    }

    // 對 P \ N(pivot) 中的每個節點遞迴
    for (int v : P_minus_N) {
        // 計算 N(v) ∩ P（v 的鄰居與候選節點的交集）
        std::set<int> N_v_intersect_P;
        for (int u : P) {
            if (u != v && (distance_similarity(v, u) >= 0)) {
                N_v_intersect_P.insert(u);
            }
        }

        // 計算 N(v) ∩ X
        std::set<int> N_v_intersect_X;
        for (int u : X) {
            if ((distance_similarity(v, u) >= 0)) {
                N_v_intersect_X.insert(u);
            }
        }

        // R ∪ {v}
        R.insert(v);

        // 遞迴呼叫
        bronKerbosch(R, N_v_intersect_P, N_v_intersect_X, total_weight, pair_inds, distance_similarity);

        // 回溯
        R.erase(v);
        P.erase(v);
        X.insert(v);
    }
}

typename std::vector<TLIGraph::TCSubgraph> TLIGraph::filterSubgraphs(const std::vector<Node>& graph_src, const std::vector<Node>& graph_tgt, std::vector<std::vector<std::pair<int, int>>> subgraphs_match_pairs, int topK) {
    std::vector<TCSubgraph> tcsubgraphs;
    std::vector<double> weights;
    
    Eigen::MatrixXd distance_matrix_src = buildDistanceMatrix(graph_src);
    Eigen::MatrixXd distance_matrix_tgt = buildDistanceMatrix(graph_tgt);

    for (int sub_g_i = 0; sub_g_i < topK * 2; ++sub_g_i) {
        TCSubgraph subgraph;
        std::vector<std::pair<int, int>> &sub_g_match_pairs = subgraphs_match_pairs[sub_g_i];

        Eigen::MatrixXd distance_similarity = 
            computeDistanceSimilarity(distance_matrix_src, distance_matrix_tgt, sub_g_match_pairs, dist_tolerance);
        //std::cout << "pair_inds " << sub_g_match_pairs.size() << std::endl;
        //std::cout << distance_similarity << std::endl;

        double total_weight = 0.0;
        std::vector<int> pair_inds;

        std::set<int> R;
        std::set<int> P;
        std::set<int> X;

        for (int i = 0; i < subgraphs_match_pairs[sub_g_i].size(); ++i) {
            P.insert(i);
        }

        bronKerbosch(R, P, X, total_weight, pair_inds, distance_similarity);

        subgraph.distance_similarity.resize(pair_inds.size(), pair_inds.size());
        for (int i = 0; i < pair_inds.size(); ++i) {
            subgraph.node_pairs.emplace_back(sub_g_match_pairs[pair_inds[i]]);
            subgraph.distance_similarity(i, i) = -1.0;
            for (int j = i + 1; j < pair_inds.size(); ++j) {
                subgraph.distance_similarity(i, j) =
                subgraph.distance_similarity(j, i) =
                distance_similarity(pair_inds[i], pair_inds[j]);
            }
        }

        tcsubgraphs.emplace_back(subgraph);
        weights.emplace_back(total_weight);
    }

    std::vector<size_t> indices(tcsubgraphs.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [&](size_t i, size_t j) {
        return weights[i] > weights[j]; // 比較權重
    });

    std::vector<TCSubgraph> sorted_subgraphs;

    for (int i = 0; i < topK; ++i) {
        sorted_subgraphs.emplace_back(tcsubgraphs[indices[i]]);
    }

    return sorted_subgraphs;
}

bool TLIGraph::build(const std::vector<std::tuple<double, double, double>>& lines) {
    nodes.clear();
    nodes.reserve(lines.size() * lines.size());

    // Step 1: compute all valid intersections.
    for (int i = 0; i < static_cast<int>(lines.size()); ++i) {
        for (int j = i + 1; j < static_cast<int>(lines.size()); ++j) {
            double ix = 0.0, iy = 0.0;
            if (!computeIntersection(lines[i], lines[j], ix, iy)) {
                // Lines are parallel or coincident – skip
                continue;
            }
            // Filter by distance from origin
            const double dist = std::sqrt(ix * ix + iy * iy);
            if (dist > redius) {
                continue;
            }
            // Compute acute angle between the two lines
            const double ang = computeAcuteAngleDeg(lines[i], lines[j]);
            // Discard degenerate intersections (angle very small ~ parallel)
            const double smallAngleEps = 3.0;
            if (ang < smallAngleEps) {
                continue;
            }
            // Create node
            Node n;
            n.id = static_cast<int>(nodes.size());
            // We'll fill n.lines later based on orientation ordering – initially store i and j
            n.lines = { {i, j} };
            n.x = ix;
            n.y = iy;
            n.angle = ang;
            n.is_right_angle = (std::fabs(ang - 90.0) <= right_angle_eps);
            // edges array initialised to nullptrs automatically by default constructor of shared_ptr
            // store node
            nodes.push_back(n);
        }
    }

    // If no nodes were found, return early
    if (nodes.empty()) {
        std::cout << "false" << std::endl;
        return false;
    }

    // Step 2: build adjacency: group nodes by the lines they lie on.
    const int nLines = static_cast<int>(lines.size());
    // nodesOnLine[l] holds indices of nodes formed by line l
    std::vector<std::vector<int>> nodesOnLine(nLines);
    for (const Node& n : nodes) {
        nodesOnLine[n.lines[0]].push_back(n.id);
        nodesOnLine[n.lines[1]].push_back(n.id);
    }
    // adjacency list: for each node, store neighbour id and pointer to edge
    std::vector<std::vector<std::pair<int, std::shared_ptr<Edge>>>> adjacency(nodes.size());
    // store all edges created to keep them alive
    std::vector<std::shared_ptr<Edge>> allEdges;
    // For each line, sort the nodes along the line and connect consecutive nodes
    for (int l = 0; l < nLines; ++l) {
        auto& nodeIndices = nodesOnLine[l];
        if (nodeIndices.size() < 2) {
            continue;
        }
        // To sort along the line, compute a scalar projection for each node.
        // A direction vector for the line is (b, -a).
        const double a = std::get<0>(lines[l]);
        const double b = std::get<1>(lines[l]);
        const double dx = b;
        const double dy = -a;
        // Build a vector of pairs (t, nodeIndex)
        std::vector<std::pair<double, int>> tNode;
        tNode.reserve(nodeIndices.size());
        for (int idx : nodeIndices) {
            const Node& n = nodes[idx];
            // projection along (dx,dy).  We don't normalise since we only need ordering.
            const double t = n.x * dx + n.y * dy;
            tNode.emplace_back(t, idx);
        }
        // Sort by t
        std::sort(tNode.begin(), tNode.end(), [](const auto& a, const auto& b) {
            if (a.first < b.first) return true;
            if (a.first > b.first) return false;
            return a.second < b.second;
            });
        // Connect consecutive nodes
        for (std::size_t k = 0; k + 1 < tNode.size(); ++k) {
            const int u = tNode[k].second;
            const int v = tNode[k + 1].second;
            // Compute Euclidean distance
            const double dxUV = nodes[u].x - nodes[v].x;
            const double dyUV = nodes[u].y - nodes[v].y;
            const double distUV = std::sqrt(dxUV * dxUV + dyUV * dyUV);
            // Create and store edge
            auto e = std::make_shared<Edge>();
            e->u = u;
            e->v = v;
            e->length = distUV;
            e->line = l;
            allEdges.push_back(e);
            adjacency[u].emplace_back(v, e);
            adjacency[v].emplace_back(u, e);
        }
    }
    // Step 3: assign edges to each node's edges array in clockwise order and reorder lines array.
    for (Node& n : nodes) {
        const int nodeId = n.id;
        // Retrieve the two line indices (as stored initially) – they might be in any order.
        int lineA = n.lines[0];
        int lineB = n.lines[1];
        // Generate the four orientation vectors: two for each line.
        // Each orientation entry stores (angleDeg, lineIndex, orientationIndex)
        struct OrientationEntry {
            double angleDeg;
            int lineIndex;   // 0 for first line, 1 for second line
            std::array<double, 2> vec;
        };
        std::array<OrientationEntry, 4> orientation{};
        // Line A directions
        auto dirsA = computeLineDirections(lines[lineA]);
        orientation[0].vec = { dirsA.first[0], dirsA.first[1] };
        orientation[0].lineIndex = 0;
        orientation[1].vec = { dirsA.second[0], dirsA.second[1] };
        orientation[1].lineIndex = 0;
        // Line B directions
        auto dirsB = computeLineDirections(lines[lineB]);
        orientation[2].vec = { dirsB.first[0], dirsB.first[1] };
        orientation[2].lineIndex = 1;
        orientation[3].vec = { dirsB.second[0], dirsB.second[1] };
        orientation[3].lineIndex = 1;
        // Compute angles in degrees in [0,360) for each orientation
        for (auto& entry : orientation) {
            double ang = std::atan2(entry.vec[1], entry.vec[0]); // [-pi,pi]
            ang = ang * 180.0 / M_PI;
            if (ang < 0) {
                ang += 360.0;
            }
            entry.angleDeg = ang;
        }
        // Sort orientation indices by angle descending (clockwise order)
        std::array<int, 4> sortedIdx{ {0, 1, 2, 3} };
        std::sort(sortedIdx.begin(), sortedIdx.end(), [&](int a, int b) {
            // larger angle should come first for clockwise ordering
            return orientation[a].angleDeg > orientation[b].angleDeg;
            });
        // Create arrays of angles and line indices in the sorted (cw) order.
        std::array<double, 4> orientationAnglesCwRaw;
        std::array<int, 4> orientationLineIndicesRaw;
        for (int k = 0; k < 4; ++k) {
            orientationAnglesCwRaw[k] = orientation[sortedIdx[k]].angleDeg;
            // orientation.lineIndex: 0 for lineA, 1 for lineB
            orientationLineIndicesRaw[k] = orientation[sortedIdx[k]].lineIndex;
        }
        // Compute the clockwise sector angles between consecutive directions.  Because
        // we sorted in descending order, moving from index k to k+1 (mod 4)
        // represents turning clockwise.  Each sector angle should be either the
        // acute intersection angle or its supplement (180 - angle).  We'll use
        // these to decide where the acute angles should reside in the final
        // ordering.
        std::array<double, 4> sectorAngles;
        for (int k = 0; k < 4; ++k) {
            int next = (k + 1) % 4;
            // difference between angles[k] and angles[next] in clockwise sense
            double diff = orientationAnglesCwRaw[k] - orientationAnglesCwRaw[next];
            if (diff < 0) diff += 360.0;
            sectorAngles[k] = diff;
        }
        // Determine which rotation (shift) places the acute angles between
        // positions (0,1) and (2,3).  For non‑right angles there will be two
        // sectors equal (or very close) to the acute angle.  We look for a
        // shift such that sectorAngles[shift] and sectorAngles[(shift+2)%4] are
        // closest to the node's acute angle.  For right angles, any shift is
        // acceptable.
        int chosenShift = 0;
        //if (!n.is_right_angle) {
            // tolerance when matching the node's angle to sector angles.  Use a
            // small absolute tolerance relative to the node's acute angle.
        const double angleTol = std::max(1.0, n.angle * 0.1); // dynamic tolerance
        double bestError = std::numeric_limits<double>::max();
        for (int shift = 0; shift < 4; ++shift) {
            // compute errors for acute sectors at shift and shift+2
            double e1 = std::fabs(sectorAngles[shift] - n.angle);
            double e2 = std::fabs(sectorAngles[(shift + 2) % 4] - n.angle);
            double error = e1 + e2;
            if (error < bestError) {
                bestError = error;
                chosenShift = shift;
            }
        }
        //}
        // Rotate the orientation arrays by the chosen shift.  After rotation,
        // orientationAnglesCw[0] and [1] span the acute angle, and [2],[3] span
        // the other acute angle (for right angles this is arbitrary but
        // consistent).  The corresponding orientationLineIndices indicate which
        // original line (0 or 1) each direction belongs to.
        std::array<double, 4> orientationAnglesCw;
        std::array<int, 4> orientationLineIndices;
        std::array<int, 4> rotatedIdx;
        for (int k = 0; k < 4; ++k) {
            int src = (k + chosenShift) % 4;
            orientationAnglesCw[k] = orientationAnglesCwRaw[src];
            orientationLineIndices[k] = orientationLineIndicesRaw[src];
            rotatedIdx[k] = sortedIdx[src];
        }
        // Determine the two actual line identifiers after rotation.  The pair
        // (0,2) should correspond to one line and (1,3) to the other.  The
        // orientationLineIndices array still refers to 0/1 for lineA/lineB.
        // We map those back to the actual line indices.
        int sortedLine0 = (orientationLineIndices[0] == 0 ? lineA : lineB);
        int sortedLine1 = (orientationLineIndices[1] == 0 ? lineA : lineB);
        n.lines = { {sortedLine0, sortedLine1} };
        // Reset edges to null
        n.edges = { nullptr, nullptr, nullptr, nullptr };
        // Assign edges based on neighbour directions.  Use the rotated
        // orientationAnglesCw array to match neighbour direction to the
        // appropriate slot.
        for (const auto& adj : adjacency[nodeId]) {
            const int nbId = adj.first;
            const std::shared_ptr<Edge>& edge = adj.second;
            // Vector from this node to neighbour
            const double dxN = nodes[nbId].x - n.x;
            const double dyN = nodes[nbId].y - n.y;
            double angNb = std::atan2(dyN, dxN) * 180.0 / M_PI;
            if (angNb < 0) angNb += 360.0;
            // Find best matching orientation slot
            int bestIdx = -1;
            double bestDiff = std::numeric_limits<double>::max();
            for (int k = 0; k < 4; ++k) {
                double diff = std::fabs(angNb - orientationAnglesCw[k]);
                if (diff > 180.0) diff = 360.0 - diff;
                if (diff < bestDiff) {
                    bestDiff = diff;
                    bestIdx = k;
                }
            }
            if (bestIdx >= 0) {
                n.edges[bestIdx] = edge;
            }
        }
    }

    return true;
}

typename std::vector<TLIGraph::TCSubgraph> TLIGraph::match(const TLIGraph& other, int topK) {
    std::vector<std::vector<std::pair<int, int>>> subgraphs_match_pairs;
    if (nodes.empty() || other.nodes.empty()) {
        return std::vector<TLIGraph::TCSubgraph>();
    }

    // Build list of seeds based on angle similarity
    std::vector<Seed> seeds;
    seeds.reserve(nodes.size() * other.nodes.size());

    for (const Node& ns : nodes) {
        for (const Node& nt : other.nodes) {
            double diff = std::fabs(ns.angle - nt.angle);
            if (diff > angle_tolerance) {
                continue;
            }
            double score = std::exp(-(diff * diff) / (2.0 * angle_tolerance * angle_tolerance));
            seeds.push_back({ ns.id, nt.id, score });
        }
    }
    if (seeds.empty()) {
        return std::vector<TLIGraph::TCSubgraph>();
    }
    // Sort seeds by descending score
    std::sort(seeds.begin(), seeds.end(), [](const Seed& a, const Seed& b) {
        return a.score > b.score;
        });

    std::unordered_set<NodePairWithOri, NodePairWithOri::Hash> isMapped;
    //std::cout << "Pass 1" << std::endl;
    // Determine number of seeds to explore (top 10%)
    std::size_t numSeeds = nodes.size() * other.nodes.size();
    std::size_t cutoff = std::min(static_cast<std::size_t>(std::ceil(numSeeds * 0.2)), seeds.size());
    int numSeedsToUse = cutoff;   // static_cast<int>(std::ceil(seeds.size() * 0.99));
    if (numSeedsToUse < 1) numSeedsToUse = 1;
    // For each seed, run a BFS style growth
    for (int seedIdx = 0; seedIdx < numSeedsToUse; ++seedIdx) {
        const Seed& seed = seeds[seedIdx];
        const Node& seedSrcNode = nodes[seed.srcId];
        const Node& seedTgtNode = other.nodes[seed.tgtId];

        bool isRightAnglePair = seedSrcNode.is_right_angle && seedTgtNode.is_right_angle;
        int oriStep = isRightAnglePair ? 1 : 2;
        //std::cout << "Pass 2 " << seed.srcId << " " << seed.tgtId << std::endl;
        for (int shift = 0; shift < 4; shift += oriStep) {
            if (isMapped.count({ seedSrcNode.id, seedTgtNode.id, shift })) continue;

            std::queue<NodePairWithFromEdge> q;
            std::vector<bool> srcVisited(nodes.size(), false);
            std::vector<bool> tgtVisited(other.nodes.size(), false);
            std::vector<std::pair<int, int>> currMatchPairs;
            srcVisited[seedSrcNode.id] = true;
            tgtVisited[seedTgtNode.id] = true;
            currMatchPairs.emplace_back<std::pair<int, int>>({ seedSrcNode.id, seedTgtNode.id });

            for (int oriEdgeIdx = 0; oriEdgeIdx < 4; ++oriEdgeIdx) {
                int prevSrcNodeId = seedSrcNode.id;
                int prevTgtNodeId = seedTgtNode.id;
                Edge::Ptr currSrcEdge = nodes[prevSrcNodeId].edges[oriEdgeIdx];
                Edge::Ptr currTgtEdge = other.nodes[prevTgtNodeId].edges[(oriEdgeIdx + shift) % 4];
                if (currSrcEdge == nullptr || currTgtEdge == nullptr) continue;
                double totalSrcEdgeLen = currSrcEdge->length;
                double totalTgtEdgeLen = currTgtEdge->length;
                //std::cout << "Pass 3" << std::endl;

                while (true) {
                    const Node& prevSrcNode = nodes[prevSrcNodeId];
                    const Node& prevTgtNode = other.nodes[prevTgtNodeId];
                    const Node& currSrcNode = nodes[getForwardNodeIndex(prevSrcNode, currSrcEdge)];
                    const Node& currTgtNode = other.nodes[getForwardNodeIndex(prevTgtNode, currTgtEdge)];

                    int currSrcFromEdgeId = getFromEdgeIndex(currSrcNode, currSrcEdge);
                    int currTgtFromEdgeId = getFromEdgeIndex(currTgtNode, currTgtEdge);

                    if (isSimilarLength(totalSrcEdgeLen, totalTgtEdgeLen)
                        && (std::abs(currSrcNode.angle - currTgtNode.angle) <= angle_tolerance)
                        && ((currSrcFromEdgeId % 2 == currTgtFromEdgeId % 2) || (currSrcNode.is_right_angle && currTgtNode.is_right_angle))
                        ) {
                        srcVisited[currSrcNode.id] = true;
                        tgtVisited[currTgtNode.id] = true;

                        NodePairWithFromEdge nodePair;
                        nodePair.srcNode = currSrcNode.id;
                        nodePair.tgtNode = currTgtNode.id;
                        nodePair.srcFrom = currSrcFromEdgeId;
                        nodePair.tgtFrom = currTgtFromEdgeId;
                        q.push(nodePair);
                        currMatchPairs.emplace_back<std::pair<int, int>>({ currSrcNode.id, currTgtNode.id });

                        NodePairWithOri pairWithOri;
                        pairWithOri.srcNode = currSrcNode.id;
                        pairWithOri.tgtNode = currTgtNode.id;
                        int shift = (((currTgtFromEdgeId - currSrcFromEdgeId) % 4) + 4) % 4;
                        pairWithOri.ori = shift;
                        isMapped.insert(pairWithOri);

                        break;
                    }
                    else if (totalSrcEdgeLen < totalTgtEdgeLen) {
                        int fromEdgeId = currSrcFromEdgeId;
                        int forwardEdgeId = (fromEdgeId + 2) % 4;
                        currSrcEdge = currSrcNode.edges[forwardEdgeId];
                        if (currSrcEdge == nullptr) break;
                        prevSrcNodeId = currSrcNode.id;
                        // totalSrcEdgeLen += currSrcEdge->length;
                        totalSrcEdgeLen = computeL2Distance(
                            nodes[getForwardNodeIndex(nodes[prevSrcNodeId], currSrcEdge)],
                            seedSrcNode
                        );
                    }
                    else {
                        int fromEdgeId = currTgtFromEdgeId;
                        int forwardEdgeId = (fromEdgeId + 2) % 4;
                        currTgtEdge = currTgtNode.edges[forwardEdgeId];
                        if (currTgtEdge == nullptr) break;
                        prevTgtNodeId = currTgtNode.id;
                        //totalTgtEdgeLen += currTgtEdge->length;
                        totalTgtEdgeLen = computeL2Distance(
                            other.nodes[getForwardNodeIndex(other.nodes[prevTgtNodeId], currTgtEdge)],
                            seedTgtNode
                        );
                    }
                }
            }
            //std::cout << "Pass 4" << std::endl;
            matchAlongQueue(q, nodes, other.nodes, srcVisited, tgtVisited, isMapped, currMatchPairs);
            if (!currMatchPairs.empty()) {
                subgraphs_match_pairs.push_back(std::move(currMatchPairs));
            }
        }
    }
    // Sort results by descending number of matched nodes and keep top maxResults
    std::sort(subgraphs_match_pairs.begin(), subgraphs_match_pairs.end(), [](const auto& a, const auto& b) {
        return a.size() > b.size();
    });
    //if (subgraphs_match_pairs.size() > topK) {
    //    subgraphs_match_pairs.resize(topK);
    //}
    std::cout << "sub front: " << subgraphs_match_pairs.front().size() << std::endl;

    return filterSubgraphs(nodes, other.nodes, subgraphs_match_pairs, topK);
}








