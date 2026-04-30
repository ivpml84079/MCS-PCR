#include "LineDetection.h"

std::tuple<double, double, double> leastSquaresLine(const std::vector<Eigen::Vector2f>& points) {
    // 檢查是否有足夠的點
    if (points.size() < 2) {
        std::cerr << "Error: At least 2 points are required for line fitting." << std::endl;
        return std::make_tuple(0.0, 0.0, 0.0);
    }

    // 計算質心
    Eigen::Vector2f centroid = Eigen::Vector2f::Zero();
    for (const auto& point : points) {
        centroid += point;
    }
    centroid /= static_cast<float>(points.size());

    // 構建去中心化點陣列
    Eigen::MatrixXd centered(points.size(), 2);
    for (size_t i = 0; i < points.size(); ++i) {
        centered(i, 0) = points[i].x() - centroid.x();
        centered(i, 1) = points[i].y() - centroid.y();
    }

    // 計算協方差矩陣
    Eigen::Matrix2d covariance = centered.transpose() * centered / static_cast<double>(points.size());

    // 特徵值分解
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eig(covariance);
    Eigen::Vector2d eigenvalues = eig.eigenvalues();
    Eigen::Matrix2d eigenvectors = eig.eigenvectors();

    // 選擇具有最大特徵值的特徵向量
    int maxIndex = eigenvalues(0) > eigenvalues(1) ? 0 : 1;
    double a = eigenvectors(1, maxIndex);  // y 分量
    double b = -eigenvectors(0, maxIndex); // x 分量

    // 標準化方向向量
    double norm = std::sqrt(a * a + b * b);
    a /= norm;
    b /= norm;

    // 計算 c
    double c = -(a * centroid.x() + b * centroid.y());

    // 確保 a 為正，標準化表示
    if (a < 0 || (a == 0 && b < 0)) {
        a = -a;
        b = -b;
        c = -c;
    }

    return std::make_tuple(a, b, c);
}

Eigen::Vector2f calculateCentroid(const std::vector<Eigen::Vector2f>& points) {
    // 檢查輸入點集是否為空
    if (points.empty()) {
        std::cerr << "Error: Point set is empty!" << std::endl;
        return Eigen::Vector2f::Zero();
    }

    // 初始化質心
    Eigen::Vector2f centroid = Eigen::Vector2f::Zero();

    // 累加所有點的座標
    for (const auto& point : points) {
        centroid += point;
    }

    // 計算平均值
    centroid /= static_cast<float>(points.size());

    return centroid;
}

double pointToLineDistance(const Eigen::Vector2f& point, const std::tuple<double, double, double>& lineCoeffs) {
    double a = std::get<0>(lineCoeffs);
    double b = std::get<1>(lineCoeffs);
    double c = std::get<2>(lineCoeffs);

    return std::abs(a * point.x() + b * point.y() + c) / std::sqrt(a * a + b * b);
}

bool LineDetector::detect(const pcl::PointCloud<pcl::PointXYZ>::Ptr feasible_points, PointsPixelator::ProjectionImage& proj_image, int verbose) {

    line_segments.clear();
    extractLineSegments(proj_image);
    //std::cout << line_segments.size() << std::endl;

    lines_coefs_.clear();
    lines_points_.clear();
    lines_centroid_.clear();
    lines_points_num_.clear();

    std::vector<std::pair<int, int>> pixels_in_line_segments, nxt_pixels_in_line_segments;
    for (size_t i = 0; i < line_segments.size(); ++i) {
        for (const auto& pixel : line_segments[i].pixels) {
            auto [y, x] = pixel;
            pixels_in_line_segments.push_back({ y, x });
            proj_image.pixelInfo[y][x]->line_seg_idx = i;
        }
    }
    pixel_in_segs = pixels_in_line_segments.size();

    pcl::PointCloud<pcl::PointXYZ>::Ptr points_in_lines(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr points_in_each_pixels(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(feasible_points);
    WHTransform houghTransform(proj_image.width * resolution, proj_image.height * resolution, resolution);
    for (size_t i = 0; i < max_line_num; ++i) {
        houghTransform.transform(proj_image, line_segments, pixels_in_line_segments, 100);
        /*if (verbose_) {
            std::cout << "Iter " << i << ":" << std::endl;
            std::cout << "Number of Hough Lines: " << houghTransform.getHoughLines().size() << std::endl;
        }*/

        if (houghTransform.getHoughLines().empty()) break;
        // if (houghTransform.getBestPixels().empty()) break;

        points_in_lines->clear();
        points_in_each_pixels->clear();

        // for (const auto& [y, x] : houghTransform.getBestPixels()) {
        for (const auto& [y, x] : houghTransform.bestLinePixel) {
            extract.setIndices(proj_image.pixelInfo[y][x]->point_indices);
            extract.setNegative(false); // 提取匹配點
            extract.filter(*points_in_each_pixels);
            *points_in_lines += *points_in_each_pixels;
        }

        std::vector<Eigen::Vector2f> points2D, nextPoints2D;
        for (const auto& point : *points_in_lines) {
            points2D.emplace_back(Eigen::Vector2f(point.x, point.y));
        }
        auto lsLine = leastSquaresLine(points2D);
        for (size_t i = 2; i < 4; ++i) {
            for (const auto& point : points2D) {
                if (pointToLineDistance(point, lsLine) < resolution / i) {
                    nextPoints2D.emplace_back(point);
                }
            }
            points2D = nextPoints2D;
            nextPoints2D.clear();
            lsLine = leastSquaresLine(points2D);
        }

        auto [a, b, c] = lsLine;
        lines_coefs_.push_back(lsLine);
        lines_centroid_.push_back(calculateCentroid(points2D));
        lines_points_num_.push_back(points2D.size());
        lines_points_.push_back(std::vector<Eigen::Vector2f>());
        for (const auto& point : *points_in_lines) {
            lines_points_.back().emplace_back(Eigen::Vector2f(point.x, point.y));
        }
        nxt_pixels_in_line_segments.clear();
        for (const auto& [y, x] : pixels_in_line_segments) {
            if (pointToLineDistance(proj_image.pixelInfo[y][x]->centroid.head(2), lsLine) > resolution) {
                nxt_pixels_in_line_segments.emplace_back(std::make_pair(y, x));
            }
        }

        pixels_in_line_segments = nxt_pixels_in_line_segments;
    }
    angleCounter = houghTransform.angleCounter;

	return true;
}

float calculateAngleBtwnVectors(const Eigen::Vector2f& v1, const Eigen::Vector2f& v2) {
    float dot = v1.dot(v2);
    float norm1 = v1.norm();
    float norm2 = v2.norm();

    if (norm1 < 1e-6 || norm2 < 1e-6) {
        return 0.0f; // 避免除以零
    }

    // 確保點積在有效範圍內（-1 到 1）
    dot = std::max(-1.0f, std::min(1.0f, dot / (norm1 * norm2)));
    return std::acos(dot);
}

float vectorOrientation(const Eigen::Vector2f& vec) {
    // 使用 atan2 計算與 x 軸夾角 (範圍 -180 到 180)
    float theta = atan2(vec.y(), vec.x());

    // 將角度範圍限制在 -90 到 90
    float angle = theta * 180.0 / M_PI;

    // 如果角度超過 90 或小於 -90，轉換到對稱範圍
    if (angle > 90.0) {
        angle = -180.0f + angle;
    }
    else if (angle < -90.0) {
        angle = 180.0 + angle;
    }
    angle = std::max(-90.0f, std::min(90.0f, angle));

    return angle;
}

float calculateMean(const std::vector<float>& data) {
    float sum = 0.0f;
    for (const auto& value : data) {
        sum += value;
    }
    return sum / data.size();
}

float calculateStd(const std::vector<float>& data) {
    float mean = calculateMean(data);
    float sum = 0.0f;
    for (const auto& value : data) {
        sum += (value - mean) * (value - mean);
    }
    return std::sqrt(sum / data.size());  // 樣本標準差
}

Eigen::Vector2f directionVectorMean(const std::vector<Eigen::Vector2f>& directionVectors, std::vector<int> directionWeights) {
    Eigen::Vector2f meanVector(0.f, 0.f);
    int weightSum = 0;
    for (size_t i = 0; i < directionVectors.size(); ++i) {
        meanVector += directionWeights[i] * directionVectors[i];
    }
    meanVector.normalize();
    return meanVector;
}

float directionVectorStd(const std::vector<Eigen::Vector2f>& directionVectors, std::vector<int> directionWeights) {
    Eigen::Vector2f meanVector(0.f, 0.f);
    int weightSum = 0;
    for (size_t i = 0; i < directionVectors.size(); ++i) {
        meanVector += directionWeights[i] * directionVectors[i];
        weightSum += directionWeights[i];
    }
    meanVector.normalize();
    //std::cout << "Mean Dir Vector: " << meanVector.transpose() << std::endl;
    float sum = 0.0f;
    for (size_t i = 0; i < directionVectors.size(); ++i) {
        float diff = calculateAngleBtwnVectors(directionVectors[i], meanVector) * 180.f / M_PI;
        sum += directionWeights[i] * diff * diff;
    }
    return std::sqrt(sum / weightSum);  // 樣本標準差
}

bool LineDetector::extractLineSegments(PointsPixelator::ProjectionImage& proj_image, int minLineLength, float maxAngleDeviation) {
    int height = proj_image.height;
    int width = proj_image.width;

    // 複製二值圖像，避免修改原始數據
    std::vector<std::vector<bool>> valid_image_copy = proj_image.binaryImage;

    // 標記已訪問的像素
    std::vector<std::vector<bool>> visited(height, std::vector<bool>(width, false));

    // 定義8個方向的偏移量（順時針方向）
    const int dx[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
    const int dy[8] = { 0, 1, 1, 1, 0, -1, -1, -1 };
    const int inv_dx[8] = { -1, 0, 1, 1, 1, 0, -1, -1 };
    const int inv_dy[8] = { 1, 1, 1, 0, -1, -1, -1, 0 };

    // 將搜索一條線的函數定義在這裡，方便訪問外部變量
    auto growLine = [&](int startY, int startX) -> LineSegment {
        LineSegment line;
        std::vector<Eigen::Vector2f> directionVectors;
        std::vector<int> directionWeights;

        // 起點一定是一個前景像素
        if (!valid_image_copy[startY][startX]) {
            return line;
        }

        // 加入起點
        line.pixels.push_back({ startY, startX });
        visited[startY][startX] = true;
        valid_image_copy[startY][startX] = false; // 從複製的圖像中移除

        // 如果起點有質心資訊，記錄為線段起點
        if (proj_image.pixelInfo[startY][startX]) {
            line.startPoint = proj_image.pixelInfo[startY][startX]->centroid;
        }

        // 使用BFS或DFS擴展線段
        // 這裡使用queue實現BFS
        std::queue<std::pair<int, int>> queue;
        queue.push({ startY, startX });

        // 記錄目前擴展方向 (僅xy平面)
        Eigen::Vector2f currentDirection(0, 0);
        bool hasDirection = false;
        int growthCount = 0; // 記錄生長次數

        while (!queue.empty()) {
            // 取出當前點
            auto [y, x] = queue.front();
            queue.pop();

            // 查找相鄰的線段像素
            std::vector<std::pair<int, int>> neighbors;
            std::vector<Eigen::Vector2f> neighborDirections;

            for (int i = 0; i < 8; ++i) {
                int ny = y + dy[i];
                int nx = x + dx[i];

                // 檢查邊界
                if (ny < 0 || ny >= height || nx < 0 || nx >= width) {
                    continue;
                }

                // 檢查是否為前景且未訪問
                if (valid_image_copy[ny][nx] && !visited[ny][nx]) {
                    neighbors.push_back({ ny, nx });

                    // 計算方向（如果有質心資訊）
                    if (proj_image.pixelInfo[ny][nx] && proj_image.pixelInfo[y][x]) {
                        // 僅考慮xy平面的方向，忽略z軸
                        Eigen::Vector2f dir(
                            proj_image.pixelInfo[ny][nx]->centroid.x() - proj_image.pixelInfo[y][x]->centroid.x(),
                            proj_image.pixelInfo[ny][nx]->centroid.y() - proj_image.pixelInfo[y][x]->centroid.y()
                        );
                        if (dir.norm() > 1e-5) {
                            dir.normalize();
                            neighborDirections.push_back(dir);
                        }
                        else {
                            // 如果質心非常接近，則使用像素坐標作為方向
                            Eigen::Vector2f pixelDir(nx - x, ny - y);
                            pixelDir.normalize();
                            neighborDirections.push_back(pixelDir);
                        }
                    }
                    else {
                        // 如果沒有質心資訊，使用像素坐標的方向
                        Eigen::Vector2f dir(nx - x, ny - y);
                        dir.normalize();
                        neighborDirections.push_back(dir);
                    }
                }
            }

            // 根據鄰居數量決定如何擴展
            if (!neighbors.empty()) {
                int bestNeighbor = 0;
                float bestScore = -1.0f;

                if (hasDirection && growthCount >= 2) {
                    // 已經有穩定方向，檢查角度約束
                    for (size_t i = 0; i < neighbors.size(); ++i) {
                        float angle = calculateAngleBtwnVectors(currentDirection, neighborDirections[i]);

                        // 如果角度在允許範圍內，選擇最符合當前方向的
                        if (angle <= maxAngleDeviation) {
                            float alignment = currentDirection.dot(neighborDirections[i]);
                            if (alignment > bestScore) {
                                bestScore = alignment;
                                bestNeighbor = i;
                            }
                        }
                    }

                    // 如果沒有符合角度約束的鄰居，結束生長
                    if (bestScore < 0) {
                        break;
                    }
                }
                else {
                    // 前3次生長或尚未有穩定方向，不受角度約束
                    for (size_t i = 0; i < neighbors.size(); ++i) {
                        float score = 1.0f; // 預設分數
                        if (hasDirection) {
                            score = currentDirection.dot(neighborDirections[i]);
                        }

                        if (score > bestScore) {
                            bestScore = score;
                            bestNeighbor = i;
                        }
                    }
                }

                // 添加最佳鄰居到線段
                auto [ny, nx] = neighbors[bestNeighbor];
                visited[ny][nx] = true;
                valid_image_copy[ny][nx] = false; // 從複製的圖像中移除
                line.pixels.push_back({ ny, nx });
                queue.push({ ny, nx });

                // 更新當前方向（基於生長次數的加權）
                Eigen::Vector2f newDirection = neighborDirections[bestNeighbor];
                directionVectors.emplace_back(newDirection);
                directionWeights.emplace_back(
                    std::min(proj_image.pixelInfo[y][x]->count, proj_image.pixelInfo[ny][nx]->count)
                );
                growthCount++;

                if (!hasDirection) {
                    currentDirection = newDirection;
                    hasDirection = true;
                }
                else {
                    // 使用加權平均更新方向
                    // 給予當前方向更大的權重，新方向的權重逐漸減小
                    float weight = 1.0f / growthCount;
                    currentDirection = ((1.0f - weight) * currentDirection + weight * newDirection).normalized();
                }
            }
            // 若沒有符合條件的鄰居，當前點是線段的端點，不需要進一步處理
        }

        //bool firstInv = true;
        if (line.pixels.size() >= 3) {
            queue.push({ startY, startX });

            for (auto& dirVec : directionVectors) {
                dirVec = -dirVec;
            }
            // 記錄目前擴展方向 (僅xy平面)
            currentDirection = -currentDirection;

            std::reverse(directionVectors.begin(), directionVectors.end());
            std::reverse(directionWeights.begin(), directionWeights.end());
            std::reverse(line.pixels.begin(), line.pixels.end());

            int newStartY = line.pixels.begin()->first;
            int newStartX = line.pixels.begin()->second;
            if (proj_image.pixelInfo[newStartY][newStartX]) {
                line.startPoint = proj_image.pixelInfo[newStartY][newStartX]->centroid;
            }

            while (!queue.empty()) {
                // 取出當前點
                auto [y, x] = queue.front();
                queue.pop();

                // 查找相鄰的線段像素
                std::vector<std::pair<int, int>> neighbors;
                std::vector<Eigen::Vector2f> neighborDirections;

                for (int i = 0; i < 8; ++i) {
                    int ny = y + inv_dy[i];
                    int nx = x + inv_dx[i];

                    // 檢查邊界
                    if (ny < 0 || ny >= height || nx < 0 || nx >= width) {
                        continue;
                    }

                    // 檢查是否為前景且未訪問
                    if (valid_image_copy[ny][nx] && !visited[ny][nx]) {
                        neighbors.push_back({ ny, nx });

                        // 計算方向（如果有質心資訊）
                        if (proj_image.pixelInfo[ny][nx] && proj_image.pixelInfo[y][x]) {
                            // 僅考慮xy平面的方向，忽略z軸
                            Eigen::Vector2f dir(
                                proj_image.pixelInfo[ny][nx]->centroid.x() - proj_image.pixelInfo[y][x]->centroid.x(),
                                proj_image.pixelInfo[ny][nx]->centroid.y() - proj_image.pixelInfo[y][x]->centroid.y()
                            );
                            if (dir.norm() > 1e-5) {
                                dir.normalize();
                                neighborDirections.push_back(dir);
                            }
                            else {
                                // 如果質心非常接近，則使用像素坐標作為方向
                                Eigen::Vector2f pixelDir(nx - x, ny - y);
                                pixelDir.normalize();
                                neighborDirections.push_back(pixelDir);
                            }
                        }
                        else {
                            // 如果沒有質心資訊，使用像素坐標的方向
                            Eigen::Vector2f dir(nx - x, ny - y);
                            dir.normalize();
                            neighborDirections.push_back(dir);
                        }
                    }
                }

                // 根據鄰居數量決定如何擴展
                if (!neighbors.empty()) {
                    int bestNeighbor = 0;
                    float bestScore = -1.0f;

                    if (hasDirection && growthCount >= 3) {
                        // 已經有穩定方向，檢查角度約束
                        for (size_t i = 0; i < neighbors.size(); ++i) {
                            float angle = calculateAngleBtwnVectors(currentDirection, neighborDirections[i]);

                            // 如果角度在允許範圍內，選擇最符合當前方向的
                            if (angle <= maxAngleDeviation) {
                                float alignment = currentDirection.dot(neighborDirections[i]);
                                if (alignment > bestScore) {
                                    bestScore = alignment;
                                    bestNeighbor = i;
                                }
                            }
                        }

                        // 如果沒有符合角度約束的鄰居，結束生長
                        if (bestScore < 0) {
                            break;
                        }
                    }
                    else {
                        // 前3次生長或尚未有穩定方向，不受角度約束
                        for (size_t i = 0; i < neighbors.size(); ++i) {
                            float score = 1.0f; // 預設分數
                            if (hasDirection) {
                                score = currentDirection.dot(neighborDirections[i]);
                            }

                            if (score > bestScore) {
                                bestScore = score;
                                bestNeighbor = i;
                            }
                        }
                    }

                    /*if (firstInv) {

                    }*/

                    // 添加最佳鄰居到線段
                    auto [ny, nx] = neighbors[bestNeighbor];
                    visited[ny][nx] = true;
                    valid_image_copy[ny][nx] = false; // 從複製的圖像中移除
                    line.pixels.push_back({ ny, nx });
                    queue.push({ ny, nx });

                    // 更新當前方向（基於生長次數的加權）
                    Eigen::Vector2f newDirection = neighborDirections[bestNeighbor];
                    directionVectors.emplace_back(newDirection);
                    directionWeights.emplace_back(
                        std::min(proj_image.pixelInfo[y][x]->count, proj_image.pixelInfo[ny][nx]->count)
                    );
                    growthCount++;

                    if (!hasDirection) {
                        currentDirection = newDirection;
                        hasDirection = true;
                    }
                    else {
                        // 使用加權平均更新方向
                        // 給予當前方向更大的權重，新方向的權重逐漸減小
                        float weight = 1.0f / growthCount;
                        currentDirection = ((1.0f - weight) * currentDirection + weight * newDirection).normalized();
                    }
                }
                // 若沒有符合條件的鄰居，當前點是線段的端點，不需要進一步處理
            }
        }

        // 設置線段的終點為最後一個點
        if (!line.pixels.empty() && line.pixels.size() > 1) {
            auto [lastY, lastX] = line.pixels.back();
            if (proj_image.pixelInfo[lastY][lastX]) {
                line.endPoint = proj_image.pixelInfo[lastY][lastX]->centroid;
            }
            line.directionMean = vectorOrientation(directionVectorMean(directionVectors, directionWeights));
            line.directionStd = directionVectorStd(directionVectors, directionWeights);
            //line.directionStd = calculateStd(directionAngles);
            for (int i = 0; i < directionVectors.size(); i++) {
                line.directions.emplace_back(vectorOrientation(directionVectors[i]));
            }

            // 計算線段方向和長度
            if (proj_image.pixelInfo[line.pixels.front().first][line.pixels.front().second] &&
                proj_image.pixelInfo[lastY][lastX]) {
                // 僅考慮xy平面上的方向和長度
                Eigen::Vector2f start2D(line.startPoint.x(), line.startPoint.y());
                Eigen::Vector2f end2D(line.endPoint.x(), line.endPoint.y());
                line.directionVector = currentDirection;//(end2D - start2D).normalized();
                line.length = (end2D - start2D).norm();
            }
            else {
                // 如果沒有質心資訊，使用像素數量作為長度近似
                line.length = line.pixels.size();

                // 使用最終計算得到的生長方向
                if (hasDirection) {
                    line.directionVector = currentDirection;
                }
            }
        }

        return line;
        };

    // 掃描整個圖像尋找線段
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (valid_image_copy[y][x] && !visited[y][x]) {
                // 找到一個未訪問的前景點，開始生長一條新線段
                LineSegment line = growLine(y, x);

                // 如果線段長度達到閾值，則保存它
                if (line.pixels.size() >= minLineLength) {
                    line_segments.push_back(line);
                }
                // 不符合閾值的線段已經在growLine中從複製的圖像中移除了
            }
        }
    }

    std::sort(line_segments.begin(), line_segments.end(),
        [](const LineSegment& a, const LineSegment& b) {
            return a.pixels.size() > b.pixels.size();
        }
    );
	
	return true;
}

int WHTransform::angleToThetaIndex(float angleDegrees) {
    // 將角度從度轉為弧度並加π/2，使範圍變為 [0, π]
    double angleRadians = (angleDegrees * M_PI / 180.0) + M_PI / 2.0;

    // 標準化角度到 [0, π) 範圍
    while (angleRadians < 0) angleRadians += M_PI;
    while (angleRadians >= M_PI) angleRadians -= M_PI;

    // 轉換為索引
    return static_cast<int>(roundf(angleRadians / thetaStep));
}

double WHTransform::calculateRho(float x, float y, int thetaIndex) {
    double theta = thetaIndex * thetaStep;
    return x * std::cos(theta) + y * std::sin(theta);
}

int WHTransform::wrapThetaIndex(int thetaIndex) {
    while (thetaIndex < 0) thetaIndex += numTheta;
    while (thetaIndex >= numTheta) thetaIndex -= numTheta;
    return thetaIndex;
}

void WHTransform::voteForPoint(float x, float y, int thetaIndex, int thetaRange, int pixelY, int pixelX, size_t pointCount) {
    // 處理環狀範圍，如果範圍橫跨邊界，需要分兩段處理
    if (thetaRange >= numTheta / 2) {
        // 範圍太大，涵蓋整個環
        for (int t = 0; t < numTheta; ++t) {
            voteForPointAtTheta(x, y, t, pixelY, pixelX, pointCount);
        }
    }
    else {
        // 一般情況，需要處理環狀範圍
        int startTheta = thetaIndex - thetaRange;
        int endTheta = thetaIndex + thetaRange;

        if (startTheta < 0) {
            // 左邊界下溢，分兩段
            for (int t = wrapThetaIndex(startTheta); t < numTheta; ++t) {
                voteForPointAtTheta(x, y, t, pixelY, pixelX, pointCount);
                if (first_time) angleCounter[t]++;
            }
            for (int t = 0; t <= endTheta; ++t) {
                voteForPointAtTheta(x, y, t, pixelY, pixelX, pointCount);
                if (first_time) angleCounter[t]++;
            }
        }
        else if (endTheta >= numTheta) {
            // 右邊界上溢，分兩段
            for (int t = startTheta; t < numTheta; ++t) {
                voteForPointAtTheta(x, y, t, pixelY, pixelX, pointCount);
                if (first_time) angleCounter[t]++;
            }
            for (int t = 0; t <= wrapThetaIndex(endTheta); ++t) {
                voteForPointAtTheta(x, y, t, pixelY, pixelX, pointCount);
                if (first_time) angleCounter[t]++;
            }
        }
        else {
            // 沒有跨越邊界，一般情況
            for (int t = startTheta; t <= endTheta; ++t) {
                voteForPointAtTheta(x, y, t, pixelY, pixelX, pointCount);
                if (first_time) angleCounter[t]++;
            }
        }
    }
}

void WHTransform::voteForPointAtTheta(float x, float y, int thetaIndex, int pixelY, int pixelX, size_t pointCount) {
    double rho = calculateRho(x, y, thetaIndex);

    // 將 ρ 映射到累加器的索引
    int rhoIndex = static_cast<int>((rho + rhoMax) / rhoStep);

    // 確保 ρ 索引在有效範圍內
    if (rhoIndex >= 0 && rhoIndex < numRho) {
        // 增加投票數，使用像素的點數量而不是固定增加1
        accumulator[rhoIndex][thetaIndex] += pointCount;

        // 記錄投票的像素
        if (!pixelAccumulator[rhoIndex][thetaIndex]) {
            // 首次投票，創建新的像素集合
            pixelAccumulator[rhoIndex][thetaIndex] = std::make_shared<std::vector<std::pair<int, int>>>();
        }
        // 添加像素座標
        pixelAccumulator[rhoIndex][thetaIndex]->push_back(std::make_pair(pixelY, pixelX));
    }
}

void WHTransform::findPeaks(int threshold) {
    float maxAcc = 0;
    for (int r = 0; r < numRho; ++r) {
        for (int t = 0; t < numTheta; ++t) {
            if (accumulator[r][t] >= threshold) {
                // 檢查是否為局部最大值
                bool isLocalMax = true;

                // 在 3x3 的窗口內檢查
                for (int dr = -1; dr <= 1 && isLocalMax; ++dr) {
                    for (int dt = -1; dt <= 1; ++dt) {
                        int nr = r + dr;
                        int nt = wrapThetaIndex(t + dt); // 使用環狀索引

                        // 跳過中心點和rho範圍外的點
                        if ((dr == 0 && dt == 0) || nr < 0 || nr >= numRho) {
                            continue;
                        }

                        // 如果鄰居的值更大，則不是局部最大值
                        if (accumulator[nr][nt] > accumulator[r][t]) {
                            isLocalMax = false;
                            break;
                        }
                    }
                }

                // 如果是局部最大值，則加入結果
                if (isLocalMax) {
                    double rho = (r * rhoStep) - rhoMax;
                    double theta = t * thetaStep;
                    houghLines.push_back(std::make_pair(rho, theta));

                    // 添加對應的像素集合
                    linePixels.push_back(pixelAccumulator[r][t]);

                    if (accumulator[r][t] > maxAcc) {
                        maxAcc = accumulator[r][t];
                        bestHoughLine = std::make_pair(rho, theta);
                        bestLinePixel = *pixelAccumulator[r][t];
                    }

                }
            }
        }
    }
}

void WHTransform::transform(const PointsPixelator::ProjectionImage& result, const std::vector<LineDetector::LineSegment>& lineSegments, const std::vector<std::pair<int, int>>& pixelsInLines, int threshold) {
    // 重置累加器
    for (auto& row : accumulator) {
        std::fill(row.begin(), row.end(), 0);
    }

    // 重置像素累加器
    for (auto& row : pixelAccumulator) {
        for (auto& cell : row) {
            cell = nullptr;
        }
    }

    // 清空之前檢測到的線
    houghLines.clear();
    linePixels.clear();

    // 為每個線段中的像素投票
    for (const auto& [y, x] : pixelsInLines) {
        // 確保像素在有效範圍內
        if (y < 0 || y >= result.height || x < 0 || x >= result.width) {
            continue;
        }

        // 獲取像素所屬的線段索引
        const auto& pixelInfo = result.pixelInfo[y][x];
        if (!pixelInfo || pixelInfo->line_seg_idx < 0 ||
            pixelInfo->line_seg_idx >= lineSegments.size()) {
            continue;
        }

        // 獲取線段資訊
        const auto& lineSegment = lineSegments[pixelInfo->line_seg_idx];

        // 獲取線段角度和標準差
        float angleDegrees = lineSegment.directionMean;
        float stdDev = lineSegment.directionStd;

        // 將角度轉換為 Hough 空間中的 θ 索引
        int thetaIndex = angleToThetaIndex(angleDegrees);

        // 計算標準差對應的 θ 索引範圍
        int thetaRange = static_cast<int>(stdDevMultiplier * stdDev / (thetaStep * 180.0 / M_PI));
        thetaRange = std::max(1, thetaRange); // 確保至少有一個範圍

        // 使用質心的 xy 座標進行投票
        float centroidX = pixelInfo->centroid.x();
        float centroidY = pixelInfo->centroid.y();

        // 使用像素的點數量進行投票
        size_t pointCount = pixelInfo->count;

        // 在指定範圍內投票，並記錄像素
        voteForPoint(centroidX, centroidY, thetaIndex, thetaRange, y, x, pointCount);
    }

    // 尋找超過閾值的峰值
    findPeaks(threshold);
    first_time = false;
}



