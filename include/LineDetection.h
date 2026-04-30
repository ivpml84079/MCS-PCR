#pragma once

#include <vector>
#include <iostream>
#include <cmath>
#include <queue>
#include <list>
#include <memory>
#include <algorithm>

#include <Eigen/Core>

#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>

#include "Projection.h"

class LineDetector {
public:
    struct LineSegment {
        std::vector<std::pair<int, int>> pixels;  // 線段中的像素座標 (y, x)
        Eigen::Vector3f startPoint;               // 線段起點的3D座標
        Eigen::Vector3f endPoint;                 // 線段終點的3D座標
        Eigen::Vector2f directionVector;          // 線段的方向向量 (只考慮xy平面)
        float length;                             // 線段的長度
        float directionMean;
        float directionStd;

        std::vector<float> directions;

        LineSegment() : length(0.0f), directionMean(0.0f), directionStd(0.0) {}
    };

	double resolution;
	int max_line_num;
	bool refine;

    std::vector<std::tuple<double, double, double>> lines_coefs_;
    std::vector<Eigen::Vector2f> lines_centroid_;
    std::vector<std::vector<Eigen::Vector2f>> lines_points_;
    std::vector<int> lines_points_num_;

    std::vector<int> angleCounter;
    int pixel_in_segs = 0;

    std::vector<LineSegment> line_segments;

	LineDetector(double res_param, int max_line): resolution(res_param), max_line_num(max_line), refine(true) { }

	bool detect(const pcl::PointCloud<pcl::PointXYZ>::Ptr feasible_points, PointsPixelator::ProjectionImage &proj_image, int verbose);

    bool extractLineSegments(PointsPixelator::ProjectionImage& proj_image, int minLineLength = 5, float maxAngleDeviation = M_PI_4);

    const std::vector<std::tuple<double, double, double>>& getLinesCoef() const {
        return lines_coefs_;
    }

    const std::vector<Eigen::Vector2f>& getEstimatedLinesCentroid() const {
        return lines_centroid_;
    }

    const std::vector<int>& getEstimatedLinesPointsNum() const {
        return lines_points_num_;
    }

    const std::vector<std::vector<Eigen::Vector2f>>& getEstimatedLinesPoints() const {
        return lines_points_;
    }
};

class WHTransform {
public:
    int numRho;            // ρ (距離) 的離散數量
    int numTheta;          // θ (角度) 的離散數量
    double rhoStep;        // ρ 的步長
    double thetaStep;      // θ 的步長
    double rhoMax;         // ρ 的最大值
    double stdDevMultiplier; // 角度標準差的乘數

    // 投票累加器
    std::vector<std::vector<int>> accumulator;

    // 每個投票箱對應的像素集合
    std::vector<std::vector<std::shared_ptr<std::vector<std::pair<int, int>>>>> pixelAccumulator;

    // 檢測到的線 (ρ, θ)
    std::vector<std::pair<double, double>> houghLines;

    // 每條線對應的像素集合
    std::vector<std::shared_ptr<std::vector<std::pair<int, int>>>> linePixels;

    std::pair<double, double> bestHoughLine;
    std::vector<std::pair<int, int>> bestLinePixel;
    std::vector<int> angleCounter;
    bool first_time = true;

    WHTransform(int width, int height, float rhoStep_ = 0.1, int numRhoSteps = 180,
        int numThetaSteps = 180, double stdDevMult = 1.5) {
        numRho = numRhoSteps;
        numTheta = numThetaSteps;
        rhoMax = std::sqrt(width * width + height * height);
        rhoStep = rhoStep_;   //(2.0 * rhoMax) / numRho;
        thetaStep = M_PI / numTheta;
        stdDevMultiplier = stdDevMult;
        numRho = (2.0 * rhoMax) / rhoStep;
        // 初始化累加器
        accumulator.resize(numRho, std::vector<int>(numTheta, 0));

        // 初始化像素累加器
        pixelAccumulator.resize(numRho);
        for (auto& row : pixelAccumulator) {
            row.resize(numTheta);
            for (auto& cell : row) {
                cell = nullptr; // 初始化為 nullptr，僅在有投票時才創建
            }
        }
        angleCounter.resize(numThetaSteps, 0);
    }

    int angleToThetaIndex(float angleDegrees);

    double calculateRho(float x, float y, int thetaIndex);

    int wrapThetaIndex(int thetaIndex);

    void voteForPoint(float x, float y, int thetaIndex, int thetaRange,
        int pixelY, int pixelX, size_t pointCount);

    void voteForPointAtTheta(float x, float y, int thetaIndex,
        int pixelY, int pixelX, size_t pointCount);

    void findPeaks(int threshold);

    void transform(const PointsPixelator::ProjectionImage& result,
        const std::vector<LineDetector::LineSegment>& lineSegments,
        const std::vector<std::pair<int, int>>& pixelsInLines,
        int threshold = 100);

    const std::vector<std::pair<double, double>>& getHoughLines() const {
        return houghLines;
    }

    const std::vector<std::shared_ptr<std::vector<std::pair<int, int>>>>& getLinePixels() const {
        return linePixels;
    }

    const std::vector<std::vector<int>>& getAccumulator() const {
        return accumulator;
    }

    const std::vector<std::vector<std::shared_ptr<std::vector<std::pair<int, int>>>>>& getPixelAccumulator() const {
        return pixelAccumulator;
    }
};