#include "Registration.h"
#include "Projection.h"
#include "LineDetection.h"
#include "Graph.h"
#include "LoopRegistration2D.h"
#include "ZEstimation.h"

void WHTTC::init() {
	est_rotation = Eigen::Matrix3d::Identity();
	est_translation = Eigen::Vector3d::Zero();
	successful = false;
}

bool WHTTC::regis(
	const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src,
	const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tgt,
	int verbose
) {
	init();
	auto start = std::chrono::high_resolution_clock::now();

	if (verbose > 0) {
		std::cout << "\n--- Processing Source Point Cloud ---" << std::endl;
		std::cout << "Source point cloud size: " << cloud_src->points.size() << std::endl;
	}

	auto src_proc_bgn = std::chrono::high_resolution_clock::now();

	IFVRProcessor ifvr_proc_src(2. * param.resolution);
	ifvr_proc_src.process(cloud_src, verbose);
	pcl::PointCloud<pcl::PointXYZ>::Ptr feasible_points_src = ifvr_proc_src.getFeasiblePoints();
	if (verbose > 1) {
		std::cout << "Source feasible point size: " << feasible_points_src->points.size() << std::endl;
	}

	PointsPixelator pixelator_src(2. * param.resolution);
	pixelator_src.pixelize(feasible_points_src, verbose);
	PointsPixelator::ProjectionImage proj_image_src = pixelator_src.getProjectionImage();

	LineDetector line_detector_src(2. * param.resolution, param.max_line_num);
	line_detector_src.detect(feasible_points_src, proj_image_src, verbose);
	std::vector<std::tuple<double, double, double>> line_src = line_detector_src.getLinesCoef();

	auto src_proc_end = std::chrono::high_resolution_clock::now();
	auto src_proc_duration = std::chrono::duration_cast<std::chrono::milliseconds>(src_proc_end - src_proc_bgn);

	if (verbose > 0) {
		std::cout << "--------------- Finish --------------" << std::endl;
		std::cout << "Source point cloud total processing time: " << src_proc_duration.count() << std::endl << std::endl;
	}



	if (verbose > 0) {
		std::cout << "\n--- Processing Target Point Cloud ---" << std::endl;
		std::cout << "Target point cloud size: " << cloud_tgt->points.size() << std::endl;
	}

	auto tgt_proc_bgn = std::chrono::high_resolution_clock::now();

	IFVRProcessor ifvr_proc_tgt(2. * param.resolution);
	ifvr_proc_tgt.process(cloud_tgt, verbose);
	pcl::PointCloud<pcl::PointXYZ>::Ptr feasible_points_tgt = ifvr_proc_tgt.getFeasiblePoints();
	if (verbose > 1) {
		std::cout << "Target feasible point size: " << feasible_points_tgt->points.size() << std::endl;
	}

	PointsPixelator pixelator_tgt(2. * param.resolution);
	pixelator_tgt.pixelize(feasible_points_tgt, verbose);
	PointsPixelator::ProjectionImage proj_image_tgt = pixelator_tgt.getProjectionImage();

	LineDetector line_detector_tgt(2. * param.resolution, param.max_line_num);
	line_detector_tgt.detect(feasible_points_tgt, proj_image_tgt, verbose);
	std::vector<std::tuple<double, double, double>> line_tgt = line_detector_tgt.getLinesCoef();

	auto tgt_proc_end = std::chrono::high_resolution_clock::now();
	auto tgt_proc_duration = std::chrono::duration_cast<std::chrono::milliseconds>(tgt_proc_end - tgt_proc_bgn);

	if (verbose > 0) {
		std::cout << "--------------- Finish --------------" << std::endl;
		std::cout << "Target point cloud total processing time: " << tgt_proc_duration.count() << std::endl << std::endl;
	}



	double redius_src = std::sqrt(std::pow(proj_image_src.height * 2. * param.resolution, 2.0) + std::pow(proj_image_src.width * 2. * param.resolution, 2.0)) / 2.0;
	double redius_tgt = std::sqrt(std::pow(proj_image_tgt.height * 2. * param.resolution, 2.0) + std::pow(proj_image_tgt.width * 2. * param.resolution, 2.0)) / 2.0;
	double graph_redius = (redius_src + redius_tgt) / 2;

	if (verbose > 0) {
		std::cout << "\n--- Building up Topological Line-Intersection Graphs ---" << std::endl;
		std::cout << "Graph Redius: " << graph_redius << std::endl;
	}

	auto graph_build_bgn = std::chrono::high_resolution_clock::now();

	TLIGraph graph_src(param.resolution, param.angle_tolerance, param.fac_epsilon, graph_redius);
	graph_src.build(line_src);
	std::cout << line_src.size() << " " << graph_src.nodes.size() << std::endl;
	TLIGraph graph_tgt(param.resolution, param.angle_tolerance, param.fac_epsilon, graph_redius);
	graph_tgt.build(line_tgt);
	std::cout << line_tgt.size() << " " << graph_tgt.nodes.size() << std::endl;

	auto graph_build_end = std::chrono::high_resolution_clock::now();
	auto graph_build_duration = std::chrono::duration_cast<std::chrono::milliseconds>(graph_build_end - graph_build_bgn);

	if (verbose > 0) {
		std::cout << "------------------------ Finish ------------------------" << std::endl;
		std::cout << "Topological Line-Intersection Graphs building time: " << graph_build_duration.count() << std::endl << std::endl;
	}

	if (verbose > 0) {
		std::cout << "\n--- Building up Topological Subgraphs ---" << std::endl;
	}

	auto subgraph_build_bgn = std::chrono::high_resolution_clock::now();
	std::vector<TLIGraph::TCSubgraph> topk_tcsubgraphs = graph_src.match(graph_tgt, 5);
	auto subgraph_build_end = std::chrono::high_resolution_clock::now();
	auto subgraph_build_duration = std::chrono::duration_cast<std::chrono::milliseconds>(subgraph_build_end - subgraph_build_bgn);

	if (verbose > 0) {
		std::cout << "----------------- Finish ----------------" << std::endl;
		std::cout << "Subgraphs building time: " << subgraph_build_duration.count() << std::endl << std::endl;
	}


	if (verbose > 0) {
		std::cout << "\n--- TCLoop Registration ---" << std::endl;
	}
	auto loop_riges_bgn = std::chrono::high_resolution_clock::now();
	LoopBased2DRegister lb2dregister(param.fac_tau * param.resolution);
	successful = lb2dregister.regis(proj_image_src.points2D, proj_image_tgt.points2D, graph_src, graph_tgt, topk_tcsubgraphs);
	auto loop_riges_end = std::chrono::high_resolution_clock::now();
	auto loop_riges_duration = std::chrono::duration_cast<std::chrono::milliseconds>(loop_riges_end - loop_riges_bgn);
	std::cout << lb2dregister.getRotation2D() << std::endl;
	std::cout << lb2dregister.getTranslation2D() << std::endl;

	if (verbose > 0) {
		std::cout << "----------------- Finish ----------------" << std::endl;
		std::cout << "Subgraphs building time: " << loop_riges_duration.count() << std::endl << std::endl;
	}
	
	if (successful == false) return false;

	auto ground_voxels_src = ifvr_proc_src.getVoxels();
	auto ground_voxels_tgt = ifvr_proc_tgt.getVoxels();
	ZEstimator z_estimator(param.fac_tau * param.resolution);
	double dz = z_estimator.estimate(cloud_src, cloud_tgt, ground_voxels_src, ground_voxels_tgt, lb2dregister.getRotation2D(), lb2dregister.getTranslation2D());
	Eigen::Matrix4d T = integrate(lb2dregister.getRotation2D(), lb2dregister.getTranslation2D(), dz);
	
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

	regis_time = duration.count() / 1000.0;

	est_rotation = T.block<3, 3>(0, 0);
	est_translation = T.block<3, 1>(0, 3);
	est_transformation = T;

	successful = true;

	return successful;
}
