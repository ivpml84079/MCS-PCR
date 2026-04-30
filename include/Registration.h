#pragma once

#include <iostream>
#include <string>
#include <fstream>

#include <Eigen/Dense>

#include "Utils.h"

class WHTTC {
public:
	struct Param {
		double resolution = 0.2;
		double angle_tolerance = 3.0;
		double fac_epsilon = 2.0;
		double fac_tau = 2.0;
		int max_line_num = 20;
	} param;

	Eigen::Matrix3d est_rotation;
	Eigen::Vector3d est_translation;
	Eigen::Matrix4d est_transformation;

	double regis_time;
	bool successful;

	WHTTC() : est_rotation(Eigen::Matrix3d::Identity()), est_translation(Eigen::Vector3d::Zero()), successful(false), regis_time(-1) { }
	WHTTC(Param p) : param(p), est_rotation(Eigen::Matrix3d::Identity()), est_translation(Eigen::Vector3d::Zero()), successful(false), regis_time(-1) {}

	void init();

	bool regis(
		const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src, 
		const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tgt,
		int verbose = false
	);

	const Eigen::Matrix3d& getEstRotation() const { return est_rotation; }
	const Eigen::Vector3d& getEstTranslation() const { return est_translation; }
	const Eigen::Matrix4d& getEstTransformation() const { return est_transformation; }

	const double getTime() const { return regis_time; }
};