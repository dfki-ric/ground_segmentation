#pragma once
#include <base/Eigen.hpp>

namespace pointcloud_obstacle_detection{

struct OrientedBoundingBox
{
	double x_min;
	double y_min;
	double z_min;
	double x_max;
	double y_max;
	double z_max;
	base::Vector3d position;
	base::Quaterniond rotation; 
};

} //namespace pointcloud_obstacle_detection
