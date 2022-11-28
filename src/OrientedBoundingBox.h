#ifndef ORIENTED_BOUNDING_BOX_H
#define ORIENTED_BOUNDING_BOX_H

#include <Eigen/Geometry>

namespace pointcloud_obstacle_detection{

struct OrientedBoundingBox
{
	double x_min;
	double y_min;
	double z_min;
	double x_max;
	double y_max;
	double z_max;
	Eigen::Vector3d position;
	Eigen::Quaterniond rotation; 
};

} //namespace pointcloud_obstacle_detection
#endif
