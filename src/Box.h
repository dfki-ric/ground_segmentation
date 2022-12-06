#pragma once

namespace pointcloud_obstacle_detection{
struct Box
{
	double x_min;
	double y_min;
	double z_min;
	double x_max;
	double y_max;
	double z_max;
};

} //namespace pointcloud_obstacle_detection
