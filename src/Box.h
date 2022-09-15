#ifndef BOX_H
#define BOX_H

namespace pointcloud_obstacle_detection{
struct Box
{
	float x_min;
	float y_min;
	float z_min;
	float x_max;
	float y_max;
	float z_max;
};
} //namespace pointcloud_obstacle_detection
#endif
