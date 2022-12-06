#pragma once
#include <vector>
#include "Point.h"
#include "OrientedBoundingBox.h"

namespace pointcloud_obstacle_detection{

class OrientedBoundingBox;

class ComputeConvexHulls{
public:
  ComputeConvexHulls();
  double isLeft(Point P0, Point P1, Point P2);
  double cross(const Point &O, const Point &A, const Point &B);
  std::vector<Point> convex_hull(std::vector<Point> P);
  std::vector<std::vector<Point>> computeOBBPoints(std::vector<pointcloud_obstacle_detection::OrientedBoundingBox> boxes);
  std::vector<std::vector<Point>> computeConvexHulls(std::vector<pointcloud_obstacle_detection::OrientedBoundingBox> boxes);  
};
} // namespace pointcloud_obstacle_detection

