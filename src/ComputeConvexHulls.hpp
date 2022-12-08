#pragma once
#include <vector>
#include "Point.h"
#include "OrientedBoundingBox.h"
#include <base/samples/OrientedBoundingBox.hpp>

namespace pointcloud_obstacle_detection{

class OrientedBoundingBox;

class ComputeConvexHulls{
public:
  ComputeConvexHulls();
  std::vector<std::vector<Point>> computeOBBPoints3D(std::vector<base::samples::OrientedBoundingBox>  boxes);
  std::vector<std::vector<Point>> computeConvexHulls3D(std::vector<base::samples::OrientedBoundingBox>  boxes);  
private:
  double isLeft(Point P0, Point P1, Point P2);
  double cross(const Point &O, const Point &A, const Point &B);
  std::vector<Point> convexHull3D(std::vector<Point> P);

};
} // namespace pointcloud_obstacle_detection

