#include "ComputeConvexHulls.hpp"
#include <algorithm>

namespace pointcloud_obstacle_detection{

ComputeConvexHulls::ComputeConvexHulls(){}

//source: https://github.com/sasamil/PointInPolygon
//===================================================================
// isLeft(): tests if a point is Left|On|Right of an infinite line.
//    Input:  three points P0, P1, and P2
//    Return: >0 for P2 left of the line through P0 and P1
//            =0 for P2  on the line
//            <0 for P2  right of the line
double ComputeConvexHulls::isLeft(Point P0, Point P1, Point P2) {
  return ((P1.x - P0.x) * (P2.y - P0.y) - (P2.x - P0.x) * (P1.y - P0.y));
}

double ComputeConvexHulls::cross(const Point &O, const Point &A, const Point &B)
{
	return (A.x - O.x) * (double)(B.y - O.y) - (A.y - O.y) * (double)(B.x - O.x);
}

std::vector<Point> ComputeConvexHulls::convexHull3D(std::vector<Point> P)
{
	int n = P.size(), k = 0;
	std::vector<Point> H(2*n);
 
	// Sort points lexicographically
	std::sort(P.begin(), P.end());
 
	// Build lower hull
	for (int i = 0; i < n; i++) {
		while (k >= 2 && cross(H[k-2], H[k-1], P[i]) <= 0) k--;
		H[k++] = P[i];
	}
 
	// Build upper hull
	for (int i = n-2, t = k+1; i >= 0; i--) {
		while (k >= t && cross(H[k-2], H[k-1], P[i]) <= 0) k--;
		H[k++] = P[i];
	}
 
	H.resize(k);
	return H;
}

std::vector<std::vector<pointcloud_obstacle_detection::Point>> ComputeConvexHulls::computeOBBPoints3D(std::vector<base::samples::OrientedBoundingBox>  boxes){

	std::vector<std::vector<pointcloud_obstacle_detection::Point>> boxes_points; 
	for(base::samples::OrientedBoundingBox box : boxes)
	{
		double x_min = -box.dimension.x()/2;
		double y_min = -box.dimension.y()/2;
		double z_min = -box.dimension.z()/2;
		double x_max = box.dimension.x()/2;
		double y_max = box.dimension.y()/2;
		double z_max = box.dimension.z()/2;
		
		Eigen::Vector3d p1{x_min,y_min,z_min};
		Eigen::Vector3d p2{x_min,y_min,z_max};
		Eigen::Vector3d p3{x_max,y_min,z_max};
		Eigen::Vector3d p4{x_max,y_min,z_min};
		Eigen::Vector3d p5{x_min,y_max,z_min};
		Eigen::Vector3d p6{x_min,y_max,z_max};
		Eigen::Vector3d p7{x_max,y_max,z_max};
		Eigen::Vector3d p8{x_max,y_max,z_min};

		p1 = box.orientation * p1 + box.position;
		p2 = box.orientation * p2 + box.position;
		p3 = box.orientation * p3 + box.position;
		p4 = box.orientation * p4 + box.position;
		p5 = box.orientation * p5 + box.position;
		p6 = box.orientation * p6 + box.position;
		p7 = box.orientation * p7 + box.position;
		p8 = box.orientation * p8 + box.position;

		std::vector<Eigen::Vector3d> points{p1,p2,p3,p4,p5,p6,p7,p8};
		std::vector<pointcloud_obstacle_detection::Point> hull;

		for (auto val : points){
            pointcloud_obstacle_detection::Point pt(val.x(), val.y(),val.z());
            hull.push_back(pt);
		}
		boxes_points.push_back(hull);	
	}
	return boxes_points;
}

std::vector<std::vector<Point>> ComputeConvexHulls::computeConvexHulls3D(std::vector<base::samples::OrientedBoundingBox>  boxes){
	std::vector<std::vector<Point>> obstacle_edges = computeOBBPoints3D(boxes);
	std::vector<std::vector<Point>> convex_hulls;
	for (auto edges : obstacle_edges){
		std::vector<Point> hull;
		hull = convexHull3D(edges);		
		convex_hulls.push_back(hull);
	}
	return convex_hulls;
}
} //namespace pointcloud_obstacle_detection