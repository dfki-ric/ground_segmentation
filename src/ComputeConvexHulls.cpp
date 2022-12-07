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

std::vector<std::vector<pointcloud_obstacle_detection::Point>> ComputeConvexHulls::computeOBBPoints3D(std::vector<pointcloud_obstacle_detection::OrientedBoundingBox> boxes){

	std::vector<std::vector<pointcloud_obstacle_detection::Point>> boxes_points; 
	for(pointcloud_obstacle_detection::OrientedBoundingBox box : boxes)
	{
		Eigen::Vector3d p1{box.x_min,box.y_min,box.z_min};
		Eigen::Vector3d p2{box.x_min,box.y_min,box.z_max};
		Eigen::Vector3d p3{box.x_max,box.y_min,box.z_max};
		Eigen::Vector3d p4{box.x_max,box.y_min,box.z_min};
		Eigen::Vector3d p5{box.x_min,box.y_max,box.z_min};
		Eigen::Vector3d p6{box.x_min,box.y_max,box.z_max};
		Eigen::Vector3d p7{box.x_max,box.y_max,box.z_max};
		Eigen::Vector3d p8{box.x_max,box.y_max,box.z_min};

		p1 = box.rotation * p1 + box.position;
		p2 = box.rotation * p2 + box.position;
		p3 = box.rotation * p3 + box.position;
		p4 = box.rotation * p4 + box.position;
		p5 = box.rotation * p5 + box.position;
		p6 = box.rotation * p6 + box.position;
		p7 = box.rotation * p7 + box.position;
		p8 = box.rotation * p8 + box.position;

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

std::vector<std::vector<Point>> ComputeConvexHulls::computeConvexHulls3D(std::vector<pointcloud_obstacle_detection::OrientedBoundingBox> boxes){
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