// PCL lib Functions for processing point clouds
// Source: https://github.com/enginBozkurt/LidarObstacleDetection

#ifndef PROCESSPOINTCLOUDS_H_
#define PROCESSPOINTCLOUDS_H_

#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/common/transforms.h>
#include <iostream>
#include <string>
#include <vector>
#include <ctime>
#include <chrono>
#include "Box.h"
#include "OrientedBoundingBox.h"
#include <unordered_set>
#include <base-logging/Logging.hpp>
#include <base/samples/OrientedBoundingBox.hpp>

namespace pointcloud_obstacle_detection {

// Structure to represent node of kd tree
struct Node
{
	std::vector<float> point;
	int id;
	Node* left;
	Node* right;

	Node(std::vector<float> arr, int setId)
	:	point(arr), id(setId), left(NULL), right(NULL)
	{}
};

template <typename PointT>
struct KdTree_euclidean
{
	Node* root;


	KdTree_euclidean()
	: root(NULL)
	{}
	/*inserthelper function is called recursively to create a KDTree
	 *Algo: Check if new data point is greater than or less than root.
	 *		Insert if the child is null as left child if less than root else right child.
	 *		For above steps at level 0:x coordinates, level 1: y coordinates , leve 2: z coordinates are compared	 *
	 * */
	void inserthelper(Node *&node, uint level, PointT point, int id)
	{
		/*Identify the axis*/
	    uint index = level%3;
	    /*If the node is NULL insert the point along with index by creating a new node*/
		if(node == NULL)
		{
		// convert point.data arr to vector
		 std::vector<float> v_point(point.data, point.data+3);
		 node = new Node(v_point,id);
		}
		else if(point.data[index] < node->point[index])
		{
		/*data point is less than root insert in left child*/
		inserthelper(node->left,level+1,point,id);
		}
		else
		{
		/*data point is greater than root insert in right child*/
		inserthelper(node->right,level+1,point,id);
		}
	}
	/*insert_cloud function helps creating KDTree from a cloud.
	 * This function shall loop through each of the cloud points
	 * and call inserthelper function for each point
	 * */
	void insert_cloud(typename pcl::PointCloud<PointT>::Ptr cloud)
	{
		for(uint index = 0; index < cloud->points.size(); index++)
		{
		   inserthelper(root,0,cloud->points[index],index);
		}

	}
	/*helpersearch function looks for the target point in the KDTree
	 * Algo: Check if the target point x,y,z are within node+/-distanceTol ,
	 * 		 	if they are then check if the distance b/w node and target is with in distanceTol then add it to the list
	 * 		 If x,y,z of target are not with in distanceTol of node then check if target-/+distanceTol is less or greater node and
	 * 		 call helpersearch with left or right child node.
	 *
	 * */
	void helpersearch(Node *&node,uint depth,std::vector<int> *ids,PointT target, float distanceTol)
	{
		uint id = depth%3;
		if(node!=NULL)
		{
			/*Check if nodes x,y,z are with in target+/-distanceTol */
			if(((node->point[0]<target.data[0]+distanceTol)&&(node->point[0]>target.data[0]-distanceTol))&&
					((node->point[1]<target.data[1]+distanceTol)&&(node->point[1]>target.data[1]-distanceTol))&&
						((node->point[2]<target.data[2]+distanceTol)&&(node->point[2]>target.data[2]-distanceTol)))
			{
				/*calculate distance b/w node and point*/
				uint dis=sqrt((node->point[0]-target.data[0])*(node->point[0]-target.data[0])+
						(node->point[1]-target.data[1])*(node->point[1]-target.data[1])+
						(node->point[2]-target.data[2])*(node->point[2]-target.data[2]));

				/*is distance b/w node and point less than distanceTol then add it to vector*/
				if(dis<distanceTol)
				{
					ids->push_back(node->id);
				}
			}

			if(target.data[id]-distanceTol<node->point[id])
			{
				helpersearch(node->left,depth+1,ids,target,distanceTol);

			}
			if(target.data[id]+distanceTol>node->point[id])
			{
				helpersearch(node->right,depth+1,ids,target,distanceTol);

			}

		}
	}
	/*This is the API for KDTree search. It calls helpersearch function.
	 * */
	std::vector<int> search(PointT target, float distanceTol)
	{
		std::vector<int> ids;
		uint depth =0;
		uint maxdistance=0;

		helpersearch(root,depth,&ids,target,distanceTol);
        //cout<<"helpersearch end"<<endl;
		return ids;
	}


};

template<typename PointT>
class ProcessPointClouds {
public:

    //constructor
    ProcessPointClouds();
    //deconstructor
    ~ProcessPointClouds();

    void numPoints(typename pcl::PointCloud<PointT>::Ptr cloud);

    typename pcl::PointCloud<PointT>::Ptr FilterCloud(typename pcl::PointCloud<PointT>::Ptr cloud, bool downSampleInputCloud, float filterRes, Eigen::Vector4f minPoint, Eigen::Vector4f maxPoint);

    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> SeparateClouds(pcl::PointIndices::Ptr inliers, typename pcl::PointCloud<PointT>::Ptr cloud);

    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> SegmentPlane(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceThreshold);

    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> SegmentPlane_RANSAC(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceThreshold);

    std::vector<typename pcl::PointCloud<PointT>::Ptr> Clustering(typename pcl::PointCloud<PointT>::Ptr cloud, float clusterTolerance, int minSize, int maxSize);

    std::vector<typename pcl::PointCloud<PointT>::Ptr> Clustering_euclideanCluster(typename pcl::PointCloud<PointT>::Ptr cloud, float clusterTolerance, int minSize, int maxSize);

	  std::vector<std::vector<int>> euclideanCluster(typename pcl::PointCloud<PointT>::Ptr cloud, typename KdTree_euclidean<PointT>::KdTree_euclidean* tree, float distanceTol, int minSize, int maxSize);

	  void Proximity(typename pcl::PointCloud<PointT>::Ptr cloud,std::vector<int> &cluster,std::vector<bool> &processed_f,int idx,typename KdTree_euclidean<PointT>::KdTree_euclidean* tree,float distanceTol, int maxSize);

    pointcloud_obstacle_detection::Box BoundingBox(typename pcl::PointCloud<PointT>::Ptr cluster);

		base::samples::OrientedBoundingBox OrientedBoundingBox(typename pcl::PointCloud<PointT>::Ptr cluster, const base::Time& ts);

		//pointcloud_obstacle_detection::Box AxisAlignedBoundingBox(typename pcl::PointCloud<PointT>::Ptr cluster);

    void savePcd(typename pcl::PointCloud<PointT>::Ptr cloud, std::string file);

    typename pcl::PointCloud<PointT>::Ptr loadPcd(std::string file);

    std::vector<boost::filesystem::path> streamPcd(std::string dataPath);

};


//constructor:
template<typename PointT>
ProcessPointClouds<PointT>::ProcessPointClouds() {}


//de-constructor:
template<typename PointT>
ProcessPointClouds<PointT>::~ProcessPointClouds() {}


template<typename PointT>
void ProcessPointClouds<PointT>::numPoints(typename pcl::PointCloud<PointT>::Ptr cloud)
{
    LOG_INFO_S << cloud->points.size();
}
/*FilterCloud function filters the given cloud. Following operations are performed
 * Downsampling: points are converted to voxels using the dimensions provided.
 * Crop: Remove all the points that are outside the min , max limits
 * RoofCrop: Remove roof points , dimensions of roof are given.
 * */

template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr ProcessPointClouds<PointT>::FilterCloud(typename pcl::PointCloud<PointT>::Ptr cloud, bool downSampleInputCloud, float filterRes, Eigen::Vector4f minPoint, Eigen::Vector4f maxPoint)
{

    // Time segmentation process
    auto startTime = std::chrono::steady_clock::now();
    typename pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>());
	// Crop the scene to create ROI
    pcl::CropBox<PointT> roi;
    roi.setMin(minPoint);
    roi.setMax(maxPoint);
	
	if (downSampleInputCloud){
		// Convert the points to voxel grid points
		pcl::VoxelGrid<PointT> sor;
		sor.setInputCloud (cloud);
		sor.setLeafSize (filterRes, filterRes, filterRes);
		sor.filter (*cloud_filtered);
		LOG_INFO_S << "Voxeled " << cloud_filtered->points.size();
		roi.setInputCloud(cloud_filtered);
	}
	else{
		roi.setInputCloud(cloud);
	}
	
    roi.filter(*cloud_filtered);
    LOG_INFO_S << "ROI " << cloud_filtered->points.size();

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    LOG_INFO_S << "filtering took " << elapsedTime.count() << " milliseconds";

    return cloud_filtered;

}


template<typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::SeparateClouds(pcl::PointIndices::Ptr inliers, typename pcl::PointCloud<PointT>::Ptr cloud)
{
  // TODO: Create two new point clouds, one cloud with obstacles and other with segmented plane
	typename pcl::PointCloud<PointT>::Ptr obstCloud(new pcl::PointCloud<PointT>());
	typename pcl::PointCloud<PointT>::Ptr planeCloud(new pcl::PointCloud<PointT>());

	for(int it : inliers->indices)
	{
		planeCloud->points.push_back(cloud->points[it]);
	}

	pcl::ExtractIndices<PointT> extract;
	extract.setInputCloud (cloud);
	extract.setIndices (inliers);
	extract.setNegative (true);
	extract.filter (*obstCloud);

    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult(obstCloud, planeCloud);
    return segResult;
}


template<typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::SegmentPlane(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceThreshold)
{
    // Time segmentation process
    auto startTime = std::chrono::steady_clock::now();
	pcl::PointIndices::Ptr inliers{new pcl::PointIndices};
    // TODO:: Fill in this function to find inliers for the cloud.
	pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
	// Create the segmentation object
	pcl::SACSegmentation<PointT> seg;
	// Optional
	seg.setOptimizeCoefficients (true);
	// Mandatory
	seg.setModelType (pcl::SACMODEL_PLANE);
	seg.setMethodType (pcl::SAC_RANSAC);
	seg.setMaxIterations (maxIterations);
	seg.setDistanceThreshold (distanceThreshold);

	// Segment the largest planar component from the remaining cloud
	seg.setInputCloud (cloud);
	seg.segment (*inliers, *coefficients);
	if (inliers->indices.size () == 0)
	{
	  LOG_ERROR_S << "Could not estimate a planar model for the given dataset.";
	}
	std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult = SeparateClouds(inliers,cloud);
    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    LOG_INFO_S << "plane segmentation took " << elapsedTime.count() << " milliseconds";


    return segResult;
}
/*SegmentPlane_RANSAC function implements RANSAC function.
 * It is used for identifying the road surface.
 * Algo: Randomly choose 3 points from the cloud, form a plane using these points.
 * 		 Loop through all the points in the cloud, for each of the point calculate
 * 		 	the distance to the plane created above. If the distance is below distanceTol
 * 		 	then add the index to a temporary set
 * 		 Store the temporary vector if the size is more than previously identified indices.
 * 		 Repeat above steps for maxIterations
 * */
template<typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::SegmentPlane_RANSAC(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceThreshold)
{
    // Time segmentation process
    auto startTime = std::chrono::steady_clock::now();

    /*Buffer to hold the indices of the points within distanceTol , it shall hold max identified indices*/
	std::unordered_set<int> inliersResult;
	srand(time(NULL));
	PointT point1;
	PointT point2;
	PointT point3;
	int idx1;
	int idx2;
	int idx3;
	float a,b,c,d,dis,len;

	// For max iterations
	for(int it=0;it<maxIterations;it++)
	{
		/*Temporary buffer to hold identified points in current loop*/
		std::unordered_set<int> tempIndices;
		/*Identify 3 points randomly*/
		while(tempIndices.size()<3)
			tempIndices.insert((rand() % cloud->points.size()));
		auto iter = tempIndices.begin();
		idx1 = *iter;
		++iter;
		idx2 = *iter;
		++iter;
		idx3 = *iter;

		point1 = cloud->points[idx1];
		point2 = cloud->points[idx2];
		point3 = cloud->points[idx3];

		/*Fit a plane using the above 3 points*/
		a = (((point2.y-point1.y)*(point3.z-point1.z))-((point2.z-point1.z)*(point3.y-point1.y)));
		b = (((point2.z-point1.z)*(point3.x-point1.x))-((point2.x-point1.x)*(point3.z-point1.z)));
		c = (((point2.x-point1.x)*(point3.y-point1.y))-((point2.y-point1.y)*(point3.x-point1.x)));
		d = -(a*point1.x+b*point1.y+c*point1.z);
		len = sqrt(a*a+b*b+c*c);

		// Measure distance between every point and fitted plane
		for(int pt_cnt=0;pt_cnt<cloud->points.size();pt_cnt++)
		{
			if(pt_cnt!=idx1||pt_cnt!=idx2||pt_cnt!=idx3)
			{
				dis = (fabs(a*cloud->points[pt_cnt].x+b*cloud->points[pt_cnt].y+c*cloud->points[pt_cnt].z+d)/len);
				// If distance is smaller than threshold count it as inlier
				if(dis<=distanceThreshold)
				{
					tempIndices.insert(pt_cnt);
				}
			}
		}

		/*Store the temporary buffer if the size if more than previously idenfitied points */
		if(tempIndices.size()>inliersResult.size())
		{
			inliersResult.clear();
			inliersResult = tempIndices;

		}

	}

	// Segment the largest planar component from the remaining cloud
	if (inliersResult.size () == 0)
	{
	  LOG_ERROR_S << "Could not estimate a planar model for the given dataset.";
	}
	/*Buffers to hold cloud and object points*/
	typename pcl::PointCloud<PointT>::Ptr cloudInliers(new pcl::PointCloud<PointT>());
	typename pcl::PointCloud<PointT>::Ptr cloudOutliers(new pcl::PointCloud<PointT>());

	/*Copy the points from inputcloud in to cloudInliers if the indices is in inliersResult vector
	 * or else copy the point to cloudOutliers*/
	for(int index = 0; index < cloud->points.size(); index++)
	{
		PointT point = cloud->points[index];
		if(inliersResult.count(index))
			cloudInliers->points.push_back(point);
		else
			cloudOutliers->points.push_back(point);
	}
	/*Create a pair using inlier and outlier points*/
	std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult(cloudOutliers, cloudInliers);

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    LOG_INFO_S << "plane segmentation took " << elapsedTime.count() << " milliseconds";


    return segResult;
}



template<typename PointT>
std::vector<typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::Clustering(typename pcl::PointCloud<PointT>::Ptr cloud, float clusterTolerance, int minSize, int maxSize)
{

    // Time clustering process
    auto startTime = std::chrono::steady_clock::now();

    std::vector<typename pcl::PointCloud<PointT>::Ptr> clusters;

    // TODO:: Fill in the function to perform euclidean clustering to group detected obstacles
    // Creating the KdTree object for the search method of the extraction
	typename pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
	tree->setInputCloud (cloud);

	std::vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<PointT> ec;
	ec.setClusterTolerance (clusterTolerance); // 2cm
	ec.setMinClusterSize (minSize);
	ec.setMaxClusterSize (maxSize);
	ec.setSearchMethod (tree);
	ec.setInputCloud (cloud);
	ec.extract (cluster_indices);

	for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
	  {
		typename pcl::PointCloud<PointT>::Ptr cloud_cluster (new pcl::PointCloud<PointT>);
	    for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
	      cloud_cluster->points.push_back (cloud->points[*pit]); //*
	    cloud_cluster->width = cloud_cluster->points.size ();
	    cloud_cluster->height = 1;
	    cloud_cluster->is_dense = true;

	    clusters.push_back(cloud_cluster);
	  }
    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    LOG_INFO_S << "clustering took " << elapsedTime.count() << " milliseconds and found " << clusters.size() << " clusters";

    return clusters;
}

/* Proximity function shall identify all the points that are within distanceTol
 * distance from the given point in the cloud and return the indices of the points
 * This is a recurssive function.
 * Algo: If the target point is not processed then set is as processed and search
 * 			the KDTree for all the points within the distanceTol
 * 		 Use each of the nearby points and search for other points that are within
 * 		 distanceTol distance from this points
 *
 * */
template<typename PointT>
void ProcessPointClouds<PointT>::Proximity(typename pcl::PointCloud<PointT>::Ptr cloud,std::vector<int> &cluster,std::vector<bool> &processed_f,int idx,typename KdTree_euclidean<PointT>::KdTree_euclidean* tree,float distanceTol, int maxSize)
{
	if((processed_f[idx]==false)&&
			(cluster.size()<maxSize))
	{
		processed_f[idx]=true;
		cluster.push_back(idx);
		std::vector<int> nearby = tree->search(cloud->points[idx],distanceTol);
		for(int index : nearby)
		{
			if(processed_f[index]==false)
			{
				Proximity(cloud, cluster,processed_f,index,tree,distanceTol,maxSize);
			}
		}
	}

}
/* euclideanCluster function shall identify clusters that have points with in min and max limits
 * Algo: Take one point at a time from the cluster , call Proximity function to identify the
 * 			list of points that are within distanceTol limits
 * 		 Check if the no of points in cluster ,returned by proximity function, are in (minSize, maxSize)
 * 		 	limits if not discard
 * */
template<typename PointT>
std::vector<std::vector<int>> ProcessPointClouds<PointT>::euclideanCluster(typename pcl::PointCloud<PointT>::Ptr cloud, typename KdTree_euclidean<PointT>::KdTree_euclidean* tree, float distanceTol, int minSize, int maxSize)
{
	std::vector<std::vector<int>> clusters;
	/*Create a flag for each point in the cloud, to identified if the point is processed or not, and set it to false*/
	std::vector<bool> processed_flag(cloud->points.size(),false);

	/*Loop through each point of the cloud*/
	for(int idx=0;idx<cloud->points.size();idx++)
	{
		/*Pass the point to Proximity function only if it was not processed
		 * (either added to a cluster or discarded)*/
		if(processed_flag[idx]==false)
		{
			std::vector<int> cluster;
			/*Call Proximity function to identify all the points that are
			 * within in distanceTol distance*/
			Proximity(cloud, cluster,processed_flag,idx,tree,distanceTol,maxSize);
			/*Check if the number of points in the identified cluster are with in limits */
			if((cluster.size()>=minSize)&&cluster.size()<=maxSize)
				clusters.push_back(cluster);
			/*else
				LOG_INFO_S <<"discarted cluster"<<cluster.size();*/
		}

	}
	/*LOG_INFO_S<<"Distance Tolerance"<<distanceTol;
	LOG_INFO_S<<"Max Distance "<<tree->max_distance;*/
	return clusters;

}

/* Clustering_euclideanCluster function shall identify the cluster of point that have given
 * no of min, max points and meet the cluster tolerance requirement.
 * Algo: Using points in the given cloud KDTree is formed.
 * 		 Using Euclidena Clustering, clusters are searched in the created KDTree
 * 		 Identified clusters are filtered, clusters that dont have points in min, max points are discarded.
 * */
template<typename PointT>
std::vector<typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::Clustering_euclideanCluster(typename pcl::PointCloud<PointT>::Ptr cloud, float clusterTolerance, int minSize, int maxSize)
{

    // Time clustering process
    auto startTime = std::chrono::steady_clock::now();

    std::vector<typename pcl::PointCloud<PointT>::Ptr> clusters;

    // Create the KdTree object using the points in cloud.
    typename KdTree_euclidean<PointT>::KdTree_euclidean *tree =new KdTree_euclidean<PointT>;
    tree->insert_cloud(cloud);

    //perform euclidean clustering to group detected obstacles
	std::vector<std::vector<int>> cluster_indices = euclideanCluster(cloud, tree,clusterTolerance ,minSize,maxSize);

	for (std::vector<std::vector<int>>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
	  {
		typename pcl::PointCloud<PointT>::Ptr cloud_cluster (new pcl::PointCloud<PointT>);
	    for (std::vector<int>::const_iterator pit = it->begin (); pit != it->end (); ++pit)
	      cloud_cluster->points.push_back (cloud->points[*pit]); //*
	    cloud_cluster->width = cloud_cluster->points.size ();
	    cloud_cluster->height = 1;
	    cloud_cluster->is_dense = true;

	    clusters.push_back(cloud_cluster);
	  }
    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    LOG_INFO_S << "euclideanClustering took " << elapsedTime.count() << " milliseconds and found " << clusters.size() << " clusters";

    return clusters;
}

/*BoundingBox function shall identify the min and max coordinates
 *in the provided cluster, a box shall be fitted using these min
 *and max coordinates
 * */
template<typename PointT>
pointcloud_obstacle_detection::Box ProcessPointClouds<PointT>::BoundingBox(typename pcl::PointCloud<PointT>::Ptr cluster)
{

    // Find bounding box for one of the clusters
    PointT minPoint, maxPoint;
    /*Get min and max coordinates in the cluster*/
    pcl::getMinMax3D(*cluster, minPoint, maxPoint);

    pointcloud_obstacle_detection::Box box;
    box.x_min = minPoint.x;
    box.y_min = minPoint.y;
    box.z_min = minPoint.z;
    box.x_max = maxPoint.x;
    box.y_max = maxPoint.y;
    box.z_max = maxPoint.z;

	/*LOG_INFO_S << "Max x: " << maxPoint.x;
	LOG_INFO_S << "Max y: " << maxPoint.y;
	LOG_INFO_S << "Max z: " << maxPoint.z;
	LOG_INFO_S << "Min x: " << minPoint.x;
	LOG_INFO_S << "Min y: " << minPoint.y;
	LOG_INFO_S << "Min z: " << minPoint.z;*/

	return box;
}

/*OrientedBoundingBox function shall identify the min and max coordinates
 *in the provided cluster, a box shall be fitted using these min
 *and max coordinates
 * */
template<typename PointT>
base::samples::OrientedBoundingBox ProcessPointClouds<PointT>::OrientedBoundingBox(typename pcl::PointCloud<PointT>::Ptr cluster, const base::Time& ts)
{
		pcl::MomentOfInertiaEstimation <pcl::PointXYZI> feature_extractor;
		feature_extractor.setInputCloud(cluster);
		feature_extractor.compute();

  	pcl::PointXYZI min_point_OBB;
		pcl::PointXYZI max_point_OBB;
		pcl::PointXYZI position_OBB;
		Eigen::Matrix3f rotational_matrix_OBB;
		feature_extractor.getOBB (min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);

		Eigen::Vector3d position{position_OBB.x, position_OBB.y, position_OBB.z};
		Eigen::Quaternionf quat(rotational_matrix_OBB);
		Eigen::Vector3d size{max_point_OBB.x - min_point_OBB.x, max_point_OBB.y - min_point_OBB.y, max_point_OBB.z - min_point_OBB.z};
		Eigen::Quaterniond orientation;
		orientation.x() = quat.x();
		orientation.y() = quat.y();
		orientation.z() = quat.z();
		orientation.w() = quat.w();

    base::samples::OrientedBoundingBox box;
		box.initOrientedBoundingBox(ts, position, size, orientation);
	return box;
}

/*AxisAlignedBoundingBox function shall identify the min and max coordinates
 *in the provided cluster, a box shall be fitted using these min
 *and max coordinates
 * */
/*
template<typename PointT>
pointcloud_obstacle_detection::Box ProcessPointClouds<PointT>::AxisAlignedBoundingBox(typename pcl::PointCloud<PointT>::Ptr cluster)
{
		pcl::MomentOfInertiaEstimation <pcl::PointXYZI> feature_extractor;
		feature_extractor.setInputCloud(cluster);
		feature_extractor.compute();

		pcl::PointXYZI min_point_AABB;
		pcl::PointXYZI max_point_AABB;
		feature_extractor.getAABB (min_point_AABB, max_point_AABB);

    pointcloud_obstacle_detection::Box box;
    box.x_min = min_point_AABB.x;
    box.y_min = min_point_AABB.y;
    box.z_min = min_point_AABB.z;
    box.x_max = max_point_AABB.x;
    box.y_max = max_point_AABB.y;
    box.z_max = max_point_AABB.z;

	return box;
}
*/
template<typename PointT>
void ProcessPointClouds<PointT>::savePcd(typename pcl::PointCloud<PointT>::Ptr cloud, std::string file)
{
    pcl::io::savePCDFileASCII (file, *cloud);
    LOG_INFO_S << "Saved " << cloud->points.size () << " data points to "+file;
}


template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr ProcessPointClouds<PointT>::loadPcd(std::string file)
{

    typename pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);

    if (pcl::io::loadPCDFile<PointT> (file, *cloud) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file \n");
    }
    LOG_INFO_S << "Loaded " << cloud->points.size () << " data points from "+file;

    return cloud;
}


template<typename PointT>
std::vector<boost::filesystem::path> ProcessPointClouds<PointT>::streamPcd(std::string dataPath)
{

    std::vector<boost::filesystem::path> paths(boost::filesystem::directory_iterator{dataPath}, boost::filesystem::directory_iterator{});

    // sort files in accending order so playback is chronological
    sort(paths.begin(), paths.end());

    return paths;

}
} //namespace pointcloud_obstacle_detection

#endif /* PROCESSPOINTCLOUDS_H_ */
