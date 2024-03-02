// PCL lib Functions for processing point clouds 

#include "process_pointcloud.hpp"

namespace pointcloud_obstacle_detection{
    //constructor:
  ProcessPointCloud::ProcessPointCloud() {}


  //de-constructor:  
  ProcessPointCloud::~ProcessPointCloud() {}
  
  /*FilterCloud function filters the given cloud. Following operations are performed
  * Downsampling: points are converted to voxels using the dimensions provided.
  * Crop: Remove all the points that are outside the min , max limits
  * RoofCrop: Remove roof points , dimensions of roof are given.
  * */
 
  pcl::PointCloud<pcl::PointXYZ>::Ptr ProcessPointCloud::FilterCloud( pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, bool downSampleInputCloud, float filterRes, Eigen::Vector4f minPoint, Eigen::Vector4f maxPoint)
  {

      // Time segmentation process
      auto startTime = std::chrono::steady_clock::now();
       pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>());
    // Crop the scene to create ROI
      pcl::CropBox<pcl::PointXYZ> roi;
      roi.setMin(minPoint);
      roi.setMax(maxPoint);
    
    if (downSampleInputCloud){
      // Convert the points to voxel grid points
      pcl::VoxelGrid<pcl::PointXYZ> sor;
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


  
  std::pair< pcl::PointCloud<pcl::PointXYZ>::Ptr,  pcl::PointCloud<pcl::PointXYZ>::Ptr> ProcessPointCloud::SeparateClouds(pcl::PointIndices::Ptr inliers,  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
  {
    // TODO: Create two new point clouds, one cloud with obstacles and other with segmented plane
     pcl::PointCloud<pcl::PointXYZ>::Ptr obstCloud(new pcl::PointCloud<pcl::PointXYZ>());
     pcl::PointCloud<pcl::PointXYZ>::Ptr planeCloud(new pcl::PointCloud<pcl::PointXYZ>());

    for(int it : inliers->indices)
    {
      planeCloud->points.push_back(cloud->points[it]);
    }

    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud (cloud);
    extract.setIndices (inliers);
    extract.setNegative (true);
    extract.filter (*obstCloud);

      std::pair< pcl::PointCloud<pcl::PointXYZ>::Ptr,  pcl::PointCloud<pcl::PointXYZ>::Ptr> segResult(obstCloud, planeCloud);
      return segResult;
  }


  
  std::pair< pcl::PointCloud<pcl::PointXYZ>::Ptr,  pcl::PointCloud<pcl::PointXYZ>::Ptr> ProcessPointCloud::SegmentPlane( pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int maxIterations, float distanceThreshold)
  {
      // Time segmentation process
      auto startTime = std::chrono::steady_clock::now();
    pcl::PointIndices::Ptr inliers{new pcl::PointIndices};
      // TODO:: Fill in this function to find inliers for the cloud.
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZ> seg;
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
    std::pair< pcl::PointCloud<pcl::PointXYZ>::Ptr,  pcl::PointCloud<pcl::PointXYZ>::Ptr> segResult = SeparateClouds(inliers,cloud);
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
  
  std::pair< pcl::PointCloud<pcl::PointXYZ>::Ptr,  pcl::PointCloud<pcl::PointXYZ>::Ptr> ProcessPointCloud::SegmentPlane_RANSAC( pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int maxIterations, float distanceThreshold)
  {
      // Time segmentation process
      auto startTime = std::chrono::steady_clock::now();

      /*Buffer to hold the indices of the points within distanceTol , it shall hold max identified indices*/
    std::unordered_set<int> inliersResult;
    srand(time(NULL));
    pcl::PointXYZ point1;
    pcl::PointXYZ point2;
    pcl::PointXYZ point3;
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
     pcl::PointCloud<pcl::PointXYZ>::Ptr cloudInliers(new pcl::PointCloud<pcl::PointXYZ>());
     pcl::PointCloud<pcl::PointXYZ>::Ptr cloudOutliers(new pcl::PointCloud<pcl::PointXYZ>());

    /*Copy the points from inputcloud in to cloudInliers if the indices is in inliersResult vector
    * or else copy the point to cloudOutliers*/
    for(int index = 0; index < cloud->points.size(); index++)
    {
      pcl::PointXYZ point = cloud->points[index];
      if(inliersResult.count(index))
        cloudInliers->points.push_back(point);
      else
        cloudOutliers->points.push_back(point);
    }
    /*Create a pair using inlier and outlier points*/
    std::pair< pcl::PointCloud<pcl::PointXYZ>::Ptr,  pcl::PointCloud<pcl::PointXYZ>::Ptr> segResult(cloudOutliers, cloudInliers);

      auto endTime = std::chrono::steady_clock::now();
      auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
      LOG_INFO_S << "plane segmentation took " << elapsedTime.count() << " milliseconds";


      return segResult;
  }

  std::vector< pcl::PointCloud<pcl::PointXYZ>::Ptr> ProcessPointCloud::Clustering( pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float clusterTolerance, int minSize, int maxSize)
  {

      // Time clustering process
      auto startTime = std::chrono::steady_clock::now();

      std::vector< pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters;

      // TODO:: Fill in the function to perform euclidean clustering to group detected obstacles
      // Creating the KdTree object for the search method of the extraction
     pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud (cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance (clusterTolerance); // 2cm
    ec.setMinClusterSize (minSize);
    ec.setMaxClusterSize (maxSize);
    ec.setSearchMethod (tree);
    ec.setInputCloud (cloud);
    ec.extract (cluster_indices);

    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
      {
       pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
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
  
  void ProcessPointCloud::Proximity( pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,std::vector<int> &cluster,std::vector<bool> &processed_f,int idx, KdTree_euclidean* tree,float distanceTol, int maxSize)
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
  
  std::vector<std::vector<int>> ProcessPointCloud::euclideanCluster( pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,  KdTree_euclidean* tree, float distanceTol, int minSize, int maxSize)
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
  
  std::vector< pcl::PointCloud<pcl::PointXYZ>::Ptr> ProcessPointCloud::Clustering_euclideanCluster( pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float clusterTolerance, int minSize, int maxSize)
  {

      // Time clustering process
      auto startTime = std::chrono::steady_clock::now();

      std::vector< pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters;

      // Create the KdTree object using the points in cloud.
      KdTree_euclidean *tree =new KdTree_euclidean;
      tree->insert_cloud(cloud);

      //perform euclidean clustering to group detected obstacles
    std::vector<std::vector<int>> cluster_indices = euclideanCluster(cloud, tree,clusterTolerance ,minSize,maxSize);

    for (std::vector<std::vector<int>>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
      {
       pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
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

  std::vector<base::samples::Pointcloud> ProcessPointCloud::Clustering_euclideanCluster_Base( pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float clusterTolerance, int minSize, int maxSize)
  {

      // Time clustering process
      auto startTime = std::chrono::steady_clock::now();

      std::vector<base::samples::Pointcloud> clusters;

      // Create the KdTree object using the points in cloud.
      KdTree_euclidean *tree =new KdTree_euclidean;
      tree->insert_cloud(cloud);

      //perform euclidean clustering to group detected obstacles
    std::vector<std::vector<int>> cluster_indices = euclideanCluster(cloud, tree,clusterTolerance ,minSize,maxSize);

    for (std::vector<std::vector<int>>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
      {
       base::samples::Pointcloud cloud_cluster;
        for (std::vector<int>::const_iterator pit = it->begin (); pit != it->end (); ++pit)
          cloud_cluster.points.emplace_back(base::Point(cloud->points[*pit].x,cloud->points[*pit].y,cloud->points[*pit].z)); //*

        clusters.push_back(cloud_cluster);
      }
      auto endTime = std::chrono::steady_clock::now();
      auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
      LOG_INFO_S << "euclideanClustering took " << elapsedTime.count() << " milliseconds and found " << clusters.size() << " clusters";

      return clusters;
  }

  /*OrientedBoundingBox function shall identify the min and max coordinates
  *in the provided cluster, a box shall be fitted using these min
  *and max coordinates
  * */
  
  base::samples::OrientedBoundingBox ProcessPointCloud::OrientedBoundingBox( pcl::PointCloud<pcl::PointXYZ>::Ptr cluster, const base::Time& ts)
  {
      pcl::MomentOfInertiaEstimation <pcl::PointXYZ> feature_extractor;
      feature_extractor.setInputCloud(cluster);
      feature_extractor.compute();

      pcl::PointXYZ min_point_OBB;
      pcl::PointXYZ max_point_OBB;
      pcl::PointXYZ position_OBB;
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
 
  void ProcessPointCloud::savePcd( pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::string file)
  {
      pcl::io::savePCDFileASCII (file, *cloud);
      LOG_INFO_S << "Saved " << cloud->points.size () << " data points to "+file;
  }
  
  pcl::PointCloud<pcl::PointXYZ>::Ptr ProcessPointCloud::loadPcd(std::string file)
  {
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

      if (pcl::io::loadPCDFile<pcl::PointXYZ> (file, *cloud) == -1) //* load the file
      {
          LOG_ERROR_S << "Couldn't read file: " << file;
      }
      LOG_INFO_S << "Loaded " << cloud->points.size () << " data points from " << file;
      return cloud;
  }

} //namespace pointcloud_obstacle_detection
