/**
 * @file ProcessCloudProcessor.hpp
 * @brief Preprocessing utilities for point cloud filtering and clustering.
 *
 * Includes:
 *  - ROI filtering
 *  - Voxel downsampling
 *  - Euclidean clustering
 *  - PCD I/O
 */

#pragma once

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
#include <unordered_set>

namespace ground_segmentation
{

template<typename PointT>
class ProcessCloudProcessor
{

public:
  typename pcl::PointCloud<PointT>::Ptr filterCloud(
    typename pcl::PointCloud<PointT>::Ptr cloud, bool downSampleInputCloud, float filterRes,
    Eigen::Vector4f minPoint,
    Eigen::Vector4f maxPoint,
    bool invertCropBox);

  std::vector<typename pcl::PointCloud<PointT>::Ptr> euclideanClustering(
    typename pcl::PointCloud<PointT>::Ptr cloud,
    float clusterTolerance, int minSize, int maxSize);

  void savePcd(typename pcl::PointCloud<PointT>::Ptr cloud, std::string file);
  typename pcl::PointCloud<PointT>::Ptr loadPcd(std::string file);
};

template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr ProcessCloudProcessor<PointT>::filterCloud(
  typename pcl::PointCloud<PointT>::Ptr cloud,
  bool downSampleInputCloud,
  float filterRes,
  Eigen::Vector4f minPoint,
  Eigen::Vector4f maxPoint,
  bool invertCropBox
)
{
  typename pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>());
  pcl::CropBox<PointT> roi;
  roi.setMin(minPoint);
  roi.setMax(maxPoint);
  roi.setNegative(invertCropBox);

  if (downSampleInputCloud) {
    pcl::VoxelGrid<PointT> sor;
    sor.setInputCloud(cloud);
    sor.setLeafSize(filterRes, filterRes, filterRes);
    sor.filter(*cloud_filtered);
    roi.setInputCloud(cloud_filtered);
  } else {
    roi.setInputCloud(cloud);
  }
  roi.filter(*cloud_filtered);
  return cloud_filtered;
}

template<typename PointT>
std::vector<typename pcl::PointCloud<PointT>::Ptr> ProcessCloudProcessor<PointT>::
euclideanClustering(
  typename pcl::PointCloud<PointT>::Ptr cloud,
  float clusterTolerance, int minSize, int maxSize)
{
  std::vector<typename pcl::PointCloud<PointT>::Ptr> clusters;
  typename pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
  tree->setInputCloud(cloud);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<PointT> ec;
  ec.setClusterTolerance(clusterTolerance);
  ec.setMinClusterSize(minSize);
  ec.setMaxClusterSize(maxSize);
  ec.setSearchMethod(tree);
  ec.setInputCloud(cloud);
  ec.extract(cluster_indices);

  for (const auto & cluster : cluster_indices) {
    typename pcl::PointCloud<PointT>::Ptr cloud_cluster(new pcl::PointCloud<PointT>);
    for (const auto & idx : cluster.indices) {
      cloud_cluster->push_back((*cloud)[idx]);
    }
    cloud_cluster->width = cloud_cluster->size();
    cloud_cluster->height = 1;
    cloud_cluster->is_dense = true;
    clusters.push_back(cloud_cluster);
  }
  return clusters;
}

template<typename PointT>
void ProcessCloudProcessor<PointT>::savePcd(
  typename pcl::PointCloud<PointT>::Ptr cloud,
  std::string file)
{
  pcl::io::savePCDFileASCII(file, *cloud);
}

template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr ProcessCloudProcessor<PointT>::loadPcd(std::string file)
{
  typename pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
  if (pcl::io::loadPCDFile<PointT>(file, *cloud) == -1) {    //* load the file
    std::cerr << "Failed to read PCD file: " << file << std::endl;
  }
  return cloud;
}
} //namespace ground_segmentation
