#pragma once

#include "ground_detection_types.hpp"
#include <nanoflann.hpp>
#include <unordered_map>

namespace ground_segmentation
{

template<typename PointT>
struct PCLPointCloudAdaptor
{
  typedef pcl::PointCloud<PointT> PointCloudType;

  PointCloudType & cloud;

  PCLPointCloudAdaptor(PointCloudType & cloud)
  : cloud(cloud) {}

  // Must return the number of data points
  inline size_t kdtree_get_point_count() const {return cloud.points.size();}

  // Returns the dim'th component of the idx'th point in the point cloud
  inline double kdtree_get_pt(const size_t idx, const size_t dim) const
  {
    if (dim == 0) {return cloud.points[idx].x;} else if (dim == 1) {
      return cloud.points[idx].y;
    } else {return cloud.points[idx].z;}
  }

  // Optional bounding-box computation
  template<class BBOX>
  bool kdtree_get_bbox(BBOX & /*bb*/) const {return false;}
};

template<typename PointT>
class PointCloudGrid
{

public:
  PointCloudGrid(const GridConfig & config);
  void clear();
  typedef GridCell<PointT> CellType;
  typedef std::unordered_map<Index3D, CellType, Index3D::HashFunction> GridCellsType;
  void setInputCloud(
    typename pcl::PointCloud<PointT>::Ptr input,
    const Eigen::Quaterniond & R_body2World);
  std::pair<typename pcl::PointCloud<PointT>::Ptr,
    typename pcl::PointCloud<PointT>::Ptr> segmentPoints();
  GridCellsType & getGridCells() {return gridCells;}

  // Build KD-Tree
  typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double,
      PCLPointCloudAdaptor<PointT>>,
      PCLPointCloudAdaptor<PointT>, 3> KDTree;
  bool checkIndex3DInGrid(const Index3D & index);

private:
  std::vector<Index3D> generateIndices(const uint16_t & z_threshold);
  void cleanUp();
  void addPoint(const PointT & point);
  void getGroundCells();
  std::vector<Index3D> getNeighbors(
    const GridCell<PointT> & cell, const TerrainType & type,
    const std::vector<Index3D> & neighbor_offsets);
  double computeSlope(const Eigen::Hyperplane<double, int(3)> & plane) const;
  double computeSlope(const Eigen::Vector3d & normal);
  bool fitGroundPlane(GridCell<PointT> & cell, const double & inlier_threshold);
  void expandGrid(std::queue<Index3D> q);
  std::string classifySparsityBoundingBox(
    const GridCell<PointT> & cell,
    typename pcl::PointCloud<PointT>::Ptr cloud);
  bool classifySparsityNormalDist(const GridCell<PointT> & cell);
  std::vector<Index3D> neighbor_offsets;

  GridCellsType gridCells;
  GridConfig grid_config;
  std::vector<Index3D> ground_cells;
  std::vector<Index3D> non_ground_cells;
  Eigen::Quaterniond orientation;

  // Add these members to your class:
  typename pcl::PointCloud<PointT>::Ptr centroid_cloud;
  std::vector<Index3D> centroid_indices;                // Corresponding Index3D for each centroid
  std::unordered_map<Index3D, size_t, Index3D::HashFunction> index_to_centroid_idx;

};
template<typename PointT>
PointCloudGrid<PointT>::PointCloudGrid(const GridConfig & config)
{

  centroid_cloud.reset(new pcl::PointCloud<PointT>());
  grid_config = config;

  if (grid_config.processing_phase == 2) {
    neighbor_offsets = generateIndices(1);
  } else {
    neighbor_offsets = generateIndices(0);
  }
}

template<typename PointT>
std::vector<Index3D> PointCloudGrid<PointT>::generateIndices(const uint16_t & z_threshold)
{
  std::vector<Index3D> idxs;

  for (int dx = -1; dx <= 1; ++dx) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dz = -1; dz <= z_threshold; ++dz) {
        if (dx == 0 && dy == 0 && dz == 0) {
          continue;
        }
        Index3D idx;
        idx.x = dx;
        idx.y = dy;
        idx.z = dz;
        idxs.push_back(idx);
      }
    }
  }
  return idxs;
}

template<typename PointT>
void PointCloudGrid<PointT>::clear()
{
  gridCells.clear();
}

template<typename PointT>
void PointCloudGrid<PointT>::cleanUp()
{
  ground_cells.clear();
  non_ground_cells.clear();
  centroid_cloud->clear();
  centroid_indices.clear();
  index_to_centroid_idx.clear();
}

template<typename PointT>
void PointCloudGrid<PointT>::addPoint(const PointT & point)
{
  double cell_x = point.x / grid_config.cellSizeX;
  double cell_y = point.y / grid_config.cellSizeY;
  double cell_z = point.z / grid_config.cellSizeZ;

  int x = static_cast<int>(std::floor(cell_x));
  int y = static_cast<int>(std::floor(cell_y));
  int z = static_cast<int>(std::floor(cell_z));

  CellType & cell = gridCells[{x, y, z}];
  // information is redundant:
  cell.x = x;
  cell.y = y;
  cell.z = z;
  cell.points->push_back(point);
}

template<typename PointT>
double PointCloudGrid<PointT>::computeSlope(const Eigen::Vector3d & normal)
{
  const Eigen::Vector3d zNormal(Eigen::Vector3d::UnitZ());
  Eigen::Vector3d planeNormal = normal;
  planeNormal = orientation * planeNormal;
  planeNormal.normalize();
  return acos(std::abs(planeNormal.dot(zNormal)));
}

template<typename PointT>
double PointCloudGrid<PointT>::computeSlope(const Eigen::Hyperplane<double, int(3)> & plane) const
{
  const Eigen::Vector3d zNormal(Eigen::Vector3d::UnitZ());
  Eigen::Vector3d planeNormal = plane.normal();
  planeNormal = orientation * planeNormal;
  planeNormal.normalize();
  return acos(std::abs(planeNormal.dot(zNormal)));
}

template<typename PointT>
std::string PointCloudGrid<PointT>::classifySparsityBoundingBox(
  const GridCell<PointT> & cell,
  typename pcl::PointCloud<PointT>::Ptr cloud)
{
  if (cell.points->empty() || cloud->empty()) {return "Empty";}

  PointT min_pt, max_pt;
  pcl::getMinMax3D(*cell.points, min_pt, max_pt);

  double volume = (max_pt.x - min_pt.x) *
    (max_pt.y - min_pt.y) *
    (max_pt.z - min_pt.z);

  if (volume <= 0.0) {return "Degenerate";}

  double sparsity = volume / static_cast<double>(cloud->size());

  if (sparsity < 0.001) {
    return "Low sparsity";
  } else if (sparsity < 0.01) {
    return "Medium sparsity";
  } else {
    return "High sparsity";
  }
}

template<typename PointT>
bool PointCloudGrid<PointT>::classifySparsityNormalDist(const GridCell<PointT> & cell)
{
  if (cell.points->empty()) {return false;}

  // Assume cell.centroid and cell.normal are Eigen::Vector3d and cell.normal is normalized
  double sum_abs_proj = 0.0;

  for (const auto & pt : cell.points->points) {
    Eigen::Vector3d diff(pt.x - cell.centroid[0], pt.y - cell.centroid[1], pt.z - cell.centroid[2]);
    double proj = diff.dot(cell.normal);     // projection along normal
    sum_abs_proj += std::abs(proj);
  }

  double mean_abs_proj = sum_abs_proj / cell.points->size();

  // Threshold: tune for your use case (0.1 = flat/ground)
  return mean_abs_proj < 0.1;
}

template<typename PointT>
std::vector<Index3D> PointCloudGrid<PointT>::getNeighbors(
  const GridCell<PointT> & cell,
  const TerrainType & type,
  const std::vector<Index3D> & idx)
{

  std::vector<Index3D> neighbors;

  Index3D cell_id;
  cell_id.x = cell.x;
  cell_id.y = cell.y;
  cell_id.z = cell.z;

  for (uint i = 0; i < idx.size(); ++i) {
    Index3D neighbor_id = cell_id + idx[i];

    if (!checkIndex3DInGrid(neighbor_id)) {
      continue;
    }

    const GridCell<PointT> & neighbor = gridCells[neighbor_id];

    if (neighbor.points->size() > 0 && neighbor.terrain_type == type) {
      Index3D id;
      id.x = neighbor.x;
      id.y = neighbor.y;
      id.z = neighbor.z;
      neighbors.push_back(id);
    }
  }
  return neighbors;
}

template<typename PointT>
bool PointCloudGrid<PointT>::fitGroundPlane(GridCell<PointT> & cell, const double & threshold)
{

  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  pcl::SACSegmentation<PointT> seg;
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_PROSAC);
  seg.setMaxIterations(1000);
  seg.setInputCloud(cell.points);
  seg.setDistanceThreshold(threshold);   // Adjust this threshold based on your needs
  seg.segment(*inliers, *coefficients);
  cell.inliers = inliers;
  if (cell.inliers->indices.size() == 0) {
    return false;
  }

  Eigen::Vector3d plane_normal(coefficients->values[0], coefficients->values[1],
    coefficients->values[2]);
  double distToOrigin = coefficients->values[3];
  auto plane = Eigen::Hyperplane<double, 3>(plane_normal, distToOrigin);
  cell.slope = computeSlope(plane);
  return true;
}

template<typename PointT>
void PointCloudGrid<PointT>::getGroundCells()
{

  if (gridCells.empty()) {
    return;
  }

  this->cleanUp();

  for (auto & cellPair : gridCells) {
    GridCell<PointT> & cell = cellPair.second;

    //Too few points
    if ((cell.points->size() < 3)) {continue;}

    Index3D cell_id = cellPair.first;
    pcl::compute3DCentroid(*(cell.points), cell.centroid);

    if (cell.points->size() <= 5) {
      Eigen::Vector4f squared_diff_sum(0, 0, 0, 0);

      for (typename pcl::PointCloud<PointT>::iterator it = cell.points->begin();
        it != cell.points->end(); ++it)
      {
        Eigen::Vector4f diff = (*it).getVector4fMap() - cell.centroid.template cast<float>();
        squared_diff_sum += diff.array().square().matrix();
      }

      Eigen::Vector4f variance = squared_diff_sum / cell.points->size();

      if (variance[0] < variance[2] && variance[1] < variance[2]) {
        cell.terrain_type = TerrainType::OBSTACLE;
        non_ground_cells.push_back(cell_id);
      } else {
        cell.terrain_type = TerrainType::GROUND;
        PointT centroid3d;

        centroid3d.x = cell.centroid[0];
        centroid3d.y = cell.centroid[1];
        centroid3d.z = cell.centroid[2];

        centroid_cloud->points.push_back(centroid3d);
        centroid_indices.push_back(cell_id);
        index_to_centroid_idx[cell_id] = centroid_cloud->size() - 1;
      }
      continue;
    }

    Eigen::Matrix3d covariance_matrix;
    pcl::computeCovarianceMatrixNormalized(*cell.points, cell.centroid, covariance_matrix);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(covariance_matrix,
      Eigen::ComputeEigenvectors);
    cell.eigenvectors = eigen_solver.eigenvectors();
    cell.eigenvalues = eigen_solver.eigenvalues();

    Eigen::Vector3d normal = cell.eigenvectors.col(0);

    // Ensure all normals point upward
    if (normal(2) < 0) {
      normal *= -1;       // flip the normal direction
    }

    normal.normalize();
    cell.normal = normal;

    double ratio = cell.eigenvalues[2] / cell.eigenvalues.sum();
    if (ratio > 0.950) {
      cell.primitive_type = PrimitiveType::LINE;

      Eigen::Vector3d v = cell.eigenvectors.col(2);
      if (v(2) < 0) {
        v *= -1;         // flip the normal direction
      }
      v = orientation * v;
      v.normalize();

      double angle_rad = acos(std::abs(v.dot(Eigen::Vector3d::UnitZ())));

      if (angle_rad > ((90 - grid_config.slopeThresholdDegrees) * (M_PI / 180))) {
        cell.terrain_type = TerrainType::GROUND;
        PointT centroid3d;

        centroid3d.x = cell.centroid[0];
        centroid3d.y = cell.centroid[1];
        centroid3d.z = cell.centroid[2];

        centroid_cloud->points.push_back(centroid3d);
        centroid_indices.push_back(cell_id);
        index_to_centroid_idx[cell_id] = centroid_cloud->size() - 1;
      } else {
        cell.terrain_type = TerrainType::OBSTACLE;
        non_ground_cells.push_back(cell_id);
      }
      continue;
    } else if (ratio > 0.4) {
      cell.primitive_type = PrimitiveType::PLANE;
      if (std::abs(computeSlope(cell.normal)) > grid_config.slopeThresholdDegrees) {
        cell.terrain_type = TerrainType::OBSTACLE;
        non_ground_cells.push_back(cell_id);
        continue;
      }
    } else {
      cell.terrain_type = TerrainType::OBSTACLE;
      cell.primitive_type = PrimitiveType::NOISE;
      non_ground_cells.push_back(cell_id);
      continue;
    }

    if (!fitGroundPlane(cell, grid_config.groundInlierThreshold)) {
      cell.terrain_type = TerrainType::OBSTACLE;
      non_ground_cells.push_back(cell_id);
      continue;
    }

    if (cell.slope < (grid_config.slopeThresholdDegrees * (M_PI / 180)) ) {
      cell.terrain_type = TerrainType::GROUND;
      PointT centroid3d;

      centroid3d.x = cell.centroid[0];
      centroid3d.y = cell.centroid[1];
      centroid3d.z = cell.centroid[2];

      centroid_cloud->points.push_back(centroid3d);
      centroid_indices.push_back(cell_id);
      index_to_centroid_idx[cell_id] = centroid_cloud->size() - 1;
    } else {
      cell.terrain_type = TerrainType::OBSTACLE;
      non_ground_cells.push_back(cell_id);
    }
  }

  std::queue<Index3D> q;

  bool found = false;
  Index3D best_robot_cell;
  int z_start = 0;       // Your starting z value
  int z_min = -100;      // The lowest z to check

  for (int z = z_start; z >= z_min; --z) {
    Index3D idx{0, 0, z};
    if (checkIndex3DInGrid(idx) && !gridCells[idx].points->empty()) {
      best_robot_cell = idx;
      found = true;
      break;       // stop at the first populated cell
    }
  }

  if (found) {
    GridCell<PointT> & robot_cell = gridCells[best_robot_cell];
    robot_cell.terrain_type = TerrainType::GROUND;

    PointT centroid3d;
    centroid3d.x = 0;
    centroid3d.y = 0;
    centroid3d.z = -grid_config.distToGround;     // Use your actual robot base height if needed

    centroid_cloud->points.push_back(centroid3d);
    centroid_indices.push_back(best_robot_cell);
    index_to_centroid_idx[best_robot_cell] = centroid_cloud->size() - 1;

    robot_cell.in_queue = true;     // <--- Set if using the in_queue pattern

    q.push(best_robot_cell);
    expandGrid(q);
  }
}

template<typename PointT>
void PointCloudGrid<PointT>::expandGrid(std::queue<Index3D> q)
{
  // Wrap the PCL point cloud with nanoflann adaptor
  PCLPointCloudAdaptor<PointT> pclAdaptor(*centroid_cloud);

  nanoflann::SearchParams search_params;
  search_params.eps = 0.0;      // Larger tolerance for faster results
  search_params.sorted = false;   // No need to sort

  nanoflann::KDTreeSingleIndexAdaptorParams build_params(10);   // leaf size
  KDTree tree(3, pclAdaptor, build_params);
  tree.buildIndex();

  const float radius = grid_config.centroidSearchRadius * grid_config.centroidSearchRadius;   // nanoflann uses squared radius

  while (!q.empty()) {
    Index3D idx = q.front();
    q.pop();

    GridCell<PointT> & current_cell = gridCells[idx];
    current_cell.in_queue = false;     // Mark as not in queue now that we're processing it

    if (current_cell.expanded || current_cell.points->empty()) {continue;}
    current_cell.expanded = true;

    // Find current centroid index
    size_t curr_centroid_idx = index_to_centroid_idx[idx];
    const PointT & curr_centroid = centroid_cloud->points.at(curr_centroid_idx);

    // Prepare radius search
    std::vector<std::pair<unsigned int, double>> neighbors;     // pair<index, squared distance>
    double query_pt[3] = {static_cast<double>(curr_centroid.x),
      static_cast<double>(curr_centroid.y),
      static_cast<double>(curr_centroid.z)};

    tree.radiusSearch(query_pt, radius, neighbors, search_params);

    for (const auto & nb : neighbors) {
      size_t ni = nb.first;
      if (ni == curr_centroid_idx) {
        continue;                                  // skip self

      }
      Index3D neighbor_id = centroid_indices[ni];
      if (neighbor_id == idx) {
        continue;                             // redundant but safe

      }
      GridCell<PointT> & neighbor = gridCells[neighbor_id];

      if (neighbor.points->empty() || neighbor.expanded || neighbor.in_queue) {continue;}

      if (grid_config.processing_phase == 2) {
        // Reject neighbor if centroid height difference is too large
        double dz = std::abs(current_cell.centroid[2] - neighbor.centroid[2]);
        if (dz > grid_config.groundInlierThreshold) {
          continue;
        }
        neighbor.terrain_type = TerrainType::GROUND;
      }

      if (neighbor.terrain_type == TerrainType::GROUND) {
        q.push(neighbor_id);
        neighbor.in_queue = true;
      }
    }
    ground_cells.emplace_back(idx);
  }
}

template<typename PointT>
void PointCloudGrid<PointT>::setInputCloud(
  typename pcl::PointCloud<PointT>::Ptr input,
  const Eigen::Quaterniond & R_body2World)
{

  this->clear();
  orientation = R_body2World;
  unsigned int index = 0;
  for (typename pcl::PointCloud<PointT>::iterator it = input->begin(); it != input->end(); ++it) {
    this->addPoint(*it);
    index++;
  }
}

template<typename PointT>
bool PointCloudGrid<PointT>::checkIndex3DInGrid(const Index3D & index)
{
  if (auto search = gridCells.find(index); search != gridCells.end()) {
    return true;
  } else {
    return false;
  }
}

template<typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr,
  typename pcl::PointCloud<PointT>::Ptr> PointCloudGrid<PointT>::segmentPoints()
{

  typename pcl::PointCloud<PointT>::Ptr ground_points(new pcl::PointCloud<PointT>());
  typename pcl::PointCloud<PointT>::Ptr ground_inliers(new pcl::PointCloud<PointT>());
  typename pcl::PointCloud<PointT>::Ptr non_ground_points(new pcl::PointCloud<PointT>());
  typename pcl::PointCloud<PointT>::Ptr non_ground_inliers(new pcl::PointCloud<PointT>());

  pcl::ExtractIndices<PointT> extract_ground;

  getGroundCells();
  for (auto & cell_id : ground_cells) {

    GridCell<PointT> & cell = gridCells[cell_id];

    if ((cell.points->size() <= 5 || cell.primitive_type == PrimitiveType::LINE) &&
      cell.terrain_type == TerrainType::GROUND)
    {
      *ground_points += *cell.points;
      continue;
    }

    extract_ground.setInputCloud(cell.points);
    extract_ground.setIndices(cell.inliers);

    extract_ground.setNegative(false);
    extract_ground.filter(*ground_inliers);

    extract_ground.setNegative(true);
    extract_ground.filter(*non_ground_inliers);

    if (ground_inliers->size() == 0) {
      continue;
    }

    auto score1 = classifySparsityBoundingBox(cell, ground_inliers);
    auto score2 = classifySparsityBoundingBox(cell, non_ground_inliers);

    if (score1 == score2) {
      Eigen::Vector4d centroid;
      pcl::compute3DCentroid(*(ground_inliers), centroid);

      // For each candidate ground cell:
      double cell_z = centroid[2];
      std::vector<double> neighbor_zs;
      for (const auto & offset : neighbor_offsets) {
        Index3D nidx = cell_id + offset;
        if (!checkIndex3DInGrid(nidx)) {continue;}
        const auto & ncell = gridCells[nidx];

        if (ncell.terrain_type == TerrainType::GROUND) {
          typename pcl::PointCloud<PointT>::Ptr ninliers(new pcl::PointCloud<PointT>());
          Eigen::Vector4d ncentroid;

          extract_ground.setInputCloud(ncell.points);
          extract_ground.setIndices(ncell.inliers);

          extract_ground.setNegative(false);
          extract_ground.filter(*ninliers);

          pcl::compute3DCentroid(*(ninliers), ncentroid);
          neighbor_zs.push_back(ncell.centroid[2]);
        }
      }

      if (neighbor_zs.empty()) {
        *non_ground_points += *cell.points;
        continue;
      }

      double local_ref = *std::min_element(neighbor_zs.begin(), neighbor_zs.end());

      if (cell_z - local_ref > 0.3) {
        *non_ground_points += *cell.points;
        continue;
      }

      bool reject_as_floating = false;
      int bz = cell_id.z - 1;
      while (true) {
        Index3D below(cell_id.x, cell_id.y, bz);
        if (!checkIndex3DInGrid(below)) {
          break;                                        // Out of bound: stop looping
        }
        const auto & bcell = gridCells[below];
        if (!bcell.points->empty() && bcell.terrain_type != TerrainType::GROUND) {
          // Found an occupied non-ground cell below—reject as floating!
          reject_as_floating = true;
          break;
        }
        --bz;
      }
      if (reject_as_floating) {
        *non_ground_points += *cell.points;
        continue;
      }
    }
    *ground_points += *ground_inliers;
    *non_ground_points += *non_ground_inliers;
  }

  for (const auto & cell_id : non_ground_cells) {
    const GridCell<PointT> & cell = gridCells[cell_id];
    *non_ground_points += *cell.points;
  }
  return std::make_pair(ground_points, non_ground_points);
}

} //namespace ground_segmentation
