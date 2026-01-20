# GSeg3D / PointCloudGrid  
**High-Precision Grid-Based Ground Segmentation for Safety-Critical Robotics**

This module implements the core algorithmic components of **GSeg3D**, a high-precision, grid-based ground segmentation method for LiDAR point clouds designed for **safety-critical autonomous driving and robotics applications**.  
It follows the methodology described in the paper:

> *GSeg3D: A High-Precision Grid-Based Algorithm for Safety-Critical Ground Segmentation in LiDAR Point Clouds*  
> Lodhi, M. H. K., Hertzberg, C., DFKI Robotics Innovation Center

---

## Motivation

Reliable ground segmentation is a fundamental prerequisite for:
- Obstacle detection
- Traversability estimation
- Navigation and planning
- Mapping and localization

False positives (classifying ground as obstacles) can directly compromise safety and downstream decision-making.  
GSeg3D is explicitly designed to **maximize precision while maintaining robust recall**, even in cluttered, unstructured, or vegetation-rich environments.

---

## Algorithm Overview

GSeg3D performs **two-phase grid-based ground segmentation**:

### Phase I – Coarse Segmentation
- Uses **larger vertical grid cells**
- Aggressively captures elevated structures as non-ground
- Ensures high initial precision
- May temporarily over-segment ground in cluttered areas

### Phase II – Refinement
- Uses **smaller vertical grid cells**
- Re-evaluates ground points from Phase I
- Corrects false positives and false negatives from Phase I
- Enforces vertical consistency constraints

This dual-phase strategy achieves a strong balance between **precision and recall**, which is critical for safety-critical systems.

---

## Processing Pipeline

Each phase follows the same four core steps:

1. **Grid Representation**
   - The point cloud is discretized into a regular 3D grid
   - Each point is assigned to a cell based on configurable cell sizes

2. **Local Eigen Classification**
   - Per-cell covariance matrix and eigen decomposition
   - Cells classified as:
     - `LINE` (dominant linear structure)
     - `PLANE` (surface-like structure)
     - `NOISE` (scattered points)

3. **Surface Gradient Analysis**
   - Robust plane fitting using **PCL SACSegmentation (PROSAC)**
   - Slope estimation relative to gravity (world frame)
   - Planar cells exceeding slope thresholds are rejected as ground

4. **Ground Region Expansion**
   - Candidate ground cell centroids are indexed using a **KD-tree (nanoflann)**
   - Radius-based neighborhood expansion ensures connectivity
   - Works even when grid resolution is finer than LiDAR scan-line spacing
   - Phase II additionally enforces height-difference constraints

---

## Key Design Contributions

### KD-Tree–Based Ground Expansion
Traditional grid-neighbor expansion fails at high grid resolutions due to scan-line gaps.  
GSeg3D overcomes this by:
- Indexing ground centroids in a KD-tree
- Performing radius-based spatial expansion
- Enabling fine grid resolutions without loss of connectivity or recall

### Robust Seed Initialization
- Synthetic seed points are injected directly beneath the robot
- Ensures reliable ground initialization even with occlusions or sparse data
- Synthetic points are removed before final output and evaluation

### Multi-Step Ground Verification
Each candidate cell undergoes:
- Plane inlier / outlier separation
- Bounding-box sparsity analysis
- Neighborhood height consistency checks
- Floating-cell rejection (non-ground below)

Only physically plausible and spatially coherent cells are labeled as ground.

---

## Implementation Details

### Core Class

```cpp
template<typename PointT>
class PointCloudGrid;
```

### Main API

```cpp
PointCloudGrid(const GridConfig& config);

void setInputCloud(
    pcl::PointCloud<PointT>::Ptr input,
    const Eigen::Quaterniond& R_body2World
);

std::pair<
    pcl::PointCloud<PointT>::Ptr,
    pcl::PointCloud<PointT>::Ptr
> segmentPoints();
```

---

## Typical Configuration (from Paper)

```yaml
cellSizeX: 1.5
cellSizeY: 1.0
cellSizeZ: 1.5   # Phase I
slopeThresholdDegrees: 30.0
groundInlierThreshold: 0.125
centroidSearchRadius: 5.0
distToGround: 1.723 # (e.g. Based on pointcloud data from SemanticKITTI)
robotRadius: 2.7
```

---

## Performance Summary (SemanticKITTI)

- **Precision:** ~96–99% (low variance)
- **Recall:** ~85–91%
- **F1-score:** ~92–94%
- **Runtime:** ~48 ms (CPU, single scan average)

GSeg3D consistently demonstrates **stable, high-precision performance** across:
- Urban environments
- Highways
- Unstructured terrain

---

## Dependencies

- **PCL** (segmentation, centroid, covariance)
- **Eigen**
- **nanoflann**

---

## Intended Use

- Safety-critical autonomous driving
- Outdoor mobile robotics
- Traversability analysis
- Mapping and perception pipelines

---

## Notes & Future Work

- Recall degradation in dense vegetation remains challenging
- Planned extensions:
  - Semantic-aware refinement
  - Temporal fusion
  - GPU acceleration

---

## Citation

If you use this work, please cite the **GSeg3D** paper.

---

© DFKI Robotics Innovation Center