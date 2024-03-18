#pragma once

namespace pointcloud_obstacle_detection{

struct GroundDetectionStatistics{
    int undefined_cells;
    int ground_cells;
    int non_ground_cells;
    int unknown_cells;
    GroundDetectionStatistics()
    {
        undefined_cells = 0;
        ground_cells = 0;
        non_ground_cells = 0;
        unknown_cells = 0;
    }
    void clear(){
        undefined_cells = 0;
        ground_cells = 0;
        non_ground_cells = 0;
        unknown_cells = 0;
    }
};


} //namespace pointcloud_obstacle_detection
