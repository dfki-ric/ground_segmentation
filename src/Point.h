#pragma once

namespace pointcloud_obstacle_detection{
struct Point {
  Point() : x(0), y(0), z(0){}  
  Point(double x0, double y0, double z0) : x(x0), y(y0), z(z0) {}
  double x;
  double y;
  double z;
  bool operator <(const Point &p) const {
    return x < p.x || (x == p.x && y < p.y);
  }
};
} //namespace pointcloud_obstacle_detection

