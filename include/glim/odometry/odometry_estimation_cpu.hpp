#pragma once

#include <glim/odometry/odometry_estimation_imu.hpp>
#include <glim/odometry/scan_dcreg_diagnostics.hpp>

namespace gtsam_points {

class GaussianVoxelMapCPU;
struct FlatContainer;
template <typename VoxelContents>
class IncrementalVoxelMap;
using iVox = IncrementalVoxelMap<FlatContainer>;
}  // namespace gtsam_points

namespace glim {

/**
 * @brief Parameters for OdometryEstimationCPU
 */
struct OdometryEstimationCPUParams : public OdometryEstimationIMUParams {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  OdometryEstimationCPUParams();
  virtual ~OdometryEstimationCPUParams();

public:
  // Registration params
  std::string registration_type;    ///< Registration type (GICP or VGICP)
  int max_iterations;               ///< Maximum number of iterations
  int lru_thresh;                   ///< LRU cache threshold
  double target_downsampling_rate;  ///< Downsampling rate for points to be inserted into the target
  double scan_matching_pose_factor_precision;  ///< Precision of scan-matching pose surrogate factors
  bool scan_health_enable;                     ///< Compute scan-health score from registration diagnostics
  bool scan_health_apply_to_local_scan_precision;  ///< Scale local scan surrogate factors by scan health
  double scan_health_min_factor;               ///< Lower bound on local scan precision scaling
  int scan_health_reference_min_frames;         ///< Warmup frames before reference ratios are trusted
  double scan_health_reference_ema_alpha;       ///< EMA rate for healthy reference statistics
  double scan_health_reference_update_min_score;  ///< Minimum health score for reference updates
  int scan_health_absolute_min_points;          ///< Absolute low point-support threshold
  double scan_health_point_ratio_good;          ///< Point ratio with full health
  double scan_health_point_ratio_bad;           ///< Point ratio with minimum health
  double scan_health_scan_imu_translation_good_m;  ///< Scan-vs-IMU translation considered healthy
  double scan_health_scan_imu_translation_bad_m;   ///< Scan-vs-IMU translation considered unhealthy
  double scan_health_scan_imu_rotation_good_deg;   ///< Scan-vs-IMU rotation considered healthy
  double scan_health_scan_imu_rotation_bad_deg;    ///< Scan-vs-IMU rotation considered unhealthy
  double scan_health_error_ratio_good;          ///< Final/initial scan cost ratio considered healthy
  double scan_health_error_ratio_bad;           ///< Final/initial scan cost ratio considered unhealthy
  bool scan_health_hessian_enable;              ///< Include Hessian ratio evidence in active health scaling
  double scan_health_hessian_min_ratio_good;    ///< Hessian min-eigen ratio considered healthy
  double scan_health_hessian_min_ratio_bad;     ///< Hessian min-eigen ratio considered unhealthy
  double scan_health_hessian_frobenius_ratio_good;  ///< Hessian Frobenius ratio considered healthy
  double scan_health_hessian_frobenius_ratio_bad;   ///< Hessian Frobenius ratio considered unhealthy
  ScanDcregHealthConfig dcreg_health;  ///< Passive LiDAR observability sidecar

  double ivox_resolution;  ///< iVox resolution (for GICP)
  double ivox_min_dist;    ///< Minimum distance between points in an iVox cell (for GICP)

  double vgicp_resolution;               ///< Voxelmap resolution (for VGICP)
  int vgicp_voxelmap_levels;             ///< Multi-resolution voxelmap levesl (for VGICP)
  double vgicp_voxelmap_scaling_factor;  ///< Multi-resolution voxelmap scaling factor (for VGICP)
};

/**
 * @brief CPU-based semi-tightly coupled LiDAR-IMU odometry
 */
class OdometryEstimationCPU : public OdometryEstimationIMU {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  OdometryEstimationCPU(const OdometryEstimationCPUParams& params = OdometryEstimationCPUParams());
  virtual ~OdometryEstimationCPU() override;

private:
  virtual gtsam::NonlinearFactorGraph create_factors(const int current, const gtsam_points::shared_ptr<gtsam::ImuFactor>& imu_factor, gtsam::Values& new_values) override;

  virtual void fallback_smoother() override;

  void update_target(const int current, const Eigen::Isometry3d& T_target_imu);

private:
  // Registration params
  std::mt19937 mt;                                                                   ///< RNG
  Eigen::Isometry3d last_T_target_imu;                                               ///< Last IMU pose w.r.t. target model
  std::vector<std::shared_ptr<gtsam_points::GaussianVoxelMapCPU>> target_voxelmaps;  ///< VGICP target voxelmap
  std::shared_ptr<gtsam_points::iVox> target_ivox;                                   ///< GICP target iVox
  EstimationFrame::ConstPtr target_ivox_frame;                                       ///< Target points (just for visualization)
  std::unique_ptr<ScanDcregHealthMonitor> dcreg_health_monitor;                       ///< Optional passive log-only monitor
  int scan_health_reference_count = 0;                                                ///< Number of scan-health reference updates
  double scan_health_point_reference = -1.0;                                          ///< Healthy point-support reference
  double scan_health_hessian_min_reference = -1.0;                                    ///< Healthy Hessian min-eigen reference
  double scan_health_hessian_frobenius_reference = -1.0;                              ///< Healthy Hessian Frobenius reference
};

}  // namespace glim
