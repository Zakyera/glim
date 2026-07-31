#pragma once

#include <cstddef>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace glim {

enum class ScanDcregStatus {
  Valid = 0,
  IndefiniteRotationSchur = 1,
  IndefiniteTranslationSchur = 2,
  IndefiniteBothSchur = 3,
  NonFiniteHessian = 10,
  BlockEigendecompositionFailed = 11,
  SchurEigendecompositionFailed = 12,
};

/**
 * @brief Read-only degeneracy characterization of one LiDAR registration.
 *
 * The source Hessian uses GTSAM Pose3 local tangent order
 * [rx, ry, rz, tx, ty, tz]. Rotation and translation are decoupled with
 * complementary Schur reductions. Eigenvectors and axis weakness values are
 * therefore expressed in the estimated IMU/body-local tangent frame.
 */
struct ScanDcregDiagnostics {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  double stamp_sec = 0.0;
  std::size_t frame_index = 0u;
  Eigen::Isometry3d T_world_imu = Eigen::Isometry3d::Identity();

  bool valid = false;
  ScanDcregStatus status = ScanDcregStatus::NonFiniteHessian;
  double condition_threshold = 10.0;
  double spectral_ratio_cap = 1.0e12;

  int translation_block_rank = 0;
  int rotation_block_rank = 0;
  int rotation_negative_eigenvalues = 0;
  int translation_negative_eigenvalues = 0;

  double rotation_condition = 0.0;
  double translation_condition = 0.0;
  Eigen::Vector3d rotation_eigenvalues = Eigen::Vector3d::Zero();
  Eigen::Vector3d translation_eigenvalues = Eigen::Vector3d::Zero();
  Eigen::Vector3d rotation_spectral_ratios = Eigen::Vector3d::Zero();
  Eigen::Vector3d translation_spectral_ratios = Eigen::Vector3d::Zero();
  Eigen::Matrix3d rotation_eigenvectors = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d translation_eigenvectors = Eigen::Matrix3d::Identity();
  Eigen::Array3i rotation_weak_modes = Eigen::Array3i::Zero();
  Eigen::Array3i translation_weak_modes = Eigen::Array3i::Zero();
  Eigen::Vector3d rotation_axis_weakness = Eigen::Vector3d::Zero();
  Eigen::Vector3d translation_axis_weakness = Eigen::Vector3d::Zero();
};

/**
 * @brief Compute DCReg-style Schur diagnostics without changing optimization.
 *
 * Rank-deficient eliminated blocks are handled with a symmetric
 * eigendecomposition pseudo-inverse. Spectral ratios are capped so they remain
 * safe for CSV and Rerun scalar logging.
 */
ScanDcregDiagnostics analyzeScanDcregHessian(
  const Eigen::Matrix<double, 6, 6>& hessian,
  double condition_threshold = 10.0,
  double spectral_ratio_cap = 1.0e12);

const char* scanDcregStatusName(ScanDcregStatus status);

}  // namespace glim
