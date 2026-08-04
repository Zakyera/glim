#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

namespace glim {

enum class ScanDcregMode {
  Off = 0,
  LogOnly = 1,
};

/**
 * @brief Internal Stage 1.8 scheduling-isolation variant.
 *
 * This is deliberately not a public operating mode. Production behavior
 * remains off/log_only. The variant can only refine log_only during
 * controlled regression experiments.
 */
enum class ScanDcregTestVariant {
  FullLogOnly = 0,
  ComputeOnly = 1,
  EnqueueOnly = 2,
};

enum class ScanDcregReferenceSource {
  Session = 0,
  Offline = 1,
  Hybrid = 2,
};

enum class ScanDcregInvalidFramePolicy {
  Hold = 0,
  DecayTowardUnknown = 1,
  MarkUnavailable = 2,
};

enum class ScanDcregStatus {
  Valid = 0,
  NoLidarSupport = 1,
  IndefiniteRotationSchur = 2,
  IndefiniteTranslationSchur = 3,
  IndefiniteBothSchur = 4,
  NonFiniteHessian = 10,
  InvalidHessianDimension = 11,
  BlockEigendecompositionFailed = 12,
  SchurEigendecompositionFailed = 13,
};

struct ScanDcregDetectionConfig {
  double degeneracy_condition_threshold = 10.0;
  double epsilon_absolute = 1.0e-12;
  double epsilon_relative = 1.0e-9;
  double pseudoinverse_relative_threshold = 1.0e-8;
  double negative_eigenvalue_tolerance = 1.0e-9;
  double spectral_cluster_relative_gap = 0.05;
  double minimum_axis_alignment_confidence = 0.70;
};

struct ScanDcregReferenceConfig {
  ScanDcregReferenceSource source = ScanDcregReferenceSource::Hybrid;
  std::string offline_profile_path;
  std::string sensor_identifier;
  bool require_profile_metadata_match = true;

  int bootstrap_minimum_samples = 30;
  int bootstrap_window_size = 10;
  double bootstrap_max_rotation_condition_ratio = 10.0;
  double bootstrap_max_translation_condition_ratio = 10.0;
  double temporal_stability_max_log_ratio_range = 0.50;

  bool adaptation_enabled = true;
  double adaptation_rate = 0.01;
  bool freeze_during_degradation = true;
  double maximum_reference_change_ratio = 2.0;
  double offline_nominal_mad_multiplier = 3.0;
  double offline_nominal_min_log_half_width = 0.05;

  bool require_glim_initialized = true;
  bool require_registration_converged = true;
  bool require_linear_solve_success = true;
  int minimum_source_point_count = 100;
  int minimum_inlier_count = 100;
  double minimum_inlier_fraction = 0.05;
  double maximum_initial_cost = -1.0;
  double maximum_final_cost = -1.0;
  double minimum_relative_cost_reduction = -1.0;
  bool require_axis_alignment_confidence = true;
  bool reject_clustered_modes = true;
};

struct ScanDcregTemporalConfig {
  bool enabled = true;
  double smoothing_alpha = 0.10;
  double health_enter_threshold = 0.25;
  double health_exit_threshold = 0.50;
  int bad_frames_required = 3;
  int good_frames_required = 5;
  ScanDcregInvalidFramePolicy invalid_frame_policy =
    ScanDcregInvalidFramePolicy::Hold;
};

struct ScanDcregLoggingConfig {
  bool enabled = false;
  std::string csv_path;
  int log_every_n_frames = 1;
  bool asynchronous = true;
  std::size_t queue_capacity = 256u;
  int flush_every_n_rows = 10;
};

struct ScanDcregLoggingStats {
  std::size_t enqueue_attempt_count = 0u;
  std::size_t enqueued_sample_count = 0u;
  std::size_t dropped_sample_count = 0u;
  std::size_t dropped_row_count = 0u;
  std::size_t formatted_row_count = 0u;
  std::size_t maximum_queue_size = 0u;
  double total_enqueue_time_ms = 0.0;
  double maximum_enqueue_time_ms = 0.0;
  std::size_t worker_sample_count = 0u;
  double total_worker_time_ms = 0.0;
  double total_formatting_time_ms = 0.0;
  double total_file_write_time_ms = 0.0;
  bool producer_uses_try_lock = true;
  bool producer_waits_for_space = false;
  bool worker_discards_records = false;
  bool file_io_enabled = false;
};

struct ScanDcregExecutionStats {
  std::size_t evaluation_count = 0u;
  double total_evaluation_time_ms = 0.0;
  double maximum_evaluation_time_ms = 0.0;
  std::size_t snapshot_count = 0u;
  double total_initial_cost_time_ms = 0.0;
  double maximum_initial_cost_time_ms = 0.0;
  double total_snapshot_time_ms = 0.0;
  double maximum_snapshot_time_ms = 0.0;
};

struct ScanDcregHealthConfig {
  ScanDcregMode mode = ScanDcregMode::Off;
  ScanDcregDetectionConfig detection;
  ScanDcregReferenceConfig reference;
  ScanDcregTemporalConfig temporal;
  ScanDcregLoggingConfig logging;
  bool combine_support_with_shape_health = false;
};

struct ScanDcregRuntimeMetadata {
  std::string sensor_identifier;
  std::string registration_type;
  std::vector<double> voxel_resolutions;
  std::string pose_ordering = "rotation_translation";
  std::string tangent_convention = "local_right";
  std::string configuration_fingerprint;
};

struct ScanDcregHessianMetrics {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  bool valid = false;
  Eigen::Matrix<double, 6, 6> hessian =
    Eigen::Matrix<double, 6, 6>::Zero();
  double trace = 0.0;
  double minimum_eigenvalue = 0.0;
  double maximum_eigenvalue = 0.0;
  double condition = 0.0;
  double frobenius_norm = 0.0;
  double trace_per_source_point = 0.0;
  double trace_per_inlier = 0.0;
  int rank = 0;
};

/**
 * @brief Schur observability characterization of one 6x6 LiDAR Hessian.
 *
 * All matrices use GTSAM Pose3 local tangent order
 * [rx, ry, rz, tx, ty, tz]. Raw eigensystems are the direct ascending-order
 * SelfAdjointEigenSolver outputs. Aligned columns are the same actual
 * eigenvectors, only permuted and sign-corrected against a stable target basis.
 */
struct ScanDcregSchurDiagnostics {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  bool valid = false;
  bool factorization_ok = false;
  ScanDcregStatus status = ScanDcregStatus::NonFiniteHessian;
  double condition_threshold = 10.0;
  std::string invalid_reason;

  int translation_block_rank = 0;
  int rotation_block_rank = 0;
  int rotation_numerical_negative_eigenvalues = 0;
  int translation_numerical_negative_eigenvalues = 0;
  int rotation_indefinite_eigenvalues = 0;
  int translation_indefinite_eigenvalues = 0;

  Eigen::Vector3d raw_rotation_eigenvalues =
    Eigen::Vector3d::Zero();
  Eigen::Vector3d raw_translation_eigenvalues =
    Eigen::Vector3d::Zero();
  Eigen::Matrix3d raw_rotation_eigenvectors =
    Eigen::Matrix3d::Identity();
  Eigen::Matrix3d raw_translation_eigenvectors =
    Eigen::Matrix3d::Identity();

  Eigen::Vector3d aligned_rotation_eigenvalues =
    Eigen::Vector3d::Zero();
  Eigen::Vector3d aligned_translation_eigenvalues =
    Eigen::Vector3d::Zero();
  Eigen::Matrix3d aligned_rotation_basis =
    Eigen::Matrix3d::Identity();
  Eigen::Matrix3d aligned_translation_basis =
    Eigen::Matrix3d::Identity();
  Eigen::Array3i aligned_rotation_original_indices =
    Eigen::Array3i::Zero();
  Eigen::Array3i aligned_translation_original_indices =
    Eigen::Array3i::Zero();

  Eigen::Vector3d rotation_condition_ratios =
    Eigen::Vector3d::Ones();
  Eigen::Vector3d translation_condition_ratios =
    Eigen::Vector3d::Ones();
  Eigen::Array3i absolute_rotation_degenerate_mask =
    Eigen::Array3i::Zero();
  Eigen::Array3i absolute_translation_degenerate_mask =
    Eigen::Array3i::Zero();

  // Rows are local physical axes; columns are aligned actual eigenmodes.
  Eigen::Matrix3d rotation_axis_contribution_ratios =
    Eigen::Matrix3d::Identity();
  Eigen::Matrix3d translation_axis_contribution_ratios =
    Eigen::Matrix3d::Identity();
  Eigen::Vector3d rotation_alignment_confidence =
    Eigen::Vector3d::Ones();
  Eigen::Vector3d translation_alignment_confidence =
    Eigen::Vector3d::Ones();
  Eigen::Array3i rotation_spectral_cluster_flags =
    Eigen::Array3i::Zero();
  Eigen::Array3i translation_spectral_cluster_flags =
    Eigen::Array3i::Zero();
};

struct ScanDcregFactorInput {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  int factor_index = -1;
  double resolution = 0.0;
  int source_point_count = 0;
  int inlier_count = 0;
  double inlier_fraction = 0.0;
  double initial_cost = 0.0;
  double final_cost = 0.0;
  bool hessian_valid = false;
  Eigen::Matrix<double, 6, 6> hessian =
    Eigen::Matrix<double, 6, 6>::Zero();
  std::string invalid_reason;
};

struct ScanDcregFactorDiagnostics {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  int factor_index = -1;
  double resolution = 0.0;
  int source_point_count = 0;
  int inlier_count = 0;
  double inlier_fraction = 0.0;
  double initial_cost = 0.0;
  double final_cost = 0.0;
  double cost_change = 0.0;
  ScanDcregHessianMetrics hessian;
  ScanDcregSchurDiagnostics schur;
  std::string invalid_reason;
};

using ScanDcregFactorInputs =
  std::vector<ScanDcregFactorInput,
              Eigen::aligned_allocator<ScanDcregFactorInput>>;
using ScanDcregFactorDiagnosticsVector =
  std::vector<ScanDcregFactorDiagnostics,
              Eigen::aligned_allocator<ScanDcregFactorDiagnostics>>;

struct ScanDcregMonitorInput {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  double stamp_sec = 0.0;
  std::size_t frame_index = 0u;
  Eigen::Isometry3d T_world_imu = Eigen::Isometry3d::Identity();
  bool glim_initialized = false;
  bool registration_converged = false;
  bool linear_solve_success = false;
  int source_point_count = 0;
  double initial_cost = 0.0;
  double final_cost = 0.0;
  ScanDcregFactorInputs factors;
};

/**
 * @brief Passive, relative directional LiDAR observability health.
 *
 * These health values are not expected metric error, probabilities of
 * correctness, calibrated covariance, or estimator-control outputs.
 */
struct ScanDcregDiagnostics {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  static constexpr int kSchemaVersion = 2;

  bool enabled = false;
  bool valid = false;
  bool factorization_ok = false;
  bool reference_ready = false;
  bool health_available = false;
  bool registration_converged = false;
  bool linear_solve_success = false;
  bool baseline_update_accepted = false;

  double stamp_sec = 0.0;
  std::size_t frame_index = 0u;
  Eigen::Isometry3d T_world_imu = Eigen::Isometry3d::Identity();

  ScanDcregSchurDiagnostics aggregate_schur;
  ScanDcregHessianMetrics aggregate_hessian;
  ScanDcregFactorDiagnosticsVector per_factor;

  int source_point_count = 0;
  int minimum_factor_inlier_count = 0;
  double minimum_factor_inlier_fraction = 0.0;
  double initial_cost = 0.0;
  double final_cost = 0.0;
  double cost_change = 0.0;

  Eigen::Vector3d rotation_reference_ratios =
    Eigen::Vector3d::Constant(
      std::numeric_limits<double>::quiet_NaN());
  Eigen::Vector3d translation_reference_ratios =
    Eigen::Vector3d::Constant(
      std::numeric_limits<double>::quiet_NaN());
  Eigen::Vector3d rotation_health_raw =
    Eigen::Vector3d::Constant(
      std::numeric_limits<double>::quiet_NaN());
  Eigen::Vector3d translation_health_raw =
    Eigen::Vector3d::Constant(
      std::numeric_limits<double>::quiet_NaN());
  Eigen::Vector3d rotation_health_smoothed =
    Eigen::Vector3d::Constant(
      std::numeric_limits<double>::quiet_NaN());
  Eigen::Vector3d translation_health_smoothed =
    Eigen::Vector3d::Constant(
      std::numeric_limits<double>::quiet_NaN());
  Eigen::Array3i relative_rotation_degraded_mask =
    Eigen::Array3i::Zero();
  Eigen::Array3i relative_translation_degraded_mask =
    Eigen::Array3i::Zero();
  Eigen::Array3i rotation_bad_frame_counters =
    Eigen::Array3i::Zero();
  Eigen::Array3i translation_bad_frame_counters =
    Eigen::Array3i::Zero();
  Eigen::Array3i rotation_good_frame_counters =
    Eigen::Array3i::Zero();
  Eigen::Array3i translation_good_frame_counters =
    Eigen::Array3i::Zero();

  std::string baseline_update_reason;
  std::string reference_source;
  std::string offline_profile_status;
  std::string invalid_reason;
  double evaluation_time_ms = 0.0;
};

struct ScanDcregReferenceCandidate {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  bool monitor_valid = false;
  bool factorization_ok = false;
  bool glim_initialized = false;
  bool registration_converged = false;
  bool linear_solve_success = false;
  int source_point_count = 0;
  int minimum_factor_inlier_count = 0;
  double minimum_factor_inlier_fraction = 0.0;
  double initial_cost = 0.0;
  double final_cost = 0.0;
  Eigen::Vector3d rotation_condition_ratios =
    Eigen::Vector3d::Ones();
  Eigen::Vector3d translation_condition_ratios =
    Eigen::Vector3d::Ones();
  Eigen::Vector3d rotation_alignment_confidence =
    Eigen::Vector3d::Ones();
  Eigen::Vector3d translation_alignment_confidence =
    Eigen::Vector3d::Ones();
  Eigen::Array3i rotation_cluster_flags =
    Eigen::Array3i::Zero();
  Eigen::Array3i translation_cluster_flags =
    Eigen::Array3i::Zero();
  Eigen::Array3i absolute_rotation_degenerate_mask =
    Eigen::Array3i::Zero();
  Eigen::Array3i absolute_translation_degenerate_mask =
    Eigen::Array3i::Zero();
  bool axis_alignment_valid = true;
};

struct ScanDcregReferenceUpdate {
  bool reference_ready = false;
  bool initialized_now = false;
  bool accepted = false;
  std::string reason;
  Eigen::Vector3d rotation_reference_ratios =
    Eigen::Vector3d::Constant(
      std::numeric_limits<double>::quiet_NaN());
  Eigen::Vector3d translation_reference_ratios =
    Eigen::Vector3d::Constant(
      std::numeric_limits<double>::quiet_NaN());
};

/**
 * @brief Robust healthy-reference bootstrap and bounded adaptation manager.
 */
class ScanDcregReferenceManager {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ScanDcregReferenceManager(
    const ScanDcregReferenceConfig& config,
    const ScanDcregRuntimeMetadata& runtime_metadata);
  ~ScanDcregReferenceManager();

  bool ready() const;
  const Eigen::Vector3d& rotationReferenceRatios() const;
  const Eigen::Vector3d& translationReferenceRatios() const;
  const std::string& profileStatus() const;
  const std::string& activeSource() const;

  ScanDcregReferenceUpdate update(
    const ScanDcregReferenceCandidate& candidate,
    const Eigen::Vector3d& current_rotation_health,
    const Eigen::Vector3d& current_translation_health,
    const Eigen::Array3i& current_rotation_degraded,
    const Eigen::Array3i& current_translation_degraded);

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

/**
 * @brief Analyze a 6x6 [rotation, translation] LiDAR Hessian.
 *
 * target_*_basis may be null. If provided, raw eigenvectors are matched to
 * that prior orthonormal basis by maximum absolute inner product, then
 * sign-corrected. If null, the local physical XYZ basis is used.
 */
ScanDcregSchurDiagnostics analyzeScanDcregHessian(
  const Eigen::Matrix<double, 6, 6>& hessian,
  const ScanDcregDetectionConfig& config,
  const Eigen::Matrix3d* target_rotation_basis = nullptr,
  const Eigen::Matrix3d* target_translation_basis = nullptr);

ScanDcregHessianMetrics computeScanDcregHessianMetrics(
  const Eigen::Matrix<double, 6, 6>& hessian,
  int source_point_count,
  int inlier_count,
  const ScanDcregDetectionConfig& config);

/**
 * @brief Passive monitor. No estimator-control output exists in this API.
 */
class ScanDcregHealthMonitor {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ScanDcregHealthMonitor(
    const ScanDcregHealthConfig& config,
    const ScanDcregRuntimeMetadata& runtime_metadata,
    ScanDcregTestVariant test_variant =
      ScanDcregTestVariant::FullLogOnly);
  ~ScanDcregHealthMonitor();

  bool enabled() const;
  ScanDcregDiagnostics process(const ScanDcregMonitorInput& input);

  std::size_t evaluationCount() const;
  std::size_t formattedLogRowCount() const;
  std::size_t droppedLogSampleCount() const;
  ScanDcregLoggingStats loggingStats() const;
  ScanDcregExecutionStats executionStats() const;
  void recordProducerTiming(
    double initial_cost_time_ms,
    double snapshot_time_ms);

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

bool validateScanDcregHealthConfig(
  const ScanDcregHealthConfig& config,
  std::string* error);

bool scanDcregModeCreatesMonitor(ScanDcregMode mode);

const char* scanDcregModeName(ScanDcregMode mode);
const char* scanDcregTestVariantName(ScanDcregTestVariant variant);
const char* scanDcregReferenceSourceName(ScanDcregReferenceSource source);
const char* scanDcregInvalidFramePolicyName(
  ScanDcregInvalidFramePolicy policy);
const char* scanDcregStatusName(ScanDcregStatus status);

bool parseScanDcregMode(const std::string& value, ScanDcregMode* mode);
bool parseScanDcregTestVariant(
  const std::string& value,
  ScanDcregTestVariant* variant);
bool parseScanDcregReferenceSource(
  const std::string& value,
  ScanDcregReferenceSource* source);
bool parseScanDcregInvalidFramePolicy(
  const std::string& value,
  ScanDcregInvalidFramePolicy* policy);

}  // namespace glim
