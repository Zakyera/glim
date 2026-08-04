#include <glim/odometry/odometry_estimation_cpu.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

#include <Eigen/Eigenvalues>

#include <spdlog/spdlog.h>

#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/LinearContainerFactor.h>

#include <gtsam_points/ann/ivox.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/factors/linear_damping_factor.hpp>
#include <gtsam_points/factors/integrated_gicp_factor.hpp>
#include <gtsam_points/factors/integrated_vgicp_factor.hpp>
#include <gtsam_points/optimizers/levenberg_marquardt_ext.hpp>
#include <gtsam_points/optimizers/incremental_fixed_lag_smoother_with_fallback.hpp>

#include <glim/util/config.hpp>
#include <glim/common/imu_integration.hpp>
#include <glim/common/cloud_deskewing.hpp>
#include <glim/common/cloud_covariance_estimation.hpp>

#include <glim/odometry/callbacks.hpp>
#include <glim/odometry/scan_dcreg_diagnostics.hpp>

#ifdef GTSAM_USE_TBB
#include <tbb/task_arena.h>
#endif

namespace glim {

using Callbacks = OdometryEstimationCallbacks;

using gtsam::symbol_shorthand::B;  // IMU bias
using gtsam::symbol_shorthand::V;  // IMU velocity   (v_world_imu)
using gtsam::symbol_shorthand::X;  // IMU pose       (T_world_imu)

namespace {

constexpr double kRadToDeg = 180.0 / M_PI;

double nanValue() {
  return std::numeric_limits<double>::quiet_NaN();
}

double poseRotationDeg(const Eigen::Isometry3d& pose) {
  const Eigen::Matrix3d rotation = Eigen::Quaterniond(pose.linear()).normalized().toRotationMatrix();
  return Eigen::AngleAxisd(rotation).angle() * kRadToDeg;
}

double clamp01(const double value) {
  if (!std::isfinite(value)) {
    return 1.0;
  }
  return std::max(0.0, std::min(1.0, value));
}

double highValueHealthScore(const double value, const double good, const double bad) {
  if (!std::isfinite(value) || !std::isfinite(good) || !std::isfinite(bad) || bad <= good) {
    return 1.0;
  }
  if (value <= good) {
    return 1.0;
  }
  if (value >= bad) {
    return 0.0;
  }
  return clamp01((bad - value) / (bad - good));
}

double lowRatioHealthScore(const double ratio, const double good, const double bad) {
  if (!std::isfinite(ratio) || !std::isfinite(good) || !std::isfinite(bad) || good <= bad) {
    return 1.0;
  }
  if (ratio >= good) {
    return 1.0;
  }
  if (ratio <= bad) {
    return 0.0;
  }
  return clamp01((ratio - bad) / (good - bad));
}

double finitePositiveRatioOrOne(const double value, const double reference) {
  if (!std::isfinite(value) || !std::isfinite(reference) || reference <= 0.0) {
    return 1.0;
  }
  return value / reference;
}

void updateHealthyReference(double& reference, const double value, const double alpha) {
  if (!std::isfinite(value) || value <= 0.0) {
    return;
  }
  if (!std::isfinite(reference) || reference <= 0.0) {
    reference = value;
    return;
  }
  const double clamped_alpha = std::max(0.0, std::min(1.0, alpha));
  reference = (1.0 - clamped_alpha) * reference + clamped_alpha * value;
}

struct HessianHealthMetrics {
  bool pose_hessian_valid = false;
  Eigen::Matrix<double, 6, 6> pose_hessian =
    Eigen::Matrix<double, 6, 6>::Zero();
  double min_eigenvalue = nanValue();
  double max_eigenvalue = nanValue();
  double condition_estimate = nanValue();
  double frobenius_norm = nanValue();
  int rank_estimate = 0;
};

HessianHealthMetrics scanHessianHealthMetrics(const gtsam::NonlinearFactorGraph& graph,
                                              const gtsam::Values& values,
                                              const gtsam::Key key) {
  HessianHealthMetrics metrics;

  try {
    const auto linearized = graph.linearize(values);
    if (!linearized) {
      return metrics;
    }

    const auto blocks = linearized->hessianBlockDiagonal();
    const auto found = blocks.find(key);
    if (found == blocks.end() || !found->second.allFinite()) {
      return metrics;
    }

    const Eigen::MatrixXd hessian =
      0.5 * (found->second.template cast<double>() + found->second.transpose().template cast<double>());
    if (hessian.rows() == 0 || hessian.cols() == 0 || !hessian.allFinite()) {
      return metrics;
    }
    if (hessian.rows() == 6 && hessian.cols() == 6) {
      metrics.pose_hessian = hessian;
      metrics.pose_hessian_valid = true;
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(hessian);
    if (eigensolver.info() != Eigen::Success) {
      return metrics;
    }

    const Eigen::VectorXd eigenvalues = eigensolver.eigenvalues();
    metrics.min_eigenvalue = eigenvalues.minCoeff();
    metrics.max_eigenvalue = eigenvalues.maxCoeff();
    metrics.frobenius_norm = hessian.norm();

    const double max_abs = eigenvalues.cwiseAbs().maxCoeff();
    const double rank_tol = std::max(1e-12, max_abs * 1e-9);
    double min_positive_abs = std::numeric_limits<double>::infinity();
    metrics.rank_estimate = 0;
    for (int i = 0; i < eigenvalues.size(); i++) {
      const double abs_eig = std::abs(eigenvalues[i]);
      if (abs_eig > rank_tol) {
        metrics.rank_estimate++;
        min_positive_abs = std::min(min_positive_abs, abs_eig);
      }
    }

    if (std::isfinite(min_positive_abs) && min_positive_abs > 0.0) {
      metrics.condition_estimate = max_abs / min_positive_abs;
    }
  } catch (const std::exception&) {
  }

  return metrics;
}

template <typename T>
void readNestedConfigIfPresent(
  const Config& config,
  const std::vector<std::string>& modules,
  const std::string& name,
  T* value) {
  const auto found = config.param_nested<T>(modules, name);
  if (found) {
    *value = *found;
  }
}

bool loadScanDcregHealthConfig(
  const Config& config,
  ScanDcregHealthConfig* output,
  bool* section_present,
  std::string* error) {
  *output = ScanDcregHealthConfig();
  *section_present = false;
  try {
    const std::vector<std::string> root = {
      "odometry_estimation", "dcreg_health"};
    const auto mode_value =
      config.param_nested<std::string>(root, "mode");
    if (!mode_value) {
      return true;
    }
    *section_present = true;
    if (!parseScanDcregMode(*mode_value, &output->mode)) {
      *error = "unsupported dcreg_health mode '" +
               *mode_value + "'";
      output->mode = ScanDcregMode::Off;
      return false;
    }

    const std::vector<std::string> detection = {
      "odometry_estimation", "dcreg_health", "detection"};
    readNestedConfigIfPresent(
      config,
      detection,
      "degeneracy_condition_threshold",
      &output->detection.degeneracy_condition_threshold);
    readNestedConfigIfPresent(
      config,
      detection,
      "epsilon_absolute",
      &output->detection.epsilon_absolute);
    readNestedConfigIfPresent(
      config,
      detection,
      "epsilon_relative",
      &output->detection.epsilon_relative);
    readNestedConfigIfPresent(
      config,
      detection,
      "pseudoinverse_relative_threshold",
      &output->detection.pseudoinverse_relative_threshold);
    readNestedConfigIfPresent(
      config,
      detection,
      "negative_eigenvalue_tolerance",
      &output->detection.negative_eigenvalue_tolerance);
    readNestedConfigIfPresent(
      config,
      detection,
      "spectral_cluster_relative_gap",
      &output->detection.spectral_cluster_relative_gap);
    readNestedConfigIfPresent(
      config,
      detection,
      "minimum_axis_alignment_confidence",
      &output->detection.minimum_axis_alignment_confidence);

    const std::vector<std::string> reference = {
      "odometry_estimation", "dcreg_health", "reference"};
    output->reference.bootstrap_max_rotation_condition_ratio =
      output->detection.degeneracy_condition_threshold;
    output->reference.bootstrap_max_translation_condition_ratio =
      output->detection.degeneracy_condition_threshold;
    std::string reference_source =
      scanDcregReferenceSourceName(output->reference.source);
    readNestedConfigIfPresent(
      config, reference, "source", &reference_source);
    if (!parseScanDcregReferenceSource(
          reference_source, &output->reference.source)) {
      *error = "unsupported dcreg_health reference source '" +
               reference_source + "'";
      output->mode = ScanDcregMode::Off;
      return false;
    }
    readNestedConfigIfPresent(
      config,
      reference,
      "offline_profile_path",
      &output->reference.offline_profile_path);
    readNestedConfigIfPresent(
      config,
      reference,
      "sensor_identifier",
      &output->reference.sensor_identifier);
    readNestedConfigIfPresent(
      config,
      reference,
      "require_profile_metadata_match",
      &output->reference.require_profile_metadata_match);
    readNestedConfigIfPresent(
      config,
      reference,
      "bootstrap_minimum_samples",
      &output->reference.bootstrap_minimum_samples);
    readNestedConfigIfPresent(
      config,
      reference,
      "bootstrap_window_size",
      &output->reference.bootstrap_window_size);
    const auto legacy_bootstrap_limit =
      config.param_nested<double>(
        reference, "bootstrap_max_condition_ratio");
    if (legacy_bootstrap_limit) {
      spdlog::warn(
        "dcreg_health.reference.bootstrap_max_condition_ratio is "
        "deprecated and ignored; use the separate rotation and "
        "translation limits");
    }
    readNestedConfigIfPresent(
      config,
      reference,
      "bootstrap_max_rotation_condition_ratio",
      &output->reference
         .bootstrap_max_rotation_condition_ratio);
    readNestedConfigIfPresent(
      config,
      reference,
      "bootstrap_max_translation_condition_ratio",
      &output->reference
         .bootstrap_max_translation_condition_ratio);
    readNestedConfigIfPresent(
      config,
      reference,
      "temporal_stability_max_log_ratio_range",
      &output->reference.temporal_stability_max_log_ratio_range);
    readNestedConfigIfPresent(
      config,
      reference,
      "adaptation_enabled",
      &output->reference.adaptation_enabled);
    readNestedConfigIfPresent(
      config,
      reference,
      "adaptation_rate",
      &output->reference.adaptation_rate);
    readNestedConfigIfPresent(
      config,
      reference,
      "freeze_during_degradation",
      &output->reference.freeze_during_degradation);
    readNestedConfigIfPresent(
      config,
      reference,
      "maximum_reference_change_ratio",
      &output->reference.maximum_reference_change_ratio);
    readNestedConfigIfPresent(
      config,
      reference,
      "offline_nominal_mad_multiplier",
      &output->reference.offline_nominal_mad_multiplier);
    readNestedConfigIfPresent(
      config,
      reference,
      "offline_nominal_min_log_half_width",
      &output->reference.offline_nominal_min_log_half_width);
    readNestedConfigIfPresent(
      config,
      reference,
      "require_glim_initialized",
      &output->reference.require_glim_initialized);
    readNestedConfigIfPresent(
      config,
      reference,
      "require_registration_converged",
      &output->reference.require_registration_converged);
    readNestedConfigIfPresent(
      config,
      reference,
      "require_linear_solve_success",
      &output->reference.require_linear_solve_success);
    readNestedConfigIfPresent(
      config,
      reference,
      "minimum_source_point_count",
      &output->reference.minimum_source_point_count);
    readNestedConfigIfPresent(
      config,
      reference,
      "minimum_inlier_count",
      &output->reference.minimum_inlier_count);
    readNestedConfigIfPresent(
      config,
      reference,
      "minimum_inlier_fraction",
      &output->reference.minimum_inlier_fraction);
    readNestedConfigIfPresent(
      config,
      reference,
      "maximum_initial_cost",
      &output->reference.maximum_initial_cost);
    readNestedConfigIfPresent(
      config,
      reference,
      "maximum_final_cost",
      &output->reference.maximum_final_cost);
    readNestedConfigIfPresent(
      config,
      reference,
      "minimum_relative_cost_reduction",
      &output->reference.minimum_relative_cost_reduction);
    readNestedConfigIfPresent(
      config,
      reference,
      "require_axis_alignment_confidence",
      &output->reference.require_axis_alignment_confidence);
    readNestedConfigIfPresent(
      config,
      reference,
      "reject_clustered_modes",
      &output->reference.reject_clustered_modes);

    const std::vector<std::string> temporal = {
      "odometry_estimation", "dcreg_health", "temporal"};
    readNestedConfigIfPresent(
      config,
      temporal,
      "enabled",
      &output->temporal.enabled);
    readNestedConfigIfPresent(
      config,
      temporal,
      "smoothing_alpha",
      &output->temporal.smoothing_alpha);
    readNestedConfigIfPresent(
      config,
      temporal,
      "health_enter_threshold",
      &output->temporal.health_enter_threshold);
    readNestedConfigIfPresent(
      config,
      temporal,
      "health_exit_threshold",
      &output->temporal.health_exit_threshold);
    readNestedConfigIfPresent(
      config,
      temporal,
      "bad_frames_required",
      &output->temporal.bad_frames_required);
    readNestedConfigIfPresent(
      config,
      temporal,
      "good_frames_required",
      &output->temporal.good_frames_required);
    std::string invalid_frame_policy =
      scanDcregInvalidFramePolicyName(
        output->temporal.invalid_frame_policy);
    readNestedConfigIfPresent(
      config,
      temporal,
      "invalid_frame_policy",
      &invalid_frame_policy);
    if (!parseScanDcregInvalidFramePolicy(
          invalid_frame_policy,
          &output->temporal.invalid_frame_policy)) {
      *error =
        "unsupported dcreg_health invalid_frame_policy '" +
        invalid_frame_policy + "'";
      output->mode = ScanDcregMode::Off;
      return false;
    }

    const std::vector<std::string> support = {
      "odometry_estimation", "dcreg_health", "support"};
    readNestedConfigIfPresent(
      config,
      support,
      "combine_with_shape_health",
      &output->combine_support_with_shape_health);

    const std::vector<std::string> logging = {
      "odometry_estimation", "dcreg_health", "logging"};
    readNestedConfigIfPresent(
      config,
      logging,
      "enabled",
      &output->logging.enabled);
    readNestedConfigIfPresent(
      config,
      logging,
      "csv_path",
      &output->logging.csv_path);
    readNestedConfigIfPresent(
      config,
      logging,
      "log_every_n_frames",
      &output->logging.log_every_n_frames);
    readNestedConfigIfPresent(
      config,
      logging,
      "asynchronous",
      &output->logging.asynchronous);
    int queue_capacity =
      static_cast<int>(output->logging.queue_capacity);
    readNestedConfigIfPresent(
      config,
      logging,
      "queue_capacity",
      &queue_capacity);
    output->logging.queue_capacity =
      queue_capacity > 0
        ? static_cast<std::size_t>(queue_capacity)
        : 0u;
    readNestedConfigIfPresent(
      config,
      logging,
      "flush_every_n_rows",
      &output->logging.flush_every_n_rows);

    if (!validateScanDcregHealthConfig(*output, error)) {
      output->mode = ScanDcregMode::Off;
      return false;
    }
    return true;
  } catch (const std::exception& exception) {
    *error = std::string("failed to parse dcreg_health: ") +
             exception.what();
    output->mode = ScanDcregMode::Off;
    return false;
  }
}

bool extractUnaryPoseHessian(
  const gtsam::NonlinearFactor::shared_ptr& factor,
  const gtsam::Values& values,
  const gtsam::Key expected_key,
  Eigen::Matrix<double, 6, 6>* hessian,
  std::string* invalid_reason) {
  try {
    if (!factor || factor->keys().size() != 1u ||
        factor->keys().front() != expected_key) {
      *invalid_reason =
        "factor_is_not_unary_on_current_pose";
      return false;
    }
    const gtsam::GaussianFactor::shared_ptr linearized =
      factor->linearize(values);
    if (!linearized) {
      *invalid_reason = "factor_linearization_failed";
      return false;
    }
    gtsam::GaussianFactorGraph graph;
    graph.push_back(linearized);
    const auto blocks = graph.hessianBlockDiagonal();
    const auto found = blocks.find(expected_key);
    if (found == blocks.end() ||
        found->second.rows() != 6 ||
        found->second.cols() != 6 ||
        !found->second.allFinite()) {
      *invalid_reason =
        "factor_hessian_block_missing_or_invalid";
      return false;
    }
    *hessian =
      0.5 *
      (found->second.template cast<double>() +
       found->second.transpose().template cast<double>());
    return hessian->allFinite();
  } catch (const std::exception& exception) {
    *invalid_reason =
      std::string("factor_hessian_exception:") +
      exception.what();
    return false;
  }
}

}  // namespace

OdometryEstimationCPUParams::OdometryEstimationCPUParams() : OdometryEstimationIMUParams() {
  // odometry config
  Config config(GlobalConfig::get_config_path("config_odometry"));

  registration_type = config.param<std::string>("odometry_estimation", "registration_type", "VGICP");
  max_iterations = config.param<int>("odometry_estimation", "max_iterations", 5);
  lru_thresh = config.param<int>("odometry_estimation", "lru_thresh", 100);
  target_downsampling_rate = config.param<double>("odometry_estimation", "target_downsampling_rate", 0.1);
  scan_matching_pose_factor_precision = config.param<double>("odometry_estimation", "scan_matching_pose_factor_precision", 1e3);
  scan_health_enable = config.param<bool>("odometry_estimation", "scan_health_enable", false);
  scan_health_apply_to_local_scan_precision = config.param<bool>("odometry_estimation", "scan_health_apply_to_local_scan_precision", false);
  scan_health_min_factor = config.param<double>("odometry_estimation", "scan_health_min_factor", 0.03);
  scan_health_reference_min_frames = config.param<int>("odometry_estimation", "scan_health_reference_min_frames", 30);
  scan_health_reference_ema_alpha = config.param<double>("odometry_estimation", "scan_health_reference_ema_alpha", 0.05);
  scan_health_reference_update_min_score = config.param<double>("odometry_estimation", "scan_health_reference_update_min_score", 0.8);
  scan_health_absolute_min_points = config.param<int>("odometry_estimation", "scan_health_absolute_min_points", 3000);
  scan_health_point_ratio_good = config.param<double>("odometry_estimation", "scan_health_point_ratio_good", 0.8);
  scan_health_point_ratio_bad = config.param<double>("odometry_estimation", "scan_health_point_ratio_bad", 0.4);
  scan_health_scan_imu_translation_good_m = config.param<double>("odometry_estimation", "scan_health_scan_imu_translation_good_m", 0.05);
  scan_health_scan_imu_translation_bad_m = config.param<double>("odometry_estimation", "scan_health_scan_imu_translation_bad_m", 0.15);
  scan_health_scan_imu_rotation_good_deg = config.param<double>("odometry_estimation", "scan_health_scan_imu_rotation_good_deg", 1.0);
  scan_health_scan_imu_rotation_bad_deg = config.param<double>("odometry_estimation", "scan_health_scan_imu_rotation_bad_deg", 5.0);
  scan_health_error_ratio_good = config.param<double>("odometry_estimation", "scan_health_error_ratio_good", 1.0);
  scan_health_error_ratio_bad = config.param<double>("odometry_estimation", "scan_health_error_ratio_bad", 1.05);
  scan_health_hessian_enable = config.param<bool>("odometry_estimation", "scan_health_hessian_enable", false);
  scan_health_hessian_min_ratio_good = config.param<double>("odometry_estimation", "scan_health_hessian_min_ratio_good", 0.5);
  scan_health_hessian_min_ratio_bad = config.param<double>("odometry_estimation", "scan_health_hessian_min_ratio_bad", 0.1);
  scan_health_hessian_frobenius_ratio_good = config.param<double>("odometry_estimation", "scan_health_hessian_frobenius_ratio_good", 0.5);
  scan_health_hessian_frobenius_ratio_bad = config.param<double>("odometry_estimation", "scan_health_hessian_frobenius_ratio_bad", 0.1);
  bool dcreg_section_present = false;
  std::string dcreg_config_error;
  if (!loadScanDcregHealthConfig(
        config,
        &dcreg_health,
        &dcreg_section_present,
        &dcreg_config_error)) {
    spdlog::error(
      "DCReg health configuration is invalid; disabling only the "
      "passive sidecar: {}",
      dcreg_config_error);
    dcreg_health.mode = ScanDcregMode::Off;
  }
  const auto legacy_dcreg_enabled =
    config.param<bool>(
      "odometry_estimation",
      "scan_dcreg_diagnostics_enable");
  if (legacy_dcreg_enabled && *legacy_dcreg_enabled) {
    if (dcreg_section_present &&
        dcreg_health.mode == ScanDcregMode::LogOnly) {
      spdlog::warn(
        "legacy scan_dcreg_diagnostics_enable is ignored because "
        "dcreg_health.mode=log_only is authoritative");
    } else {
      spdlog::warn(
        "legacy scan_dcreg_diagnostics_enable is ignored; the optional "
        "dcreg_health section is absent or off, so the sidecar remains off");
    }
  }

  ivox_resolution = config.param<double>("odometry_estimation", "ivox_resolution", 0.5);
  ivox_min_dist = config.param<double>("odometry_estimation", "ivox_min_dist", 0.1);

  vgicp_resolution = config.param<double>("odometry_estimation", "vgicp_resolution", 0.2);
  vgicp_voxelmap_levels = config.param<int>("odometry_estimation", "vgicp_voxelmap_levels", 2);
  vgicp_voxelmap_scaling_factor = config.param<double>("odometry_estimation", "vgicp_voxelmap_scaling_factor", 2.0);
}

OdometryEstimationCPUParams::~OdometryEstimationCPUParams() {}

OdometryEstimationCPU::OdometryEstimationCPU(const OdometryEstimationCPUParams& params) : OdometryEstimationIMU(std::make_unique<OdometryEstimationCPUParams>(params)) {
  last_T_target_imu.setIdentity();
  ScanDcregTestVariant dcreg_test_variant =
    ScanDcregTestVariant::FullLogOnly;
  bool dcreg_test_variant_valid = true;
  if (params.dcreg_health.mode == ScanDcregMode::LogOnly) {
    const char* test_variant =
      std::getenv("GLIM_DCREG_TEST_VARIANT");
    if (test_variant != nullptr && test_variant[0] != '\0') {
      dcreg_test_variant_valid = parseScanDcregTestVariant(
        test_variant, &dcreg_test_variant);
      if (!dcreg_test_variant_valid) {
        spdlog::error(
          "invalid test-only GLIM_DCREG_TEST_VARIANT='{}'; "
          "disabling the passive sidecar",
          test_variant);
      } else {
        spdlog::warn(
          "Stage 1.8 test-only DCReg scheduling variant enabled: {}",
          scanDcregTestVariantName(dcreg_test_variant));
      }
    }
  }
  if (scanDcregModeCreatesMonitor(params.dcreg_health.mode) &&
      dcreg_test_variant_valid) {
    ScanDcregRuntimeMetadata metadata;
    metadata.sensor_identifier =
      params.dcreg_health.reference.sensor_identifier;
    metadata.registration_type = params.registration_type;
    if (params.registration_type == "VGICP") {
      for (int level = 0; level < params.vgicp_voxelmap_levels; ++level) {
        metadata.voxel_resolutions.push_back(
          params.vgicp_resolution *
          std::pow(params.vgicp_voxelmap_scaling_factor, level));
      }
    } else {
      metadata.voxel_resolutions.push_back(params.ivox_resolution);
    }
    std::ostringstream fingerprint;
    fingerprint << std::setprecision(17)
                << params.registration_type << "|"
                << params.num_threads << "|"
                << params.dcreg_health.detection.epsilon_absolute << "|"
                << params.dcreg_health.detection.epsilon_relative << "|"
                << params.dcreg_health.detection
                     .pseudoinverse_relative_threshold;
    for (const double resolution : metadata.voxel_resolutions) {
      fingerprint << "|" << resolution;
    }
    metadata.configuration_fingerprint = fingerprint.str();
    dcreg_health_monitor =
      std::make_unique<ScanDcregHealthMonitor>(
        params.dcreg_health, metadata, dcreg_test_variant);
    spdlog::info(
      "DCReg observability-health sidecar mode=log_only, "
      "test_variant={}, reference_source={}, estimator control disabled",
      scanDcregTestVariantName(
        dcreg_test_variant),
      scanDcregReferenceSourceName(
        params.dcreg_health.reference.source));
  }
  if (params.registration_type == "GICP") {
    target_ivox.reset(new gtsam_points::iVox(params.ivox_resolution));
    target_ivox->voxel_insertion_setting().set_min_dist_in_cell(params.ivox_min_dist);
    target_ivox->set_lru_horizon(params.lru_thresh);
    target_ivox->set_neighbor_voxel_mode(1);
  } else if (params.registration_type == "VGICP") {
    target_voxelmaps.resize(params.vgicp_voxelmap_levels);
    for (int i = 0; i < params.vgicp_voxelmap_levels; i++) {
      const double resolution = params.vgicp_resolution * std::pow(params.vgicp_voxelmap_scaling_factor, i);
      target_voxelmaps[i] = std::make_shared<gtsam_points::GaussianVoxelMapCPU>(resolution);
      target_voxelmaps[i]->set_lru_horizon(params.lru_thresh);
    }
  } else {
    spdlog::error("unknown registration type for odometry_estimation_cpu ({})", params.registration_type);
    abort();
  }
}

OdometryEstimationCPU::~OdometryEstimationCPU() {}

gtsam::NonlinearFactorGraph OdometryEstimationCPU::create_factors(const int current, const gtsam_points::shared_ptr<gtsam::ImuFactor>& imu_factor, gtsam::Values& new_values) {
  const auto params = static_cast<const OdometryEstimationCPUParams*>(this->params.get());
  const int last = current - 1;

  if (current == 0) {
    last_T_target_imu = frames[current]->T_world_imu;
    update_target(current, frames[current]->T_world_imu);
    return gtsam::NonlinearFactorGraph();
  }

  const Eigen::Isometry3d pred_T_last_current = frames[last]->T_world_imu.inverse() * frames[current]->T_world_imu;
  const Eigen::Isometry3d pred_T_target_imu = last_T_target_imu * pred_T_last_current;

  gtsam::Values values;
  values.insert(X(current), gtsam::Pose3(pred_T_target_imu.matrix()));

  // Create frame-to-model matching factor
  gtsam::NonlinearFactorGraph matching_cost_factors;
  if (params->registration_type == "GICP") {
    auto gicp_factor = gtsam::make_shared<gtsam_points::IntegratedGICPFactor_<gtsam_points::iVox, gtsam_points::PointCloud>>(
      gtsam::Pose3(),
      X(current),
      target_ivox,
      frames[current]->frame,
      target_ivox);
    gicp_factor->set_max_correspondence_distance(params->ivox_resolution * 2.0);
    gicp_factor->set_num_threads(params->num_threads);
    matching_cost_factors.add(gicp_factor);
  } else if (params->registration_type == "VGICP") {
    for (const auto& voxelmap : target_voxelmaps) {
      auto vgicp_factor = gtsam::make_shared<gtsam_points::IntegratedVGICPFactor>(gtsam::Pose3(), X(current), voxelmap, frames[current]->frame);
      vgicp_factor->set_num_threads(params->num_threads);
      matching_cost_factors.add(vgicp_factor);
    }
  }

  gtsam::NonlinearFactorGraph graph;
  graph.add(matching_cost_factors);

  gtsam_points::LevenbergMarquardtExtParams lm_params;
  lm_params.setMaxIterations(params->max_iterations);
  lm_params.setAbsoluteErrorTol(0.1);

  int lm_iterations = 0;
  int lm_inner_iterations = 0;
  double lm_error = nanValue();
  double lm_cost_change = nanValue();
  double lm_lambda = nanValue();
  bool lm_solve_success = false;
  bool registration_converged = false;
  double lm_linearization_time = nanValue();
  double lm_linear_solver_time = nanValue();
  lm_params.callback = [&](const gtsam_points::LevenbergMarquardtOptimizationStatus& status, const gtsam::Values&) {
    lm_iterations = status.iterations;
    lm_inner_iterations = status.total_inner_iterations;
    lm_error = status.error;
    lm_cost_change = status.cost_change;
    lm_lambda = status.lambda;
    lm_solve_success = status.solve_success;
    lm_linearization_time = status.linearization_time;
    lm_linear_solver_time = status.linear_solver_time;
  };

  gtsam::Pose3 last_estimate = values.at<gtsam::Pose3>(X(current));
  lm_params.termination_criteria = [&](const gtsam::Values& values) {
    const gtsam::Pose3 current_pose = values.at<gtsam::Pose3>(X(current));
    const gtsam::Pose3 delta = last_estimate.inverse() * current_pose;

    const double delta_t = delta.translation().norm();
    const double delta_r = Eigen::AngleAxisd(delta.rotation().matrix()).angle();
    last_estimate = current_pose;

    if (delta_t < 1e-10 && delta_r < 1e-10) {
      // Maybe failed to solve the linear system
      return false;
    }

    // Preserve the existing convergence check and only observe its result.
    const bool pose_increment_converged =
      delta_t < 1e-3 &&
      delta_r < 1e-3 * M_PI / 180.0;
    registration_converged =
      registration_converged || pose_increment_converged;
    return pose_increment_converged;
  };

  std::vector<double> dcreg_initial_factor_costs;
  double dcreg_initial_cost_elapsed_ms = 0.0;
  if (dcreg_health_monitor) {
    const auto dcreg_initial_cost_start =
      std::chrono::steady_clock::now();
    dcreg_initial_factor_costs.reserve(
      matching_cost_factors.size());
    for (const auto& factor : matching_cost_factors) {
      try {
        dcreg_initial_factor_costs.push_back(
          factor ? factor->error(values) : nanValue());
      } catch (const std::exception&) {
        dcreg_initial_factor_costs.push_back(nanValue());
      }
    }
    dcreg_initial_cost_elapsed_ms =
      std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() -
        dcreg_initial_cost_start).count();
  }
  const double scan_initial_error = matching_cost_factors.error(values);

  // Optimize
  // lm_params.setDiagonalDamping(true);
  gtsam_points::LevenbergMarquardtOptimizerExt optimizer(graph, values, lm_params);

#ifdef GTSAM_USE_TBB
  auto arena = static_cast<tbb::task_arena*>(this->tbb_task_arena.get());
  arena->execute([&] {
#endif
    values = optimizer.optimize();
#ifdef GTSAM_USE_TBB
  });
#endif

  const Eigen::Isometry3d T_target_imu = Eigen::Isometry3d(values.at<gtsam::Pose3>(X(current)).matrix());
  const double scan_final_error = matching_cost_factors.error(values);
  const double scan_error_ratio =
    std::isfinite(scan_initial_error) && std::abs(scan_initial_error) > 1e-12 ? scan_final_error / scan_initial_error : nanValue();

  Eigen::Isometry3d T_last_current = last_T_target_imu.inverse() * T_target_imu;
  T_last_current.linear() = Eigen::Quaterniond(T_last_current.linear()).normalized().toRotationMatrix();

  const Eigen::Isometry3d scan_correction = pred_T_target_imu.inverse() * T_target_imu;
  const Eigen::Isometry3d scan_minus_imu = pred_T_last_current.inverse() * T_last_current;
  const size_t preprocessed_points = frames[current]->frame ? frames[current]->frame->size() : 0u;
  // Preserve the pre-Stage-1 scan-health path exactly. This is separate from
  // dcreg_health; mode=off does not collect any Hessian for the new sidecar.
  const HessianHealthMetrics hessian_metrics =
    scanHessianHealthMetrics(
      matching_cost_factors, values, X(current));

  ScanDcregDiagnostics dcreg_diagnostics;
  if (dcreg_health_monitor) {
    const auto dcreg_snapshot_start =
      std::chrono::steady_clock::now();
    ScanDcregMonitorInput monitor_input;
    monitor_input.stamp_sec = frames[current]->stamp;
    monitor_input.frame_index =
      static_cast<std::size_t>(current);
    monitor_input.T_world_imu =
      frames[last]->T_world_imu * T_last_current;
    monitor_input.glim_initialized = true;
    monitor_input.registration_converged =
      registration_converged;
    monitor_input.linear_solve_success = lm_solve_success;
    monitor_input.source_point_count =
      static_cast<int>(preprocessed_points);
    monitor_input.initial_cost = scan_initial_error;
    monitor_input.final_cost = scan_final_error;
    monitor_input.factors.reserve(matching_cost_factors.size());

    for (std::size_t factor_index = 0u;
         factor_index < matching_cost_factors.size();
         ++factor_index) {
      ScanDcregFactorInput factor_input;
      factor_input.factor_index =
        static_cast<int>(factor_index);
      factor_input.source_point_count =
        static_cast<int>(preprocessed_points);
      factor_input.initial_cost =
        factor_index < dcreg_initial_factor_costs.size()
          ? dcreg_initial_factor_costs[factor_index]
          : nanValue();

      const auto& factor =
        matching_cost_factors.at(factor_index);
      try {
        factor_input.final_cost =
          factor ? factor->error(values) : nanValue();
      } catch (const std::exception& exception) {
        factor_input.final_cost = nanValue();
        factor_input.invalid_reason =
          std::string("factor_final_cost_exception:") +
          exception.what();
      }
      factor_input.hessian_valid =
        extractUnaryPoseHessian(
          factor,
          values,
          X(current),
          &factor_input.hessian,
          &factor_input.invalid_reason);

      if (const auto* vgicp =
            dynamic_cast<
              const gtsam_points::IntegratedVGICPFactor*>(
                factor.get())) {
        factor_input.resolution =
          vgicp->get_target()
            ? vgicp->get_target()->voxel_resolution()
            : nanValue();
        if (factor_input.hessian_valid) {
          factor_input.inlier_count = vgicp->num_inliers();
          factor_input.inlier_fraction =
            preprocessed_points > 0u
              ? vgicp->inlier_fraction()
              : 0.0;
        }
      } else {
        using IvoxGicpFactor =
          gtsam_points::IntegratedGICPFactor_<
            gtsam_points::iVox,
            gtsam_points::PointCloud>;
        if (const auto* gicp =
              dynamic_cast<const IvoxGicpFactor*>(
                factor.get())) {
          factor_input.resolution =
            params->ivox_resolution;
          if (factor_input.hessian_valid &&
              preprocessed_points > 0u) {
            factor_input.inlier_fraction =
              gicp->inlier_fraction();
            factor_input.inlier_count =
              static_cast<int>(std::llround(
                factor_input.inlier_fraction *
                static_cast<double>(preprocessed_points)));
          }
        } else if (factor_input.invalid_reason.empty()) {
          factor_input.invalid_reason =
            "unsupported_matching_factor_type";
          factor_input.hessian_valid = false;
        }
      }
      monitor_input.factors.push_back(
        std::move(factor_input));
    }

    const double snapshot_elapsed_ms =
      std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() -
        dcreg_snapshot_start).count();
    dcreg_health_monitor->recordProducerTiming(
      dcreg_initial_cost_elapsed_ms,
      snapshot_elapsed_ms);

    try {
      dcreg_diagnostics =
        dcreg_health_monitor->process(monitor_input);
    } catch (const std::exception& exception) {
      spdlog::error(
        "passive DCReg health monitor failed at frame {}: {}",
        current,
        exception.what());
    }
  }
  const bool scan_health_reference_ready = scan_health_reference_count >= params->scan_health_reference_min_frames;
  const double point_reference = scan_health_point_reference;
  const double hessian_min_reference = scan_health_hessian_min_reference;
  const double hessian_frobenius_reference = scan_health_hessian_frobenius_reference;
  const double point_ratio = scan_health_reference_ready
                               ? finitePositiveRatioOrOne(static_cast<double>(preprocessed_points), point_reference)
                               : 1.0;
  const double hessian_min_ratio = scan_health_reference_ready
                                     ? finitePositiveRatioOrOne(hessian_metrics.min_eigenvalue, hessian_min_reference)
                                     : 1.0;
  const double hessian_frobenius_ratio = scan_health_reference_ready
                                           ? finitePositiveRatioOrOne(hessian_metrics.frobenius_norm, hessian_frobenius_reference)
                                           : 1.0;

  double point_score = lowRatioHealthScore(point_ratio, params->scan_health_point_ratio_good, params->scan_health_point_ratio_bad);
  if (params->scan_health_absolute_min_points > 0 && preprocessed_points < static_cast<size_t>(params->scan_health_absolute_min_points)) {
    point_score = std::min(point_score, static_cast<double>(preprocessed_points) / static_cast<double>(params->scan_health_absolute_min_points));
  }
  const double translation_score = highValueHealthScore(
    scan_minus_imu.translation().norm(),
    params->scan_health_scan_imu_translation_good_m,
    params->scan_health_scan_imu_translation_bad_m);
  const double rotation_score = highValueHealthScore(
    poseRotationDeg(scan_minus_imu),
    params->scan_health_scan_imu_rotation_good_deg,
    params->scan_health_scan_imu_rotation_bad_deg);
  const double error_ratio_score = highValueHealthScore(scan_error_ratio, params->scan_health_error_ratio_good, params->scan_health_error_ratio_bad);
  const double hessian_min_score = lowRatioHealthScore(
    hessian_min_ratio,
    params->scan_health_hessian_min_ratio_good,
    params->scan_health_hessian_min_ratio_bad);
  const double hessian_frobenius_score = lowRatioHealthScore(
    hessian_frobenius_ratio,
    params->scan_health_hessian_frobenius_ratio_good,
    params->scan_health_hessian_frobenius_ratio_bad);
  double scan_health_raw = 1.0;
  if (params->scan_health_enable && scan_health_reference_ready) {
    scan_health_raw = std::min({point_score, translation_score, rotation_score, error_ratio_score});
    if (params->scan_health_hessian_enable) {
      scan_health_raw = std::min({scan_health_raw, hessian_min_score, hessian_frobenius_score});
    }
  }
  const double scan_health_clamped = params->scan_health_enable ? std::max(params->scan_health_min_factor, clamp01(scan_health_raw)) : 1.0;
  const double effective_scan_precision =
    params->scan_health_enable && params->scan_health_apply_to_local_scan_precision
      ? params->scan_matching_pose_factor_precision * scan_health_clamped
      : params->scan_matching_pose_factor_precision;

  if (params->scan_health_enable &&
      (!scan_health_reference_ready || scan_health_raw >= params->scan_health_reference_update_min_score)) {
    updateHealthyReference(scan_health_point_reference, static_cast<double>(preprocessed_points), params->scan_health_reference_ema_alpha);
    updateHealthyReference(scan_health_hessian_min_reference, hessian_metrics.min_eigenvalue, params->scan_health_reference_ema_alpha);
    updateHealthyReference(scan_health_hessian_frobenius_reference, hessian_metrics.frobenius_norm, params->scan_health_reference_ema_alpha);
    scan_health_reference_count++;
  }

  spdlog::info(
    "GLIM_SCAN_HEALTH_ROW,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
    frames[current]->stamp,
    current,
    params->registration_type,
    preprocessed_points,
    matching_cost_factors.size(),
    scan_initial_error,
    scan_final_error,
    scan_error_ratio,
    lm_iterations,
    lm_inner_iterations,
    lm_error,
    lm_cost_change,
    lm_lambda,
    lm_solve_success ? 1 : 0,
    lm_linearization_time,
    lm_linear_solver_time,
    pred_T_last_current.translation().norm(),
    poseRotationDeg(pred_T_last_current),
    T_last_current.translation().norm(),
    poseRotationDeg(T_last_current),
    scan_minus_imu.translation().norm(),
    poseRotationDeg(scan_minus_imu),
    hessian_metrics.min_eigenvalue,
    hessian_metrics.max_eigenvalue,
    hessian_metrics.condition_estimate,
    hessian_metrics.rank_estimate,
    hessian_metrics.frobenius_norm,
    params->scan_health_enable ? 1 : 0,
    params->scan_health_apply_to_local_scan_precision ? 1 : 0,
    scan_health_reference_ready ? 1 : 0,
    scan_health_reference_count,
    point_reference,
    hessian_min_reference,
    hessian_frobenius_reference,
    point_ratio,
    point_score,
    translation_score,
    rotation_score,
    error_ratio_score,
    hessian_min_ratio,
    hessian_min_score,
    hessian_frobenius_ratio,
    hessian_frobenius_score,
    scan_health_raw,
    scan_health_clamped,
    params->scan_matching_pose_factor_precision,
    effective_scan_precision,
    params->scan_health_hessian_enable ? 1 : 0);

  frames[current]->T_world_imu = frames[last]->T_world_imu * T_last_current;
  new_values.insert_or_assign(X(current), gtsam::Pose3(frames[current]->T_world_imu.matrix()));

  if (dcreg_diagnostics.enabled) {
    dcreg_diagnostics.T_world_imu = frames[current]->T_world_imu;
    Callbacks::on_scan_dcreg_diagnostics(dcreg_diagnostics);
  }

  gtsam::NonlinearFactorGraph factors;

  // Get linearized matching cost factors
  // const auto linearized = optimizer.last_linearized();
  // for (int i = linearized->size() - matching_cost_factors.size(); i < linearized->size(); i++) {
  //   factors.emplace_shared<gtsam::LinearContainerFactor>(linearized->at(i), values);
  // }

  // TODO: Extract a relative pose covariance from a frame-to-model matching result?
  factors.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
    X(last), X(current), gtsam::Pose3(T_last_current.matrix()), gtsam::noiseModel::Isotropic::Precision(6, effective_scan_precision));
  factors.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
    X(current), gtsam::Pose3(T_target_imu.matrix()), gtsam::noiseModel::Isotropic::Precision(6, effective_scan_precision));

  spdlog::info(
    "GLIM_TARGET_UPDATE_ROW,{},{},update,scan_matched,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
    frames[current]->stamp,
    current,
    params->scan_health_enable ? 1 : 0,
    scan_health_reference_ready ? 1 : 0,
    scan_health_raw,
    scan_health_clamped,
    preprocessed_points,
    scan_error_ratio,
    scan_minus_imu.translation().norm(),
    poseRotationDeg(scan_minus_imu),
    scan_correction.translation().norm(),
    poseRotationDeg(scan_correction),
    T_target_imu.translation().x(),
    T_target_imu.translation().y(),
    T_target_imu.translation().z(),
    last_T_target_imu.translation().x(),
    last_T_target_imu.translation().y(),
    last_T_target_imu.translation().z());

  update_target(current, T_target_imu);
  last_T_target_imu = T_target_imu;

  return factors;
}

void OdometryEstimationCPU::fallback_smoother() {}

void OdometryEstimationCPU::update_target(const int current, const Eigen::Isometry3d& T_target_imu) {
  const auto params = static_cast<const OdometryEstimationCPUParams*>(this->params.get());
  auto frame = frames[current]->frame;
  if (current >= 5) {
    frame = gtsam_points::random_sampling(frames[current]->frame, params->target_downsampling_rate, mt);
  }

  auto transformed = gtsam_points::transform(frame, T_target_imu);
  if (params->registration_type == "GICP") {
    target_ivox->insert(*transformed);
  } else if (params->registration_type == "VGICP") {
    for (auto& target_voxelmap : target_voxelmaps) {
      target_voxelmap->insert(*transformed);
    }
  }

  // Update target point cloud (just for visualization).
  // This is not necessary for mapping and can safely be removed.
  if (frames.size() % 50 == 0) {
    EstimationFrame::Ptr frame(new EstimationFrame);
    frame->id = frames.size() - 1;
    frame->stamp = frames.back()->stamp;

    frame->T_lidar_imu = frames.back()->T_lidar_imu;
    frame->T_world_lidar = frame->T_lidar_imu.inverse();
    frame->T_world_imu.setIdentity();

    frame->v_world_imu.setZero();
    frame->imu_bias.setZero();

    frame->frame_id = FrameID::IMU;

    if (params->registration_type == "GICP") {
      frame->frame = std::make_shared<gtsam_points::PointCloudCPU>(target_ivox->voxel_points());
    } else if (params->registration_type == "VGICP") {
      frame->frame = std::make_shared<gtsam_points::PointCloudCPU>(target_voxelmaps[0]->voxel_points());
    }

    std::vector<EstimationFrame::ConstPtr> keyframes = {frame};
    Callbacks::on_update_keyframes(keyframes);

    if (target_ivox_frame) {
      std::vector<EstimationFrame::ConstPtr> marginalized_keyframes = {target_ivox_frame};
      Callbacks::on_marginalized_keyframes(marginalized_keyframes);
    }

    target_ivox_frame = frame;
  }
}

}  // namespace glim
