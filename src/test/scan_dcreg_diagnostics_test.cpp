#include <glim/odometry/scan_dcreg_diagnostics.hpp>

#include <cmath>
#include <fstream>
#include <string>

#include <unistd.h>

#include <gtest/gtest.h>
#include <gtsam/geometry/Pose3.h>

namespace glim {
namespace {

using Matrix6d = Eigen::Matrix<double, 6, 6>;

Matrix6d blockHessian(const Eigen::Matrix3d& rotation,
                      const Eigen::Matrix3d& translation) {
  Matrix6d hessian = Matrix6d::Zero();
  hessian.block<3, 3>(0, 0) = rotation;
  hessian.block<3, 3>(3, 3) = translation;
  return hessian;
}

Matrix6d diagonalHessian(const Eigen::Vector3d& rotation,
                         const Eigen::Vector3d& translation) {
  return blockHessian(
    rotation.asDiagonal(), translation.asDiagonal());
}

Eigen::Matrix3d spectrumMatrix(const Eigen::Vector3d& eigenvalues,
                               const Eigen::Matrix3d& basis) {
  return basis * eigenvalues.asDiagonal() * basis.transpose();
}

ScanDcregHealthConfig monitorConfig() {
  ScanDcregHealthConfig config;
  config.mode = ScanDcregMode::LogOnly;
  config.logging.enabled = false;
  config.reference.source = ScanDcregReferenceSource::Session;
  config.reference.bootstrap_window_size = 2;
  config.reference.bootstrap_minimum_samples = 3;
  config.reference.bootstrap_max_rotation_condition_ratio = 10.0;
  config.reference.bootstrap_max_translation_condition_ratio = 10.0;
  config.reference.temporal_stability_max_log_ratio_range = 0.10;
  config.reference.minimum_source_point_count = 10;
  config.reference.minimum_inlier_count = 10;
  config.reference.minimum_inlier_fraction = 0.10;
  config.reference.maximum_reference_change_ratio = 2.0;
  config.temporal.smoothing_alpha = 1.0;
  config.temporal.bad_frames_required = 2;
  config.temporal.good_frames_required = 2;
  return config;
}

ScanDcregRuntimeMetadata runtimeMetadata() {
  ScanDcregRuntimeMetadata metadata;
  metadata.sensor_identifier = "test_lidar";
  metadata.registration_type = "VGICP";
  metadata.voxel_resolutions = {0.5, 1.0};
  metadata.configuration_fingerprint = "test";
  return metadata;
}

ScanDcregReferenceCandidate healthyReferenceCandidate() {
  ScanDcregReferenceCandidate candidate;
  candidate.monitor_valid = true;
  candidate.factorization_ok = true;
  candidate.glim_initialized = true;
  candidate.registration_converged = true;
  candidate.linear_solve_success = true;
  candidate.source_point_count = 1000;
  candidate.minimum_factor_inlier_count = 700;
  candidate.minimum_factor_inlier_fraction = 0.7;
  candidate.initial_cost = 100.0;
  candidate.final_cost = 50.0;
  candidate.rotation_condition_ratios =
    (Eigen::Vector3d() << 4.0, 2.0, 1.0).finished();
  candidate.translation_condition_ratios =
    (Eigen::Vector3d() << 4.0, 2.0, 1.0).finished();
  return candidate;
}

class TemporaryOfflineProfile {
public:
  explicit TemporaryOfflineProfile(
    const std::string& fingerprint = "test") {
    char path_template[] = "/tmp/glim_dcreg_profile_XXXXXX";
    const int descriptor = ::mkstemp(path_template);
    EXPECT_GE(descriptor, 0);
    if (descriptor >= 0) {
      ::close(descriptor);
    }
    path_ = path_template;
    std::ofstream stream(path_);
    stream
      << "{\n"
      << "  \"schema_version\": 2,\n"
      << "  \"metadata\": {\n"
      << "    \"sensor_identifier\": \"test_lidar\",\n"
      << "    \"registration_type\": \"VGICP\",\n"
      << "    \"voxel_resolutions\": [0.5, 1.0],\n"
      << "    \"pose_ordering\": \"rotation_translation\",\n"
      << "    \"tangent_convention\": \"local_right\",\n"
      << "    \"configuration_fingerprint\": \""
      << fingerprint << "\"\n"
      << "  },\n"
      << "  \"rotation_log_condition_ratios\": ["
      << std::log(4.0) << "," << std::log(2.0) << ",0.0],\n"
      << "  \"translation_log_condition_ratios\": ["
      << std::log(4.0) << "," << std::log(2.0) << ",0.0],\n"
      << "  \"rotation_log_condition_mad\": [0.1,0.1,0.1],\n"
      << "  \"translation_log_condition_mad\": [0.1,0.1,0.1]\n"
      << "}\n";
  }

  ~TemporaryOfflineProfile() {
    if (!path_.empty()) {
      ::unlink(path_.c_str());
    }
  }

  const std::string& path() const {
    return path_;
  }

private:
  std::string path_;
};

ScanDcregMonitorInput monitorInput(const Matrix6d& aggregate,
                                   const std::size_t frame_index) {
  ScanDcregMonitorInput input;
  input.stamp_sec = 0.1 * static_cast<double>(frame_index);
  input.frame_index = frame_index;
  input.glim_initialized = true;
  input.registration_converged = true;
  input.linear_solve_success = true;
  input.source_point_count = 1000;
  input.initial_cost = 100.0;
  input.final_cost = 50.0;
  for (int factor_index = 0; factor_index < 2; ++factor_index) {
    ScanDcregFactorInput factor;
    factor.factor_index = factor_index;
    factor.resolution = factor_index == 0 ? 0.5 : 1.0;
    factor.source_point_count = 1000;
    factor.inlier_count = factor_index == 0 ? 800 : 700;
    factor.inlier_fraction =
      factor_index == 0 ? 0.8 : 0.7;
    factor.initial_cost = 50.0;
    factor.final_cost = 25.0;
    factor.hessian_valid = true;
    factor.hessian = 0.5 * aggregate;
    input.factors.push_back(factor);
  }
  return input;
}

Matrix6d healthyHessian() {
  return diagonalHessian(
    (Eigen::Vector3d() << 10.0, 20.0, 40.0).finished(),
    (Eigen::Vector3d() << 12.0, 24.0, 48.0).finished());
}

ScanDcregDiagnostics bootstrapHealthy(
  ScanDcregHealthMonitor* monitor,
  std::size_t* frame_index) {
  ScanDcregDiagnostics diagnostics;
  for (int sample = 0; sample < 4; ++sample) {
    diagnostics =
      monitor->process(
        monitorInput(healthyHessian(), ++(*frame_index)));
  }
  return diagnostics;
}

TEST(ScanDcregDiagnostics, WellConditionedHessian) {
  const ScanDcregSchurDiagnostics diagnostics =
    analyzeScanDcregHessian(
      healthyHessian(), ScanDcregDetectionConfig());

  ASSERT_TRUE(diagnostics.valid);
  EXPECT_EQ(diagnostics.status, ScanDcregStatus::Valid);
  EXPECT_TRUE(
    diagnostics.rotation_condition_ratios.isApprox(
      (Eigen::Vector3d() << 4.0, 2.0, 1.0).finished(),
      1e-12));
  EXPECT_TRUE(
    diagnostics.translation_condition_ratios.isApprox(
      (Eigen::Vector3d() << 4.0, 2.0, 1.0).finished(),
      1e-12));
}

TEST(ScanDcregDiagnostics, WeakTranslationMode) {
  const Matrix6d hessian = diagonalHessian(
    (Eigen::Vector3d() << 10.0, 20.0, 40.0).finished(),
    (Eigen::Vector3d() << 0.01, 20.0, 40.0).finished());
  const ScanDcregSchurDiagnostics diagnostics =
    analyzeScanDcregHessian(
      hessian, ScanDcregDetectionConfig());

  ASSERT_TRUE(diagnostics.valid);
  EXPECT_EQ(
    diagnostics.absolute_rotation_degenerate_mask.sum(), 0);
  EXPECT_EQ(
    diagnostics.absolute_translation_degenerate_mask[0], 1);
  EXPECT_EQ(
    diagnostics.absolute_translation_degenerate_mask[1], 0);
  EXPECT_EQ(
    diagnostics.absolute_translation_degenerate_mask[2], 0);
}

TEST(ScanDcregDiagnostics, WeakRotationMode) {
  const Matrix6d hessian = diagonalHessian(
    (Eigen::Vector3d() << 0.01, 20.0, 40.0).finished(),
    (Eigen::Vector3d() << 10.0, 20.0, 40.0).finished());
  const ScanDcregSchurDiagnostics diagnostics =
    analyzeScanDcregHessian(
      hessian, ScanDcregDetectionConfig());

  ASSERT_TRUE(diagnostics.valid);
  EXPECT_EQ(
    diagnostics.absolute_rotation_degenerate_mask[0], 1);
  EXPECT_EQ(
    diagnostics.absolute_translation_degenerate_mask.sum(), 0);
}

TEST(ScanDcregDiagnostics, MixedWeakEigenvector) {
  const double inverse_sqrt_two = 1.0 / std::sqrt(2.0);
  Eigen::Matrix3d basis;
  basis << inverse_sqrt_two, -inverse_sqrt_two, 0.0,
           inverse_sqrt_two,  inverse_sqrt_two, 0.0,
           0.0,               0.0,              1.0;
  const Eigen::Matrix3d translation =
    spectrumMatrix(
      (Eigen::Vector3d() << 0.1, 20.0, 40.0).finished(),
      basis);
  const ScanDcregSchurDiagnostics diagnostics =
    analyzeScanDcregHessian(
      blockHessian(
        (Eigen::Vector3d() << 10.0, 20.0, 40.0)
          .finished().asDiagonal(),
        translation),
      ScanDcregDetectionConfig());

  ASSERT_TRUE(diagnostics.valid);
  Eigen::Index weak_mode = 0;
  diagnostics.translation_condition_ratios.maxCoeff(&weak_mode);
  EXPECT_NEAR(
    diagnostics.translation_axis_contribution_ratios(
      0, weak_mode),
    0.5,
    1e-12);
  EXPECT_NEAR(
    diagnostics.translation_axis_contribution_ratios(
      1, weak_mode),
    0.5,
    1e-12);
  EXPECT_NEAR(
    diagnostics.translation_axis_contribution_ratios(
      2, weak_mode),
    0.0,
    1e-12);
}

TEST(ScanDcregDiagnostics, EigenvectorSignFlipIsInvariant) {
  const ScanDcregDetectionConfig config;
  const Matrix6d hessian = healthyHessian();
  const ScanDcregSchurDiagnostics first =
    analyzeScanDcregHessian(hessian, config);
  Eigen::Matrix3d sign_flipped_rotation =
    first.aligned_rotation_basis;
  Eigen::Matrix3d sign_flipped_translation =
    first.aligned_translation_basis;
  sign_flipped_rotation.col(0) *= -1.0;
  sign_flipped_translation.col(2) *= -1.0;
  const ScanDcregSchurDiagnostics second =
    analyzeScanDcregHessian(
      hessian,
      config,
      &sign_flipped_rotation,
      &sign_flipped_translation);

  EXPECT_TRUE(
    first.rotation_condition_ratios.isApprox(
      second.rotation_condition_ratios, 1e-12));
  EXPECT_TRUE(
    first.translation_axis_contribution_ratios.isApprox(
      second.translation_axis_contribution_ratios, 1e-12));
  EXPECT_GT(
    second.aligned_rotation_basis.col(0).dot(
      sign_flipped_rotation.col(0)),
    0.999999);
  EXPECT_GT(
    second.aligned_translation_basis.col(2).dot(
      sign_flipped_translation.col(2)),
    0.999999);
}

TEST(ScanDcregDiagnostics, EigenvectorOrderingSwapIsAligned) {
  const ScanDcregDetectionConfig config;
  Eigen::Matrix3d swapped_target;
  swapped_target.col(0) = Eigen::Vector3d::UnitY();
  swapped_target.col(1) = Eigen::Vector3d::UnitX();
  swapped_target.col(2) = Eigen::Vector3d::UnitZ();
  const ScanDcregSchurDiagnostics diagnostics =
    analyzeScanDcregHessian(
      healthyHessian(),
      config,
      &swapped_target,
      &swapped_target);

  EXPECT_EQ(
    diagnostics.aligned_rotation_original_indices[0], 1);
  EXPECT_EQ(
    diagnostics.aligned_rotation_original_indices[1], 0);
  EXPECT_NEAR(
    diagnostics.rotation_alignment_confidence.minCoeff(),
    1.0,
    1e-12);
  const Eigen::Matrix3d reconstructed =
    diagnostics.aligned_rotation_basis *
    diagnostics.aligned_rotation_eigenvalues.asDiagonal() *
    diagnostics.aligned_rotation_basis.transpose();
  EXPECT_TRUE(
    reconstructed.isApprox(
      healthyHessian().block<3, 3>(0, 0), 1e-12));
}

TEST(ScanDcregDiagnostics, ClusteredEigenspaceIsReported) {
  const Matrix6d hessian = diagonalHessian(
    (Eigen::Vector3d() << 1.0, 1.01, 10.0).finished(),
    (Eigen::Vector3d() << 2.0, 2.02, 20.0).finished());
  const ScanDcregSchurDiagnostics diagnostics =
    analyzeScanDcregHessian(
      hessian, ScanDcregDetectionConfig());

  ASSERT_TRUE(diagnostics.valid);
  EXPECT_EQ(
    diagnostics.rotation_spectral_cluster_flags[0], 1);
  EXPECT_EQ(
    diagnostics.rotation_spectral_cluster_flags[1], 1);
  EXPECT_EQ(
    diagnostics.translation_spectral_cluster_flags[0], 1);
  EXPECT_EQ(
    diagnostics.translation_spectral_cluster_flags[1], 1);
}

TEST(ScanDcregDiagnostics, UniformScalingPreservesShapeHealth) {
  const Matrix6d hessian = healthyHessian();
  const ScanDcregDetectionConfig config;
  const ScanDcregSchurDiagnostics first =
    analyzeScanDcregHessian(hessian, config);
  const ScanDcregSchurDiagnostics scaled =
    analyzeScanDcregHessian(100.0 * hessian, config);
  const ScanDcregHessianMetrics first_metrics =
    computeScanDcregHessianMetrics(
      hessian, 1000, 800, config);
  const ScanDcregHessianMetrics scaled_metrics =
    computeScanDcregHessianMetrics(
      100.0 * hessian, 1000, 800, config);

  EXPECT_TRUE(
    first.rotation_condition_ratios.isApprox(
      scaled.rotation_condition_ratios, 1e-12));
  EXPECT_TRUE(
    first.translation_condition_ratios.isApprox(
      scaled.translation_condition_ratios, 1e-12));
  EXPECT_NEAR(
    scaled_metrics.trace_per_source_point,
    100.0 * first_metrics.trace_per_source_point,
    1e-12);
}

TEST(ScanDcregDiagnostics, NearSingularSchurIsFiniteAndSevere) {
  const Matrix6d hessian = diagonalHessian(
    (Eigen::Vector3d() << 1.0e-14, 1.0, 2.0).finished(),
    (Eigen::Vector3d() << 2.0e-14, 2.0, 4.0).finished());
  const ScanDcregSchurDiagnostics diagnostics =
    analyzeScanDcregHessian(
      hessian, ScanDcregDetectionConfig());

  ASSERT_TRUE(diagnostics.valid);
  EXPECT_TRUE(
    diagnostics.rotation_condition_ratios.allFinite());
  EXPECT_GT(
    diagnostics.rotation_condition_ratios.maxCoeff(), 1.0e8);
  EXPECT_GT(
    diagnostics.translation_condition_ratios.maxCoeff(),
    1.0e8);
}

TEST(ScanDcregDiagnostics, SmallNumericalNegativeIsTolerated) {
  const Matrix6d hessian = diagonalHessian(
    (Eigen::Vector3d() << -1.0e-10, 5.0, 10.0).finished(),
    (Eigen::Vector3d() << -1.0e-10, 5.0, 10.0).finished());
  const ScanDcregSchurDiagnostics diagnostics =
    analyzeScanDcregHessian(
      hessian, ScanDcregDetectionConfig());

  ASSERT_TRUE(diagnostics.valid);
  EXPECT_EQ(
    diagnostics.rotation_numerical_negative_eigenvalues, 1);
  EXPECT_EQ(
    diagnostics.translation_numerical_negative_eigenvalues, 1);
  EXPECT_EQ(
    diagnostics.rotation_indefinite_eigenvalues, 0);
}

TEST(ScanDcregDiagnostics, InvalidIndefiniteInputIsPassive) {
  const Matrix6d hessian = diagonalHessian(
    (Eigen::Vector3d() << -1.0, 5.0, 10.0).finished(),
    (Eigen::Vector3d() << 1.0, 5.0, 10.0).finished());
  const ScanDcregSchurDiagnostics diagnostics =
    analyzeScanDcregHessian(
      hessian, ScanDcregDetectionConfig());

  EXPECT_FALSE(diagnostics.valid);
  EXPECT_TRUE(diagnostics.factorization_ok);
  EXPECT_EQ(
    diagnostics.status,
    ScanDcregStatus::IndefiniteRotationSchur);
}

TEST(ScanDcregHealth, HealthySessionBootstrapAndHealthOne) {
  ScanDcregHealthMonitor monitor(
    monitorConfig(), runtimeMetadata());
  std::size_t frame = 0u;
  const ScanDcregDiagnostics diagnostics =
    bootstrapHealthy(&monitor, &frame);

  ASSERT_TRUE(diagnostics.reference_ready);
  ASSERT_TRUE(diagnostics.health_available);
  EXPECT_NEAR(
    diagnostics.rotation_health_raw.minCoeff(), 1.0, 1e-12);
  EXPECT_NEAR(
    diagnostics.translation_health_raw.minCoeff(), 1.0, 1e-12);
  EXPECT_EQ(
    diagnostics.baseline_update_reason,
    "session_reference_initialized");
}

TEST(ScanDcregHealth,
     IntermittentRejectedScansDoNotEraseGatePassingWindow) {
  ScanDcregHealthMonitor monitor(
    monitorConfig(), runtimeMetadata());
  ScanDcregDiagnostics diagnostics;
  std::size_t frame = 0u;
  for (int sample = 0; sample < 4; ++sample) {
    diagnostics =
      monitor.process(
        monitorInput(healthyHessian(), ++frame));
    if (sample < 3) {
      ScanDcregMonitorInput rejected =
        monitorInput(healthyHessian(), ++frame);
      rejected.registration_converged = false;
      diagnostics = monitor.process(rejected);
      EXPECT_EQ(
        diagnostics.baseline_update_reason,
        "registration_not_converged");
    }
  }

  EXPECT_TRUE(diagnostics.reference_ready);
  EXPECT_TRUE(diagnostics.health_available);
  EXPECT_EQ(
    diagnostics.baseline_update_reason,
    "session_reference_initialized");
}

TEST(ScanDcregHealth, SessionBeginningDegenerateIsNotLearned) {
  ScanDcregHealthConfig config = monitorConfig();
  ScanDcregHealthMonitor monitor(config, runtimeMetadata());
  const Matrix6d degenerate = diagonalHessian(
    (Eigen::Vector3d() << 1.0e-6, 10.0, 20.0).finished(),
    (Eigen::Vector3d() << 1.0e-6, 10.0, 20.0).finished());
  ScanDcregDiagnostics diagnostics;
  for (std::size_t frame = 1u; frame <= 20u; ++frame) {
    diagnostics =
      monitor.process(monitorInput(degenerate, frame));
  }

  EXPECT_FALSE(diagnostics.reference_ready);
  EXPECT_FALSE(diagnostics.health_available);
  EXPECT_EQ(
    diagnostics.baseline_update_reason,
    "absolute_rotation_degeneracy+"
    "absolute_translation_degeneracy");
}

TEST(ScanDcregReference, RotationConditionAboveLimitIsRejected) {
  ScanDcregHealthConfig config = monitorConfig();
  config.reference.bootstrap_window_size = 1;
  config.reference.bootstrap_minimum_samples = 1;
  config.reference.bootstrap_max_rotation_condition_ratio = 8.0;
  ScanDcregReferenceManager manager(
    config.reference, runtimeMetadata());
  ScanDcregReferenceCandidate candidate =
    healthyReferenceCandidate();
  candidate.rotation_condition_ratios[0] = 9.0;

  const ScanDcregReferenceUpdate update = manager.update(
    candidate,
    Eigen::Vector3d::Constant(
      std::numeric_limits<double>::quiet_NaN()),
    Eigen::Vector3d::Constant(
      std::numeric_limits<double>::quiet_NaN()),
    Eigen::Array3i::Zero(),
    Eigen::Array3i::Zero());

  EXPECT_FALSE(update.reference_ready);
  EXPECT_FALSE(update.accepted);
  EXPECT_EQ(update.reason, "absolute_rotation_degeneracy");
}

TEST(ScanDcregReference, TranslationConditionAboveLimitIsRejected) {
  ScanDcregHealthConfig config = monitorConfig();
  config.reference.bootstrap_window_size = 1;
  config.reference.bootstrap_minimum_samples = 1;
  config.reference.bootstrap_max_translation_condition_ratio = 8.0;
  ScanDcregReferenceManager manager(
    config.reference, runtimeMetadata());
  ScanDcregReferenceCandidate candidate =
    healthyReferenceCandidate();
  candidate.translation_condition_ratios[0] = 9.0;

  const ScanDcregReferenceUpdate update = manager.update(
    candidate,
    Eigen::Vector3d::Constant(
      std::numeric_limits<double>::quiet_NaN()),
    Eigen::Vector3d::Constant(
      std::numeric_limits<double>::quiet_NaN()),
    Eigen::Array3i::Zero(),
    Eigen::Array3i::Zero());

  EXPECT_FALSE(update.reference_ready);
  EXPECT_FALSE(update.accepted);
  EXPECT_EQ(update.reason, "absolute_translation_degeneracy");
}

TEST(ScanDcregReference, AbsoluteMaskAlwaysPreventsBootstrap) {
  ScanDcregHealthConfig config = monitorConfig();
  config.reference.bootstrap_window_size = 1;
  config.reference.bootstrap_minimum_samples = 1;
  config.reference.bootstrap_max_rotation_condition_ratio = 100.0;
  config.reference.bootstrap_max_translation_condition_ratio = 100.0;
  ScanDcregReferenceManager manager(
    config.reference, runtimeMetadata());
  ScanDcregReferenceCandidate candidate =
    healthyReferenceCandidate();
  candidate.absolute_translation_degenerate_mask[1] = 1;

  const ScanDcregReferenceUpdate update = manager.update(
    candidate,
    Eigen::Vector3d::Constant(
      std::numeric_limits<double>::quiet_NaN()),
    Eigen::Vector3d::Constant(
      std::numeric_limits<double>::quiet_NaN()),
    Eigen::Array3i::Zero(),
    Eigen::Array3i::Zero());

  EXPECT_FALSE(update.reference_ready);
  EXPECT_EQ(update.reason, "absolute_translation_degeneracy");
}

TEST(ScanDcregReference, CompatibleOfflineReferenceIsReadyAtStartup) {
  TemporaryOfflineProfile profile;
  ScanDcregHealthConfig config = monitorConfig();
  config.reference.source = ScanDcregReferenceSource::Hybrid;
  config.reference.sensor_identifier = "test_lidar";
  config.reference.offline_profile_path = profile.path();
  ScanDcregReferenceManager manager(
    config.reference, runtimeMetadata());

  EXPECT_TRUE(manager.ready());
  EXPECT_EQ(manager.profileStatus(), "offline_profile_loaded");
  EXPECT_EQ(manager.activeSource(), "hybrid_offline");
  EXPECT_TRUE(
    manager.rotationReferenceRatios().isApprox(
      (Eigen::Vector3d() << 4.0, 2.0, 1.0).finished(),
      1e-5));
}

TEST(ScanDcregReference, IncompatibleOfflineReferenceIsRejected) {
  TemporaryOfflineProfile profile("different_configuration");
  ScanDcregHealthConfig config = monitorConfig();
  config.reference.source = ScanDcregReferenceSource::Hybrid;
  config.reference.sensor_identifier = "test_lidar";
  config.reference.offline_profile_path = profile.path();
  ScanDcregReferenceManager manager(
    config.reference, runtimeMetadata());

  EXPECT_FALSE(manager.ready());
  EXPECT_EQ(
    manager.profileStatus(),
    "offline_profile_configuration_fingerprint_mismatch");
  EXPECT_EQ(manager.activeSource(), "session_pending");
}

TEST(ScanDcregReference, HybridRejectsOutsideOfflineEnvelope) {
  TemporaryOfflineProfile profile;
  ScanDcregHealthConfig config = monitorConfig();
  config.reference.source = ScanDcregReferenceSource::Hybrid;
  config.reference.sensor_identifier = "test_lidar";
  config.reference.offline_profile_path = profile.path();
  config.reference.bootstrap_window_size = 1;
  ScanDcregReferenceManager manager(
    config.reference, runtimeMetadata());
  ScanDcregReferenceCandidate candidate =
    healthyReferenceCandidate();
  candidate.rotation_condition_ratios[0] = 5.5;

  const ScanDcregReferenceUpdate update = manager.update(
    candidate,
    Eigen::Vector3d::Ones(),
    Eigen::Vector3d::Ones(),
    Eigen::Array3i::Zero(),
    Eigen::Array3i::Zero());

  EXPECT_TRUE(update.reference_ready);
  EXPECT_FALSE(update.accepted);
  EXPECT_EQ(update.reason, "outside_offline_nominal_envelope");
}

TEST(ScanDcregHealth,
     CompatibleOfflineReferenceSupportsDegenerateStartup) {
  TemporaryOfflineProfile profile;
  ScanDcregHealthConfig config = monitorConfig();
  config.reference.source = ScanDcregReferenceSource::Hybrid;
  config.reference.sensor_identifier = "test_lidar";
  config.reference.offline_profile_path = profile.path();
  ScanDcregHealthMonitor monitor(config, runtimeMetadata());
  const Matrix6d degenerate = diagonalHessian(
    (Eigen::Vector3d() << 1.0e-6, 10.0, 20.0).finished(),
    (Eigen::Vector3d() << 1.0e-6, 10.0, 20.0).finished());

  const ScanDcregDiagnostics diagnostics =
    monitor.process(monitorInput(degenerate, 1u));

  EXPECT_TRUE(diagnostics.reference_ready);
  EXPECT_TRUE(diagnostics.health_available);
  EXPECT_FALSE(diagnostics.baseline_update_accepted);
  EXPECT_EQ(
    diagnostics.baseline_update_reason,
    "absolute_rotation_degeneracy+"
    "absolute_translation_degeneracy");
  EXPECT_EQ(diagnostics.reference_source, "hybrid_offline");
}

TEST(ScanDcregHealth, HybridDoesNotAdaptFromDegenerateSample) {
  TemporaryOfflineProfile profile;
  ScanDcregHealthConfig config = monitorConfig();
  config.reference.source = ScanDcregReferenceSource::Hybrid;
  config.reference.sensor_identifier = "test_lidar";
  config.reference.offline_profile_path = profile.path();
  ScanDcregHealthMonitor monitor(config, runtimeMetadata());
  const Matrix6d degenerate = diagonalHessian(
    (Eigen::Vector3d() << 1.0e-6, 10.0, 20.0).finished(),
    (Eigen::Vector3d() << 10.0, 20.0, 40.0).finished());

  const ScanDcregDiagnostics diagnostics =
    monitor.process(monitorInput(degenerate, 1u));

  EXPECT_FALSE(diagnostics.baseline_update_accepted);
  EXPECT_EQ(
    diagnostics.baseline_update_reason,
    "absolute_rotation_degeneracy");
  EXPECT_TRUE(
    diagnostics.rotation_reference_ratios.isApprox(
      (Eigen::Vector3d() << 4.0, 2.0, 1.0).finished(),
      1e-5));
}

TEST(ScanDcregHealth, HealthyDegradedHealthyFreezesAndRecovers) {
  ScanDcregHealthMonitor monitor(
    monitorConfig(), runtimeMetadata());
  std::size_t frame = 0u;
  const ScanDcregDiagnostics reference =
    bootstrapHealthy(&monitor, &frame);
  ASSERT_TRUE(reference.reference_ready);
  const Eigen::Vector3d reference_ratios =
    reference.translation_reference_ratios;

  const Matrix6d degraded = diagonalHessian(
    (Eigen::Vector3d() << 10.0, 20.0, 40.0).finished(),
    (Eigen::Vector3d() << 0.1, 24.0, 48.0).finished());
  ScanDcregDiagnostics diagnostics;
  diagnostics =
    monitor.process(monitorInput(degraded, ++frame));
  EXPECT_EQ(
    diagnostics.relative_translation_degraded_mask.sum(), 0);
  diagnostics =
    monitor.process(monitorInput(degraded, ++frame));
  EXPECT_EQ(
    diagnostics.relative_translation_degraded_mask[0], 1);
  EXPECT_TRUE(
    diagnostics.translation_reference_ratios.isApprox(
      reference_ratios, 1e-12));

  diagnostics =
    monitor.process(monitorInput(healthyHessian(), ++frame));
  EXPECT_EQ(
    diagnostics.relative_translation_degraded_mask[0], 1);
  diagnostics =
    monitor.process(monitorInput(healthyHessian(), ++frame));
  EXPECT_EQ(
    diagnostics.relative_translation_degraded_mask[0], 0);
  EXPECT_NEAR(
    diagnostics.translation_health_smoothed[0], 1.0, 1e-12);
}

TEST(ScanDcregHealth, OneFrameOutlierDoesNotSwitchState) {
  ScanDcregHealthConfig config = monitorConfig();
  config.temporal.bad_frames_required = 3;
  ScanDcregHealthMonitor monitor(config, runtimeMetadata());
  std::size_t frame = 0u;
  bootstrapHealthy(&monitor, &frame);
  const Matrix6d degraded = diagonalHessian(
    (Eigen::Vector3d() << 10.0, 20.0, 40.0).finished(),
    (Eigen::Vector3d() << 0.1, 24.0, 48.0).finished());

  ScanDcregDiagnostics diagnostics =
    monitor.process(monitorInput(degraded, ++frame));
  EXPECT_EQ(
    diagnostics.relative_translation_degraded_mask.sum(), 0);
  diagnostics =
    monitor.process(monitorInput(healthyHessian(), ++frame));
  EXPECT_EQ(
    diagnostics.relative_translation_degraded_mask.sum(), 0);
  EXPECT_EQ(diagnostics.translation_bad_frame_counters.sum(), 0);
}

TEST(ScanDcregHealth, RichSceneDoesNotRatchetReference) {
  ScanDcregHealthConfig config = monitorConfig();
  config.reference.bootstrap_window_size = 1;
  config.reference.bootstrap_minimum_samples = 3;
  config.reference.temporal_stability_max_log_ratio_range = 10.0;
  ScanDcregHealthMonitor monitor(config, runtimeMetadata());
  std::size_t frame = 0u;
  ScanDcregDiagnostics reference;
  for (int sample = 0; sample < 3; ++sample) {
    reference =
      monitor.process(
        monitorInput(healthyHessian(), ++frame));
  }
  ASSERT_TRUE(reference.reference_ready);
  const Eigen::Vector3d rotation_reference =
    reference.rotation_reference_ratios;
  const Matrix6d unusually_rich = diagonalHessian(
    (Eigen::Vector3d() << 28.0, 35.0, 42.0).finished(),
    (Eigen::Vector3d() << 34.0, 42.5, 51.0).finished());
  const ScanDcregDiagnostics diagnostics =
    monitor.process(monitorInput(unusually_rich, ++frame));

  EXPECT_FALSE(diagnostics.baseline_update_accepted);
  EXPECT_EQ(
    diagnostics.baseline_update_reason,
    "reference_change_too_large");
  EXPECT_TRUE(
    diagnostics.rotation_reference_ratios.isApprox(
      rotation_reference, 1e-12));
}

TEST(ScanDcregHealth, ModeOffReturnsBeforeEvaluation) {
  ScanDcregHealthConfig config = monitorConfig();
  config.mode = ScanDcregMode::Off;
  ScanDcregHealthMonitor monitor(config, runtimeMetadata());
  const ScanDcregDiagnostics diagnostics =
    monitor.process(monitorInput(healthyHessian(), 1u));

  EXPECT_FALSE(monitor.enabled());
  EXPECT_FALSE(diagnostics.enabled);
  EXPECT_EQ(monitor.evaluationCount(), 0u);
  EXPECT_EQ(monitor.formattedLogRowCount(), 0u);
}

TEST(ScanDcregHealth, AbsentAndExplicitOffShareConstructionBypass) {
  const ScanDcregHealthConfig section_absent_defaults;
  ScanDcregHealthConfig explicit_off;
  explicit_off.mode = ScanDcregMode::Off;

  EXPECT_FALSE(
    scanDcregModeCreatesMonitor(section_absent_defaults.mode));
  EXPECT_FALSE(scanDcregModeCreatesMonitor(explicit_off.mode));
  EXPECT_TRUE(scanDcregModeCreatesMonitor(ScanDcregMode::LogOnly));
}

TEST(ScanDcregHealth, LogOnlyDoesNotMutateEstimatorInput) {
  ScanDcregHealthMonitor monitor(
    monitorConfig(), runtimeMetadata());
  ScanDcregMonitorInput input =
    monitorInput(healthyHessian(), 1u);
  const ScanDcregMonitorInput original = input;
  const ScanDcregDiagnostics diagnostics =
    monitor.process(input);

  ASSERT_TRUE(diagnostics.enabled);
  ASSERT_EQ(input.factors.size(), original.factors.size());
  EXPECT_DOUBLE_EQ(input.initial_cost, original.initial_cost);
  EXPECT_DOUBLE_EQ(input.final_cost, original.final_cost);
  for (std::size_t index = 0u;
       index < input.factors.size();
       ++index) {
    EXPECT_TRUE(
      input.factors[index].hessian.isApprox(
        original.factors[index].hessian, 0.0));
  }
}

TEST(ScanDcregHealth, MissingReferenceRemainsPassive) {
  ScanDcregHealthMonitor monitor(
    monitorConfig(), runtimeMetadata());
  const ScanDcregDiagnostics diagnostics =
    monitor.process(monitorInput(healthyHessian(), 1u));

  EXPECT_TRUE(diagnostics.valid);
  EXPECT_FALSE(diagnostics.reference_ready);
  EXPECT_FALSE(diagnostics.health_available);
  EXPECT_FALSE(diagnostics.rotation_health_raw.allFinite());
  EXPECT_EQ(
    diagnostics.relative_rotation_degraded_mask.sum(), 0);
}

TEST(ScanDcregHealth, LoggingDisabledFormatsNothing) {
  ScanDcregHealthConfig config = monitorConfig();
  config.logging.enabled = false;
  config.logging.csv_path = "/path/that/must/not/be/opened.csv";
  ScanDcregHealthMonitor monitor(config, runtimeMetadata());
  monitor.process(monitorInput(healthyHessian(), 1u));

  EXPECT_EQ(monitor.formattedLogRowCount(), 0u);
  EXPECT_EQ(monitor.droppedLogSampleCount(), 0u);
  const ScanDcregLoggingStats stats = monitor.loggingStats();
  EXPECT_EQ(stats.enqueue_attempt_count, 0u);
  EXPECT_EQ(stats.maximum_queue_size, 0u);
  EXPECT_FALSE(stats.producer_waits_for_space);
}

TEST(ScanDcregHealth, ComputeOnlyCreatesNoQueueAndWritesNoFile) {
  char path_template[] = "/tmp/glim_dcreg_compute_only_XXXXXX";
  const int descriptor = ::mkstemp(path_template);
  ASSERT_GE(descriptor, 0);
  ::close(descriptor);
  ASSERT_EQ(::unlink(path_template), 0);

  ScanDcregHealthConfig config = monitorConfig();
  config.logging.enabled = true;
  config.logging.csv_path = path_template;
  ScanDcregHealthMonitor monitor(
    config,
    runtimeMetadata(),
    ScanDcregTestVariant::ComputeOnly);
  const ScanDcregDiagnostics diagnostics =
    monitor.process(monitorInput(healthyHessian(), 1u));

  EXPECT_TRUE(diagnostics.valid);
  EXPECT_EQ(::access(path_template, F_OK), -1);
  EXPECT_EQ(monitor.formattedLogRowCount(), 0u);
  const ScanDcregLoggingStats stats = monitor.loggingStats();
  EXPECT_EQ(stats.enqueue_attempt_count, 0u);
  EXPECT_EQ(stats.worker_sample_count, 0u);
  EXPECT_FALSE(stats.file_io_enabled);
}

TEST(ScanDcregHealth, EnqueueOnlyDiscardsWithoutFormattingOrFileIo) {
  char path_template[] = "/tmp/glim_dcreg_enqueue_only_XXXXXX";
  const int descriptor = ::mkstemp(path_template);
  ASSERT_GE(descriptor, 0);
  ::close(descriptor);
  ASSERT_EQ(::unlink(path_template), 0);

  ScanDcregHealthConfig config = monitorConfig();
  config.logging.enabled = true;
  config.logging.csv_path = path_template;
  ScanDcregHealthMonitor monitor(
    config,
    runtimeMetadata(),
    ScanDcregTestVariant::EnqueueOnly);
  monitor.process(monitorInput(healthyHessian(), 1u));

  const ScanDcregLoggingStats stats = monitor.loggingStats();
  EXPECT_EQ(stats.enqueue_attempt_count, 1u);
  EXPECT_EQ(stats.formatted_row_count, 0u);
  EXPECT_DOUBLE_EQ(stats.total_formatting_time_ms, 0.0);
  EXPECT_DOUBLE_EQ(stats.total_file_write_time_ms, 0.0);
  EXPECT_TRUE(stats.worker_discards_records);
  EXPECT_FALSE(stats.file_io_enabled);
  EXPECT_EQ(::access(path_template, F_OK), -1);
}

TEST(ScanDcregHealth, EnqueueOnlyOverflowDropsDiagnosticsNotProducer) {
  ScanDcregHealthConfig config = monitorConfig();
  config.logging.queue_capacity = 1u;
  ScanDcregHealthMonitor monitor(
    config,
    runtimeMetadata(),
    ScanDcregTestVariant::EnqueueOnly);
  for (std::size_t index = 0u; index < 5000u; ++index) {
    monitor.process(monitorInput(healthyHessian(), index + 1u));
  }
  const ScanDcregLoggingStats stats = monitor.loggingStats();
  EXPECT_EQ(stats.enqueue_attempt_count, 5000u);
  EXPECT_LE(stats.maximum_queue_size, 1u);
  EXPECT_TRUE(stats.producer_uses_try_lock);
  EXPECT_FALSE(stats.producer_waits_for_space);
  EXPECT_GT(stats.dropped_sample_count, 0u);
}

TEST(ScanDcregHealth, QueuedDiagnosticIsAnImmutableCopy) {
  ScanDcregHealthConfig config = monitorConfig();
  ScanDcregHealthMonitor monitor(
    config,
    runtimeMetadata(),
    ScanDcregTestVariant::EnqueueOnly);
  ScanDcregMonitorInput input =
    monitorInput(healthyHessian(), 1u);
  const Matrix6d captured = input.factors.front().hessian;
  monitor.process(input);
  input.factors.front().hessian.setConstant(-123.0);

  EXPECT_FALSE(input.factors.front().hessian.isApprox(captured, 0.0));
  EXPECT_EQ(monitor.evaluationCount(), 1u);
  EXPECT_EQ(monitor.loggingStats().enqueue_attempt_count, 1u);
}

TEST(ScanDcregHealth, LoggerInitializationFailureRemainsPassive) {
  ScanDcregHealthConfig config = monitorConfig();
  config.logging.enabled = true;
  config.logging.csv_path = "${GLIM_DCREG_UNRESOLVED}/health.csv";
  ScanDcregHealthMonitor monitor(
    config,
    runtimeMetadata(),
    ScanDcregTestVariant::FullLogOnly);
  const ScanDcregDiagnostics diagnostics =
    monitor.process(monitorInput(healthyHessian(), 1u));

  EXPECT_TRUE(diagnostics.valid);
  EXPECT_EQ(monitor.evaluationCount(), 1u);
  EXPECT_EQ(monitor.loggingStats().enqueue_attempt_count, 0u);
}

TEST(ScanDcregHealth, ShutdownDrainsOrDropsQueuedDiagnosticsSafely) {
  ScanDcregHealthConfig config = monitorConfig();
  config.logging.queue_capacity = 8u;
  {
    ScanDcregHealthMonitor monitor(
      config,
      runtimeMetadata(),
      ScanDcregTestVariant::EnqueueOnly);
    for (std::size_t index = 0u; index < 500u; ++index) {
      monitor.process(monitorInput(healthyHessian(), index + 1u));
    }
    EXPECT_EQ(monitor.loggingStats().enqueue_attempt_count, 500u);
  }
  SUCCEED();
}

TEST(ScanDcregHealth, AsyncLoggerIsBoundedAndDropsWithoutWaiting) {
  char path_template[] = "/tmp/glim_dcreg_async_XXXXXX";
  const int descriptor = ::mkstemp(path_template);
  ASSERT_GE(descriptor, 0);
  ::close(descriptor);

  ScanDcregHealthConfig config = monitorConfig();
  config.logging.enabled = true;
  config.logging.csv_path = path_template;
  config.logging.queue_capacity = 1u;
  config.logging.flush_every_n_rows = 100000;
  ScanDcregHealthMonitor monitor(config, runtimeMetadata());
  for (std::size_t index = 0u; index < 2000u; ++index) {
    monitor.process(monitorInput(healthyHessian(), index + 1u));
  }
  const ScanDcregLoggingStats stats = monitor.loggingStats();
  EXPECT_EQ(stats.enqueue_attempt_count, 2000u);
  EXPECT_LE(stats.maximum_queue_size, 1u);
  EXPECT_TRUE(stats.producer_uses_try_lock);
  EXPECT_FALSE(stats.producer_waits_for_space);
  EXPECT_GT(stats.dropped_sample_count, 0u);
  EXPECT_GT(stats.dropped_row_count, 0u);
  EXPECT_LT(stats.maximum_enqueue_time_ms, 100.0);
  ::unlink(path_template);
}

TEST(ScanDcregDiagnostics, TargetPoseAndBetweenResidualUseSameLocalTangent) {
  const gtsam::Pose3 from(
    gtsam::Rot3::RzRyRx(0.2, -0.1, 0.3),
    gtsam::Point3(1.0, -2.0, 0.5));
  const gtsam::Pose3 to(
    gtsam::Rot3::RzRyRx(-0.15, 0.25, -0.05),
    gtsam::Point3(2.0, 1.0, -0.25));
  gtsam::Vector6 target_local_delta;
  target_local_delta << 1.0e-5, -2.0e-5, 3.0e-5,
                        4.0e-5, -5.0e-5, 6.0e-5;

  const gtsam::Pose3 relative = from.between(to);
  const gtsam::Pose3 perturbed_relative =
    from.between(to.retract(target_local_delta));
  const gtsam::Vector6 relative_local_delta =
    relative.localCoordinates(perturbed_relative);

  EXPECT_TRUE(
    relative_local_delta.isApprox(
      target_local_delta, 1.0e-10));
}

TEST(ScanDcregDiagnostics, OfflineSe3LogFormulaMatchesGtsamConvention) {
  const gtsam::Pose3 pose(
    gtsam::Rot3::Quaternion(
      0.9393727128473789,
      0.0916432938695913,
      0.1832865877391826,
      0.2749298816087739),
    gtsam::Point3(1.0, -2.0, 0.5));
  gtsam::Vector6 expected;
  expected << 0.18708286933869706,
              0.3741657386773941,
              0.5612486080160912,
              0.2996280618166691,
             -2.1603344003864504,
              0.8403469129854103;

  EXPECT_TRUE(
    gtsam::Pose3::Logmap(pose).isApprox(
      expected, 1.0e-12));
}

}  // namespace
}  // namespace glim
