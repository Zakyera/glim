#include <glim/odometry/scan_dcreg_diagnostics.hpp>

#include <gtest/gtest.h>
#include <gtsam/geometry/Pose3.h>

namespace glim {
namespace {

TEST(ScanDcregDiagnostics, DecouplesDiagonalPoseHessian) {
  Eigen::Matrix<double, 6, 6> hessian =
    Eigen::Matrix<double, 6, 6>::Zero();
  hessian.diagonal() << 1.0, 10.0, 100.0, 2.0, 20.0, 200.0;

  const ScanDcregDiagnostics diagnostics =
    analyzeScanDcregHessian(hessian, 10.0);

  ASSERT_TRUE(diagnostics.valid);
  EXPECT_EQ(diagnostics.status, ScanDcregStatus::Valid);
  EXPECT_EQ(diagnostics.rotation_block_rank, 3);
  EXPECT_EQ(diagnostics.translation_block_rank, 3);
  EXPECT_NEAR(diagnostics.rotation_condition, 100.0, 1e-9);
  EXPECT_NEAR(diagnostics.translation_condition, 100.0, 1e-9);
  EXPECT_EQ(diagnostics.rotation_weak_modes[0], 1);
  EXPECT_EQ(diagnostics.rotation_weak_modes[1], 0);
  EXPECT_EQ(diagnostics.rotation_weak_modes[2], 0);
  EXPECT_EQ(diagnostics.translation_weak_modes[0], 1);
  EXPECT_NEAR(diagnostics.rotation_axis_weakness[0], 1.0, 1e-9);
  EXPECT_NEAR(diagnostics.translation_axis_weakness[0], 1.0, 1e-9);
}

TEST(ScanDcregDiagnostics, UsesComplementarySchurReductions) {
  Eigen::Matrix<double, 6, 6> hessian =
    Eigen::Matrix<double, 6, 6>::Zero();
  const Eigen::Matrix3d rotation =
    (Eigen::Vector3d() << 8.0, 10.0, 12.0).finished().asDiagonal();
  const Eigen::Matrix3d translation =
    (Eigen::Vector3d() << 4.0, 5.0, 6.0).finished().asDiagonal();
  const Eigen::Matrix3d coupling =
    (Eigen::Vector3d() << 2.0, 1.0, 3.0).finished().asDiagonal();
  hessian.block<3, 3>(0, 0) = rotation;
  hessian.block<3, 3>(0, 3) = coupling;
  hessian.block<3, 3>(3, 0) = coupling.transpose();
  hessian.block<3, 3>(3, 3) = translation;

  const ScanDcregDiagnostics diagnostics =
    analyzeScanDcregHessian(hessian, 10.0);

  ASSERT_TRUE(diagnostics.valid);
  const Eigen::Vector3d expected_rotation =
    (Eigen::Vector3d() << 7.0, 9.8, 10.5).finished();
  const Eigen::Vector3d expected_translation =
    (Eigen::Vector3d() << 3.5, 4.9, 5.25).finished();
  EXPECT_TRUE(
    diagnostics.rotation_eigenvalues.isApprox(expected_rotation, 1e-9));
  EXPECT_TRUE(
    diagnostics.translation_eigenvalues.isApprox(expected_translation, 1e-9));
}

TEST(ScanDcregDiagnostics, HandlesRankDeficientEliminatedBlocks) {
  Eigen::Matrix<double, 6, 6> hessian =
    Eigen::Matrix<double, 6, 6>::Zero();
  hessian.diagonal() << 0.0, 1.0, 1.0, 0.0, 2.0, 2.0;

  const ScanDcregDiagnostics diagnostics =
    analyzeScanDcregHessian(hessian, 10.0);

  ASSERT_TRUE(diagnostics.valid);
  EXPECT_EQ(diagnostics.rotation_block_rank, 2);
  EXPECT_EQ(diagnostics.translation_block_rank, 2);
  EXPECT_EQ(diagnostics.rotation_weak_modes[0], 1);
  EXPECT_EQ(diagnostics.translation_weak_modes[0], 1);
  EXPECT_DOUBLE_EQ(diagnostics.rotation_condition,
                   diagnostics.spectral_ratio_cap);
  EXPECT_DOUBLE_EQ(diagnostics.translation_condition,
                   diagnostics.spectral_ratio_cap);
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

  EXPECT_TRUE(relative_local_delta.isApprox(target_local_delta, 1.0e-10));
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

  EXPECT_TRUE(gtsam::Pose3::Logmap(pose).isApprox(expected, 1.0e-12));
}

}  // namespace
}  // namespace glim
