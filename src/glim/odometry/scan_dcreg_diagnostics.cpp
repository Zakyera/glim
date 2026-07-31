#include <glim/odometry/scan_dcreg_diagnostics.hpp>

#include <algorithm>
#include <cmath>

#include <Eigen/Eigenvalues>

namespace glim {
namespace {

constexpr double kAbsoluteEigenvalueTolerance = 1.0e-12;
constexpr double kRelativeEigenvalueTolerance = 1.0e-9;

bool symmetricPseudoInverse(const Eigen::Matrix3d& input,
                            Eigen::Matrix3d* inverse,
                            int* rank) {
  const Eigen::Matrix3d symmetric = 0.5 * (input + input.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(symmetric);
  if (solver.info() != Eigen::Success ||
      !solver.eigenvalues().allFinite() ||
      !solver.eigenvectors().allFinite()) {
    return false;
  }

  const double max_abs = solver.eigenvalues().cwiseAbs().maxCoeff();
  const double tolerance =
    std::max(kAbsoluteEigenvalueTolerance,
             kRelativeEigenvalueTolerance * max_abs);
  Eigen::Vector3d inverse_eigenvalues = Eigen::Vector3d::Zero();
  *rank = 0;
  for (int i = 0; i < 3; ++i) {
    if (std::abs(solver.eigenvalues()[i]) > tolerance) {
      inverse_eigenvalues[i] = 1.0 / solver.eigenvalues()[i];
      ++(*rank);
    }
  }

  *inverse = solver.eigenvectors() *
             inverse_eigenvalues.asDiagonal() *
             solver.eigenvectors().transpose();
  return inverse->allFinite();
}

bool characterizeSchur(const Eigen::Matrix3d& schur,
                       const double condition_threshold,
                       const double spectral_ratio_cap,
                       Eigen::Vector3d* eigenvalues,
                       Eigen::Matrix3d* eigenvectors,
                       Eigen::Vector3d* spectral_ratios,
                       Eigen::Array3i* weak_modes,
                       Eigen::Vector3d* axis_weakness,
                       double* condition,
                       int* negative_eigenvalues) {
  const Eigen::Matrix3d symmetric = 0.5 * (schur + schur.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(symmetric);
  if (solver.info() != Eigen::Success ||
      !solver.eigenvalues().allFinite() ||
      !solver.eigenvectors().allFinite()) {
    return false;
  }

  *eigenvalues = solver.eigenvalues();
  *eigenvectors = solver.eigenvectors();
  const double maximum = (*eigenvalues)[2];
  const double scale = std::max(1.0, std::abs(maximum));
  const double floor = std::max(kAbsoluteEigenvalueTolerance,
                                kRelativeEigenvalueTolerance * scale);
  *negative_eigenvalues = 0;
  weak_modes->setZero();
  axis_weakness->setZero();

  for (int mode = 0; mode < 3; ++mode) {
    const double eigenvalue = (*eigenvalues)[mode];
    if (eigenvalue < -floor) {
      ++(*negative_eigenvalues);
    }

    double ratio = spectral_ratio_cap;
    if (maximum > floor && eigenvalue > floor) {
      ratio = std::min(spectral_ratio_cap, maximum / eigenvalue);
    }
    (*spectral_ratios)[mode] = ratio;
    (*weak_modes)[mode] =
      (eigenvalue <= floor || ratio > condition_threshold) ? 1 : 0;
    if ((*weak_modes)[mode] != 0) {
      *axis_weakness += eigenvectors->col(mode).array().square().matrix();
    }
  }

  *condition = (*spectral_ratios)[0];
  return true;
}

}  // namespace

ScanDcregDiagnostics analyzeScanDcregHessian(
  const Eigen::Matrix<double, 6, 6>& hessian,
  const double condition_threshold,
  const double spectral_ratio_cap) {
  ScanDcregDiagnostics diagnostics;
  diagnostics.condition_threshold =
    std::max(1.0, condition_threshold);
  diagnostics.spectral_ratio_cap =
    std::max(diagnostics.condition_threshold, spectral_ratio_cap);

  if (!hessian.allFinite()) {
    diagnostics.status = ScanDcregStatus::NonFiniteHessian;
    return diagnostics;
  }

  const Eigen::Matrix<double, 6, 6> symmetric =
    0.5 * (hessian + hessian.transpose());
  const Eigen::Matrix3d H_rotation_rotation =
    symmetric.block<3, 3>(0, 0);
  const Eigen::Matrix3d H_rotation_translation =
    symmetric.block<3, 3>(0, 3);
  const Eigen::Matrix3d H_translation_rotation =
    symmetric.block<3, 3>(3, 0);
  const Eigen::Matrix3d H_translation_translation =
    symmetric.block<3, 3>(3, 3);

  Eigen::Matrix3d translation_inverse;
  Eigen::Matrix3d rotation_inverse;
  if (!symmetricPseudoInverse(H_translation_translation,
                              &translation_inverse,
                              &diagnostics.translation_block_rank) ||
      !symmetricPseudoInverse(H_rotation_rotation,
                              &rotation_inverse,
                              &diagnostics.rotation_block_rank)) {
    diagnostics.status =
      ScanDcregStatus::BlockEigendecompositionFailed;
    return diagnostics;
  }

  const Eigen::Matrix3d rotation_schur =
    H_rotation_rotation -
    H_rotation_translation * translation_inverse *
      H_translation_rotation;
  const Eigen::Matrix3d translation_schur =
    H_translation_translation -
    H_translation_rotation * rotation_inverse *
      H_rotation_translation;

  if (!characterizeSchur(rotation_schur,
                         diagnostics.condition_threshold,
                         diagnostics.spectral_ratio_cap,
                         &diagnostics.rotation_eigenvalues,
                         &diagnostics.rotation_eigenvectors,
                         &diagnostics.rotation_spectral_ratios,
                         &diagnostics.rotation_weak_modes,
                         &diagnostics.rotation_axis_weakness,
                         &diagnostics.rotation_condition,
                         &diagnostics.rotation_negative_eigenvalues) ||
      !characterizeSchur(translation_schur,
                         diagnostics.condition_threshold,
                         diagnostics.spectral_ratio_cap,
                         &diagnostics.translation_eigenvalues,
                         &diagnostics.translation_eigenvectors,
                         &diagnostics.translation_spectral_ratios,
                         &diagnostics.translation_weak_modes,
                         &diagnostics.translation_axis_weakness,
                         &diagnostics.translation_condition,
                         &diagnostics.translation_negative_eigenvalues)) {
    diagnostics.status =
      ScanDcregStatus::SchurEigendecompositionFailed;
    return diagnostics;
  }

  diagnostics.valid = true;
  const bool indefinite_rotation =
    diagnostics.rotation_negative_eigenvalues > 0;
  const bool indefinite_translation =
    diagnostics.translation_negative_eigenvalues > 0;
  if (indefinite_rotation && indefinite_translation) {
    diagnostics.status = ScanDcregStatus::IndefiniteBothSchur;
  } else if (indefinite_rotation) {
    diagnostics.status =
      ScanDcregStatus::IndefiniteRotationSchur;
  } else if (indefinite_translation) {
    diagnostics.status =
      ScanDcregStatus::IndefiniteTranslationSchur;
  } else {
    diagnostics.status = ScanDcregStatus::Valid;
  }
  return diagnostics;
}

const char* scanDcregStatusName(const ScanDcregStatus status) {
  switch (status) {
    case ScanDcregStatus::Valid:
      return "valid";
    case ScanDcregStatus::IndefiniteRotationSchur:
      return "indefinite_rotation_schur";
    case ScanDcregStatus::IndefiniteTranslationSchur:
      return "indefinite_translation_schur";
    case ScanDcregStatus::IndefiniteBothSchur:
      return "indefinite_both_schur";
    case ScanDcregStatus::NonFiniteHessian:
      return "nonfinite_hessian";
    case ScanDcregStatus::BlockEigendecompositionFailed:
      return "block_eigendecomposition_failed";
    case ScanDcregStatus::SchurEigendecompositionFailed:
      return "schur_eigendecomposition_failed";
  }
  return "unknown";
}

}  // namespace glim
