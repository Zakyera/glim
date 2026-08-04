#include <glim/odometry/scan_dcreg_diagnostics.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <thread>
#include <utility>

#include <Eigen/Eigenvalues>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace glim {
namespace {

using Matrix6d = Eigen::Matrix<double, 6, 6>;

double nanValue() {
  return std::numeric_limits<double>::quiet_NaN();
}

bool finitePositiveVector(const Eigen::Vector3d& values) {
  return values.allFinite() && (values.array() > 0.0).all();
}

double clamp01(const double value) {
  return std::max(0.0, std::min(1.0, value));
}

double safeMaximumAbsolute(const Eigen::Vector3d& values) {
  if (!values.allFinite()) {
    return nanValue();
  }
  return values.cwiseAbs().maxCoeff();
}

double eigenvalueFloor(const double maximum,
                       const ScanDcregDetectionConfig& config) {
  return config.epsilon_absolute +
         config.epsilon_relative * std::max(0.0, maximum);
}

double negativeEigenvalueTolerance(
  const Eigen::Vector3d& eigenvalues,
  const ScanDcregDetectionConfig& config) {
  return config.epsilon_absolute +
         config.negative_eigenvalue_tolerance *
           std::max(1.0, safeMaximumAbsolute(eigenvalues));
}

struct SymmetricPseudoInverseResult {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  bool factorization_ok = false;
  Eigen::Matrix3d inverse = Eigen::Matrix3d::Zero();
  int rank = 0;
};

SymmetricPseudoInverseResult symmetricPseudoInverse(
  const Eigen::Matrix3d& input,
  const ScanDcregDetectionConfig& config) {
  SymmetricPseudoInverseResult result;
  const Eigen::Matrix3d symmetric =
    0.5 * (input + input.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(symmetric);
  if (solver.info() != Eigen::Success ||
      !solver.eigenvalues().allFinite() ||
      !solver.eigenvectors().allFinite()) {
    return result;
  }

  const double maximum_absolute =
    solver.eigenvalues().cwiseAbs().maxCoeff();
  const double threshold =
    std::max(config.epsilon_absolute,
             config.pseudoinverse_relative_threshold * maximum_absolute);
  Eigen::Vector3d inverse_eigenvalues = Eigen::Vector3d::Zero();
  for (int index = 0; index < 3; ++index) {
    // A Hessian block is positive semidefinite in the valid case. Tiny or
    // negative modes are not inverted; significant indefiniteness is detected
    // from the Schur spectra below and reported as invalid.
    if (solver.eigenvalues()[index] > threshold) {
      inverse_eigenvalues[index] =
        1.0 / solver.eigenvalues()[index];
      ++result.rank;
    }
  }

  result.inverse =
    solver.eigenvectors() * inverse_eigenvalues.asDiagonal() *
    solver.eigenvectors().transpose();
  result.factorization_ok = result.inverse.allFinite();
  return result;
}

struct AlignedModes {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Vector3d eigenvalues = Eigen::Vector3d::Zero();
  Eigen::Vector3d condition_ratios = Eigen::Vector3d::Ones();
  Eigen::Matrix3d basis = Eigen::Matrix3d::Identity();
  Eigen::Array3i original_indices = Eigen::Array3i::Zero();
  Eigen::Matrix3d contribution_ratios = Eigen::Matrix3d::Identity();
  Eigen::Vector3d alignment_confidence = Eigen::Vector3d::Ones();
  Eigen::Array3i cluster_flags = Eigen::Array3i::Zero();
};

const std::array<std::array<int, 3>, 6>& permutations3() {
  static const std::array<std::array<int, 3>, 6> permutations = {{
    {{0, 1, 2}},
    {{0, 2, 1}},
    {{1, 0, 2}},
    {{1, 2, 0}},
    {{2, 0, 1}},
    {{2, 1, 0}},
  }};
  return permutations;
}

Eigen::Array3i rawClusterFlags(
  const Eigen::Vector3d& eigenvalues,
  const ScanDcregDetectionConfig& config) {
  Eigen::Array3i flags = Eigen::Array3i::Zero();
  for (int first = 0; first < 3; ++first) {
    for (int second = first + 1; second < 3; ++second) {
      const double denominator =
        std::max({config.epsilon_absolute,
                  std::abs(eigenvalues[first]),
                  std::abs(eigenvalues[second])});
      const double relative_gap =
        std::abs(eigenvalues[first] - eigenvalues[second]) /
        denominator;
      if (relative_gap <= config.spectral_cluster_relative_gap) {
        flags[first] = 1;
        flags[second] = 1;
      }
    }
  }
  return flags;
}

AlignedModes alignModes(
  const Eigen::Vector3d& raw_eigenvalues,
  const Eigen::Vector3d& raw_condition_ratios,
  const Eigen::Matrix3d& raw_basis,
  const Eigen::Matrix3d& target_basis,
  const ScanDcregDetectionConfig& config) {
  AlignedModes result;

  double best_score = -1.0;
  std::array<int, 3> best_permutation = {{0, 1, 2}};
  for (const auto& permutation : permutations3()) {
    double score = 0.0;
    for (int aligned_index = 0; aligned_index < 3; ++aligned_index) {
      score += std::abs(
        target_basis.col(aligned_index).dot(
          raw_basis.col(permutation[aligned_index])));
    }
    if (score > best_score) {
      best_score = score;
      best_permutation = permutation;
    }
  }

  const Eigen::Array3i raw_cluster_flags =
    rawClusterFlags(raw_eigenvalues, config);
  for (int aligned_index = 0; aligned_index < 3; ++aligned_index) {
    const int raw_index = best_permutation[aligned_index];
    Eigen::Vector3d direction = raw_basis.col(raw_index);
    double signed_alignment =
      target_basis.col(aligned_index).dot(direction);
    if (signed_alignment < 0.0) {
      direction = -direction;
      signed_alignment = -signed_alignment;
    } else if (std::abs(signed_alignment) <=
               config.epsilon_absolute) {
      Eigen::Index largest_axis = 0;
      direction.cwiseAbs().maxCoeff(&largest_axis);
      if (direction[largest_axis] < 0.0) {
        direction = -direction;
      }
    }

    result.eigenvalues[aligned_index] =
      raw_eigenvalues[raw_index];
    result.condition_ratios[aligned_index] =
      raw_condition_ratios[raw_index];
    result.basis.col(aligned_index) = direction;
    result.original_indices[aligned_index] = raw_index;
    result.alignment_confidence[aligned_index] =
      std::abs(target_basis.col(aligned_index).dot(direction));
    result.cluster_flags[aligned_index] =
      raw_cluster_flags[raw_index];
  }

  result.contribution_ratios =
    result.basis.array().square().matrix();
  return result;
}

struct SchurSpectrum {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  bool factorization_ok = false;
  bool no_support = false;
  Eigen::Vector3d raw_eigenvalues = Eigen::Vector3d::Zero();
  Eigen::Matrix3d raw_eigenvectors = Eigen::Matrix3d::Identity();
  Eigen::Vector3d raw_condition_ratios = Eigen::Vector3d::Ones();
  int numerical_negative_count = 0;
  int indefinite_count = 0;
};

SchurSpectrum characterizeSchur(
  const Eigen::Matrix3d& schur,
  const ScanDcregDetectionConfig& config) {
  SchurSpectrum result;
  const Eigen::Matrix3d symmetric =
    0.5 * (schur + schur.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(symmetric);
  if (solver.info() != Eigen::Success ||
      !solver.eigenvalues().allFinite() ||
      !solver.eigenvectors().allFinite()) {
    return result;
  }

  result.factorization_ok = true;
  result.raw_eigenvalues = solver.eigenvalues();
  result.raw_eigenvectors = solver.eigenvectors();

  const double negative_tolerance =
    negativeEigenvalueTolerance(result.raw_eigenvalues, config);
  for (int index = 0; index < 3; ++index) {
    if (result.raw_eigenvalues[index] < -negative_tolerance) {
      ++result.indefinite_count;
    } else if (result.raw_eigenvalues[index] < 0.0) {
      ++result.numerical_negative_count;
    }
  }

  const double maximum =
    std::max(0.0, result.raw_eigenvalues.maxCoeff());
  const double floor = eigenvalueFloor(maximum, config);
  result.no_support = maximum <= config.epsilon_absolute;
  for (int index = 0; index < 3; ++index) {
    if (result.no_support) {
      result.raw_condition_ratios[index] = 1.0;
      continue;
    }
    const double denominator =
      std::max(result.raw_eigenvalues[index], floor);
    result.raw_condition_ratios[index] = maximum / denominator;
  }
  return result;
}

Eigen::Vector3d medianVector(
  const std::vector<Eigen::Vector3d,
                    Eigen::aligned_allocator<Eigen::Vector3d>>& values) {
  Eigen::Vector3d median = Eigen::Vector3d::Zero();
  if (values.empty()) {
    return median;
  }
  for (int index = 0; index < 3; ++index) {
    std::vector<double> axis_values;
    axis_values.reserve(values.size());
    for (const auto& value : values) {
      axis_values.push_back(value[index]);
    }
    const std::size_t middle = axis_values.size() / 2u;
    std::nth_element(
      axis_values.begin(),
      axis_values.begin() + static_cast<std::ptrdiff_t>(middle),
      axis_values.end());
    median[index] = axis_values[middle];
    if (axis_values.size() % 2u == 0u) {
      const double upper = median[index];
      std::nth_element(
        axis_values.begin(),
        axis_values.begin() +
          static_cast<std::ptrdiff_t>(middle - 1u),
        axis_values.end());
      median[index] =
        0.5 * (upper + axis_values[middle - 1u]);
    }
  }
  return median;
}

std::string expandEnvironmentVariables(const std::string& input) {
  std::string output = input;
  std::size_t cursor = 0u;
  while ((cursor = output.find("${", cursor)) != std::string::npos) {
    const std::size_t end = output.find('}', cursor + 2u);
    if (end == std::string::npos) {
      break;
    }
    const std::string variable =
      output.substr(cursor + 2u, end - cursor - 2u);
    const char* value = std::getenv(variable.c_str());
    if (value == nullptr) {
      cursor = end + 1u;
      continue;
    }
    output.replace(cursor, end - cursor + 1u, value);
    cursor += std::char_traits<char>::length(value);
  }
  return output;
}

bool approximatelyEqual(const double first,
                        const double second,
                        const double relative_tolerance = 1.0e-9) {
  const double scale =
    std::max({1.0, std::abs(first), std::abs(second)});
  return std::abs(first - second) <= relative_tolerance * scale;
}

std::string csvEscape(const std::string& value) {
  std::string escaped;
  escaped.reserve(value.size() + 2u);
  escaped.push_back('"');
  for (const char character : value) {
    if (character == '"') {
      escaped.push_back('"');
    }
    escaped.push_back(character);
  }
  escaped.push_back('"');
  return escaped;
}

void appendVectorHeader(std::ostringstream* stream,
                        const std::string& prefix) {
  for (int index = 0; index < 3; ++index) {
    *stream << "," << prefix << "_" << index;
  }
}

void appendMatrixHeader(std::ostringstream* stream,
                        const std::string& prefix) {
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      *stream << "," << prefix << "_" << row << column;
    }
  }
}

void appendVector(std::ostringstream* stream,
                  const Eigen::Vector3d& values) {
  for (int index = 0; index < 3; ++index) {
    *stream << "," << values[index];
  }
}

void appendArray(std::ostringstream* stream,
                 const Eigen::Array3i& values) {
  for (int index = 0; index < 3; ++index) {
    *stream << "," << values[index];
  }
}

void appendMatrix(std::ostringstream* stream,
                  const Eigen::Matrix3d& values) {
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      *stream << "," << values(row, column);
    }
  }
}

std::string csvHeader() {
  std::ostringstream stream;
  stream
    << "schema_version,configuration_fingerprint,sensor_identifier"
    << ",registration_type"
    << ",pose_ordering,tangent_convention"
    << ",timestamp,frame_id,record_kind,factor_id,resolution"
    << ",valid,factorization_ok,status"
    << ",degeneracy_condition_threshold"
    << ",registration_converged,linear_solve_success"
    << ",reference_ready,health_available"
    << ",reference_source,offline_profile_status"
    << ",source_point_count,inlier_count,inlier_fraction"
    << ",minimum_factor_inlier_count,minimum_factor_inlier_fraction"
    << ",initial_cost,final_cost,cost_change"
    << ",hessian_trace,hessian_min_eigenvalue,hessian_max_eigenvalue"
    << ",hessian_rank,hessian_condition,hessian_frobenius_norm"
    << ",hessian_trace_per_source_point,hessian_trace_per_inlier"
    << ",rotation_block_rank,translation_block_rank"
    << ",rotation_numerical_negative_count"
    << ",translation_numerical_negative_count"
    << ",rotation_indefinite_count,translation_indefinite_count";
  appendVectorHeader(&stream, "raw_rotation_eigenvalue");
  appendVectorHeader(&stream, "raw_translation_eigenvalue");
  appendVectorHeader(&stream, "aligned_rotation_eigenvalue");
  appendVectorHeader(&stream, "aligned_translation_eigenvalue");
  appendVectorHeader(&stream, "rotation_condition_ratio");
  appendVectorHeader(&stream, "translation_condition_ratio");
  appendVectorHeader(&stream, "rotation_reference_ratio");
  appendVectorHeader(&stream, "translation_reference_ratio");
  appendVectorHeader(&stream, "rotation_health_raw");
  appendVectorHeader(&stream, "translation_health_raw");
  appendVectorHeader(&stream, "rotation_health_smoothed");
  appendVectorHeader(&stream, "translation_health_smoothed");
  appendVectorHeader(&stream, "absolute_rotation_mask");
  appendVectorHeader(&stream, "absolute_translation_mask");
  appendVectorHeader(&stream, "relative_rotation_mask");
  appendVectorHeader(&stream, "relative_translation_mask");
  appendVectorHeader(&stream, "rotation_bad_counter");
  appendVectorHeader(&stream, "translation_bad_counter");
  appendVectorHeader(&stream, "rotation_good_counter");
  appendVectorHeader(&stream, "translation_good_counter");
  appendVectorHeader(&stream, "aligned_rotation_original_index");
  appendVectorHeader(&stream, "aligned_translation_original_index");
  appendMatrixHeader(&stream, "raw_rotation_basis");
  appendMatrixHeader(&stream, "raw_translation_basis");
  appendMatrixHeader(&stream, "aligned_rotation_basis");
  appendMatrixHeader(&stream, "aligned_translation_basis");
  appendMatrixHeader(&stream, "rotation_axis_contribution");
  appendMatrixHeader(&stream, "translation_axis_contribution");
  appendVectorHeader(&stream, "rotation_alignment_confidence");
  appendVectorHeader(&stream, "translation_alignment_confidence");
  appendVectorHeader(&stream, "rotation_cluster_flag");
  appendVectorHeader(&stream, "translation_cluster_flag");
  stream
    << ",baseline_update_accepted,baseline_update_reason,invalid_reason"
    << ",evaluation_time_ms";
  return stream.str();
}

std::string csvRow(const ScanDcregDiagnostics& diagnostics,
                   const ScanDcregRuntimeMetadata& runtime_metadata,
                   const ScanDcregFactorDiagnostics* factor) {
  const bool aggregate = factor == nullptr;
  const ScanDcregSchurDiagnostics& schur =
    aggregate ? diagnostics.aggregate_schur : factor->schur;
  const ScanDcregHessianMetrics& hessian =
    aggregate ? diagnostics.aggregate_hessian : factor->hessian;
  const Eigen::Vector3d unavailable =
    Eigen::Vector3d::Constant(nanValue());
  const Eigen::Array3i unavailable_array =
    Eigen::Array3i::Constant(-1);

  std::ostringstream stream;
  stream << std::setprecision(17);
  stream
    << ScanDcregDiagnostics::kSchemaVersion
    << "," << csvEscape(runtime_metadata.configuration_fingerprint)
    << "," << csvEscape(runtime_metadata.sensor_identifier)
    << "," << csvEscape(runtime_metadata.registration_type)
    << "," << csvEscape(runtime_metadata.pose_ordering)
    << "," << csvEscape(runtime_metadata.tangent_convention)
    << "," << diagnostics.stamp_sec
    << "," << diagnostics.frame_index
    << "," << (aggregate ? "aggregate" : "factor")
    << "," << (aggregate ? -1 : factor->factor_index)
    << "," << (aggregate ? nanValue() : factor->resolution)
    << "," << (schur.valid ? 1 : 0)
    << "," << (schur.factorization_ok ? 1 : 0)
    << "," << scanDcregStatusName(schur.status)
    << "," << schur.condition_threshold
    << "," << (diagnostics.registration_converged ? 1 : 0)
    << "," << (diagnostics.linear_solve_success ? 1 : 0)
    << "," << (diagnostics.reference_ready ? 1 : 0)
    << "," << (diagnostics.health_available ? 1 : 0)
    << "," << csvEscape(
      aggregate ? diagnostics.reference_source : std::string())
    << "," << csvEscape(
      aggregate ? diagnostics.offline_profile_status : std::string())
    << "," << (aggregate ? diagnostics.source_point_count
                         : factor->source_point_count)
    << "," << (aggregate ? -1 : factor->inlier_count)
    << "," << (aggregate ? nanValue() : factor->inlier_fraction)
    << "," << diagnostics.minimum_factor_inlier_count
    << "," << diagnostics.minimum_factor_inlier_fraction
    << "," << (aggregate ? diagnostics.initial_cost
                         : factor->initial_cost)
    << "," << (aggregate ? diagnostics.final_cost
                         : factor->final_cost)
    << "," << (aggregate ? diagnostics.cost_change
                         : factor->cost_change)
    << "," << hessian.trace
    << "," << hessian.minimum_eigenvalue
    << "," << hessian.maximum_eigenvalue
    << "," << hessian.rank
    << "," << hessian.condition
    << "," << hessian.frobenius_norm
    << "," << hessian.trace_per_source_point
    << "," << hessian.trace_per_inlier
    << "," << schur.rotation_block_rank
    << "," << schur.translation_block_rank
    << "," << schur.rotation_numerical_negative_eigenvalues
    << "," << schur.translation_numerical_negative_eigenvalues
    << "," << schur.rotation_indefinite_eigenvalues
    << "," << schur.translation_indefinite_eigenvalues;
  appendVector(&stream, schur.raw_rotation_eigenvalues);
  appendVector(&stream, schur.raw_translation_eigenvalues);
  appendVector(&stream, schur.aligned_rotation_eigenvalues);
  appendVector(&stream, schur.aligned_translation_eigenvalues);
  appendVector(&stream, schur.rotation_condition_ratios);
  appendVector(&stream, schur.translation_condition_ratios);
  appendVector(
    &stream,
    aggregate ? diagnostics.rotation_reference_ratios : unavailable);
  appendVector(
    &stream,
    aggregate ? diagnostics.translation_reference_ratios : unavailable);
  appendVector(
    &stream,
    aggregate ? diagnostics.rotation_health_raw : unavailable);
  appendVector(
    &stream,
    aggregate ? diagnostics.translation_health_raw : unavailable);
  appendVector(
    &stream,
    aggregate ? diagnostics.rotation_health_smoothed : unavailable);
  appendVector(
    &stream,
    aggregate ? diagnostics.translation_health_smoothed : unavailable);
  appendArray(&stream, schur.absolute_rotation_degenerate_mask);
  appendArray(&stream, schur.absolute_translation_degenerate_mask);
  appendArray(
    &stream,
    aggregate ? diagnostics.relative_rotation_degraded_mask
              : unavailable_array);
  appendArray(
    &stream,
    aggregate ? diagnostics.relative_translation_degraded_mask
              : unavailable_array);
  appendArray(
    &stream,
    aggregate ? diagnostics.rotation_bad_frame_counters
              : unavailable_array);
  appendArray(
    &stream,
    aggregate ? diagnostics.translation_bad_frame_counters
              : unavailable_array);
  appendArray(
    &stream,
    aggregate ? diagnostics.rotation_good_frame_counters
              : unavailable_array);
  appendArray(
    &stream,
    aggregate ? diagnostics.translation_good_frame_counters
              : unavailable_array);
  appendArray(&stream, schur.aligned_rotation_original_indices);
  appendArray(&stream, schur.aligned_translation_original_indices);
  appendMatrix(&stream, schur.raw_rotation_eigenvectors);
  appendMatrix(&stream, schur.raw_translation_eigenvectors);
  appendMatrix(&stream, schur.aligned_rotation_basis);
  appendMatrix(&stream, schur.aligned_translation_basis);
  appendMatrix(&stream, schur.rotation_axis_contribution_ratios);
  appendMatrix(&stream, schur.translation_axis_contribution_ratios);
  appendVector(&stream, schur.rotation_alignment_confidence);
  appendVector(&stream, schur.translation_alignment_confidence);
  appendArray(&stream, schur.rotation_spectral_cluster_flags);
  appendArray(&stream, schur.translation_spectral_cluster_flags);
  stream
    << "," << (aggregate && diagnostics.baseline_update_accepted ? 1 : 0)
    << "," << csvEscape(
      aggregate ? diagnostics.baseline_update_reason : std::string())
    << "," << csvEscape(
      aggregate
        ? diagnostics.invalid_reason
        : factor->invalid_reason)
    << "," << diagnostics.evaluation_time_ms;
  return stream.str();
}

class AsyncScanDcregCsvLogger {
public:
  explicit AsyncScanDcregCsvLogger(
    const ScanDcregLoggingConfig& config,
    const ScanDcregRuntimeMetadata& runtime_metadata,
    const bool discard_records)
  : config_(config),
    runtime_metadata_(runtime_metadata),
    discard_records_(discard_records) {
    if (discard_records_) {
      enabled_ = true;
      worker_ = std::thread(&AsyncScanDcregCsvLogger::run, this);
      return;
    }
    if (!config_.enabled) {
      return;
    }

    const std::string expanded_path =
      expandEnvironmentVariables(config_.csv_path);
    if (expanded_path.empty() ||
        expanded_path.find("${") != std::string::npos) {
      spdlog::error(
        "DCReg health CSV path is empty or contains an unresolved "
        "environment variable: '{}'",
        config_.csv_path);
      return;
    }

    try {
      const std::filesystem::path path(expanded_path);
      if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
      }
      stream_.open(path, std::ios::out | std::ios::trunc);
      if (!stream_) {
        spdlog::error(
          "failed to open DCReg health CSV '{}'", expanded_path);
        return;
      }
      stream_ << csvHeader() << '\n';
      stream_.flush();
      enabled_ = true;
      worker_ = std::thread(&AsyncScanDcregCsvLogger::run, this);
      spdlog::info(
        "DCReg health structured logger enabled: '{}'", expanded_path);
    } catch (const std::exception& error) {
      spdlog::error(
        "failed to initialize DCReg health CSV '{}': {}",
        expanded_path,
        error.what());
    }
  }

  ~AsyncScanDcregCsvLogger() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    condition_.notify_one();
    if (worker_.joinable()) {
      worker_.join();
    }
    if (stream_) {
      stream_.flush();
    }
    const ScanDcregLoggingStats final_stats = stats();
    spdlog::info(
      "GLIM_DCREG_LOGGER_STATS_ROW,{},{},{},{},{},{},"
      "{:.9f},{:.9f},1,0",
      final_stats.enqueue_attempt_count,
      final_stats.enqueued_sample_count,
      final_stats.dropped_sample_count,
      final_stats.dropped_row_count,
      final_stats.formatted_row_count,
      final_stats.maximum_queue_size,
      final_stats.total_enqueue_time_ms,
      final_stats.maximum_enqueue_time_ms);
    spdlog::info(
      "GLIM_DCREG_WORKER_STATS_ROW,{},{:.9f},{:.9f},{:.9f},{}",
      final_stats.worker_sample_count,
      final_stats.total_worker_time_ms,
      final_stats.total_formatting_time_ms,
      final_stats.total_file_write_time_ms,
      final_stats.worker_discards_records ? 1 : 0);
  }

  bool enabled() const {
    return enabled_;
  }

  void enqueue(const ScanDcregDiagnostics& diagnostics) {
    if (!enabled_) {
      return;
    }
    const auto start = std::chrono::steady_clock::now();
    ++enqueue_attempts_;
    const auto finish = [&]() {
      const std::uint64_t elapsed_ns =
        static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start).count());
      total_enqueue_time_ns_.fetch_add(elapsed_ns);
      std::uint64_t previous = maximum_enqueue_time_ns_.load();
      while (previous < elapsed_ns &&
             !maximum_enqueue_time_ns_.compare_exchange_weak(
               previous, elapsed_ns)) {}
    };
    std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
    if (!lock.owns_lock() ||
        queue_.size() >= config_.queue_capacity) {
      ++dropped_samples_;
      dropped_rows_.fetch_add(1u + diagnostics.per_factor.size());
      finish();
      return;
    }
    queue_.push_back(diagnostics);
    ++enqueued_samples_;
    const std::size_t queue_size = queue_.size();
    std::size_t previous_maximum = maximum_queue_size_.load();
    while (previous_maximum < queue_size &&
           !maximum_queue_size_.compare_exchange_weak(
             previous_maximum, queue_size)) {}
    lock.unlock();
    condition_.notify_one();
    finish();
  }

  std::size_t formattedRowCount() const {
    return formatted_rows_.load();
  }

  std::size_t droppedSampleCount() const {
    return dropped_samples_.load();
  }

  ScanDcregLoggingStats stats() const {
    ScanDcregLoggingStats result;
    result.enqueue_attempt_count = enqueue_attempts_.load();
    result.enqueued_sample_count = enqueued_samples_.load();
    result.dropped_sample_count = dropped_samples_.load();
    result.dropped_row_count = dropped_rows_.load();
    result.formatted_row_count = formatted_rows_.load();
    result.maximum_queue_size = maximum_queue_size_.load();
    result.total_enqueue_time_ms =
      static_cast<double>(total_enqueue_time_ns_.load()) / 1.0e6;
    result.maximum_enqueue_time_ms =
      static_cast<double>(maximum_enqueue_time_ns_.load()) / 1.0e6;
    result.worker_sample_count = worker_samples_.load();
    result.total_worker_time_ms =
      static_cast<double>(total_worker_time_ns_.load()) / 1.0e6;
    result.total_formatting_time_ms =
      static_cast<double>(total_formatting_time_ns_.load()) / 1.0e6;
    result.total_file_write_time_ms =
      static_cast<double>(total_file_write_time_ns_.load()) / 1.0e6;
    result.worker_discards_records = discard_records_;
    result.file_io_enabled = !discard_records_ && stream_.is_open();
    return result;
  }

private:
  void run() {
    std::size_t rows_since_flush = 0u;
    while (true) {
      ScanDcregDiagnostics diagnostics;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(
          lock,
          [&] { return stopping_ || !queue_.empty(); });
        if (stopping_ && queue_.empty()) {
          break;
        }
        diagnostics = std::move(queue_.front());
        queue_.pop_front();
      }

      const auto worker_start = std::chrono::steady_clock::now();
      ++worker_samples_;
      if (!discard_records_) {
        const auto formatting_start =
          std::chrono::steady_clock::now();
        std::vector<std::string> rows;
        rows.reserve(1u + diagnostics.per_factor.size());
        rows.push_back(csvRow(
          diagnostics, runtime_metadata_, nullptr));
        for (const auto& factor : diagnostics.per_factor) {
          rows.push_back(csvRow(
            diagnostics, runtime_metadata_, &factor));
        }
        const auto formatting_finish =
          std::chrono::steady_clock::now();
        total_formatting_time_ns_.fetch_add(
          static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
              formatting_finish - formatting_start).count()));

        const auto write_start = std::chrono::steady_clock::now();
        for (const auto& row : rows) {
          stream_ << row << '\n';
          ++formatted_rows_;
          ++rows_since_flush;
        }
        if (rows_since_flush >=
            static_cast<std::size_t>(config_.flush_every_n_rows)) {
          stream_.flush();
          rows_since_flush = 0u;
        }
        const auto write_finish = std::chrono::steady_clock::now();
        total_file_write_time_ns_.fetch_add(
          static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
              write_finish - write_start).count()));
        if (!stream_) {
          enabled_ = false;
          spdlog::error(
            "DCReg health structured logger encountered a write failure");
          break;
        }
      }
      total_worker_time_ns_.fetch_add(
        static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - worker_start).count()));
    }
  }

private:
  ScanDcregLoggingConfig config_;
  ScanDcregRuntimeMetadata runtime_metadata_;
  std::ofstream stream_;
  std::atomic<bool> enabled_{false};
  std::atomic<std::size_t> formatted_rows_{0u};
  std::atomic<std::size_t> enqueue_attempts_{0u};
  std::atomic<std::size_t> enqueued_samples_{0u};
  std::atomic<std::size_t> dropped_samples_{0u};
  std::atomic<std::size_t> dropped_rows_{0u};
  std::atomic<std::size_t> maximum_queue_size_{0u};
  std::atomic<std::uint64_t> total_enqueue_time_ns_{0u};
  std::atomic<std::uint64_t> maximum_enqueue_time_ns_{0u};
  std::atomic<std::size_t> worker_samples_{0u};
  std::atomic<std::uint64_t> total_worker_time_ns_{0u};
  std::atomic<std::uint64_t> total_formatting_time_ns_{0u};
  std::atomic<std::uint64_t> total_file_write_time_ns_{0u};
  std::mutex mutex_;
  std::condition_variable condition_;
  std::deque<ScanDcregDiagnostics> queue_;
  bool stopping_ = false;
  bool discard_records_ = false;
  std::thread worker_;
};

struct TemporalModeState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  bool initialized = false;
  Eigen::Vector3d smoothed_log_deterioration =
    Eigen::Vector3d::Zero();
  Eigen::Array3i degraded = Eigen::Array3i::Zero();
  Eigen::Array3i bad_counters = Eigen::Array3i::Zero();
  Eigen::Array3i good_counters = Eigen::Array3i::Zero();
  Eigen::Vector3d last_smoothed_health =
    Eigen::Vector3d::Constant(nanValue());
  double availability_confidence = 0.0;
};

void updateTemporalModes(
  const Eigen::Vector3d& raw_log_deterioration,
  const ScanDcregTemporalConfig& config,
  TemporalModeState* state,
  Eigen::Vector3d* smoothed_health) {
  if (!state->initialized) {
    state->smoothed_log_deterioration = raw_log_deterioration;
    state->initialized = true;
  } else if (config.enabled) {
    state->smoothed_log_deterioration =
      (1.0 - config.smoothing_alpha) *
        state->smoothed_log_deterioration +
      config.smoothing_alpha * raw_log_deterioration;
  } else {
    state->smoothed_log_deterioration = raw_log_deterioration;
  }

  *smoothed_health =
    (-state->smoothed_log_deterioration.array()).exp().matrix();
  state->availability_confidence = 1.0;
  for (int mode = 0; mode < 3; ++mode) {
    const double health = (*smoothed_health)[mode];
    if (state->degraded[mode] == 0) {
      state->good_counters[mode] = 0;
      if (health < config.health_enter_threshold) {
        ++state->bad_counters[mode];
      } else {
        state->bad_counters[mode] = 0;
      }
      const int required =
        config.enabled ? config.bad_frames_required : 1;
      if (state->bad_counters[mode] >= required) {
        state->degraded[mode] = 1;
        state->bad_counters[mode] = 0;
      }
    } else {
      state->bad_counters[mode] = 0;
      if (health > config.health_exit_threshold) {
        ++state->good_counters[mode];
      } else {
        state->good_counters[mode] = 0;
      }
      const int required =
        config.enabled ? config.good_frames_required : 1;
      if (state->good_counters[mode] >= required) {
        state->degraded[mode] = 0;
        state->good_counters[mode] = 0;
      }
    }
  }
  state->last_smoothed_health = *smoothed_health;
}

Eigen::Vector3d invalidTemporalHealth(
  const ScanDcregTemporalConfig& config,
  TemporalModeState* state) {
  if (!state->initialized) {
    return Eigen::Vector3d::Constant(nanValue());
  }
  switch (config.invalid_frame_policy) {
    case ScanDcregInvalidFramePolicy::Hold:
      return state->last_smoothed_health;
    case ScanDcregInvalidFramePolicy::DecayTowardUnknown:
      state->availability_confidence *=
        (1.0 - config.smoothing_alpha);
      return state->last_smoothed_health;
    case ScanDcregInvalidFramePolicy::MarkUnavailable:
      return Eigen::Vector3d::Constant(nanValue());
  }
  return Eigen::Vector3d::Constant(nanValue());
}

}  // namespace

struct ScanDcregReferenceManager::Impl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  struct RatioPair {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector3d rotation_log_ratios = Eigen::Vector3d::Zero();
    Eigen::Vector3d translation_log_ratios = Eigen::Vector3d::Zero();
  };

  ScanDcregReferenceConfig config;
  ScanDcregRuntimeMetadata runtime_metadata;
  bool reference_ready = false;
  bool offline_loaded = false;
  Eigen::Vector3d rotation_log_reference =
    Eigen::Vector3d::Zero();
  Eigen::Vector3d translation_log_reference =
    Eigen::Vector3d::Zero();
  Eigen::Vector3d rotation_anchor =
    Eigen::Vector3d::Zero();
  Eigen::Vector3d translation_anchor =
    Eigen::Vector3d::Zero();
  Eigen::Vector3d offline_rotation_log_mad =
    Eigen::Vector3d::Zero();
  Eigen::Vector3d offline_translation_log_mad =
    Eigen::Vector3d::Zero();
  std::string profile_status;
  std::string active_source = "unavailable";
  std::deque<RatioPair> stability_window;
  std::vector<Eigen::Vector3d,
              Eigen::aligned_allocator<Eigen::Vector3d>>
    bootstrap_rotation_logs;
  std::vector<Eigen::Vector3d,
              Eigen::aligned_allocator<Eigen::Vector3d>>
    bootstrap_translation_logs;

  Impl(const ScanDcregReferenceConfig& config_,
       const ScanDcregRuntimeMetadata& runtime_metadata_)
  : config(config_), runtime_metadata(runtime_metadata_) {
    loadOfflineProfile();
  }

  bool sessionFallbackAllowed() const {
    return config.source == ScanDcregReferenceSource::Session ||
           config.source == ScanDcregReferenceSource::Hybrid;
  }

  void loadOfflineProfile() {
    if (config.source == ScanDcregReferenceSource::Session) {
      profile_status = "session_reference_selected";
      active_source = "session_pending";
      return;
    }
    if (config.offline_profile_path.empty()) {
      profile_status =
        config.source == ScanDcregReferenceSource::Hybrid
          ? "offline_profile_not_configured_session_fallback"
          : "offline_profile_path_empty";
      active_source =
        config.source == ScanDcregReferenceSource::Hybrid
          ? "session_pending"
          : "unavailable";
      return;
    }

    const std::string path =
      expandEnvironmentVariables(config.offline_profile_path);
    if (config.source == ScanDcregReferenceSource::Hybrid) {
      active_source = "session_pending";
    }
    try {
      std::ifstream stream(path);
      if (!stream) {
        profile_status = "offline_profile_open_failed";
        return;
      }
      nlohmann::json profile;
      stream >> profile;
      if (profile.value("schema_version", 0) != 2) {
        profile_status = "offline_profile_schema_mismatch";
        return;
      }

      if (!profile.contains("metadata") ||
          !profile["metadata"].is_object()) {
        profile_status = "offline_profile_metadata_missing";
        return;
      }
      const auto& metadata = profile["metadata"];
      if (!metadata.contains("sensor_identifier") ||
          !metadata.contains("registration_type") ||
          !metadata.contains("voxel_resolutions") ||
          !metadata.contains("pose_ordering") ||
          !metadata.contains("tangent_convention") ||
          !metadata.contains("configuration_fingerprint")) {
        profile_status = "offline_profile_metadata_incomplete";
        return;
      }

      if (config.require_profile_metadata_match) {
        if (!config.sensor_identifier.empty() &&
            metadata.value("sensor_identifier", std::string()) !=
              config.sensor_identifier) {
          profile_status =
            "offline_profile_sensor_identifier_mismatch";
          return;
        }
        if (metadata.value("registration_type", std::string()) !=
              runtime_metadata.registration_type ||
            metadata.value("pose_ordering", std::string()) !=
              runtime_metadata.pose_ordering ||
            metadata.value("tangent_convention", std::string()) !=
              runtime_metadata.tangent_convention) {
          profile_status = "offline_profile_metadata_mismatch";
          return;
        }
        const std::vector<double> profile_resolutions =
          metadata["voxel_resolutions"].get<std::vector<double>>();
        if (profile_resolutions.size() !=
            runtime_metadata.voxel_resolutions.size()) {
          profile_status =
            "offline_profile_voxel_resolution_count_mismatch";
          return;
        }
        for (std::size_t index = 0u;
             index < profile_resolutions.size();
             ++index) {
          if (!approximatelyEqual(
                profile_resolutions[index],
                runtime_metadata.voxel_resolutions[index])) {
            profile_status =
              "offline_profile_voxel_resolution_mismatch";
            return;
          }
        }
        const std::string profile_fingerprint =
          metadata.value("configuration_fingerprint", std::string());
        if (profile_fingerprint.empty() ||
            runtime_metadata.configuration_fingerprint.empty() ||
            profile_fingerprint !=
              runtime_metadata.configuration_fingerprint) {
          profile_status =
            "offline_profile_configuration_fingerprint_mismatch";
          return;
        }
      }

      const std::vector<double> rotation_values =
        profile.at("rotation_log_condition_ratios")
          .get<std::vector<double>>();
      const std::vector<double> translation_values =
        profile.at("translation_log_condition_ratios")
          .get<std::vector<double>>();
      const std::vector<double> rotation_mad_values =
        profile.at("rotation_log_condition_mad")
          .get<std::vector<double>>();
      const std::vector<double> translation_mad_values =
        profile.at("translation_log_condition_mad")
          .get<std::vector<double>>();
      if (rotation_values.size() != 3u ||
          translation_values.size() != 3u ||
          rotation_mad_values.size() != 3u ||
          translation_mad_values.size() != 3u) {
        profile_status = "offline_profile_ratio_dimension_mismatch";
        return;
      }
      for (int index = 0; index < 3; ++index) {
        rotation_log_reference[index] = rotation_values[index];
        translation_log_reference[index] =
          translation_values[index];
        offline_rotation_log_mad[index] =
          rotation_mad_values[index];
        offline_translation_log_mad[index] =
          translation_mad_values[index];
      }
      if (!rotation_log_reference.allFinite() ||
          !translation_log_reference.allFinite() ||
          !offline_rotation_log_mad.allFinite() ||
          !offline_translation_log_mad.allFinite() ||
          (offline_rotation_log_mad.array() < 0.0).any() ||
          (offline_translation_log_mad.array() < 0.0).any()) {
        profile_status = "offline_profile_nonfinite_ratios";
        return;
      }
      rotation_anchor = rotation_log_reference;
      translation_anchor = translation_log_reference;
      reference_ready = true;
      offline_loaded = true;
      profile_status = "offline_profile_loaded";
      active_source =
        config.source == ScanDcregReferenceSource::Hybrid
          ? "hybrid_offline"
          : "offline";
    } catch (const std::exception& error) {
      profile_status =
        std::string("offline_profile_invalid:") + error.what();
    }
  }

  std::optional<std::string> candidateGateFailure(
    const ScanDcregReferenceCandidate& candidate) const {
    if (!candidate.monitor_valid ||
        !candidate.factorization_ok) {
      return "numerical_failure";
    }
    if (!finitePositiveVector(
          candidate.rotation_condition_ratios) ||
        !finitePositiveVector(
          candidate.translation_condition_ratios)) {
      return "numerical_failure";
    }
    const bool rotation_degenerate =
      candidate.rotation_condition_ratios.maxCoeff() >
        config.bootstrap_max_rotation_condition_ratio ||
      (candidate.absolute_rotation_degenerate_mask != 0).any();
    const bool translation_degenerate =
      candidate.translation_condition_ratios.maxCoeff() >
        config.bootstrap_max_translation_condition_ratio ||
      (candidate.absolute_translation_degenerate_mask != 0).any();
    if (rotation_degenerate && translation_degenerate) {
      return
        "absolute_rotation_degeneracy+"
        "absolute_translation_degeneracy";
    }
    if (rotation_degenerate) {
      return "absolute_rotation_degeneracy";
    }
    if (translation_degenerate) {
      return "absolute_translation_degeneracy";
    }
    if (config.require_glim_initialized &&
        !candidate.glim_initialized) {
      return "glim_not_initialized";
    }
    if (config.require_registration_converged &&
        !candidate.registration_converged) {
      return "registration_not_converged";
    }
    if (config.require_linear_solve_success &&
        !candidate.linear_solve_success) {
      return "numerical_failure";
    }
    if (candidate.source_point_count <
        config.minimum_source_point_count) {
      return "insufficient_support";
    }
    if (candidate.minimum_factor_inlier_count <
        config.minimum_inlier_count) {
      return "insufficient_support";
    }
    if (!std::isfinite(candidate.minimum_factor_inlier_fraction) ||
        candidate.minimum_factor_inlier_fraction <
          config.minimum_inlier_fraction) {
      return "insufficient_support";
    }
    if (!std::isfinite(candidate.initial_cost) ||
        !std::isfinite(candidate.final_cost)) {
      return "numerical_failure";
    }
    if (config.maximum_initial_cost >= 0.0 &&
        candidate.initial_cost > config.maximum_initial_cost) {
      return "matching_quality_gate_failed";
    }
    if (config.maximum_final_cost >= 0.0 &&
        candidate.final_cost > config.maximum_final_cost) {
      return "matching_quality_gate_failed";
    }
    if (config.minimum_relative_cost_reduction >= 0.0) {
      if (std::abs(candidate.initial_cost) <=
          std::numeric_limits<double>::epsilon()) {
        return "numerical_failure";
      }
      const double relative_reduction =
        (candidate.initial_cost - candidate.final_cost) /
        std::abs(candidate.initial_cost);
      if (relative_reduction <
          config.minimum_relative_cost_reduction) {
        return "matching_quality_gate_failed";
      }
    }
    if (offline_loaded &&
        config.source == ScanDcregReferenceSource::Hybrid) {
      const Eigen::Vector3d rotation_log =
        candidate.rotation_condition_ratios.array().log().matrix();
      const Eigen::Vector3d translation_log =
        candidate.translation_condition_ratios.array().log().matrix();
      const Eigen::Vector3d rotation_half_width =
        (config.offline_nominal_mad_multiplier *
           offline_rotation_log_mad.array())
          .max(config.offline_nominal_min_log_half_width)
          .matrix();
      const Eigen::Vector3d translation_half_width =
        (config.offline_nominal_mad_multiplier *
           offline_translation_log_mad.array())
          .max(config.offline_nominal_min_log_half_width)
          .matrix();
      if (((rotation_log - rotation_anchor).array().abs() >
           rotation_half_width.array()).any() ||
          ((translation_log - translation_anchor).array().abs() >
           translation_half_width.array()).any()) {
        return "outside_offline_nominal_envelope";
      }
    }
    if (config.require_axis_alignment_confidence &&
        !candidate.axis_alignment_valid) {
      return "axis_alignment_confidence_low";
    }
    if (config.reject_clustered_modes &&
        (candidate.rotation_cluster_flags != 0).any()) {
      return "rotation_spectral_cluster";
    }
    if (config.reject_clustered_modes &&
        (candidate.translation_cluster_flags != 0).any()) {
      return "translation_spectral_cluster";
    }
    return std::nullopt;
  }

  bool temporalWindowStable(const RatioPair& current,
                            std::string* reason) {
    stability_window.push_back(current);
    while (stability_window.size() >
           static_cast<std::size_t>(config.bootstrap_window_size)) {
      stability_window.pop_front();
    }
    if (stability_window.size() <
        static_cast<std::size_t>(config.bootstrap_window_size)) {
      *reason = "unstable_condition_history";
      return false;
    }

    for (int mode = 0; mode < 3; ++mode) {
      double rotation_min = std::numeric_limits<double>::infinity();
      double rotation_max = -std::numeric_limits<double>::infinity();
      double translation_min = std::numeric_limits<double>::infinity();
      double translation_max = -std::numeric_limits<double>::infinity();
      for (const auto& sample : stability_window) {
        rotation_min =
          std::min(rotation_min, sample.rotation_log_ratios[mode]);
        rotation_max =
          std::max(rotation_max, sample.rotation_log_ratios[mode]);
        translation_min =
          std::min(
            translation_min,
            sample.translation_log_ratios[mode]);
        translation_max =
          std::max(
            translation_max,
            sample.translation_log_ratios[mode]);
      }
      if (rotation_max - rotation_min >
            config.temporal_stability_max_log_ratio_range ||
          translation_max - translation_min >
            config.temporal_stability_max_log_ratio_range) {
        *reason = "unstable_condition_history";
        return false;
      }
    }
    return true;
  }

  ScanDcregReferenceUpdate makeUpdate(
    const bool accepted,
    const bool initialized_now,
    const std::string& reason) const {
    ScanDcregReferenceUpdate update;
    update.reference_ready = reference_ready;
    update.initialized_now = initialized_now;
    update.accepted = accepted;
    update.reason = reason;
    if (reference_ready) {
      update.rotation_reference_ratios =
        rotation_log_reference.array().exp().matrix();
      update.translation_reference_ratios =
        translation_log_reference.array().exp().matrix();
    }
    return update;
  }

  ScanDcregReferenceUpdate update(
    const ScanDcregReferenceCandidate& candidate,
    const Eigen::Vector3d& current_rotation_health,
    const Eigen::Vector3d& current_translation_health,
    const Eigen::Array3i& current_rotation_degraded,
    const Eigen::Array3i& current_translation_degraded) {
    const auto gate_failure = candidateGateFailure(candidate);
    if (gate_failure) {
      // The window contains only gate-passing candidates. A rejected scan is
      // never learned, but it also does not erase already accepted evidence;
      // GLIM's explicit pose-increment convergence signal can legitimately be
      // intermittent even during otherwise steady registration.
      return makeUpdate(false, false, *gate_failure);
    }

    RatioPair current;
    current.rotation_log_ratios =
      candidate.rotation_condition_ratios.array().log().matrix();
    current.translation_log_ratios =
      candidate.translation_condition_ratios.array().log().matrix();
    std::string stability_reason;
    if (!temporalWindowStable(current, &stability_reason)) {
      return makeUpdate(false, false, stability_reason);
    }

    if (!reference_ready) {
      if (!sessionFallbackAllowed()) {
        return makeUpdate(
          false, false, "offline_reference_unavailable");
      }
      bootstrap_rotation_logs.push_back(
        current.rotation_log_ratios);
      bootstrap_translation_logs.push_back(
        current.translation_log_ratios);
      if (bootstrap_rotation_logs.size() <
          static_cast<std::size_t>(
            config.bootstrap_minimum_samples)) {
        return makeUpdate(
          true, false, "bootstrap_sample_accepted");
      }

      rotation_log_reference =
        medianVector(bootstrap_rotation_logs);
      translation_log_reference =
        medianVector(bootstrap_translation_logs);
      rotation_anchor = rotation_log_reference;
      translation_anchor = translation_log_reference;
      reference_ready = true;
      active_source = "session";
      return makeUpdate(
        true, true, "session_reference_initialized");
    }

    if (config.source == ScanDcregReferenceSource::Offline) {
      return makeUpdate(
        false, false, "offline_reference_fixed");
    }
    if (!config.adaptation_enabled) {
      return makeUpdate(
        false, false, "adaptation_disabled");
    }
    if (config.freeze_during_degradation &&
        ((current_rotation_degraded != 0).any() ||
         (current_translation_degraded != 0).any())) {
      return makeUpdate(
        false, false, "adaptation_frozen_relative_degradation");
    }

    const double minimum_adaptation_health =
      1.0 / config.maximum_reference_change_ratio;
    if (!current_rotation_health.allFinite() ||
        !current_translation_health.allFinite() ||
        current_rotation_health.minCoeff() <
          minimum_adaptation_health ||
        current_translation_health.minCoeff() <
          minimum_adaptation_health) {
      return makeUpdate(
        false, false, "adaptation_frozen_low_current_health");
    }

    const double maximum_log_change =
      std::log(config.maximum_reference_change_ratio);
    if (((current.rotation_log_ratios -
          rotation_log_reference).array().abs() >
         maximum_log_change).any() ||
        ((current.translation_log_ratios -
          translation_log_reference).array().abs() >
         maximum_log_change).any()) {
      return makeUpdate(
        false, false, "reference_change_too_large");
    }

    Eigen::Vector3d proposed_rotation =
      (1.0 - config.adaptation_rate) * rotation_log_reference +
      config.adaptation_rate * current.rotation_log_ratios;
    Eigen::Vector3d proposed_translation =
      (1.0 - config.adaptation_rate) *
        translation_log_reference +
      config.adaptation_rate * current.translation_log_ratios;
    Eigen::Vector3d rotation_anchor_half_width =
      Eigen::Vector3d::Constant(maximum_log_change);
    Eigen::Vector3d translation_anchor_half_width =
      Eigen::Vector3d::Constant(maximum_log_change);
    if (offline_loaded &&
        config.source == ScanDcregReferenceSource::Hybrid) {
      rotation_anchor_half_width =
        rotation_anchor_half_width.cwiseMin(
          (config.offline_nominal_mad_multiplier *
             offline_rotation_log_mad.array())
            .max(config.offline_nominal_min_log_half_width)
            .matrix());
      translation_anchor_half_width =
        translation_anchor_half_width.cwiseMin(
          (config.offline_nominal_mad_multiplier *
             offline_translation_log_mad.array())
            .max(config.offline_nominal_min_log_half_width)
            .matrix());
    }
    proposed_rotation =
      proposed_rotation
        .cwiseMax(rotation_anchor - rotation_anchor_half_width)
        .cwiseMin(rotation_anchor + rotation_anchor_half_width);
    proposed_translation =
      proposed_translation
        .cwiseMax(
          translation_anchor - translation_anchor_half_width)
        .cwiseMin(
          translation_anchor + translation_anchor_half_width);
    rotation_log_reference = proposed_rotation;
    translation_log_reference = proposed_translation;
    return makeUpdate(
      true, false, "reference_adaptation_accepted");
  }
};

ScanDcregReferenceManager::ScanDcregReferenceManager(
  const ScanDcregReferenceConfig& config,
  const ScanDcregRuntimeMetadata& runtime_metadata)
: impl(new Impl(config, runtime_metadata)) {}

ScanDcregReferenceManager::~ScanDcregReferenceManager() = default;

bool ScanDcregReferenceManager::ready() const {
  return impl->reference_ready;
}

const Eigen::Vector3d&
ScanDcregReferenceManager::rotationReferenceRatios() const {
  static const Eigen::Vector3d unavailable =
    Eigen::Vector3d::Constant(nanValue());
  if (!impl->reference_ready) {
    return unavailable;
  }
  // Cache-free reference-return storage is kept thread-local only for this
  // synchronous monitor API.
  static thread_local Eigen::Vector3d values;
  values = impl->rotation_log_reference.array().exp().matrix();
  return values;
}

const Eigen::Vector3d&
ScanDcregReferenceManager::translationReferenceRatios() const {
  static const Eigen::Vector3d unavailable =
    Eigen::Vector3d::Constant(nanValue());
  if (!impl->reference_ready) {
    return unavailable;
  }
  static thread_local Eigen::Vector3d values;
  values = impl->translation_log_reference.array().exp().matrix();
  return values;
}

const std::string& ScanDcregReferenceManager::profileStatus() const {
  return impl->profile_status;
}

const std::string& ScanDcregReferenceManager::activeSource() const {
  return impl->active_source;
}

ScanDcregReferenceUpdate ScanDcregReferenceManager::update(
  const ScanDcregReferenceCandidate& candidate,
  const Eigen::Vector3d& current_rotation_health,
  const Eigen::Vector3d& current_translation_health,
  const Eigen::Array3i& current_rotation_degraded,
  const Eigen::Array3i& current_translation_degraded) {
  return impl->update(
    candidate,
    current_rotation_health,
    current_translation_health,
    current_rotation_degraded,
    current_translation_degraded);
}

ScanDcregSchurDiagnostics analyzeScanDcregHessian(
  const Matrix6d& hessian,
  const ScanDcregDetectionConfig& config,
  const Eigen::Matrix3d* target_rotation_basis,
  const Eigen::Matrix3d* target_translation_basis) {
  ScanDcregSchurDiagnostics diagnostics;
  diagnostics.condition_threshold =
    config.degeneracy_condition_threshold;
  if (!hessian.allFinite()) {
    diagnostics.status = ScanDcregStatus::NonFiniteHessian;
    diagnostics.invalid_reason = "nonfinite_hessian";
    return diagnostics;
  }

  const Matrix6d symmetric =
    0.5 * (hessian + hessian.transpose());
  const Eigen::Matrix3d H_rotation_rotation =
    symmetric.block<3, 3>(0, 0);
  const Eigen::Matrix3d H_rotation_translation =
    symmetric.block<3, 3>(0, 3);
  const Eigen::Matrix3d H_translation_rotation =
    symmetric.block<3, 3>(3, 0);
  const Eigen::Matrix3d H_translation_translation =
    symmetric.block<3, 3>(3, 3);

  const SymmetricPseudoInverseResult translation_inverse =
    symmetricPseudoInverse(H_translation_translation, config);
  const SymmetricPseudoInverseResult rotation_inverse =
    symmetricPseudoInverse(H_rotation_rotation, config);
  diagnostics.translation_block_rank =
    translation_inverse.rank;
  diagnostics.rotation_block_rank = rotation_inverse.rank;
  if (!translation_inverse.factorization_ok ||
      !rotation_inverse.factorization_ok) {
    diagnostics.status =
      ScanDcregStatus::BlockEigendecompositionFailed;
    diagnostics.invalid_reason =
      "block_eigendecomposition_failed";
    return diagnostics;
  }

  const Eigen::Matrix3d rotation_schur =
    H_rotation_rotation -
    H_rotation_translation * translation_inverse.inverse *
      H_translation_rotation;
  const Eigen::Matrix3d translation_schur =
    H_translation_translation -
    H_translation_rotation * rotation_inverse.inverse *
      H_rotation_translation;

  const SchurSpectrum rotation =
    characterizeSchur(rotation_schur, config);
  const SchurSpectrum translation =
    characterizeSchur(translation_schur, config);
  if (!rotation.factorization_ok ||
      !translation.factorization_ok) {
    diagnostics.status =
      ScanDcregStatus::SchurEigendecompositionFailed;
    diagnostics.invalid_reason =
      "schur_eigendecomposition_failed";
    return diagnostics;
  }

  diagnostics.factorization_ok = true;
  diagnostics.raw_rotation_eigenvalues =
    rotation.raw_eigenvalues;
  diagnostics.raw_translation_eigenvalues =
    translation.raw_eigenvalues;
  diagnostics.raw_rotation_eigenvectors =
    rotation.raw_eigenvectors;
  diagnostics.raw_translation_eigenvectors =
    translation.raw_eigenvectors;
  diagnostics.rotation_numerical_negative_eigenvalues =
    rotation.numerical_negative_count;
  diagnostics.translation_numerical_negative_eigenvalues =
    translation.numerical_negative_count;
  diagnostics.rotation_indefinite_eigenvalues =
    rotation.indefinite_count;
  diagnostics.translation_indefinite_eigenvalues =
    translation.indefinite_count;

  const Eigen::Matrix3d rotation_target =
    target_rotation_basis == nullptr
      ? Eigen::Matrix3d::Identity()
      : *target_rotation_basis;
  const Eigen::Matrix3d translation_target =
    target_translation_basis == nullptr
      ? Eigen::Matrix3d::Identity()
      : *target_translation_basis;
  const AlignedModes aligned_rotation =
    alignModes(
      rotation.raw_eigenvalues,
      rotation.raw_condition_ratios,
      rotation.raw_eigenvectors,
      rotation_target,
      config);
  const AlignedModes aligned_translation =
    alignModes(
      translation.raw_eigenvalues,
      translation.raw_condition_ratios,
      translation.raw_eigenvectors,
      translation_target,
      config);

  diagnostics.aligned_rotation_eigenvalues =
    aligned_rotation.eigenvalues;
  diagnostics.aligned_translation_eigenvalues =
    aligned_translation.eigenvalues;
  diagnostics.aligned_rotation_basis =
    aligned_rotation.basis;
  diagnostics.aligned_translation_basis =
    aligned_translation.basis;
  diagnostics.aligned_rotation_original_indices =
    aligned_rotation.original_indices;
  diagnostics.aligned_translation_original_indices =
    aligned_translation.original_indices;
  diagnostics.rotation_condition_ratios =
    aligned_rotation.condition_ratios;
  diagnostics.translation_condition_ratios =
    aligned_translation.condition_ratios;
  diagnostics.rotation_axis_contribution_ratios =
    aligned_rotation.contribution_ratios;
  diagnostics.translation_axis_contribution_ratios =
    aligned_translation.contribution_ratios;
  diagnostics.rotation_alignment_confidence =
    aligned_rotation.alignment_confidence;
  diagnostics.translation_alignment_confidence =
    aligned_translation.alignment_confidence;
  diagnostics.rotation_spectral_cluster_flags =
    aligned_rotation.cluster_flags;
  diagnostics.translation_spectral_cluster_flags =
    aligned_translation.cluster_flags;
  for (int mode = 0; mode < 3; ++mode) {
    diagnostics.absolute_rotation_degenerate_mask[mode] =
      diagnostics.rotation_condition_ratios[mode] >
          config.degeneracy_condition_threshold
        ? 1
        : 0;
    diagnostics.absolute_translation_degenerate_mask[mode] =
      diagnostics.translation_condition_ratios[mode] >
          config.degeneracy_condition_threshold
        ? 1
        : 0;
  }

  if (rotation.no_support || translation.no_support) {
    diagnostics.status = ScanDcregStatus::NoLidarSupport;
    diagnostics.invalid_reason = "no_lidar_support";
    return diagnostics;
  }

  const bool indefinite_rotation =
    rotation.indefinite_count > 0;
  const bool indefinite_translation =
    translation.indefinite_count > 0;
  if (indefinite_rotation && indefinite_translation) {
    diagnostics.status = ScanDcregStatus::IndefiniteBothSchur;
    diagnostics.invalid_reason = "indefinite_both_schur";
    return diagnostics;
  }
  if (indefinite_rotation) {
    diagnostics.status =
      ScanDcregStatus::IndefiniteRotationSchur;
    diagnostics.invalid_reason = "indefinite_rotation_schur";
    return diagnostics;
  }
  if (indefinite_translation) {
    diagnostics.status =
      ScanDcregStatus::IndefiniteTranslationSchur;
    diagnostics.invalid_reason =
      "indefinite_translation_schur";
    return diagnostics;
  }

  diagnostics.valid = true;
  diagnostics.status = ScanDcregStatus::Valid;
  return diagnostics;
}

ScanDcregHessianMetrics computeScanDcregHessianMetrics(
  const Matrix6d& hessian,
  const int source_point_count,
  const int inlier_count,
  const ScanDcregDetectionConfig& config) {
  ScanDcregHessianMetrics metrics;
  if (!hessian.allFinite()) {
    return metrics;
  }
  metrics.hessian = 0.5 * (hessian + hessian.transpose());
  Eigen::SelfAdjointEigenSolver<Matrix6d> solver(metrics.hessian);
  if (solver.info() != Eigen::Success ||
      !solver.eigenvalues().allFinite()) {
    return metrics;
  }

  metrics.valid = true;
  metrics.trace = metrics.hessian.trace();
  metrics.frobenius_norm = metrics.hessian.norm();
  metrics.minimum_eigenvalue = solver.eigenvalues().minCoeff();
  metrics.maximum_eigenvalue = solver.eigenvalues().maxCoeff();
  const double maximum_absolute =
    solver.eigenvalues().cwiseAbs().maxCoeff();
  const double rank_threshold =
    std::max(config.epsilon_absolute,
             config.epsilon_relative * maximum_absolute);
  double minimum_positive =
    std::numeric_limits<double>::infinity();
  for (int index = 0; index < solver.eigenvalues().size(); ++index) {
    if (solver.eigenvalues()[index] > rank_threshold) {
      ++metrics.rank;
      minimum_positive =
        std::min(minimum_positive, solver.eigenvalues()[index]);
    }
  }
  if (metrics.maximum_eigenvalue > 0.0 &&
      std::isfinite(minimum_positive)) {
    metrics.condition =
      metrics.maximum_eigenvalue / minimum_positive;
  } else {
    metrics.condition = nanValue();
  }
  metrics.trace_per_source_point =
    source_point_count > 0
      ? metrics.trace / static_cast<double>(source_point_count)
      : nanValue();
  metrics.trace_per_inlier =
    inlier_count > 0
      ? metrics.trace / static_cast<double>(inlier_count)
      : nanValue();
  return metrics;
}

struct ScanDcregHealthMonitor::Impl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ScanDcregHealthConfig config;
  ScanDcregTestVariant test_variant =
    ScanDcregTestVariant::FullLogOnly;
  ScanDcregRuntimeMetadata runtime_metadata;
  ScanDcregReferenceManager reference_manager;
  std::unique_ptr<AsyncScanDcregCsvLogger> logger;
  std::size_t evaluation_count = 0u;
  std::uint64_t total_evaluation_time_ns = 0u;
  std::uint64_t maximum_evaluation_time_ns = 0u;
  std::size_t snapshot_count = 0u;
  double total_initial_cost_time_ms = 0.0;
  double maximum_initial_cost_time_ms = 0.0;
  double total_snapshot_time_ms = 0.0;
  double maximum_snapshot_time_ms = 0.0;

  bool have_previous_aggregate_basis = false;
  Eigen::Matrix3d previous_rotation_basis =
    Eigen::Matrix3d::Identity();
  Eigen::Matrix3d previous_translation_basis =
    Eigen::Matrix3d::Identity();
  std::vector<Eigen::Matrix3d,
              Eigen::aligned_allocator<Eigen::Matrix3d>>
    previous_factor_rotation_bases;
  std::vector<Eigen::Matrix3d,
              Eigen::aligned_allocator<Eigen::Matrix3d>>
    previous_factor_translation_bases;
  std::vector<bool> have_previous_factor_basis;

  TemporalModeState rotation_temporal;
  TemporalModeState translation_temporal;

  Impl(const ScanDcregHealthConfig& config_,
       const ScanDcregRuntimeMetadata& runtime_metadata_,
       const ScanDcregTestVariant test_variant_)
  : config(config_),
    test_variant(test_variant_),
    runtime_metadata(runtime_metadata_),
    reference_manager(config.reference, runtime_metadata) {
    if (config.mode == ScanDcregMode::LogOnly &&
        test_variant ==
          ScanDcregTestVariant::EnqueueOnly) {
      logger.reset(
        new AsyncScanDcregCsvLogger(
          config.logging, runtime_metadata, true));
    } else if (config.mode == ScanDcregMode::LogOnly &&
               test_variant ==
                 ScanDcregTestVariant::FullLogOnly &&
               config.logging.enabled) {
      logger.reset(
        new AsyncScanDcregCsvLogger(
          config.logging, runtime_metadata, false));
    }
  }

  ScanDcregReferenceCandidate referenceCandidate(
    const ScanDcregMonitorInput& input,
    const ScanDcregDiagnostics& diagnostics) const {
    ScanDcregReferenceCandidate candidate;
    candidate.monitor_valid = diagnostics.valid;
    candidate.factorization_ok = diagnostics.factorization_ok;
    candidate.glim_initialized = input.glim_initialized;
    candidate.registration_converged =
      input.registration_converged;
    candidate.linear_solve_success =
      input.linear_solve_success;
    candidate.source_point_count =
      diagnostics.source_point_count;
    candidate.minimum_factor_inlier_count =
      diagnostics.minimum_factor_inlier_count;
    candidate.minimum_factor_inlier_fraction =
      diagnostics.minimum_factor_inlier_fraction;
    candidate.initial_cost = diagnostics.initial_cost;
    candidate.final_cost = diagnostics.final_cost;
    candidate.rotation_condition_ratios =
      diagnostics.aggregate_schur.rotation_condition_ratios;
    candidate.translation_condition_ratios =
      diagnostics.aggregate_schur.translation_condition_ratios;
    candidate.rotation_alignment_confidence =
      diagnostics.aggregate_schur.rotation_alignment_confidence;
    candidate.translation_alignment_confidence =
      diagnostics.aggregate_schur.translation_alignment_confidence;
    candidate.rotation_cluster_flags =
      diagnostics.aggregate_schur.rotation_spectral_cluster_flags;
    candidate.translation_cluster_flags =
      diagnostics.aggregate_schur.translation_spectral_cluster_flags;
    candidate.absolute_rotation_degenerate_mask =
      diagnostics.aggregate_schur.absolute_rotation_degenerate_mask;
    candidate.absolute_translation_degenerate_mask =
      diagnostics.aggregate_schur.absolute_translation_degenerate_mask;
    candidate.axis_alignment_valid =
      (candidate.rotation_alignment_confidence.array() >=
       config.detection.minimum_axis_alignment_confidence).all() &&
      (candidate.translation_alignment_confidence.array() >=
       config.detection.minimum_axis_alignment_confidence).all();
    return candidate;
  }

  void computeCurrentHealth(
    ScanDcregDiagnostics* diagnostics) {
    diagnostics->rotation_reference_ratios =
      reference_manager.rotationReferenceRatios();
    diagnostics->translation_reference_ratios =
      reference_manager.translationReferenceRatios();
    if (!finitePositiveVector(
          diagnostics->rotation_reference_ratios) ||
        !finitePositiveVector(
          diagnostics->translation_reference_ratios)) {
      diagnostics->health_available = false;
      return;
    }

    Eigen::Vector3d rotation_log_deterioration =
      (diagnostics->aggregate_schur.rotation_condition_ratios.array()
         .log() -
       diagnostics->rotation_reference_ratios.array().log())
        .max(0.0)
        .matrix();
    Eigen::Vector3d translation_log_deterioration =
      (diagnostics->aggregate_schur.translation_condition_ratios
         .array().log() -
       diagnostics->translation_reference_ratios.array().log())
        .max(0.0)
        .matrix();
    diagnostics->rotation_health_raw =
      (-rotation_log_deterioration.array()).exp().matrix();
    diagnostics->translation_health_raw =
      (-translation_log_deterioration.array()).exp().matrix();
    updateTemporalModes(
      rotation_log_deterioration,
      config.temporal,
      &rotation_temporal,
      &diagnostics->rotation_health_smoothed);
    updateTemporalModes(
      translation_log_deterioration,
      config.temporal,
      &translation_temporal,
      &diagnostics->translation_health_smoothed);
    diagnostics->relative_rotation_degraded_mask =
      rotation_temporal.degraded;
    diagnostics->relative_translation_degraded_mask =
      translation_temporal.degraded;
    diagnostics->rotation_bad_frame_counters =
      rotation_temporal.bad_counters;
    diagnostics->translation_bad_frame_counters =
      translation_temporal.bad_counters;
    diagnostics->rotation_good_frame_counters =
      rotation_temporal.good_counters;
    diagnostics->translation_good_frame_counters =
      translation_temporal.good_counters;
    diagnostics->health_available = true;
  }

  void applyInvalidTemporalPolicy(
    ScanDcregDiagnostics* diagnostics) {
    if (!reference_manager.ready()) {
      return;
    }
    diagnostics->rotation_reference_ratios =
      reference_manager.rotationReferenceRatios();
    diagnostics->translation_reference_ratios =
      reference_manager.translationReferenceRatios();
    diagnostics->rotation_health_smoothed =
      invalidTemporalHealth(config.temporal, &rotation_temporal);
    diagnostics->translation_health_smoothed =
      invalidTemporalHealth(config.temporal, &translation_temporal);
    diagnostics->relative_rotation_degraded_mask =
      rotation_temporal.degraded;
    diagnostics->relative_translation_degraded_mask =
      translation_temporal.degraded;
    diagnostics->rotation_bad_frame_counters =
      rotation_temporal.bad_counters;
    diagnostics->translation_bad_frame_counters =
      translation_temporal.bad_counters;
    diagnostics->rotation_good_frame_counters =
      rotation_temporal.good_counters;
    diagnostics->translation_good_frame_counters =
      translation_temporal.good_counters;
    diagnostics->health_available = false;
  }

  ScanDcregDiagnostics process(
    const ScanDcregMonitorInput& input) {
    ScanDcregDiagnostics diagnostics;
    if (config.mode == ScanDcregMode::Off) {
      return diagnostics;
    }

    const auto start = std::chrono::steady_clock::now();
    ++evaluation_count;
    diagnostics.enabled = true;
    diagnostics.stamp_sec = input.stamp_sec;
    diagnostics.frame_index = input.frame_index;
    diagnostics.T_world_imu = input.T_world_imu;
    diagnostics.registration_converged =
      input.registration_converged;
    diagnostics.linear_solve_success =
      input.linear_solve_success;
    diagnostics.source_point_count = input.source_point_count;
    diagnostics.initial_cost = input.initial_cost;
    diagnostics.final_cost = input.final_cost;
    diagnostics.cost_change =
      input.initial_cost - input.final_cost;
    diagnostics.reference_source = reference_manager.activeSource();
    diagnostics.offline_profile_status =
      reference_manager.profileStatus();

    Matrix6d aggregate = Matrix6d::Zero();
    bool aggregate_capture_valid = !input.factors.empty();
    diagnostics.minimum_factor_inlier_count =
      input.factors.empty()
        ? 0
        : std::numeric_limits<int>::max();
    diagnostics.minimum_factor_inlier_fraction =
      input.factors.empty()
        ? 0.0
        : std::numeric_limits<double>::infinity();

    if (previous_factor_rotation_bases.size() <
        input.factors.size()) {
      previous_factor_rotation_bases.resize(
        input.factors.size(), Eigen::Matrix3d::Identity());
      previous_factor_translation_bases.resize(
        input.factors.size(), Eigen::Matrix3d::Identity());
      have_previous_factor_basis.resize(
        input.factors.size(), false);
    }

    diagnostics.per_factor.reserve(input.factors.size());
    for (std::size_t index = 0u;
         index < input.factors.size();
         ++index) {
      const ScanDcregFactorInput& factor_input =
        input.factors[index];
      ScanDcregFactorDiagnostics factor;
      factor.factor_index = factor_input.factor_index;
      factor.resolution = factor_input.resolution;
      factor.source_point_count =
        factor_input.source_point_count;
      factor.inlier_count = factor_input.inlier_count;
      factor.inlier_fraction = factor_input.inlier_fraction;
      factor.initial_cost = factor_input.initial_cost;
      factor.final_cost = factor_input.final_cost;
      factor.cost_change =
        factor_input.initial_cost - factor_input.final_cost;
      factor.invalid_reason = factor_input.invalid_reason;
      diagnostics.minimum_factor_inlier_count =
        std::min(
          diagnostics.minimum_factor_inlier_count,
          factor.inlier_count);
      diagnostics.minimum_factor_inlier_fraction =
        std::min(
          diagnostics.minimum_factor_inlier_fraction,
          factor.inlier_fraction);

      if (!factor_input.hessian_valid ||
          !factor_input.hessian.allFinite()) {
        aggregate_capture_valid = false;
        if (factor.invalid_reason.empty()) {
          factor.invalid_reason = "factor_hessian_unavailable";
        }
        factor.schur.invalid_reason = factor.invalid_reason;
        diagnostics.per_factor.push_back(std::move(factor));
        continue;
      }

      const Matrix6d symmetric_factor =
        0.5 * (factor_input.hessian +
               factor_input.hessian.transpose());
      factor.hessian = computeScanDcregHessianMetrics(
        symmetric_factor,
        factor.source_point_count,
        factor.inlier_count,
        config.detection);
      const Eigen::Matrix3d* factor_rotation_target =
        have_previous_factor_basis[index]
          ? &previous_factor_rotation_bases[index]
          : nullptr;
      const Eigen::Matrix3d* factor_translation_target =
        have_previous_factor_basis[index]
          ? &previous_factor_translation_bases[index]
          : nullptr;
      factor.schur = analyzeScanDcregHessian(
        symmetric_factor,
        config.detection,
        factor_rotation_target,
        factor_translation_target);
      if (factor.schur.valid) {
        previous_factor_rotation_bases[index] =
          factor.schur.aligned_rotation_basis;
        previous_factor_translation_bases[index] =
          factor.schur.aligned_translation_basis;
        have_previous_factor_basis[index] = true;
      }
      aggregate += symmetric_factor;
      diagnostics.per_factor.push_back(std::move(factor));
    }

    if (!aggregate_capture_valid) {
      diagnostics.invalid_reason =
        input.factors.empty()
          ? "no_matching_factors"
          : "one_or_more_factor_hessians_unavailable";
      diagnostics.aggregate_schur.invalid_reason =
        diagnostics.invalid_reason;
    } else {
      diagnostics.aggregate_hessian =
        computeScanDcregHessianMetrics(
          aggregate,
          diagnostics.source_point_count,
          -1,
          config.detection);
      const Eigen::Matrix3d* rotation_target =
        have_previous_aggregate_basis
          ? &previous_rotation_basis
          : nullptr;
      const Eigen::Matrix3d* translation_target =
        have_previous_aggregate_basis
          ? &previous_translation_basis
          : nullptr;
      diagnostics.aggregate_schur =
        analyzeScanDcregHessian(
          aggregate,
          config.detection,
          rotation_target,
          translation_target);
      diagnostics.factorization_ok =
        diagnostics.aggregate_schur.factorization_ok;
      diagnostics.valid =
        diagnostics.aggregate_schur.valid &&
        diagnostics.aggregate_hessian.valid;
      diagnostics.invalid_reason =
        diagnostics.valid
          ? std::string()
          : diagnostics.aggregate_schur.invalid_reason;
      if (diagnostics.aggregate_schur.valid) {
        previous_rotation_basis =
          diagnostics.aggregate_schur.aligned_rotation_basis;
        previous_translation_basis =
          diagnostics.aggregate_schur.aligned_translation_basis;
        have_previous_aggregate_basis = true;
      }
    }

    const ScanDcregReferenceCandidate candidate =
      referenceCandidate(input, diagnostics);
    const bool reference_was_ready =
      reference_manager.ready();
    if (reference_was_ready && diagnostics.valid) {
      computeCurrentHealth(&diagnostics);
    } else if (!diagnostics.valid) {
      applyInvalidTemporalPolicy(&diagnostics);
    }

    const ScanDcregReferenceUpdate reference_update =
      reference_manager.update(
        candidate,
        diagnostics.rotation_health_raw,
        diagnostics.translation_health_raw,
        diagnostics.relative_rotation_degraded_mask,
        diagnostics.relative_translation_degraded_mask);
    diagnostics.reference_ready =
      reference_update.reference_ready;
    diagnostics.reference_source = reference_manager.activeSource();
    diagnostics.offline_profile_status =
      reference_manager.profileStatus();
    diagnostics.baseline_update_accepted =
      reference_update.accepted;
    diagnostics.baseline_update_reason =
      reference_update.reason;
    if (reference_update.initialized_now &&
        diagnostics.valid) {
      computeCurrentHealth(&diagnostics);
    } else if (!reference_was_ready &&
               !reference_update.initialized_now) {
      diagnostics.health_available = false;
    }

    const auto finish = std::chrono::steady_clock::now();
    diagnostics.evaluation_time_ms =
      std::chrono::duration<double, std::milli>(
        finish - start).count();
    const std::uint64_t evaluation_time_ns =
      static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
          finish - start).count());
    total_evaluation_time_ns += evaluation_time_ns;
    maximum_evaluation_time_ns =
      std::max(maximum_evaluation_time_ns, evaluation_time_ns);

    if (logger && logger->enabled() &&
        diagnostics.frame_index %
            static_cast<std::size_t>(
              config.logging.log_every_n_frames) ==
          0u) {
      logger->enqueue(diagnostics);
    }
    return diagnostics;
  }
};

ScanDcregHealthMonitor::ScanDcregHealthMonitor(
  const ScanDcregHealthConfig& config,
  const ScanDcregRuntimeMetadata& runtime_metadata,
  const ScanDcregTestVariant test_variant)
: impl(new Impl(config, runtime_metadata, test_variant)) {}

ScanDcregHealthMonitor::~ScanDcregHealthMonitor() {
  const ScanDcregExecutionStats stats = executionStats();
  spdlog::info(
    "GLIM_DCREG_MONITOR_STATS_ROW,{},{},{:.9f},{:.9f}",
    scanDcregTestVariantName(impl->test_variant),
    stats.evaluation_count,
    stats.total_evaluation_time_ms,
    stats.maximum_evaluation_time_ms);
  spdlog::info(
    "GLIM_DCREG_PRODUCER_STATS_ROW,{},{:.9f},{:.9f},{:.9f},{:.9f}",
    stats.snapshot_count,
    stats.total_initial_cost_time_ms,
    stats.maximum_initial_cost_time_ms,
    stats.total_snapshot_time_ms,
    stats.maximum_snapshot_time_ms);
}

bool ScanDcregHealthMonitor::enabled() const {
  return impl->config.mode == ScanDcregMode::LogOnly;
}

ScanDcregDiagnostics ScanDcregHealthMonitor::process(
  const ScanDcregMonitorInput& input) {
  return impl->process(input);
}

std::size_t ScanDcregHealthMonitor::evaluationCount() const {
  return impl->evaluation_count;
}

std::size_t ScanDcregHealthMonitor::formattedLogRowCount() const {
  return impl->logger
    ? impl->logger->formattedRowCount()
    : 0u;
}

std::size_t ScanDcregHealthMonitor::droppedLogSampleCount() const {
  return impl->logger
    ? impl->logger->droppedSampleCount()
    : 0u;
}

ScanDcregLoggingStats ScanDcregHealthMonitor::loggingStats() const {
  return impl->logger
    ? impl->logger->stats()
    : ScanDcregLoggingStats();
}

ScanDcregExecutionStats ScanDcregHealthMonitor::executionStats() const {
  ScanDcregExecutionStats result;
  result.evaluation_count = impl->evaluation_count;
  result.total_evaluation_time_ms =
    static_cast<double>(impl->total_evaluation_time_ns) / 1.0e6;
  result.maximum_evaluation_time_ms =
    static_cast<double>(impl->maximum_evaluation_time_ns) / 1.0e6;
  result.snapshot_count = impl->snapshot_count;
  result.total_initial_cost_time_ms =
    impl->total_initial_cost_time_ms;
  result.maximum_initial_cost_time_ms =
    impl->maximum_initial_cost_time_ms;
  result.total_snapshot_time_ms = impl->total_snapshot_time_ms;
  result.maximum_snapshot_time_ms =
    impl->maximum_snapshot_time_ms;
  return result;
}

void ScanDcregHealthMonitor::recordProducerTiming(
  const double initial_cost_time_ms,
  const double snapshot_time_ms) {
  ++impl->snapshot_count;
  impl->total_initial_cost_time_ms += initial_cost_time_ms;
  impl->maximum_initial_cost_time_ms =
    std::max(
      impl->maximum_initial_cost_time_ms,
      initial_cost_time_ms);
  impl->total_snapshot_time_ms += snapshot_time_ms;
  impl->maximum_snapshot_time_ms =
    std::max(
      impl->maximum_snapshot_time_ms,
      snapshot_time_ms);
}

bool validateScanDcregHealthConfig(
  const ScanDcregHealthConfig& config,
  std::string* error) {
  const auto fail = [&](const std::string& reason) {
    if (error != nullptr) {
      *error = reason;
    }
    return false;
  };
  if (config.mode != ScanDcregMode::Off &&
      config.mode != ScanDcregMode::LogOnly) {
    return fail("unsupported Stage 1 mode");
  }
  if (!(config.detection.degeneracy_condition_threshold > 1.0) ||
      !(config.detection.epsilon_absolute > 0.0) ||
      !(config.detection.epsilon_relative >= 0.0) ||
      !(config.detection.pseudoinverse_relative_threshold > 0.0) ||
      !(config.detection.negative_eigenvalue_tolerance >= 0.0) ||
      !(config.detection.spectral_cluster_relative_gap >= 0.0) ||
      !(config.detection.spectral_cluster_relative_gap < 1.0) ||
      !(config.detection.minimum_axis_alignment_confidence >= 0.0) ||
      !(config.detection.minimum_axis_alignment_confidence <= 1.0)) {
    return fail("invalid detection configuration");
  }
  if (config.reference.bootstrap_minimum_samples <= 0 ||
      config.reference.bootstrap_window_size <= 0 ||
      !(config.reference
          .bootstrap_max_rotation_condition_ratio > 1.0) ||
      !(config.reference
          .bootstrap_max_translation_condition_ratio > 1.0) ||
      !(config.reference.temporal_stability_max_log_ratio_range >=
        0.0) ||
      !(config.reference.adaptation_rate >= 0.0) ||
      !(config.reference.adaptation_rate <= 1.0) ||
      !(config.reference.maximum_reference_change_ratio > 1.0) ||
      !(config.reference.offline_nominal_mad_multiplier > 0.0) ||
      !(config.reference.offline_nominal_min_log_half_width >= 0.0) ||
      config.reference.minimum_source_point_count < 0 ||
      config.reference.minimum_inlier_count < 0 ||
      !(config.reference.minimum_inlier_fraction >= 0.0) ||
      !(config.reference.minimum_inlier_fraction <= 1.0)) {
    return fail("invalid reference configuration");
  }
  if (config.reference.source ==
        ScanDcregReferenceSource::Offline &&
      config.reference.offline_profile_path.empty()) {
    return fail("offline reference requires offline_profile_path");
  }
  if (!(config.temporal.smoothing_alpha > 0.0) ||
      !(config.temporal.smoothing_alpha <= 1.0) ||
      !(config.temporal.health_enter_threshold > 0.0) ||
      !(config.temporal.health_enter_threshold < 1.0) ||
      !(config.temporal.health_exit_threshold > 0.0) ||
      !(config.temporal.health_exit_threshold <= 1.0) ||
      !(config.temporal.health_exit_threshold >
        config.temporal.health_enter_threshold) ||
      config.temporal.bad_frames_required <= 0 ||
      config.temporal.good_frames_required <= 0) {
    return fail("invalid temporal configuration");
  }
  if (config.combine_support_with_shape_health) {
    return fail(
      "combine_support_with_shape_health is not supported in Stage 1");
  }
  if (config.logging.log_every_n_frames <= 0 ||
      config.logging.queue_capacity == 0u ||
      config.logging.flush_every_n_rows <= 0) {
    return fail("invalid logging configuration");
  }
  if (config.logging.enabled &&
      !config.logging.asynchronous) {
    return fail("Stage 1 structured logging must be asynchronous");
  }
  if (config.mode == ScanDcregMode::LogOnly &&
      config.logging.enabled &&
      config.logging.csv_path.empty()) {
    return fail("enabled structured logging requires csv_path");
  }
  if (error != nullptr) {
    error->clear();
  }
  return true;
}

bool scanDcregModeCreatesMonitor(const ScanDcregMode mode) {
  return mode == ScanDcregMode::LogOnly;
}

const char* scanDcregModeName(const ScanDcregMode mode) {
  switch (mode) {
    case ScanDcregMode::Off:
      return "off";
    case ScanDcregMode::LogOnly:
      return "log_only";
  }
  return "unknown";
}

const char* scanDcregTestVariantName(
  const ScanDcregTestVariant variant) {
  switch (variant) {
    case ScanDcregTestVariant::FullLogOnly:
      return "full_log_only";
    case ScanDcregTestVariant::ComputeOnly:
      return "compute_only";
    case ScanDcregTestVariant::EnqueueOnly:
      return "enqueue_only";
  }
  return "unknown";
}

const char* scanDcregReferenceSourceName(
  const ScanDcregReferenceSource source) {
  switch (source) {
    case ScanDcregReferenceSource::Session:
      return "session";
    case ScanDcregReferenceSource::Offline:
      return "offline";
    case ScanDcregReferenceSource::Hybrid:
      return "hybrid";
  }
  return "unknown";
}

const char* scanDcregInvalidFramePolicyName(
  const ScanDcregInvalidFramePolicy policy) {
  switch (policy) {
    case ScanDcregInvalidFramePolicy::Hold:
      return "hold";
    case ScanDcregInvalidFramePolicy::DecayTowardUnknown:
      return "decay_toward_unknown";
    case ScanDcregInvalidFramePolicy::MarkUnavailable:
      return "mark_unavailable";
  }
  return "unknown";
}

const char* scanDcregStatusName(const ScanDcregStatus status) {
  switch (status) {
    case ScanDcregStatus::Valid:
      return "valid";
    case ScanDcregStatus::NoLidarSupport:
      return "no_lidar_support";
    case ScanDcregStatus::IndefiniteRotationSchur:
      return "indefinite_rotation_schur";
    case ScanDcregStatus::IndefiniteTranslationSchur:
      return "indefinite_translation_schur";
    case ScanDcregStatus::IndefiniteBothSchur:
      return "indefinite_both_schur";
    case ScanDcregStatus::NonFiniteHessian:
      return "nonfinite_hessian";
    case ScanDcregStatus::InvalidHessianDimension:
      return "invalid_hessian_dimension";
    case ScanDcregStatus::BlockEigendecompositionFailed:
      return "block_eigendecomposition_failed";
    case ScanDcregStatus::SchurEigendecompositionFailed:
      return "schur_eigendecomposition_failed";
  }
  return "unknown";
}

bool parseScanDcregMode(const std::string& value,
                        ScanDcregMode* mode) {
  if (value == "off") {
    *mode = ScanDcregMode::Off;
    return true;
  }
  if (value == "log_only") {
    *mode = ScanDcregMode::LogOnly;
    return true;
  }
  return false;
}

bool parseScanDcregTestVariant(
  const std::string& value,
  ScanDcregTestVariant* variant) {
  if (value == "full_log_only") {
    *variant = ScanDcregTestVariant::FullLogOnly;
    return true;
  }
  if (value == "compute_only") {
    *variant = ScanDcregTestVariant::ComputeOnly;
    return true;
  }
  if (value == "enqueue_only") {
    *variant = ScanDcregTestVariant::EnqueueOnly;
    return true;
  }
  return false;
}

bool parseScanDcregReferenceSource(
  const std::string& value,
  ScanDcregReferenceSource* source) {
  if (value == "session") {
    *source = ScanDcregReferenceSource::Session;
    return true;
  }
  if (value == "offline") {
    *source = ScanDcregReferenceSource::Offline;
    return true;
  }
  if (value == "hybrid") {
    *source = ScanDcregReferenceSource::Hybrid;
    return true;
  }
  return false;
}

bool parseScanDcregInvalidFramePolicy(
  const std::string& value,
  ScanDcregInvalidFramePolicy* policy) {
  if (value == "hold") {
    *policy = ScanDcregInvalidFramePolicy::Hold;
    return true;
  }
  if (value == "decay_toward_unknown") {
    *policy =
      ScanDcregInvalidFramePolicy::DecayTowardUnknown;
    return true;
  }
  if (value == "mark_unavailable") {
    *policy = ScanDcregInvalidFramePolicy::MarkUnavailable;
    return true;
  }
  return false;
}

}  // namespace glim
