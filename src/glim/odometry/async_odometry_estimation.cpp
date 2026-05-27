#include <glim/odometry/async_odometry_estimation.hpp>

#include <spdlog/spdlog.h>
#include <glim/util/logging.hpp>
#include <glim/util/timing.hpp>

namespace glim {

AsyncOdometryEstimation::AsyncOdometryEstimation(const std::shared_ptr<OdometryEstimationBase>& odometry_estimation, bool enable_imu)
: odometry_estimation(odometry_estimation),
  logger(create_module_logger("odom")) {
  this->enable_imu = enable_imu;
  kill_switch = false;
  end_of_sequence = false;
  internal_frame_queue_size = 0;
  thread = std::thread([this] { run(); });
}

AsyncOdometryEstimation::~AsyncOdometryEstimation() {
  kill_switch = true;
  join();
}

#ifdef GLIM_USE_OPENCV
void AsyncOdometryEstimation::insert_image(const double stamp, const cv::Mat& image) {
  input_image_queue.push_back(std::make_pair(stamp, image));
}
#endif

void AsyncOdometryEstimation::insert_imu(const double stamp, const Eigen::Vector3d& linear_acc, const Eigen::Vector3d& angular_vel) {
  Eigen::Matrix<double, 7, 1> imu_data;
  imu_data << stamp, linear_acc, angular_vel;
  input_imu_queue.push_back(imu_data);
}

void AsyncOdometryEstimation::insert_frame(const PreprocessedFrame::Ptr& frame) {
  input_frame_queue.push_back(frame);
}

void AsyncOdometryEstimation::join() {
  end_of_sequence = true;
  if (thread.joinable()) {
    thread.join();
  }
}

int AsyncOdometryEstimation::workload() const {
  return input_frame_queue.size() + internal_frame_queue_size;
}

void AsyncOdometryEstimation::get_results(std::vector<EstimationFrame::ConstPtr>& estimation_results, std::vector<EstimationFrame::ConstPtr>& marginalized_frames) {
  estimation_results = output_estimation_results.get_all_and_clear();
  marginalized_frames = output_marginalized_frames.get_all_and_clear();
}

void AsyncOdometryEstimation::run() {
  double last_imu_time = enable_imu ? 0.0 : std::numeric_limits<double>::max();
#ifdef GLIM_USE_OPENCV
  std::deque<std::pair<double, cv::Mat>> images;
#endif
  std::deque<PreprocessedFrame::Ptr> raw_frames;

  while (!kill_switch) {
    auto imu_frames = input_imu_queue.get_all_and_clear();
    auto new_raw_frames = input_frame_queue.get_all_and_clear();
    const size_t imu_batch_count = imu_frames.size();
    const size_t frame_batch_count = new_raw_frames.size();
    raw_frames.insert(raw_frames.end(), new_raw_frames.begin(), new_raw_frames.end());
    internal_frame_queue_size = raw_frames.size();

#ifdef GLIM_USE_OPENCV
    auto new_images = input_image_queue.get_all_and_clear();
    images.insert(images.end(), new_images.begin(), new_images.end());
#endif

    if (
#ifdef GLIM_USE_OPENCV
      images.empty() &&
#endif
      imu_frames.empty() && raw_frames.empty()) {
      if (end_of_sequence) {
        break;
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

    for (const auto& imu : imu_frames) {
      const double stamp = imu[0];
      const Eigen::Vector3d linear_acc = imu.block<3, 1>(1, 0);
      const Eigen::Vector3d angular_vel = imu.block<3, 1>(4, 0);
      odometry_estimation->insert_imu(stamp, linear_acc, angular_vel);

      last_imu_time = stamp;
    }

#ifdef GLIM_USE_OPENCV
    while (!images.empty()) {
      if (!end_of_sequence && images.front().first > last_imu_time) {
        logger->debug("waiting for IMU data (image_time={:.6f}, last_imu_time={:.6f} |images|={})", images.front().first, last_imu_time, images.size());
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        if (images.size() > 10) {
          logger->warn("waiting for IMU data (image_time={:.6f}, last_imu_time={:.6f} |images|={})", images.front().first, last_imu_time, images.size());
        }

        break;
      }

      const auto image = images.front();
      odometry_estimation->insert_image(image.first, image.second);
      images.pop_front();
    }
#endif

    while (!raw_frames.empty()) {
      if (!end_of_sequence && raw_frames.front()->scan_end_time > last_imu_time) {
        logger->debug("waiting for IMU data (scan_end_time={:.6f}, last_imu_time={:.6f} |frames|={})", raw_frames.front()->scan_end_time, last_imu_time, raw_frames.size());
        const auto wait_start = TimingClock::now();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        const double wait_ms = timing_enabled() ? timing_elapsed_ms(wait_start) : 0.0;
        if (timing_enabled()) {
          logger->info(
            "GLIM_ASYNC_ODOM_TIMING_ROW,{:.9f},{:.9f},{:.9f},{},{},{},{:.6f},0,0,{},{},{},waiting_for_imu",
            raw_frames.front()->stamp,
            raw_frames.front()->scan_end_time,
            last_imu_time,
            raw_frames.size(),
            imu_batch_count,
            frame_batch_count,
            wait_ms,
            raw_frames.size(),
            0,
            0);
        }

        if (raw_frames.size() > 10) {
          logger->warn("waiting for IMU data (scan_end_time={:.6f}, last_imu_time={:.6f} |frames|={})", raw_frames.front()->scan_end_time, last_imu_time, raw_frames.size());
        }

        break;
      }

      const auto& frame = raw_frames.front();
      const auto process_start = TimingClock::now();
      std::vector<EstimationFrame::ConstPtr> marginalized;
      auto state = odometry_estimation->insert_frame(frame, marginalized);
      const double odom_insert_ms =
        timing_enabled() ? timing_elapsed_ms(process_start) : 0.0;

      output_estimation_results.push_back(state);
      output_marginalized_frames.insert(marginalized);
      raw_frames.pop_front();
      internal_frame_queue_size = raw_frames.size();
      if (timing_enabled()) {
        logger->info(
          "GLIM_ASYNC_ODOM_TIMING_ROW,{:.9f},{:.9f},{:.9f},{},{},{},0,{:.6f},{:.6f},{},{},{},processed",
          frame->stamp,
          frame->scan_end_time,
          last_imu_time,
          raw_frames.size() + 1u,
          imu_batch_count,
          frame_batch_count,
          odom_insert_ms,
          odom_insert_ms,
          raw_frames.size(),
          state ? 1 : 0,
          marginalized.size());
      }
    }
  }

  auto marginalized = odometry_estimation->get_remaining_frames();
  output_marginalized_frames.insert(marginalized);
}
}  // namespace glim
