// This file simualtes a spring mass damper system where
// m * x_dot_dot + b * x_dot + k * x + m * u = 0;
// u being the external acceleration provided by an external accuator
// The sensor used for measurement provides the measured x position
// Which is why the output function is just retruns the input
// estimated_pos is the estimated position by the unscented kalman filter
// measured_pos is the  x position measued by the sensor
// true_pos is the true x position
// The state vector q is [x, x_dot]

#include <array>
#include <fstream>
#include <random>

#include <Eigen/Dense>

#include "bayes_filter/bayes_filter.hpp"
#include "bayes_filter/extended_kalman.hpp"

// Log results to a csv file
constexpr bool logging = true;

// Simulation constants
constexpr double system_dt = 0.001;
constexpr double measurement_dt = 0.01;
constexpr double simulation_time = 25.0;

constexpr int N = static_cast<int>(simulation_time / system_dt);
constexpr int M = static_cast<int>(measurement_dt / system_dt);

// Process noise and measurement noise generators
constexpr double measurement_mu = 0.0;
constexpr double measurement_sigma = 0.5;

constexpr double process_mu = 0.0;
constexpr double process_sigma = 0.5;

// System variables and dimenstions
constexpr int n = 2;  // Number of state variables
constexpr int m = 1;  // Number of measurement variables

constexpr double initial_position = 1.0;
constexpr double initial_velocity = 0.0;
constexpr double initial_acceleration = 0.0;
constexpr double system_center = 2.0;

// Process constants
constexpr double mass = 1.0;  // Mass in kg
constexpr double b = 0.25;  // Friction coef. in 1/s
constexpr double k = 1.0;  // spring constant in 1/s2

// Simulate system dynamics
auto processFunction = [] (const Eigen::VectorXd& q, const Eigen::VectorXd& u) {
  return (Eigen::VectorXd(2) <<
            q(0) + q(1) * system_dt,
            q(1) * (1 - (b / mass) * system_dt) + (u(0) - k * (q(0) - system_center) / mass) * system_dt
         ).finished();
};

auto processJacobian = [] (const Eigen::VectorXd& q, const Eigen::VectorXd& u) {
  return (Eigen::MatrixXd(2, 2) <<
    1, system_dt,
    -k / mass * system_dt, (1 - (b / mass) * system_dt)
   ).finished();
};

// Convert from state space to measurement space
auto outputFunction = [] (const Eigen::VectorXd& q) {
  return (Eigen::VectorXd(1) << q(0)).finished();
};

auto outputJacobian = [] (const Eigen::VectorXd& q) {
  return (Eigen::MatrixXd(1, 2) << 1, 0).finished();
};

// Input u given to system, can be modified to be a feedback controller
auto controlInputFunction = [] () {
  return 1.0;
};

Eigen::MatrixXd P(n, n);  // Estimate error covariance
Eigen::MatrixXd Q(n, n);  // Process noise covariance
Eigen::MatrixXd R(m, m);  // Measurement noise covariance

int main() {

  // Initialize system variables
  std::default_random_engine generator;
  std::normal_distribution<double> measurement_noise(measurement_mu, measurement_sigma);
  std::normal_distribution<double> process_noise(process_mu, process_sigma);

  P.setZero(n, n);
  Q << 0.001, 0.0, 0.0, 0.001;
  R << 1.0;

  // Variables to store for comparison/record
  std::array<double, N> time;

  std::array<double, N> true_pos;
  std::array<double, N> true_vel;
  std::array<double, N> true_acc;

  std::array<Eigen::VectorXd, N> measured_pos;
  std::array<double, N> estimated_pos;
  std::array<double, N> believed_acc;

  // Initialize ekf
  ExtendedKalman ekf(P, Q, R, processFunction, outputFunction, processJacobian, outputJacobian);

  time[0] = 0.0;

  true_pos[0] = initial_position;
  true_vel[0] = initial_velocity;
  true_acc[0] = initial_acceleration;

  measured_pos[0].setZero(1);
  measured_pos[0][0] = 0.0;
  estimated_pos[0] = 0.0;
  believed_acc[0] = 0.0;

  std::ofstream file;
  if constexpr (logging) {
    file.open("position.csv");
  }

  // NOTE: Initial state set to zero to test, set to actual value as required
  ekf.SetInitialState((Eigen::VectorXd(2) << 0.0, 0.0).finished());
  // ekf.SetInitialState((Eigen::VectorXd(2) << initial_position, initial_velocity).finished());

  // Simulation
  for (int i = 1; i < N; ++i) {
    time[i] = i * system_dt;

    // The ideal input u at time step i
    believed_acc[i] = controlInputFunction();

    // The actual input u at time step i given actuator noise
    true_acc[i] = believed_acc[i] + process_noise(generator);

    // The process function simulates the system
    Eigen::VectorXd q = processFunction(
      (Eigen::VectorXd(2) << true_pos[i - 1], true_vel[i - 1]).finished(),
      (Eigen::VectorXd(1) << true_acc[i]).finished()
    );

    true_pos[i] = q(0);
    true_vel[i] = q(1);

    // A new measumerment is obtained at very Mth time step
    if(i % M == 0) {
      measured_pos[i] = outputFunction((Eigen::VectorXd(1) << true_pos[i]).finished());
      measured_pos[i](0) += measurement_noise(generator);
    } else {
      measured_pos[i] = measured_pos[i-1];
    }

    // Predict and update
    ekf.Predict((Eigen::VectorXd(1) << believed_acc[i]).finished());
    ekf.Update((Eigen::VectorXd(1) << measured_pos[i][0]).finished());

    // Get the estimated state
    estimated_pos[i] = ekf.GetState()[0];

    if constexpr (logging) {
      file << estimated_pos[i] << ',' << measured_pos[i] << ',' << true_pos[i] << '\n';
    }

  }

  if constexpr (logging) {
    file.close();
  }

  return 0;
}
