// This file simulates a simple rigid pendulum with all 3 filters for comparison

#include <iostream>
#include <array>
#include <fstream>
#include <random>

#include <Eigen/Dense>

#include "bayes_filter/bayes_filter.hpp"
#include "bayes_filter/extended_kalman.hpp"
#include "bayes_filter/kalman.hpp"
#include "bayes_filter/particle.hpp"
#include "bayes_filter/unscented_kalman.hpp"

// Log results to a csv file
constexpr bool logging = true;

// Simulation constants
constexpr double system_dt = 0.001;
constexpr double measurement_dt = 0.01;
constexpr double simulation_time = 7.50;

constexpr int N = static_cast<int>(simulation_time / system_dt);
constexpr int M = static_cast<int>(measurement_dt / system_dt);

// Process noise and measurement noise generators
constexpr double measurement_mu = 0.0;
constexpr double measurement_sigma = 0.3;

constexpr double process_mu = 0.0;
constexpr double process_sigma = 0.3;

// System variables and dimenstions
constexpr int m = 2;  // Number of measurement variables
constexpr int n = 2;  // Number of state variables
constexpr int c = 1;  // Number of control variables

constexpr double initial_position = 3.0;
constexpr double initial_velocity = 0.0;
constexpr double initial_acceleration = 0.0;
constexpr double system_center = 0.0;
// NOTE: To use system centre another matrix would need to be passded into the
// kalman filter for constants

// Design parameters for unscented kalman filter
constexpr double alpha(0.05);
constexpr double beta(2.0);
constexpr double kappa(0.0);

// Process constants
constexpr double mass = 1.0;   // Mass in kg
constexpr double g = 9.8;   // Gravitational accel.
constexpr double d = 1.0;   // Length in m
constexpr double b = 0.25;   // Friction coef. in 1/s

// Simulate system dynamics
auto processFunction = [] (const Eigen::VectorXd& q, const Eigen::VectorXd& u) {
  return (Eigen::VectorXd(2) <<
            q(0) + q(1) * system_dt,
            q(1) + (mass * (u(0) - g) * d * sin(q(0)) - b * q(1)) * system_dt /
            (mass * d * d)
         ).finished();
};

// Convert from state space to measurement space
auto outputFunction = [] (const Eigen::VectorXd& q) {
  return (Eigen::VectorXd(2) << d * sin(q(0)), d * cos(q(0))).finished();
};

auto processJacobian = [] (const Eigen::VectorXd& q, const Eigen::VectorXd& u) {
  return (Eigen::MatrixXd(2, 2) <<
            1, system_dt,
            (u(0) - g) * cos(q(0)) * system_dt / d, 1.0 - b * system_dt / (m *d *d)
         ).finished();
};

auto outputJacobian = [](const Eigen::VectorXd& q) {
    return  (Eigen::MatrixXd(2, 2) <<
              d * cos(q(0)), 0.0,
             -d * sin(q(0)), 0.0).finished();
};

// Input u given to system, can be modified to be a feedback controller
auto controlInputFunction = [] () {
  return 1.0;
};

Eigen::MatrixXd A(n, n);  // State Transition model matrix
Eigen::MatrixXd B(n, c);  // Input control matrix
Eigen::MatrixXd C(m, n);  // Observation model matrix
Eigen::MatrixXd P(n, n);  // Estimate error covariance
Eigen::MatrixXd Q(n, n);  // Process noise covariance
Eigen::MatrixXd R(m, m);  // Measurement noise covariance

int main() {

  // Initialize system variables
  std::default_random_engine generator;
  std::normal_distribution<double> measurement_noise(measurement_mu, measurement_sigma);
  std::normal_distribution<double> process_noise(process_mu, process_sigma);

  // Set up matrices for kalman filter
  A << 1, system_dt,
       mass * - g * d * system_dt, (1 - (b / mass) * system_dt);
  B << 0, mass * d *  system_dt;
  C << d, 0, 0, 0;

  P.setZero(n, n);
  Q << 0.001, 0.0, 0.0, 0.001;
  R << 1.0, 0.0, 0.0, 1.0;

  // Variables to store for comparison/record
  std::array<double, N> time;

  std::array<double, N> true_ang;
  std::array<double, N> true_vel;
  std::array<double, N> true_acc;

  std::array<Eigen::VectorXd, N> measured_pos;
  std::array<double, N> measured_ang;
  std::array<double, N> estimated_ang;
  std::array<double, N> believed_acc;

  // Set up variables for the partile filter
  constexpr int n_particles = 1000;
  const Eigen::VectorXd min_states((Eigen::VectorXd(2) << -3, -10).finished());
  const Eigen::VectorXd max_states((Eigen::VectorXd(2) << 3, 10).finished());

  // Initialize Filters
  Kalman kf(A, B, C, P, Q, R);
  ExtendedKalman ekf(P, Q, R, processFunction, outputFunction, processJacobian, outputJacobian);
  UnscentedKalman ukf(P, Q, R, processFunction, outputFunction, alpha, beta, kappa);
  Particle pf(P, Q, R, processFunction, outputFunction, min_states, max_states, n_particles);

  time[0] = 0.0;

  true_ang[0] = initial_position;
  true_vel[0] = initial_velocity;
  true_acc[0] = initial_acceleration;

  measured_pos[0].setZero(2);
  measured_pos[0][0] = 0.0;
  estimated_ang[0] = 0.0;
  believed_acc[0] = 0.0;

  std::ofstream file;
  if constexpr (logging) {
    file.open("position.csv");
  }

  // const auto initial_state = (Eigen::VectorXd(2) << 0.0, 0.0).finished();
  const auto initial_state = (Eigen::VectorXd(2) << initial_position, initial_velocity).finished();

  kf.SetInitialState(initial_state);
  ekf.SetInitialState(initial_state);
  ukf.SetInitialState(initial_state);
  pf.SetInitialState(initial_state);
  // NOTE: allowing the particle filter to randomly generate a particles over
  // the entire space rather than a single initial position when the
  // initial position might be wrong is better

  // Simulation
  for (int i = 1; i < N; ++i) {
    std::cout << "Simulation step: " << i << '\n';
    time[i] = i * system_dt;

    // The ideal input u at time step i
    believed_acc[i] = controlInputFunction();

    // The actual input u at time step i given actuator noise
    true_acc[i] = believed_acc[i] + process_noise(generator);

    // The process function simulates the system
    Eigen::VectorXd q = processFunction(
      (Eigen::VectorXd(2) << true_ang[i - 1], true_vel[i - 1]).finished(),
      (Eigen::VectorXd(1) << true_acc[i]).finished()
    );

    true_ang[i] = q(0);
    true_vel[i] = q(1);

    // A new measumerment is obtained at very Mth time step
    if(i % M == 0) {
      measured_pos[i] = outputFunction((Eigen::VectorXd(1) << true_ang[i]).finished());
      measured_pos[i](0) += measurement_noise(generator);
      measured_pos[i](1) += measurement_noise(generator);
      measured_ang[i] = atan2(measured_pos[i](0), measured_pos[i](1));
    } else {
      measured_pos[i] = measured_pos[i-1];
      measured_ang[i] = measured_ang[i-1];
    }

    // Predict and update
    // Input
    const auto u = (Eigen::VectorXd(1) << believed_acc[i]).finished();

    kf.Predict(u);
    kf.Update(measured_pos[i]);

    ekf.Predict(u);
    ekf.Update(measured_pos[i]);

    ukf.Predict(u);
    ukf.Update(measured_pos[i]);

    pf.Predict(u);
    pf.Update(measured_pos[i]);

    // Get the estimated state
    estimated_ang[i] = kf.GetState()[0];
    if constexpr (logging) {
      file << kf.GetState()[0] << ',' <<
              ekf.GetState()[0] << ',' <<
              ukf.GetState()[0] << ','  <<
              pf.GetState()[0] << ','  <<
              measured_ang[i] << ',' <<
              true_ang[i] << '\n';
    }
  }

  if constexpr (logging) {
    file.close();
  }

  return 0;
}
