// This file simualtes a simple rigid pendulum
// the state q is the [angle, elocity] ie [theta, theta_dot]
// u is the external acceleration provided by an external accuator in the upward direction
// ie moving the entire pendulum at an acceleration u vertically upward
// The sensor used for measurement provides the measured x and y coordinate
// Which is why the output function is just converts the state q into the x and y coordinates
// estimated_ang is the angle estimated by the unscented kalman filter
// measured_xy is the x and y position measued by the sensor
// true_ang is the true angle of the pendulum
// believed_acc is teh acceleration we believe that the actuator provides to the system (u)

#include <iostream>
#include <array>
#include <fstream>
#include <random>

#include <Eigen/Dense>

#include "bayes_filter/bayes_filter.hpp"
#include "bayes_filter/particle.hpp"

// Log results to a csv file
constexpr bool logging = true;

// Simulation constants
constexpr double system_dt = 0.001;
constexpr double measurement_dt = 0.01;
constexpr double simulation_time = 2.5;

constexpr int N = static_cast<int>(simulation_time / system_dt);
constexpr int M = static_cast<int>(measurement_dt / system_dt);

// Process noise and measurement noise generators
constexpr double measurement_mu = 0.0;
constexpr double measurement_sigma = 0.1;

constexpr double process_mu = 0.0;
constexpr double process_sigma = 0.1;

// System variables and dimenstions
constexpr int n = 2;  // Number of state variables
constexpr int m = 2;  // Number of measurement variables

constexpr double initial_position = 3.0;
constexpr double initial_velocity = 0.0;
constexpr double initial_acceleration = 0.0;
constexpr double system_center = 0.0;

// Process constants
constexpr double mass = 1.0;   // Mass in kg
constexpr double g = 9.8;   // Gravitational accel.
constexpr double d = 1.0;   // Length in m
constexpr double b = 0.1;   // Friction coef. in 1/s

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

// Input u given to system, can be modified to be a feedback controller
auto controlInputFunction = [] () {
  return 1.0;
};

Eigen::MatrixXd P(n, n);  // Estimate error covariance
Eigen::MatrixXd Q(n, n);  // Process noise covariance
Eigen::MatrixXd R(m, m);  // Measurement noise covariance

constexpr int n_particles = 1000;

int main() {

  // Initialize system variables
  std::default_random_engine generator;
  std::normal_distribution<double> measurement_noise(measurement_mu, measurement_sigma);
  std::normal_distribution<double> process_noise(process_mu, process_sigma);

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

  Eigen::VectorXd min_states((Eigen::VectorXd(2) << -3, -10).finished());
  Eigen::VectorXd max_states((Eigen::VectorXd(2) << 3, 10).finished());

  // Initialize UKF
  Particle pf(P, Q, R, processFunction, outputFunction, min_states, max_states, n_particles);

  time[0] = 0.0;

  true_ang[0] = initial_position;
  true_vel[0] = initial_velocity;
  true_acc[0] = initial_acceleration;

  // These are unknown to the filter
  measured_pos[0].setZero(2);
  measured_pos[0][0] = 0.0;
  estimated_ang[0] = 0.0;
  believed_acc[0] = 0.0;

  std::ofstream file;
  if constexpr (logging) {
    file.open("position.csv");
  }

  const auto initial_state = (Eigen::VectorXd(2) << 0, 0).finished();

  // Simulation
  for (int i = 1; i < N; ++i) {
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
    const auto u = (Eigen::VectorXd(1) << believed_acc[i]).finished();

    pf.Predict(u);
    pf.Update(measured_pos[i]);

    // Get the estimated state
    estimated_ang[i] = pf.GetState()[0];
    if constexpr (logging) {
      file << estimated_ang[i] << ','  <<
              measured_ang[i] << ',' <<
              true_ang[i] << '\n';
    }
  }

  if constexpr (logging) {
    file.close();
  }

  return 0;
}
