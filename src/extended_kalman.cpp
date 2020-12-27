#include <functional>

#include <Eigen/Dense>

#include "kalman_filter/extended_kalman.hpp"

ExtendedKalman::ExtendedKalman(
		const Eigen::MatrixXd& P,
		const Eigen::MatrixXd& Q,
		const Eigen::MatrixXd& R,
    std::function<Eigen::VectorXd(const Eigen::VectorXd& q, const Eigen::VectorXd& u)> processFunction,
    std::function<Eigen::VectorXd(const Eigen::VectorXd& q)> outputFunction,
    std::function<Eigen::MatrixXd(const Eigen::VectorXd& q, const Eigen::VectorXd& u)> processJacobian,
    std::function<Eigen::MatrixXd(const Eigen::VectorXd& q)> outputJacobian
  )
	: P_(P), Q_(Q), R_(R),
    processFunction(processFunction),
    outputFunction(outputFunction),
    processJacobian(processJacobian),
    outputJacobian(outputJacobian),
		m_(R.rows()), n_(P.rows()),
		I_(n_, n_), x_est_(n_), x_pred_(n_) {
	x_est_.setZero();
	I_.setIdentity();
}

void ExtendedKalman::SetInitialState(const Eigen::VectorXd& x0) {
	x_est_ = x0;
}

void ExtendedKalman::Predict(const Eigen::VectorXd& u) {
	// Calculate the process jacobian
  A_ = processJacobian(x_est_, u);

	// Predict the state based on the process function and the control input
  x_pred_ = processFunction(x_est_, u);

	// Update the error covariance matrix
  P_ = A_ * P_ * A_.transpose() + Q_;
}

void ExtendedKalman::Update(const Eigen::VectorXd& y) {
	// Calculate the output jacobian
  C_ = outputJacobian(x_est_);

	// Calculate the kalman gain
  K_ = P_ * C_.transpose() * (C_ * P_ * C_.transpose() + R_).inverse();

	// Estimate the state and the correspondeding error covariance matrix
  x_est_ = x_pred_ + K_ * (y - outputFunction(x_pred_));
  P_ = (I_ - K_ * C_) * P_;
}
