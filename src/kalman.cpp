#include <Eigen/Dense>

#include "kalman_filter/kalman.hpp"

Kalman::Kalman(
		const Eigen::MatrixXd& A,
		const Eigen::MatrixXd& B,
		const Eigen::MatrixXd& C,
		const Eigen::MatrixXd& P,
		const Eigen::MatrixXd& Q,
		const Eigen::MatrixXd& R)
	: A_(A), B_(B), C_(C), P_(P), Q_(Q), R_(R),
		m_(C.rows()), n_(A.rows()), c_(B.cols()),
		I_(n_, n_), x_est_(n_), x_pred_(n_) {
	I_.setIdentity();
	x_est_.setZero();
	x_pred_ = x_est_;
}

void Kalman::SetInitialState(const Eigen::VectorXd& x0) {
	x_est_ = x0;
}

void Kalman::Predict(const Eigen::VectorXd& u) {
	// Predict the state based on the system model and the control input
	x_pred_ = A_ * x_est_ + B_ * u;

	// Update the error covariance matrix
	P_ = A_ * P_ * A_.transpose() + Q_;
}

void Kalman::Update(const Eigen::VectorXd& y) {
	// Calculate the Kalman gain
	K_ = P_ * C_.transpose() * (C_ * P_ * C_.transpose() + R_).inverse();

	// Estimate the state and the the correspondeding error covariance matrix
	// based on the newly calculated Kalman gain
	x_est_ = x_pred_ + K_ * (y - C_ * x_pred_);
	P_ = (I_ - K_ * C_) * P_;
}
