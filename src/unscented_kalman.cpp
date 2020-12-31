#include <algorithm>
#include <numeric>

#include <Eigen/Dense>

#include "bayes_filter/unscented_kalman.hpp"

UnscentedKalman::UnscentedKalman(
		const Eigen::MatrixXd& P,
		const Eigen::MatrixXd& Q,
		const Eigen::MatrixXd& R,
    std::function<Eigen::VectorXd(const Eigen::VectorXd& q, const Eigen::VectorXd& u)> processFunction,
    std::function<Eigen::VectorXd(const Eigen::VectorXd& r)> outputFunction,
    double alpha, double beta, double kappa
  ) :
		P_(P), Q_(Q), R_(R),
		processFunction(processFunction), outputFunction(outputFunction),
		m_(R.rows()), n_(P.rows()),
		I_(n_, n_), x_est_(n_), x_pred_(n_),
		k_(2 * n_ + 1),
		alpha_(alpha), beta_(beta), kappa_(kappa),
		mean_weights_(k_, 0.0),
		covariance_weights_(k_, 0.0),
		sigma_points_(k_, Eigen::VectorXd(n_).setZero()),
		pred_sigma_points_(k_, Eigen::VectorXd(n_).setZero()),
		output_sigma_points_(k_, Eigen::VectorXd(n_).setZero()) {

	I_.setIdentity(n_,n_);
	x_est_.setZero(n_);

  lambda_ = pow(alpha_, 2.0) * (n_ + kappa_) - n_;
  mean_weights_[0] = lambda_ / (n_ + lambda_);
  covariance_weights_[0] = lambda_ / (n_ + lambda_) + 1.0 - pow(alpha_, 2.0) + beta_;

  for (size_t i = 1; i < k_; ++i) {
    mean_weights_[i] = covariance_weights_[i] = 0.5 / (n_ + lambda_);
  }
}

void UnscentedKalman::SetInitialState(const Eigen::VectorXd& x0) {
	x_est_ = x0;
}

void UnscentedKalman::Predict(const Eigen::VectorXd& u) {

	// Calculate the sigma points
	// Use Cholesky decomposition to get the lower left triangular matrix
	Eigen::MatrixXd sqrt_P(Eigen::LLT<Eigen::MatrixXd>((n_ + lambda_) * P_).matrixL());
	sigma_points_[0] << x_est_[0], x_est_[1];
  for (size_t i = 1; i < n_ + 1; ++i) {
    sigma_points_[i] = x_est_ + sqrt_P.col(i - 1);
    sigma_points_[i + n_] = x_est_ - sqrt_P.col(i - 1);
  }

	// Predict the new sigma points after passing them through the non linear state dynamics function
  for (size_t i = 0; i < k_; ++i) {
		pred_sigma_points_[i] = processFunction(sigma_points_[i], u);
	}

	// Estimate the state using the predicted sigma points (weighted mean)
  x_pred_.setZero(n_);
  for (size_t i = 0; i < k_; ++i) {
		x_pred_ += mean_weights_[i] * pred_sigma_points_[i];
	}

	// Calculate the eror covariance of the predicted points
  P_ = Q_;
  for (size_t i = 0; i < k_; ++i) {
		P_ += covariance_weights_[i] * (pred_sigma_points_[i] - x_pred_) * (pred_sigma_points_[i] - x_pred_).transpose();
	}
}

void UnscentedKalman::Update(const Eigen::VectorXd& y) {

	// Calculate the values of the sigma points after they have been passed through the output function
	// Convert the predicted sigma points to measurement space
  for (size_t i = 0; i < k_; ++i) {
		output_sigma_points_[i] = outputFunction(pred_sigma_points_[i]);
	}

	// Calculate mean outputthe mean of the sigma points after they have been passed through the output function
  Eigen::VectorXd y_pred(m_);
  y_pred.setZero(m_);
  for (size_t i = 0; i < k_; ++i) {
		y_pred += mean_weights_[i] * output_sigma_points_[i];
	}

	// Update the innovation matrix
  auto S_ = R_;
  for (size_t i = 0; i < k_; ++i) {
		S_ += covariance_weights_[i] * (output_sigma_points_[i] - y_pred) * (output_sigma_points_[i] - y_pred).transpose();
	}

	// Calculate the cross covariance
  Eigen::MatrixXd P_qy(n_, m_);
  P_qy.setZero(n_, m_);
  for (size_t i = 0; i < k_; ++i) {
		P_qy += covariance_weights_[i] * (pred_sigma_points_[i] - x_pred_) * (output_sigma_points_[i] - y_pred).transpose();
	}

	// Calculate the Kalman gain
  K_ = P_qy * S_.inverse();

	// Estimate the state and the correspondeding error covariance matrix
  x_est_ = x_pred_ + K_ * (y - y_pred);
  P_ = P_ - K_ * S_ * K_.transpose();
}
