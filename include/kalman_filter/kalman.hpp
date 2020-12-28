#ifndef KALMAN_HPP
#define KALMAN_HPP

#include <Eigen/Dense>

#include "kalman_filter/bayes_filter.hpp"

class Kalman : public BayesFilter {
public:
	Kalman(
		const Eigen::MatrixXd& A,
		const Eigen::MatrixXd& B,
		const Eigen::MatrixXd& C,
    const Eigen::MatrixXd& P,
		const Eigen::MatrixXd& Q,
		const Eigen::MatrixXd& R
	);

  void SetInitialState(const Eigen::VectorXd& x0) override;
  void Predict(const Eigen::VectorXd& u) override;
  void Update(const Eigen::VectorXd& y) override;
	Eigen::VectorXd GetState() const override { return x_est_; };

private:
  Eigen::MatrixXd A_; /* State Transition model matrix */
  Eigen::MatrixXd B_; /* Input control matrix */
  Eigen::MatrixXd C_; /* Observation model matrix */
  Eigen::MatrixXd Q_; /* Process noise covariance */
  Eigen::MatrixXd R_; /* Measurement noise covariance */
  Eigen::MatrixXd P_; /* Estimate error covariance */

  Eigen::MatrixXd K_; /* Kalman Gain */
  Eigen::VectorXd x_est_, x_pred_; /* State estimate and predicted state */

  int n_; /* Number of states */
  int m_; /* Number of measurements */
  int c_; /* Number of control inputs */

  Eigen::MatrixXd I_; /* Identity matrix */
};

#endif  // KALMAN_HPP
