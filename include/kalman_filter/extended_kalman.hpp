#ifndef EXTENDED_KALMAN_HPP
#define EXTENDED_KALMAN_HPP

#include <functional>

#include <Eigen/Dense>

#include "kalman_filter/bayes_filter.hpp"

class ExtendedKalman : public BayesFilter {
public:
  ExtendedKalman(
    const Eigen::MatrixXd& P,
    const Eigen::MatrixXd& Q,
    const Eigen::MatrixXd& R,
    std::function<Eigen::VectorXd(const Eigen::VectorXd& q, const Eigen::VectorXd& u)> processFunction,
    std::function<Eigen::VectorXd(const Eigen::VectorXd& q)> outputFunction,
    std::function<Eigen::MatrixXd(const Eigen::VectorXd& q, const Eigen::VectorXd& u)> processJacobian,
    std::function<Eigen::MatrixXd(const Eigen::VectorXd& q)> outputJacobian
  );

  void SetInitialState(const Eigen::VectorXd& x0) override;
  void Predict(const Eigen::VectorXd& u) override;
  void Update(const Eigen::VectorXd& y) override;
  Eigen::VectorXd GetState() const override { return x_est_; };

private:
  /* Calculate the estimated state */
  std::function<Eigen::VectorXd(const Eigen::VectorXd& q, const Eigen::VectorXd& u)> processFunction;

  /* convert the predicted state into measurement space */
  std::function<Eigen::VectorXd(const Eigen::VectorXd& q)> outputFunction;

  /* Jacobian of the process matrix,
     ie the linearization of the process function */
  std::function<Eigen::MatrixXd(const Eigen::VectorXd& q, const Eigen::VectorXd& u)> processJacobian;

  /* Jacobian of the matrix that converts from measurement space into state space,
     ie linearizatrion of the output function */
  std::function<Eigen::MatrixXd(const Eigen::VectorXd& q)> outputJacobian;

  Eigen::MatrixXd A_; /* Jacobian of the process function */
  Eigen::MatrixXd C_; /* Jacobian of the output function */

  Eigen::MatrixXd Q_; /* Process noise covariance */
  Eigen::MatrixXd R_; /* Measurement noise covariance */
  Eigen::MatrixXd P_; /* Estimate error covariance */

  Eigen::MatrixXd K_; /* Kalman Gain */
  Eigen::VectorXd x_est_, x_pred_; /* State estimate and predicted state */

  int n_; /* Number of states */
  int m_; /* Number of measurements */

  Eigen::MatrixXd I_; /* Identity matrix */
};

#endif  // EXTENDED_KALMAN_HPP
