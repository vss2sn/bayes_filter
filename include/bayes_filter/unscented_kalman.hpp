#ifndef UNSCENTED_KALMAN_HPP
#define UNSCENTED_KALMAN_HPP

#include <functional>
#include <vector>

#include <Eigen/Dense>

#include "bayes_filter/bayes_filter.hpp"

class UnscentedKalman : public BayesFilter {
public:

  UnscentedKalman(
    const Eigen::MatrixXd& P,
    const Eigen::MatrixXd& Q,
    const Eigen::MatrixXd& R,
    std::function<Eigen::VectorXd(const Eigen::VectorXd& q, const Eigen::VectorXd& u)> processFunction,
    std::function<Eigen::VectorXd(const Eigen::VectorXd& q)> outputFunction,
    double alpha, double beta, double kappa
  );

  void SetInitialState(const Eigen::VectorXd& x0) override;

  void Predict(const Eigen::VectorXd& u) override;
  void Update(const Eigen::VectorXd& y) override;

  Eigen::VectorXd GetState() const override { return x_est_; };

private:

  Eigen::MatrixXd P_; /* Estimate error covariance */
  Eigen::MatrixXd Q_; /* Process noise covariance */
  Eigen::MatrixXd R_; /* Measurement noise covariance */

  std::function<Eigen::VectorXd(const Eigen::VectorXd& q, const Eigen::VectorXd& u)> processFunction;
  std::function<Eigen::VectorXd(const Eigen::VectorXd& r)> outputFunction;

  int n_; /* Number of states */
  int m_; /* Number of measurements */
  int k_; /* Number of sigma points */

  Eigen::VectorXd x_est_, x_pred_; /* State estimate and predicted state */

  Eigen::MatrixXd K_; /* Kalman Gain */
  Eigen::MatrixXd I_; /* Identity matrix */

  double alpha_; /* Design parameter */
  double beta_; /* Design parameter */
  double kappa_; /* Design parameter */
  double lambda_; /* Automatically calculated parameter */

  std::vector<Eigen::VectorXd> sigma_points_; /* States representing the current probability distribution */
  std::vector<Eigen::VectorXd> pred_sigma_points_; /* States representing the predicted probability distribution */
  std::vector<Eigen::VectorXd> output_sigma_points_; /* States representing the output probability distribution */
  std::vector<double> mean_weights_; /* Weights for mean propagation */
  std::vector<double> covariance_weights_; /* Weights for covariance propagation */
};

#endif  // UNSCENTED_KALMAN_HPP
