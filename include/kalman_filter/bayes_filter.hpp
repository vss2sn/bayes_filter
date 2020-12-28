#ifndef BAYES_FILTER_HPP
#define BAYES_FILTER_HPP

#include <Eigen/Dense>

class BayesFilter {
public:
  virtual void SetInitialState(const Eigen::VectorXd& x0) = 0;
  virtual void Predict(const Eigen::VectorXd& u) = 0;
  virtual void Update(const Eigen::VectorXd& y) = 0;
  virtual Eigen::VectorXd GetState() const = 0 ;
};

#endif  // BAYES_FILTER_HPP
