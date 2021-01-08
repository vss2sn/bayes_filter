#ifndef BAYES_FILTER_HPP
#define BAYES_FILTER_HPP

#include <Eigen/Dense>

class BayesFilter {
public:

  /**
   * @brief Set the initial guess of the state
   * @param [in] x0 the initital guess of the state
   * @return void
   */
  virtual void SetInitialState(const Eigen::VectorXd& x0) = 0;

  /**
   * @brief The first step of the bayes filter
   * @param [in] u the control input given to the system
   * @return void
   * @details Use the process model to predict the next state(s) given the current states(s)
   */
  virtual void Predict(const Eigen::VectorXd& u) = 0;

  /**
   * @brief The second step of the bayes filter
   * @param [in] y the observation made using the sensor
   * @return void
   * @details the observation could be for one or more variables of the state or something that can be used to estimate state
   */
  virtual void Update(const Eigen::VectorXd& y) = 0;

  /**
   * @brief Get the current best estimate of state
   * @return the current best state
   */
  virtual Eigen::VectorXd GetState() const = 0 ;

  /**
   * @brief Virtual destructor
   */
  virtual ~BayesFilter() {};
};

#endif  // BAYES_FILTER_HPP
