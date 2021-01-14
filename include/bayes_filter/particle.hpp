#include <Eigen/Dense>

#include "bayes_filter/bayes_filter.hpp"


struct SingleParticle {
  double weight; /* probability of the particle */
  Eigen::VectorXd state; /* paricle state */
};

class Particle : public BayesFilter {
public:
  Particle(const Eigen::MatrixXd& P,
  		const Eigen::MatrixXd& Q,
  		const Eigen::MatrixXd& R,
      std::function<Eigen::VectorXd(const Eigen::VectorXd& q, const Eigen::VectorXd& u)> processFunction,
      std::function<Eigen::VectorXd(const Eigen::VectorXd& q)> outputFunction,
      const Eigen::VectorXd& min_states,
      const Eigen::VectorXd& max_states,
      double n_particles
    );

  void SetInitialState(const Eigen::VectorXd& x0) override;
  void Predict(const Eigen::VectorXd& u) override;
  void Update(const Eigen::VectorXd& y) override;
  Eigen::VectorXd GetState() const override;

private:

  /**
   * @brief Naive resampling algorithm
   * @param [in] gen_n_samples number of particles with a low probability that will be replaced with randomly generated samples
   * @return void
   */
  void NaiveResample(const int gen_n_samples); /* A naive resampling algorithm */

  /**
   * @brief Low Variance resampling
   * @return void
   * @details The particels are resampled in a way that is representative of their probability distribution.
              The new list of particles might cionatin multiple copies of the same particle based on its previous probability.
              The probability of each of the new particles is reset to 1/n_particles_
   */
  void LowVarianceResample(); /* Lo variance resampling */

  /**
   * @brief Prevent particle deprivation by adding new particles when the number of unique particles reduces significantly
   * @return void
   * @details This has the added advantage of allowing the particle filter to move towards a correct estimation when the initail state estimated is a bad estimate
   */
  void PreventParticleDeprivation(); /* Add random particles to prevent the number of unique particles from getting too low */

  /* Calculate the estimated state */
  std::function<Eigen::VectorXd(const Eigen::VectorXd& q, const Eigen::VectorXd& u)> processFunction;

  /* convert the predicted state into measurement space */
  std::function<Eigen::VectorXd(const Eigen::VectorXd& q)> outputFunction;

  Eigen::MatrixXd P_; /* Estimate error covariance */
  Eigen::MatrixXd Q_; /* Process noise covariance */
  Eigen::MatrixXd R_; /* Measurement noise covariance */

  int n_; /* Number of states */
  int m_; /* Number of measurements */

  double n_particles_; /* number of particle*/
  std::vector<SingleParticle> particles_; /* Vector of current particles */
  Eigen::VectorXd min_states_; /* Minimum value each state */
  Eigen::VectorXd max_states_; /* Maximum value each state */
};
