#include <Eigen/Dense>

#include "kalman_filter/bayes_filter.hpp"

struct SingleParticle {
  double weight;
  Eigen::VectorXd state;
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
  void resample(const int gen_n_samples);
  ~Particle() {}
private:

  std::function<Eigen::VectorXd(const Eigen::VectorXd& q, const Eigen::VectorXd& u)> processFunction;
  std::function<Eigen::VectorXd(const Eigen::VectorXd& q)> outputFunction;

  Eigen::MatrixXd P_, Q_, R_;
  void generateSamples(const int n);
  std::vector<SingleParticle> particles;
  Eigen::VectorXd min_states;
  Eigen::VectorXd max_states;
  double n_particles_;
  int m_, n_;
};