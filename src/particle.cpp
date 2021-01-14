#include <random>
#include <unordered_set>
#include <vector>

#include <Eigen/Dense>

#include "bayes_filter/particle.hpp"

Particle::Particle(
		const Eigen::MatrixXd& P,
		const Eigen::MatrixXd& Q,
		const Eigen::MatrixXd& R,
    std::function<Eigen::VectorXd(const Eigen::VectorXd& q, const Eigen::VectorXd& u)> processFunction,
    std::function<Eigen::VectorXd(const Eigen::VectorXd& q)> outputFunction,
    const Eigen::VectorXd& min_states,
    const Eigen::VectorXd& max_states,
    double n_particles
  ) : P_(P), Q_(Q), R_(R),
    processFunction(processFunction),
    outputFunction(outputFunction),
		m_(R.rows()), n_(P.rows()),
    n_particles_(n_particles),
    particles_(n_particles),
    min_states_(min_states),
    max_states_(max_states)
{
  std::default_random_engine generator;
  std::vector<std::uniform_real_distribution<double>> dists;
  dists.reserve(n_);
  for(int i = 0; i < n_; ++i) {
    dists.emplace_back((min_states_[i], max_states_[i]));
  }

  for(auto& p : particles_) {
    p.state = Eigen::VectorXd(n_);
    for(int i = 0 ; i < n_; i++) {
      p.weight = 1.0 / static_cast<double>(n_particles_);
      p.state[i] = dists[i](generator);
    }
  }
}

void Particle::Predict(const Eigen::VectorXd& u) {
	// Predict the next state for each particle
  for(auto& p : particles_) {
    p.state = processFunction(p.state, u);
  }
}

void Particle::Update(const Eigen::VectorXd& y) {

	// Calculate the new weights of all the particles as probabilities that
	// the observed state is the same as each of the particles
  for(auto& p : particles_) {
    const Eigen::VectorXd diff = outputFunction(p.state) - y;
    p.weight *= (1 / (2 * 3.14 * R_.determinant())) * std::exp(-(diff.transpose() * R_ * diff)(0));
  }
	// Normalize the new weights
  const auto weight = std::accumulate(particles_.begin(), particles_.end(), 0.0, [&](const double sum, const auto& p) { return sum + p.weight; } );
  for(auto& p : particles_) {
    p.weight /= weight;
  }

	// Calculate the effective number of particles (NOTE: this variant might be a naive measure)
  const auto eff_n = 1.0/std::accumulate(particles_.begin(), particles_.end(), 0.0, [&](const double sum, const auto& p) { return sum + std::pow(p.weight, 2); } );

	// Resample if the effective number of particles is less than half the number of particles
  if(eff_n < n_particles_/2) {
    // resample(n_particles_/2 - eff_n);
		void LowVarianceResample();
		void PreventParticleDeprivation();
  }
}

// Naive resampling algorithm
void Particle::NaiveResample(const int gen_n_samples) {

	// Calculate the threshold for teh probabiolity under which particles can be replaced
  const auto threshold = std::max_element(particles_.begin(), particles_.end(), [&](const auto& p1, const auto& p2) { return p2.weight > p1.weight; } )->weight;

	// Replace randomly selected particles that have a probability under the threshold with new particles
  std::default_random_engine generator;
  std::vector<std::uniform_real_distribution<double>> dists;
  dists.reserve(n_);
  for(int i = 0; i < n_; ++i) {
    dists.emplace_back(std::uniform_real_distribution<double>(min_states_[i], max_states_[i]));
  }

  std::uniform_int_distribution<> dist(0, n_particles_-1);
  for(int i = 0 ; i < gen_n_samples; ++i) {
    const auto ind = dist(generator);
    if(particles_[ind].weight < threshold) {
      for(int j = 0 ; j < n_; j++) {
        particles_[ind].state[j] = dists[j](generator);
      }
    }
  }
}

// Find the most likely state and return it
Eigen::VectorXd Particle::GetState() const {
  return particles_[
    std::distance(particles_.begin(),
                  std::max_element(particles_.begin(),
                                   particles_.end(),
                                   [&](const auto& p1, const auto& p2) { return p2.weight > p1.weight; } ))].state;
}

// Assumes the initial state is close to accurate and creates a set of particles whose values are normally distributed around the state passed in
void Particle::SetInitialState(const Eigen::VectorXd& x0) {
  std::default_random_engine generator;
  std::vector<std::uniform_real_distribution<double>> dists;
  dists.reserve(n_);
  for(int i = 0; i < n_; ++i) {
    dists.emplace_back(x0[i]);
  }

  for(auto& p : particles_) {
    p.state = Eigen::VectorXd(n_);
    for(int i = 0 ; i < n_; i++) {
      p.weight = 1.0 / static_cast<double>(n_particles_);
      p.state[i] = dists[i](generator);
    }
  }
}

bool operator == (const SingleParticle& p1, const SingleParticle& p2) {
	return p1.weight == p2.weight && p1.state == p2.state;
}

// Naive resampling algorithm
void Particle::PreventParticleDeprivation() {

	struct particleHashFn {
		std::size_t operator() (const SingleParticle& p) const {
			return std::hash<double>()(p.weight);
		}
	};

	std::unordered_set<SingleParticle, particleHashFn> unique_particles;
	for(const auto& p : particles_) {
		unique_particles.insert(p);
	}

	if(unique_particles.size() < particles_.size() / 4) {
		// Remove duplicates while maintaining their probability in the weight
		std::unique(std::begin(particles_), std::end(particles_));

		std::default_random_engine generator;
		std::vector<std::uniform_real_distribution<double>> dists;
		dists.reserve(n_);
		for(int i = 0; i < n_; ++i) {
			dists.emplace_back(std::uniform_real_distribution<double>(min_states_[i], max_states_[i]));
		}

		const int c_n_particles = particles_.size(); // current number of particles after nrunning unique
		for(int i = c_n_particles; i < n_particles_; ++i) {
			for(int j = 0 ; j < n_; j++) {
				particles_[i].state[j] = dists[j](generator);
			}
			particles_[i].weight = 1.0 / static_cast<double>(n_particles_);
		}
	}
	const double normalize_weight = 1.0 / std::accumulate(particles_.begin(), particles_.end(), 0.0, [&](const double sum, const auto& p) { return sum + p.weight; } );
	for(auto& p : particles_) {
		p.weight *= normalize_weight;
	}
}

void Particle::LowVarianceResample() {
	const double delta = 1.0/static_cast<double>(particles_.size());
	std::default_random_engine generator;
	std::uniform_real_distribution<double> dist(0, delta);
	std::vector<SingleParticle> new_particles;
  new_particles.reserve(particles_.size());

	int n_selected = 0;
	double cumu_prob = 0;
	int index = 0;
	const double sel_p = dist(generator);
	while (cumu_prob + particles_[index].weight < 1.0) {
		while (cumu_prob + particles_[index].weight < sel_p) {
			++index;
		}
		new_particles.push_back(particles_[index]);
		new_particles.back().weight = delta;
		cumu_prob += delta;
	}

	particles_ = new_particles;
	const double normalize_weight = 1.0 / std::accumulate(particles_.begin(), particles_.end(), 0.0, [&](const double sum, const auto& p) { return sum + p.weight; } );
	for(auto& p : particles_) {
		p.weight *= normalize_weight;
	}
}
