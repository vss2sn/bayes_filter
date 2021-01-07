#include <iostream>
#include <random>
#include <vector>
#include <numeric>

#include <Eigen/Dense>

#include "kalman_filter/particle.hpp"

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
    particles(n_particles),
    min_states(min_states),
    max_states(max_states)
{
  std::default_random_engine generator;
  std::vector<std::uniform_real_distribution<double>> dists;
  dists.reserve(n_);
  for(int i = 0; i < n_; ++i) {
    dists.emplace_back((min_states[i], max_states[i]));
  }

  for(auto& p : particles) {
    p.state = Eigen::VectorXd(n_);
    for(int i = 0 ; i < n_; i++) {
      p.weight = 1.0 / static_cast<double>(n_particles_);
      p.state[i] = dists[i](generator);
    }
  }
}

void Particle::Predict(const Eigen::VectorXd& u) {
  for(auto& p : particles) {
    p.state = processFunction(p.state, u);
  }
}

void Particle::Update(const Eigen::VectorXd& y) {
  for(auto& p : particles) {
    const Eigen::VectorXd diff = outputFunction(p.state) - y;
    p.weight *= (1 / (2 * 3.14 * R_.determinant())) * std::exp(-(diff.transpose() * R_ * diff)(0));
  }

  const auto weight = std::accumulate(particles.begin(), particles.end(), 0.0, [&](const double sum, const auto& p) { return sum + p.weight; } );
  for(auto& p : particles) {
    p.weight /= weight;
  }
  const auto eff_n = 1.0/std::accumulate(particles.begin(), particles.end(), 0.0, [&](const double sum, const auto& p) { return sum + std::pow(p.weight, 2); } );

  if(eff_n < n_particles_/2) {
    resample(n_particles_/2 - eff_n);
  }
}

void Particle::resample(const int gen_n_samples) {
  const auto threshold = std::max_element(particles.begin(), particles.end(), [&](const auto& p1, const auto& p2) { return p2.weight > p1.weight; } )->weight;

  std::default_random_engine generator;
  std::vector<std::uniform_real_distribution<double>> dists;
  dists.reserve(n_);
  for(int i = 0; i < n_; ++i) {
    dists.emplace_back(std::uniform_real_distribution<double>(min_states[i], max_states[i]));
  }

  std::uniform_int_distribution<> dist(0, n_particles_-1);
  for(int i = 0 ; i < gen_n_samples; ++i) {
    const auto ind = dist(generator);
    if(particles[ind].weight < threshold) {
      for(int j = 0 ; j < n_; j++) {
        particles[ind].state[j] = dists[j](generator);
      }
    }
  }
}

Eigen::VectorXd Particle::GetState() const {
  return particles[
    std::distance(particles.begin(),
                  std::max_element(particles.begin(),
                                   particles.end(),
                                   [&](const auto& p1, const auto& p2) { return p2.weight > p1.weight; } ))].state;
}

void Particle::SetInitialState(const Eigen::VectorXd& x0) {
  std::default_random_engine generator;
  std::vector<std::uniform_real_distribution<double>> dists;
  dists.reserve(n_);
  for(int i = 0; i < n_; ++i) {
    dists.emplace_back(x0[i], std::abs(x0[i]));
  }

  for(auto& p : particles) {
    p.state = Eigen::VectorXd(n_);
    for(int i = 0 ; i < n_; i++) {
      p.weight = 1.0 / static_cast<double>(n_particles_);
      p.state[i] = dists[i](generator);
    }
  }
};
