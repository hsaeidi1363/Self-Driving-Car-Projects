/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 *      Modified by: Hamed Saeidi 6/2017 for the self-driving car projects
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <random>

#include "particle_filter.h"
#include "helper_functions.h"

using namespace std;


double multi_var_gauss(double sense_x, double landmark_x, double std_x,double sense_y, double landmark_y, double std_y){
	double c1 = 1/(2*M_PI*std_x*std_y);
	double c2 = pow((sense_x-landmark_x),2)/(2*std_x*std_x);
	c2 += pow((sense_y-landmark_y),2)/(2*std_y*std_y);
	return c1*exp(-c2);
}



void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	std::cout << std[0] << std::endl;
	num_particles = 2000;
	default_random_engine gen;
	double std_x, std_y, std_theta; // Standard deviations for x, y, and theta

	// Set standard deviations for x, y, and theta.
	std_x = std[0];
	std_y = std[1]; 
	std_theta = std[2];
	
	// This line creates a normal (Gaussian) distribution for x, y, theta.
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

	Particle tmp;

	for (int i = 0; i< num_particles; i++){
		tmp.id = i+1;
		tmp.x = dist_x(gen);
		tmp.y = dist_y(gen);
		tmp.theta = dist_theta(gen);
		tmp.weight = 1.0;
		particles.push_back(tmp);
		std::cout << "ID: "<< particles[i].id << ", X: "<< particles[i].x << ", Y: "<< particles[i].y << ", Theta: "<< particles[i].theta << ", Weight: "<< particles[i].weight<<std::endl;

	}
	std::cout << "Number of particles created " << num_particles << std::endl;
	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;
	double std_v, std_omega; // Standard deviations for v and omega

	// Set the standard deviations for v and omega.
	std_v = std_pos[1]/delta_t; 
	std_omega = std_pos[2]/delta_t;

	
	// This line creates a normal (Gaussian) distribution for v, omega.
	normal_distribution<double> dist_v(velocity, std_v);
	normal_distribution<double> dist_omega(yaw_rate, std_omega);
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);


	double v;
	double omega;
	for (int i = 0; i< num_particles; i++){
		v = dist_v(gen);
		omega = dist_omega(gen);
		// predict the motion using the differential equations 
		if (fabs(omega) > 0.001) {
			particles[i].x += v/omega*(sin(particles[i].theta+omega*delta_t)-sin(particles[i].theta)) + dist_x(gen);
			particles[i].y += v/omega*(-cos(particles[i].theta+omega*delta_t)+cos(particles[i].theta))+ dist_y(gen);	
		}else{
			particles[i].x += v*delta_t*sin(particles[i].theta) + dist_x(gen);
			particles[i].y += v*delta_t*cos(particles[i].theta) + dist_y(gen);
		}
		particles[i].theta +=  omega*delta_t + dist_theta(gen);
		
	}

}



void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	double std_x = std_landmark[0];
	double std_y = std_landmark[1];

		
	// step 1: transform the observations to the world coordinates for each particle
	double weight_sum = 0;
	for (int i = 0; i< num_particles; ++i){
		double weight_tmp = 1.0;
		for (int j = 0; j < observations.size(); ++j){
			double sense_x_tmp;
			double sense_y_tmp;
			int association_tmp = 0;	
			// convert the coordinates to the world (map) frame
			sense_x_tmp = particles[i].x + ( observations[j].x*cos(particles[i].theta) - observations[j].y*sin(particles[i].theta) );
			sense_y_tmp = particles[i].y + ( observations[j].x*sin(particles[i].theta) + observations[j].y*cos(particles[i].theta) );
			// associate the measurements with the landmarks
			double min_dist = sensor_range;
			for (int k = 0; k < map_landmarks.landmark_list.size(); ++k){
				double dist_tmp = dist(sense_x_tmp, sense_y_tmp, map_landmarks.landmark_list[k].x_f, map_landmarks.landmark_list[k].y_f);
				if ( dist_tmp <= min_dist){
					min_dist = dist_tmp;
					association_tmp = k;
				}
			}
			//update the weight using the current landmark measurement
			weight_tmp *= multi_var_gauss(sense_x_tmp, map_landmarks.landmark_list[association_tmp].x_f, std_x, sense_y_tmp,  map_landmarks.landmark_list[association_tmp].y_f,std_y);
				
		}
		// update the particle weight
		particles[i].weight = weight_tmp;
		//calculate the sum of weights
		weight_sum += particles[i].weight;
		
	}
	// normalize the weights
	for (int i = 0; i< num_particles; ++i){
		particles[i].weight /= weight_sum;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
//	default_random_engine generator;
	std::random_device rd;
	std::mt19937 generator(rd());
        std::vector<double> weights;
	for (int i = 0; i< num_particles; ++i){
		weights.push_back(particles[i].weight);
	}
	// initialize the distribution with particle weights
	discrete_distribution<> distribution(weights.begin(),weights.end());
	std::vector<Particle> resampled_particles;
	int ind;
	for (int i = 0; i< num_particles; ++i){
		ind = distribution(generator);
		resampled_particles.push_back(particles[ind]);
	}
	// update the particles using the resampled particles
	particles = resampled_particles;


}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}


