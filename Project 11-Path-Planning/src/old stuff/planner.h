#ifndef PLANNER_
#define PLANNER_

#include <vector>

using namespace std;

//double calc_curve(vector<double> X, vector<double> Y, double curr_x); 

void calc_curve(vector<double> & X, vector<double> & Y);

vector<double> JMT(vector< double> start, vector <double> end, double T);

vector<double> in_car_frame(double car_x, double car_y, double car_yaw, double point_x, double point_y);

vector<double> in_world_frame(double car_x, double car_y, double car_yaw, double point_x, double point_y);

#endif