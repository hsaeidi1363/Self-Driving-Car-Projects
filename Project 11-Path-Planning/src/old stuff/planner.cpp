#include "planner.h"
#include "spline.h"
#include "Eigen-3.3/Eigen/Dense"

using namespace std;

void calc_curve(vector<double> & X, vector<double> & Y){
	tk::spline s;
	s.set_points(X,Y);
	for (int i = 0; i < X.size(); ++i){
		Y[i] = s(X[i]);
	}
	
	//return y;
}

using Eigen::MatrixXd;
using Eigen::VectorXd;

// TODO - complete this function
vector<double> JMT(vector< double> start, vector <double> end, double T)
{
  
    
    MatrixXd A = MatrixXd(3, 3);
	A << T*T*T, T*T*T*T, T*T*T*T*T,
			    3*T*T, 4*T*T*T,5*T*T*T*T,
			    6*T, 12*T*T, 20*T*T*T;
		
	MatrixXd B = MatrixXd(3,1);	    
	B << end[0]-(start[0]+start[1]*T+.5*start[2]*T*T),
			    end[1]-(start[1]+start[2]*T),
			    end[2]-start[2];
			    
	MatrixXd Ai = A.inverse();
	
	MatrixXd C = Ai*B;
	
	vector <double> result = {start[0], start[1], .5*start[2]};
	for(int i = 0; i < C.size(); i++)
	{
	    result.push_back(C.data()[i]);
	}
	
    return result;
    
}

vector<double> in_car_frame(double car_x, double car_y, double car_yaw, double point_x, double point_y){

		double dx = point_x - car_x;
		double dy = point_y - car_y;
		vector<double> results;
		results.push_back(dx * cos(car_yaw) + dy * sin(car_yaw));
		results.push_back(- dx * sin(car_yaw) + dy * cos(car_yaw));
		return results;

}


vector<double> in_world_frame(double car_x, double car_y, double car_yaw, double point_x, double point_y){
		vector<double> results;
		results.push_back(car_x + point_x * cos(car_yaw) - point_y * sin(car_yaw));
		results.push_back(car_y + point_x * sin(car_yaw) + point_y * cos(car_yaw));
		return results;

}