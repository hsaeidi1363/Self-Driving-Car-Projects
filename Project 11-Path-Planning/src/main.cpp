#include <fstream>
#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "json.hpp"
#include "spline.h"

#define MAX_SPEED 22.0 // maximum speed in meters per second
#define dT 0.02 // sample time of the control loop
#define MAX_COST 1000.0// for the cost functions associated with the finiate state machine decisions

using namespace std;

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees and mile/hour to meter/sec.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }
double miph2mps(double x) { return x*0.44704; }



// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_first_of("}");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

// calculate the distance between two points on the road
double distance(double x1, double y1, double x2, double y2)
{
	return sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
}

int ClosestWaypoint(double x, double y, const vector<double> &maps_x, const vector<double> &maps_y)
{

	double closestLen = 100000; //large number
	int closestWaypoint = 0;

	for(int i = 0; i < maps_x.size(); i++)
	{
		double map_x = maps_x[i];
		double map_y = maps_y[i];
		double dist = distance(x,y,map_x,map_y);
		if(dist < closestLen)
		{
			closestLen = dist;
			closestWaypoint = i;
		}

	}

	return closestWaypoint;

}

int NextWaypoint(double x, double y, double theta, const vector<double> &maps_x, const vector<double> &maps_y)
{

	int closestWaypoint = ClosestWaypoint(x,y,maps_x,maps_y);

	double map_x = maps_x[closestWaypoint];
	double map_y = maps_y[closestWaypoint];

	double heading = atan2((map_y-y),(map_x-x));

	double angle = fabs(theta-heading);
  angle = min(2*pi() - angle, angle);

  if(angle > pi()/4)
  {
    closestWaypoint++;
  if (closestWaypoint == maps_x.size())
  {
    closestWaypoint = 0;
  }
  }

  return closestWaypoint;
}

// Transform from Cartesian x,y coordinates to Frenet s,d coordinates
vector<double> getFrenet(double x, double y, double theta, const vector<double> &maps_x, const vector<double> &maps_y)
{
	int next_wp = NextWaypoint(x,y, theta, maps_x,maps_y);

	int prev_wp;
	prev_wp = next_wp-1;
	if(next_wp == 0)
	{
		prev_wp  = maps_x.size()-1;
	}

	double n_x = maps_x[next_wp]-maps_x[prev_wp];
	double n_y = maps_y[next_wp]-maps_y[prev_wp];
	double x_x = x - maps_x[prev_wp];
	double x_y = y - maps_y[prev_wp];

	// find the projection of x onto n
	double proj_norm = (x_x*n_x+x_y*n_y)/(n_x*n_x+n_y*n_y);
	double proj_x = proj_norm*n_x;
	double proj_y = proj_norm*n_y;

	double frenet_d = distance(x_x,x_y,proj_x,proj_y);

	//see if d value is positive or negative by comparing it to a center point

	double center_x = 1000-maps_x[prev_wp];
	double center_y = 2000-maps_y[prev_wp];
	double centerToPos = distance(center_x,center_y,x_x,x_y);
	double centerToRef = distance(center_x,center_y,proj_x,proj_y);

	if(centerToPos <= centerToRef)
	{
		frenet_d *= -1;
	}

	// calculate s value
	double frenet_s = 0;
	for(int i = 0; i < prev_wp; i++)
	{
		frenet_s += distance(maps_x[i],maps_y[i],maps_x[i+1],maps_y[i+1]);
	}

	frenet_s += distance(0,0,proj_x,proj_y);

	return {frenet_s,frenet_d};

}

// Transform from Frenet s,d coordinates to Cartesian x,y
vector<double> getXY(double s, double d, const vector<double> &maps_s, const vector<double> &maps_x, const vector<double> &maps_y)
{
	int prev_wp = -1;

	while(s > maps_s[prev_wp+1] && (prev_wp < (int)(maps_s.size()-1) ))
	{
		prev_wp++;
	}

	int wp2 = (prev_wp+1)%maps_x.size();

	double heading = atan2((maps_y[wp2]-maps_y[prev_wp]),(maps_x[wp2]-maps_x[prev_wp]));
	// the x,y,s along the segment
	double seg_s = (s-maps_s[prev_wp]);

	double seg_x = maps_x[prev_wp]+seg_s*cos(heading);
	double seg_y = maps_y[prev_wp]+seg_s*sin(heading);

	double perp_heading = heading-pi()/2;

	double x = seg_x + d*cos(perp_heading);
	double y = seg_y + d*sin(perp_heading);

	return {x,y};

}

// transforms the world coordinates to the local car frame
vector<double> in_car_frame(double car_x, double car_y, double car_yaw, double point_x, double point_y){

		double dx = point_x - car_x;
		double dy = point_y - car_y;
		vector<double> results;
		results.push_back(dx * cos(car_yaw) + dy * sin(car_yaw));
		results.push_back(- dx * sin(car_yaw) + dy * cos(car_yaw));
		return results;

}

// retruns back the points in the car frame to the world frame
vector<double> in_world_frame(double car_x, double car_y, double car_yaw, double point_x, double point_y){
		vector<double> results;
		results.push_back(car_x + point_x * cos(car_yaw) - point_y * sin(car_yaw));
		results.push_back(car_y + point_x * sin(car_yaw) + point_y * cos(car_yaw));
		return results;

}


// finds cars in front/left/right of the car
vector<int> find_cars(double car_s, double car_d, vector<vector<double>> cars, string mode){
	vector<int> results;

	//finds the closest front car
	if (mode == "front"){
		int front_car = 200;
		double closest_car_dist = 500;
		for (int i = 0; i < cars.size(); i++){
			double dist = cars[i][5] - car_s;
			// check if the car is in my lane and in front of me and is the closest car
			if ( fabs(car_d - cars[i][6]) < 2 && dist < 30 && dist > 0 && dist <closest_car_dist){
				front_car = i;
				closest_car_dist = dist;
			}
		}
	  	results.push_back(front_car);
	}
	//finds the cars on the right lane
	if ( mode == "right"){
		double lane_l = ceil(car_d/4.0)*4.0;
		double lane_r = lane_l + 4.0;
		for (int i = 0; i < cars.size(); i++){
			double car_di = cars[i][6];
			if (car_di > lane_l && car_di < lane_r)
				results.push_back(i);
		}			
	}
	//finds the cars on the left lane
	if ( mode == "left"){
		double lane_r = floor(car_d/4.0)*4.0;
		double lane_l = lane_r - 4.0;
		for (int i = 0; i < cars.size(); i++){
			double car_di = cars[i][6];
			if (car_di > lane_l && car_di < lane_r)
				results.push_back(i);
		}			

	}
	return results;

}


// bould the values in a desired range
double saturate(double val, double max_val){
	if ( val > max_val)
		return max_val;
	if ( val < -max_val)
		return -max_val;
	else
		return val;
}


// current velocity of car
double vc;
// integral of speed error for driving at a desired speed
double e_integral = 0.0;


// gap between my car and the closest car in front
double gap = 0.0;
// desired value of gap for a safe drive
double gap_desired = 20.0;

// control the speed for tracking a reference speed and also avoiding collision with the front car
double control_speed(double ref_speed){
	double dist = 0;
	double kp = 0.0; // proportional control gain
	double ki = 0.0; // integral control gain
	double k_safe = 0.0; // a proportional controller for keeping the safe distance with the front car
	
	double e = ref_speed - vc; // current tracking error
	e_integral += e*dT;
	// parameters of a car model with mass m and viscous damping of b: m*v_dot + b*v = f (control force)
	double m = 800;
	double b = 10;
	// choose these gains when driving at max speed
	if (ref_speed == MAX_SPEED){
		kp = 300;
		ki = 50;
	// and switch to these gains when a car is in front
	}else{
		kp = 1000;
		ki = 50;
		k_safe = 100;
	}	
	// produce the control force to the car model 
	double force = kp*e+ki*e_integral + k_safe*(gap - gap_desired);
	
	// apply the dynamic model of the car to prevent sudden jerky motions and car teleport
	double vn = (m*vc + dT*force )/(b*dT+m);
	// saturate the velocity output
	vn = saturate(vn, MAX_SPEED);

	// calculate how much the car should move on the road (distance to travel in this step)
	dist = vn*dT;	
	std::cout << std::fixed << std::setprecision(2);
	// for debugging the controller and model://cout <<"ref: "<<ref_speed<< "vc: " << vc<<" e: "<< e<<" e_int: "<< e_integral <<" e_dot: "<< edot << " force :" << force<< " vn: " << vn<< " dd: " << dist<<endl;
	// keep the last value of vn for the next control loop. 
	vc = vn;
	return dist;
}

// this function finds out how much delta x and delta y on the trajectory spline satisfay the required travel distance
vector<double> find_dxdy(double dist, double ptx, double pty, tk::spline s){
	bool no_next_point = true;
	double eps_x = 0.5*dist;
	double xn = 0;
	double yn = 0;
	// keep searching on the spline to find the precise dx and dy				 
	while(no_next_point){				
		xn = ptx + eps_x;
		yn = s(xn);
		double eps_y = yn - pty;
		double d = sqrt(eps_x*eps_x+eps_y*eps_y);
		double lambda = 0.002;
		if (d > dist){
			// prevent overshoot
			eps_x *= (1-lambda);
		// if the desired margin of error is not satisfied keep searching
		}else if(dist - d > 0.005){
			eps_x *= (1+lambda);
		}else{
			// the next point is found
			no_next_point = false;
		}  
	}
	// return the next points on the spline for the car to track
	return {xn,yn};
}
// an exponential cost function for penalizing the slow speeds
double exp_cost1(double ds, double sigma){
	return MAX_COST*(1-exp(-fabs(ds)/(sigma)));
}

// an exponential cost function
double exp_cost2(double ds, double sigma){
	if (ds > 0) // increase the safety costs of the cars in front more than the cars at the back depending on the location
		return MAX_COST*exp(-fabs(ds)/(sigma));
	else 
		return MAX_COST*exp(-fabs(ds)/(sigma/3));
}


// an exponential cost function for testing if cars behind are speed towards me or not
// when a car is close and is driving faster than me the cost of changing lane is high
double exp_cost3(double ds, double dv){
	if(ds < 0){
		return  MAX_COST/2*(exp(-dv/ds)-1);
	}else{
		return 0;
	}
}


// find the costs of keeping lane, changing lane to left, and changing lane to right
vector<double> calc_cost(double car_s, double car_d, double car_speed, vector<vector<double>> cars){

	double KL_cost = 0.0;
	double LCL_cost = 0.0;
	double LCR_cost = 0.0;
	vector<double> costs= {0,0,0};

	KL_cost = exp_cost1((car_speed - MAX_SPEED),MAX_SPEED*2.0);
	double sigma = fabs(car_speed - MAX_SPEED)+5;

	// when at the rightmost lane
	if( car_d > 8 && car_d < 12){
		LCR_cost = MAX_COST;
	// otherwise
	}else{
		vector<int> right_cars = find_cars(car_s, car_d, cars, "right");
		for (int i = 0; i < right_cars.size(); i++){
			double ds = cars[right_cars[i]][5] - car_s ;
			double vx = cars[right_cars[i]][3];
			double vy = cars[right_cars[i]][4];
			double speed = sqrt(vx*vx + vy*vy);
			double dv = speed - car_speed;
			LCR_cost += exp_cost2(ds,sigma) + exp_cost3(ds, dv);
		}
	}
	// when at the leftmost lane
	if( car_d > 0 && car_d < 4){
		LCL_cost = MAX_COST;
	// otherwise
	}else{
		vector<int> left_cars = find_cars(car_s, car_d, cars, "left");
		for (int i = 0; i < left_cars.size(); i++){
			double ds = cars[left_cars[i]][5] - car_s ;
			double vx = cars[left_cars[i]][3];
			double vy = cars[left_cars[i]][4];
			double speed = sqrt(vx*vx + vy*vy);
			double dv = speed - car_speed;
			LCL_cost += exp_cost2(ds,sigma) + exp_cost3(ds, dv);
		}
	}
	// saturate the costs
	costs[0] = min(KL_cost, MAX_COST);
	LCL_cost = max(LCL_cost, 50.0);
	LCR_cost = max(LCR_cost, 50.0);
	costs[1] = min(LCL_cost, MAX_COST);
	costs[2] = min(LCR_cost, MAX_COST);
	return costs;
	
}

int ctr = 0;
double target_lane = 1;
double target_lane_prev = 1;

int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;

  ifstream in_map_(map_file_.c_str(), ifstream::in);

  string line;
  while (getline(in_map_, line)) {
  	istringstream iss(line);
  	double x;
  	double y;
  	float s;
  	float d_x;
  	float d_y;
  	iss >> x;
  	iss >> y;
  	iss >> s;
  	iss >> d_x;
  	iss >> d_y;
  	map_waypoints_x.push_back(x);
  	map_waypoints_y.push_back(y);
  	map_waypoints_s.push_back(s);
  	map_waypoints_dx.push_back(d_x);
  	map_waypoints_dy.push_back(d_y);
  }
  
  h.onMessage([&map_waypoints_x,&map_waypoints_y,&map_waypoints_s,&map_waypoints_dx,&map_waypoints_dy](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    //auto sdata = string(data).substr(0, length);
    //cout << sdata << endl;
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {

      auto s = hasData(data);

      if (s != "") {
        auto j = json::parse(s);
        
        string event = j[0].get<string>();
        
        if (event == "telemetry") {
          // j[1] is the data JSON object
          
        	// Main car's localization Data
          	double car_x = j[1]["x"];
          	double car_y = j[1]["y"];
          	double car_s = j[1]["s"];
          	double car_d = j[1]["d"];
          	double car_yaw = deg2rad(j[1]["yaw"]);
          	double car_speed = miph2mps(j[1]["speed"]);

          	// Previous path data given to the Planner
          	auto previous_path_x = j[1]["previous_path_x"];
          	auto previous_path_y = j[1]["previous_path_y"];
          	// Previous path's end s and d values 
          	double end_path_s = j[1]["end_path_s"];
          	double end_path_d = j[1]["end_path_d"];

          	// Sensor Fusion Data, a list of all other cars on the same side of the road.
          	auto sensor_fusion = j[1]["sensor_fusion"];

          	json msgJson;

          	vector<double> next_x_vals;
          	vector<double> next_y_vals;

			vector<double> curve_x;
			vector<double> curve_y;
			double start_x;
			double start_y;
			double start_s;

			int prev_length = previous_path_x.size();
			
			//initialize the spline points
			if ( prev_length < 2){
				double car_x_prev = car_x - cos(car_yaw);
				double car_y_prev = car_y - sin(car_yaw);
				curve_x.push_back(car_x_prev);
				curve_x.push_back(car_x);	
				curve_y.push_back(car_y_prev);
				curve_y.push_back(car_y);	
				start_x = car_x;
				start_y = car_y;
				start_s = car_s;
				vc = 0;
			//if the previouse path exists use the remaining parts of it
			}else{
				double x1 = previous_path_x[prev_length-1];
				double x2 = previous_path_x[prev_length-2];
				double y1 = previous_path_y[prev_length-1];
				double y2 = previous_path_y[prev_length-2];
				curve_x.push_back(x2);
				curve_x.push_back(x1);	
				curve_y.push_back(y2);
				curve_y.push_back(y1);				
				double dx = x1-x2;
				double dy = y1-y2;
				vc = sqrt(dx*dx + dy*dy)/dT;
				start_x = x1;
				start_y = y1;
				start_s = end_path_s;
				for (int i = 0; i < prev_length; i++){
					next_x_vals.push_back(previous_path_x[i]);
					next_y_vals.push_back(previous_path_y[i]);
				}

			}
			ctr ++;
			// every second after the 5th second make decisions about changing or keeping lanes ( initially the car has not come to speed on road to make stable decisions)
			if (ctr % 50 == 0 && ctr > 250){
				// calculate the costs of keeping lane, changing lane to left, and right
				vector<double> all_costs = calc_cost(car_s, car_d, car_speed, sensor_fusion);	
				double min_cost = MAX_COST;
				int min_ind = 0;
				cout << "from: ";
				// find the minimum cost
				for (int i = 0 ; i < all_costs.size(); i++){
					cout << all_costs[i]<< " ";
					if (all_costs[i] < min_cost){
						min_cost = all_costs[i];					
						min_ind = i;
					}
				}
				// change lane if necessary 
				cout << "chose: ";
				switch(min_ind){
					case 0:
						cout << "KL" << endl;
						break;
					case 1:
						cout << "Left" << endl;
						target_lane -= 1;
						target_lane = max(target_lane , 0.0);
						break;
					case 2:
						cout << "Right" << endl;
						target_lane += 1;
						target_lane = min(target_lane , 2.0);
						break;
				}
			}
			// find the target d value
			double target_d = target_lane*4 + 2;
			// distance between future waypoints
			double target_move = 30;
			
			//produce three distant waypoints
			for(int i = 1; i < 4; i++){
				vector<double> next_curve = getXY(start_s+i*target_move,target_d, map_waypoints_s,map_waypoints_x,map_waypoints_y);
				curve_x.push_back(next_curve[0]);
				curve_y.push_back(next_curve[1]);
			}
			// transfrom the points to the car frame 
			for (int i = 0; i < curve_x.size();i++){
				vector<double> transformed = in_car_frame(car_x,car_y,car_yaw,curve_x[i],curve_y[i]);
				curve_x[i] = transformed[0];
				curve_y[i] = transformed[1];
			}
			// fit a spline to the way points
			tk::spline spl;
			spl.set_points(curve_x,curve_y);

			double ref_speed;
			// check if a car is in front of me and change the reference speed if necessary
			vector<int> front_car = find_cars(car_s,car_d, sensor_fusion, "front");
			if (front_car[0] != 200){
				double front_car_s =  sensor_fusion[front_car[0]][5];
				int car_no = (int) front_car[0];
				double vx = sensor_fusion[car_no][3];
				double vy = sensor_fusion[car_no][4];
				double gx = sensor_fusion[car_no][1];
				gx-=car_x;
				double gy = sensor_fusion[car_no][2];
				gy-=car_y;
				ref_speed = sqrt(vx*vx + vy*vy);
				gap = sqrt(gx*gx + gy*gy);
			// otherwise drive with maximim speed
			}else{
				ref_speed = MAX_SPEED;
			}

			
			double dist_inc =0;
			
			// find the starting point of trajectory in the car frame
			vector<double> local_start_xy = in_car_frame(car_x,car_y,car_yaw,start_x,start_y);
			double xc = local_start_xy[0];
			double yc = local_start_xy[1];

		
			int N = 50;
			// produce the remaing parts of the trajectory to get 50 points
			for (int i = 0; i < N-prev_length; i++){
				// find how much car should move using the controller and car model
				dist_inc = control_speed(ref_speed);
				// find the corresponding next point on the trajectory (using the traveled distance)
				vector<double> next_tmp = find_dxdy(dist_inc,xc,yc,spl);// updates xc, and yc in the car frame to find the next points					
				xc = next_tmp[0];
				yc = next_tmp[1];
				// update the next way points in the world frame 
				next_tmp = in_world_frame(car_x, car_y,car_yaw, xc,yc);
				next_x_vals.push_back(next_tmp[0]);
				next_y_vals.push_back(next_tmp[1]);
			}
			// apply the new trajectory
          	msgJson["next_x"] = next_x_vals;
          	msgJson["next_y"] = next_y_vals;

          	auto msg = "42[\"control\","+ msgJson.dump()+"]";

          	//this_thread::sleep_for(chrono::milliseconds(1000));
          	ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
          
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
