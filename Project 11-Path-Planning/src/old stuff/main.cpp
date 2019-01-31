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
#include "planner.h"
#include "spline.h"

#define dT 0.02 //20 milliseconds is the sample rate of the loop
#define MAX_SPEED 22.0 // meters per second
#define MAX_ACC 9.5 // meters per second^2
#define MAX_JERK 9.5 // meters per second^3
#define MAX_COST 1000.0// for the cost functions associated with the finiate state machine decisions
 
using namespace std;

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }
double miph2mps(double x) { return x * 0.44704; }
double mps2miph(double x) { return x * 2.23694; }


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

	double heading = atan2( (map_y-y),(map_x-x) );

	double angle = abs(theta-heading);

	if(angle > pi()/4)
	{
		closestWaypoint++;
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



//calculate the cost of finite state machine: keep lane: KL (0), lane change left: LCL (1), lane change right: LCR (2)

double gaussian_cost(double ds){
	double sigma = 5;
	return MAX_COST*1/(sqrt(2*M_PI)*sigma)*exp(-ds*ds/(2*sigma*sigma));
}

vector<int> find_cars(double car_s, double car_d, vector<vector<double>> cars, string mode){
	vector<int> results;

	if (mode == "front"){
		int front_car = 200;
		double closest_car_dist = 500;
		for (int i = 0; i < cars.size(); i++){
			double dist = cars[i][5] - car_s;
			if ( fabs(car_d - cars[i][6]) < 2 && dist < 30 && dist > 0 && dist <closest_car_dist){
				front_car = i;
				closest_car_dist = dist;
			}
		}
	  	results.push_back(front_car);
	}
	if ( mode == "right"){
		double lane_l = ceil(car_d/4.0)*4.0;
		double lane_r = lane_l + 4.0;
//		cout << "right lane between: " << lane_l << " and " << lane_r << endl;
		for (int i = 0; i < cars.size(); i++){
			double car_di = cars[i][6];
			if (car_di > lane_l && car_di < lane_r)
				results.push_back(i);
		}			
	}
	if ( mode == "left"){
		double lane_r = floor(car_d/4.0)*4.0;
		double lane_l = lane_r - 4.0;
//		cout << "left lane between: " << lane_l << " and " << lane_r << endl;	
		for (int i = 0; i < cars.size(); i++){
			double car_di = cars[i][6];
			if (car_di > lane_l && car_di < lane_r)
				results.push_back(i);
		}			

	}

//	results.push_back(closest_car_dist);
	return results;

}


double xc = 0;
double yc = 0;
double ac = 0; // current acceleration
double vc = 0;
double prev_e = 0; // previous velocity error

double saturate(double val, double max_val){
	if ( val > max_val)
		return max_val;
	if ( val < -max_val)
		return -max_val;
	else
		return val;
}

double control_speed(double ref_speed){
	double dist = 0;
	double kp = 0.0025;
	double kd = 0.05;
	if (ref_speed == MAX_SPEED){
		kp = 1000;
		kd = 0.02;
	}
	double e = ref_speed - vc;
//	double edot = (e - prev_e)/dT;
//	edot = saturate(edot, MAX_ACC/dT);
	
	//double jc = kd*edot + kp*e;	
	//jc = saturate(jc, MAX_JERK);
	//double an = jc*dT + ac;
	//an = saturate(an, MAX_ACC);
	double m = 1000;
	double b = 50;
	
	double vn = (m*vc + (dT*kp)*e )/(b*dT+m);

	vn = saturate(vn, MAX_SPEED);


	// update the initial values for the next loop
	dist = vn*dT;	
//cout <<"ref: "<<ref_speed<< "vc: " << vc<<" e: "<< e<< " edot "<< edot <<" jc: "<<  jc <<" ac: " << ac << " vc: " << vc<< " dd: " << dist<<endl;
	cout <<"ref: "<<ref_speed<< "vc: " << vc<<" e: "<< e<< " vn: " << vn<< " dd: " << dist<<endl;
	prev_e = e;	
	vc = vn;
	//ac = an;
	return dist;
}
//find_dxdy

vector<double> find_dxdy(double dist, double ptx, double pty, tk::spline s){
	bool no_next_point = true;
	double eps_x = 0.5*dist;
	double xn = 0;
	double yn = 0;				 
	while(no_next_point){				
		xn = ptx + eps_x;
		yn = s(xn);
		double eps_y = yn - pty;
		double d = sqrt(eps_x*eps_x+eps_y*eps_y);
		double lambda = 0.002;
		if (d > dist){
			eps_x *= (1-lambda);
		}else if(dist - d > 0.02){
			eps_x *= (1+lambda);
		}else{
//			cout<< "point found at: " << xn << "  " << yn << endl;
			no_next_point = false;
		}  
	}

	return {xn,yn};
}

vector<double> calc_cost(double car_s, double car_d, double car_speed, vector<vector<double>> cars){

	double KL_cost = 0.0;
	double LCL_cost = 0.0;
	double LCR_cost = 0.0;
	vector<double> costs= {0,0,0};

	KL_cost = (car_speed - MAX_SPEED)*(car_speed - MAX_SPEED);

	if( car_d > 9 && car_d < 12){
		LCR_cost = MAX_COST;
	}else{
		vector<int> right_cars = find_cars(car_s, car_d, cars, "right");
		for (int i = 0; i < right_cars.size(); i++){
			double ds = car_s - cars[right_cars[i]][5];
			LCR_cost += gaussian_cost(ds);
		}
	}
	if( car_d > 0 && car_d < 3){
		LCL_cost = MAX_COST;
	}else{
		vector<int> left_cars = find_cars(car_s, car_d, cars, "left");
		for (int i = 0; i < left_cars.size(); i++){
			double ds = car_s - cars[left_cars[i]][5];
			LCL_cost += gaussian_cost(ds);
		}
	}
	costs[0] = min(KL_cost, MAX_COST);
	costs[1] = min(LCL_cost, MAX_COST);
	costs[2] = min(LCR_cost, MAX_COST);
	return costs;
	
}

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

	
	double car_speed_prev = 0;
	int ctr =0;
	
  h.onMessage([&map_waypoints_x,&map_waypoints_y,&map_waypoints_s,&map_waypoints_dx,&map_waypoints_dy, &car_speed_prev, &ctr](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
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
			
			
			double car_acc = (car_speed - car_speed_prev)/dT;
			car_speed_prev = car_speed;
			
		
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
			double dist_inc = 0.43;
			double traj_length = 2;
						
			//if no the previous path does not exist, initialize it using the current position of the car
			vector<double> curve_x;
			vector<double> curve_y;
			int path_length_prev = previous_path_x.size();
			//cout << "path_length_prev "<< path_length_prev<<endl;
			if ( path_length_prev == 0){
				double car_x_next = car_x + cos(car_yaw);
				double car_y_next = car_y + sin(car_yaw);
				curve_x.push_back(car_x);	
				curve_x.push_back(car_x_next);
				curve_y.push_back(car_y);	
				curve_y.push_back(car_y_next);


			}else{
				curve_x.push_back(car_x);	
				curve_x.push_back(previous_path_x[0]);
//				curve_x.push_back(previous_path_x[1]);	
				curve_y.push_back(car_y);	
				curve_y.push_back(previous_path_y[0]);
//				curve_y.push_back(previous_path_y[1]);	

			}		
			vector<double> all_costs = calc_cost(car_s, car_d, car_speed, sensor_fusion);
			//cout << "KL cost: " << all_costs[0]<< "    LCL cost: " << all_costs[1]<< "    LCR cost: " << all_costs[2]<< endl;
			
			double target_lane = 1;
			double min_cost = MAX_COST;
			for (int i = 0 ; i < all_costs.size(); i++){
				if (all_costs[i] < min_cost)
					target_lane = 1;
			}
			//ctr ++;	
			//if (ctr > 150)
			//	target_lane = 2;
			double target_d = target_lane*4 + 2;
			double target_move = 30;//traj_length*MAX_SPEED/3;
			
			//produce three distant waypoints
			for(int i = 1; i < 4; i++){
				vector<double> next_curve = getXY(car_s+i*target_move,target_d, map_waypoints_s,map_waypoints_x,map_waypoints_y);
				curve_x.push_back(next_curve[0]);
				curve_y.push_back(next_curve[1]);
			}
			for (int i = 0; i < curve_x.size();i++){
				cout <<"before: x= "<< curve_x[i]<< " y= "<<curve_y[i]<<endl;
				vector<double> transformed = in_car_frame(car_x,car_y,car_yaw,curve_x[i],curve_y[i]);
				curve_x[i] = transformed[0];
				curve_y[i] = transformed[1];
				//cout <<"after: x= "<< curve_x[i]<< " y= "<<curve_y[i]<<endl;

			}
			vector<int> front_car = find_cars(car_s,car_d, sensor_fusion, "front");
			/*vector<int> left_cars = find_cars(car_s,car_d, sensor_fusion, "left");
			for (int jj = 0 ; jj < left_cars.size(); jj++)
				cout << " left: " << left_cars[jj];
			cout << endl;
			vector<int> right_cars = find_cars(car_s,car_d, sensor_fusion, "right");
			for (int jj = 0 ; jj < right_cars.size(); jj++)
				cout << " right: " << right_cars[jj];
			cout << endl;
			*/
			
			double ref_speed =0;
			if (front_car[0] != 200){
				double front_car_s =  sensor_fusion[front_car[0]][5];
				cout << "front car: " << front_car[0]<< " distance: " <<front_car_s - car_s<< endl;
				int car_no = (int) front_car[0];
				double vx = sensor_fusion[car_no][3];
				double vy = sensor_fusion[car_no][4];
				ref_speed = sqrt(vx*vx + vy*vy);
			}else{
				ref_speed = MAX_SPEED;
			}
			tk::spline s;
			s.set_points(curve_x,curve_y);
			// vc = car_speed=>control_speed(ref_speed);
			//xc = 0, yc =0=>vector<double> find_dxdy(dist,s); vector<double> next_tmp = in_world_frame(car_x, car_y,car_yaw, xc,yc);
			double xc = 0;
			double yc = 0;
			vc = car_speed;
			double ac = car_acc;
			cout << ac<< " " << vc<< endl;
			for(int i = 0; i < traj_length/dT; i++)
			{

				double delta_d = control_speed(ref_speed);
				vector<double> next_tmp = find_dxdy(delta_d,xc,yc,s);// updates xc, and yc in the car frame to find the next points
				xc = next_tmp[0];
				yc = next_tmp[1];
				next_tmp = in_world_frame(car_x, car_y,car_yaw, xc,yc);
				//cout << "step: "<< i<< " deltad: "<< delta_d << " xcyc: " << xc << "  "<< yc << endl; 
				next_x_vals.push_back(next_tmp[0]);
			 	next_y_vals.push_back(next_tmp[1]);
				/*bool no_next_point = true;
				double eps_x = 0.5*MAX_SPEED*dT;				 
				while(no_next_point){
									
					double v_max = MAX_ACC*dT + vc;
					v_max = min(MAX_SPEED,v_max);
					double xn = xc+eps_x;
					double yn = s(xn);
					double eps_y = yn - yc;
					double v_est = sqrt(eps_x*eps_x+eps_y*eps_y)/dT;
					double lambda = 0.01;
					//cout << "eps: "<< eps_x <<" v_max: " << v_max << " v_est" << v_est<<" eps_x: "<<eps_x<<" eps_y: "<< eps_y	<<endl;	
					if (v_est > v_max){
						eps_x *= (1-lambda);
					}else if(v_max - v_est > 0.1){
						eps_x *= (1+lambda);
					}else{
						//cout<< "point " << i+1<< " found"<< endl;
						vc = v_est;
						xc = xn;
						yc = yn;	
						vector<double> next_tmp = in_world_frame(car_x, car_y,car_yaw, xc,yc);
						next_x_vals.push_back(next_tmp[0]);
					 	next_y_vals.push_back(next_tmp[1]);
						no_next_point = false;
					}  
				}*/
					
 			}
			//cout << next_x_vals.size()<<endl;
				  //vector<double> next_tmp = getXY(car_s+dist_inc*i,6,map_waypoints_s,map_waypoints_x,map_waypoints_y);
				  
				  //
				  
					//  next_x_vals.push_back(next_tmp[0]);
					 // next_y_vals.push_back(next_tmp[1]);
					
			
			//cout << car_speed<< "   "<<car_acc<<endl;
          	// TODO: define a path made up of (x,y) points that the car will visit sequentially every .02 seconds
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


/*vector<double> start = {car_s, miph2mps(car_speed), 0};
			vector<double> end = {car_s+0.7*traj_length*MAX_SPEED,0.9*MAX_SPEED,0};
			vector<double> poly_const = JMT(start,end, traj_length);
			cout << poly_const[0]<<"	" << poly_const[1]<<"	" << poly_const[2]<<"	" << poly_const[3]<<"	" << poly_const[4]<<"	" << poly_const[5]<< endl;
			vector<double> fil_prev ={car_x,car_y};

*/

 /*double t1 = dT*i;
				  double t2 = t1*t1;
				  double t3 = t2*t1;
				  double t4 = t3*t1;
				  double t5 = t4*t1;
				  double s_c = poly_const[0] + poly_const[1]*t1 + poly_const[2]*t2 + poly_const[3]*t3+ poly_const[4]*t4 + poly_const[5]*t5; 
				  double s_c_dot = poly_const[1] + 2*poly_const[2]*t1 + 3*poly_const[3]*t2+ 4*poly_const[4]*t3 + 5*poly_const[5]*t4; 
					*/


 //vector<double> next_tmp = getXY(s_c,6,map_waypoints_s,map_waypoints_x,map_waypoints_y);
				 // double tau = 0.2;
				 // next_tmp[0] = (1-tau)*next_tmp[0] + tau*fil_prev[0];
				 // next_tmp[1] = (1-tau)*next_tmp[1] + tau*fil_prev[1];
				  //fil_prev = next_tmp;