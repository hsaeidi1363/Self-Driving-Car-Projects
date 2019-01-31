#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "MPC.h"
#include "json.hpp"

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }
double mph2mps(double x) { return x * 0.447; } // convert from mile/hour to meter/second



// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.rfind("}]");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

// Evaluate a polynomial.
double polyeval(Eigen::VectorXd coeffs, double x) {
  double result = 0.0;
  for (int i = 0; i < coeffs.size(); i++) {
    result += coeffs[i] * pow(x, i);
  }
  return result;
}

// Fit a polynomial.
// Adapted from
// https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals,
                        int order) {
  assert(xvals.size() == yvals.size());
  assert(order >= 1 && order <= xvals.size() - 1);
  Eigen::MatrixXd A(xvals.size(), order + 1);

  for (int i = 0; i < xvals.size(); i++) {
    A(i, 0) = 1.0;
  }

  for (int j = 0; j < xvals.size(); j++) {
    for (int i = 0; i < order; i++) {
      A(j, i + 1) = A(j, i) * xvals(j);
    }
  }

  auto Q = A.householderQr();
  auto result = Q.solve(yvals);
  return result;
}

int main() {
  uWS::Hub h;

  // MPC is initialized here!
  MPC mpc;

  h.onMessage([&mpc](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    string sdata = string(data).substr(0, length);
    cout << sdata << endl;
    if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2') {
      string s = hasData(sdata);
      if (s != "") {
        auto j = json::parse(s);
        string event = j[0].get<string>();
        if (event == "telemetry") {
          // j[1] is the data JSON object
          vector<double> ptsx = j[1]["ptsx"];
          vector<double> ptsy = j[1]["ptsy"];
          double px = j[1]["x"];
          double py = j[1]["y"];
          double psi = j[1]["psi"];
          double v = mph2mps(j[1]["speed"]); // read and convert the velocity to meter/second
          // also read the control inputs
          double delta = j[1]["steering_angle"];
          double acc = j[1]["throttle"];
	  Eigen::VectorXd ptsx_vec(ptsx.size());
	  Eigen::VectorXd ptsy_vec(ptsy.size());
	  // compensating for the delay (predict 100 milliseconds into the future and use the results as initial values for the MPC
	  double delay = 0.15; // 150 ms delay
	  double dt = 0.05; 
	  for (double k = 0; k < int(delay/dt); ++k){
		  px += v*cos(psi)*dt;
		  py += v*sin(psi)*dt;
		  psi -= v*delta*deg2rad(25)/2.67*dt; // steering_angle = 1 means 25 degrees (0.44 radians)
		  v += acc*dt*5.0; // 5.0 is a multiplier (max acceleration) explained in details in the MPC.cpp file
	  }
          // preparing the cure fit points in the vector format (I transform all of the readings to the car's body frame for easier calculations)
          for (unsigned int k = 0; k < ptsx.size(); ++k){
		double dx = ptsx[k] - px;
		double dy = ptsy[k] - py;
		ptsx_vec[k] = dx * cos(psi) + dy * sin(psi);
		ptsy_vec[k] = - dx * sin(psi) + dy * cos(psi);
	  }
	  // find the 3rd order fit
          auto coeffs=  polyfit(ptsx_vec, ptsy_vec, 3);
	  // since I transfered the readings to the body frame, px and py are 0 in this coordinates and cte and the desired angles are simplified to
	  double cte = polyeval(coeffs, 0);
	  double epsi = -atan(coeffs[1]);// derivative of coeffs[0] + coeffs[1] * x +coeffs[2] * x^2 + coeffs[3] * x^3-> coeffs[1] + 2*coeffs[2] * x + 3*coeffs[3] * x^2 here evalutated at x = 0=> gives only coeffs[1]
	  Eigen::VectorXd states(6);
	  // use the body frame coordinate and save the initial states
	  states << 0.0, 0.0, 0.0, v, cte, epsi;

          /*
          * TODO: Calculate steering angle and throttle using MPC.
          *
          * Both are in between [-1, 1].
          *
          */
          double steer_value;
          double throttle_value;
          // solve the mpc problem and return the first control inputs as well as the predicted x,y trajectories
	  auto mpc_out = mpc.Solve(states, coeffs);
	  // normalize delta and apply the negation (since delta in the simulator has an opposite sign compared to the mathematical models
          steer_value = -mpc_out[0]/deg2rad(25);
	  // normalize the throttle with the max value chosen in the MPC file
          throttle_value = mpc_out[1]/5.0;

          json msgJson;
          // NOTE: Remember to divide by deg2rad(25) before you send the steering value back.
          // Otherwise the values will be in between [-deg2rad(25), deg2rad(25] instead of [-1, 1].
          msgJson["steering_angle"] = steer_value;
          msgJson["throttle"] = throttle_value;


          //Display the MPC predicted trajectory 
          vector<double> mpc_x_vals;
          vector<double> mpc_y_vals;

          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Green line
          int N = (mpc_out.size()-2)/2; // prediciton horizon obtained from the outputs of MPC (delta[k], throttle[k], x[k],...x[k+N], y[k]...y[k+N]
	  for (int k=0; k < N; ++k){
		mpc_x_vals.push_back(mpc_out[2+k]);
		mpc_y_vals.push_back(mpc_out[2+k+N]);
	  }


          msgJson["mpc_x"] = mpc_x_vals;
          msgJson["mpc_y"] = mpc_y_vals;

          //Display the waypoints/reference line
          vector<double> next_x_vals;
          vector<double> next_y_vals;

          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Yellow line
          for (unsigned int k = 0; k < ptsx.size() ; ++k){
		// use the polynomial to find the desired trajectory (yellow line)
		next_x_vals.push_back(ptsx_vec[k]);
		next_y_vals.push_back(polyeval(coeffs,ptsx_vec[k]));
	  }
      
          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;


          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          std::cout << msg << std::endl;
          // Latency
          // The purpose is to mimic real driving conditions where
          // the car does actuate the commands instantly.
          //
          // Feel free to play around with this value but should be to drive
          // around the track with 100ms latency.
          //
          // NOTE: REMEMBER TO SET THIS TO 100 MILLISECONDS BEFORE
          // SUBMITTING.
          this_thread::sleep_for(chrono::milliseconds(100));
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
