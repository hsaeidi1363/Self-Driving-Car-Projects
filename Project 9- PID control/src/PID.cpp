#include "PID.h"

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {
   p_error = 0;
   i_error = 0;
   d_error = 0;
   total_error = 0;


   c1 = 0.08208;
   c2 = 5.0;
   c3 = 4.541;

   comp_in_prev = 0;
   comp_o_prev = 0;
}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {
   this->Kp = Kp;
   this->Kd = Kd;
   this->Ki = Ki;   

}

double PID::UpdateControl(double cte) {
   this->d_error = cte-this->p_error; // calculate the derivative term
   this->i_error += cte; // calculate the integral term
   this->p_error = cte; // store the previous value of error
   
 
   bool comp_active = true; // change to false to deactivate the compensator
   double pid_o = -(Kp*cte + Ki*this->i_error + Kd*this->d_error); // calculate the PID controller output 
  
   if (comp_active){
	double comp_in = pid_o; 
	double comp_o;
	comp_o = c1*comp_o_prev+c2*comp_in-c3*comp_in_prev;
	comp_in_prev = comp_in;
	comp_o_prev = comp_o;
	return comp_o;	
   }else{// if the compensator is not active just return the PID output
	return pid_o;
   } 
}


