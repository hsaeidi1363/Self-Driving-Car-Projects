#ifndef PID_H
#define PID_H

class PID {
public:
  /*
  * Errors
  */
  double p_error;
  double i_error;
  double d_error;
  double total_error;

  /*
  * Coefficients
  */ 
  double Kp;
  double Ki;
  double Kd;
  /*
  * Phase-lead compensator parameters. Please see the report for the details of design and the logic behind this compensators
  */ 
  double c1;
  double c2;
  double c3;
  double comp_in_prev;
  double comp_o_prev;  

  /*
  * Constructor
  */
  PID();

  /*
  * Destructor.
  */
  virtual ~PID();

  /*
  * Initialize PID.
  */
  void Init(double Kp, double Ki, double Kd);

  /*
  * Update the PID error variables given cross track error and calculate the control input.
  */
  double UpdateControl(double cte);

  /*
  * Calculate the total PID error.
  */
  double TotalError();
};

#endif /* PID_H */
