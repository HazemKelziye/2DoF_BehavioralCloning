# 2 Degrees-of-Freedom AI Controller
This is a continuation of my previous project which is called 2DoF_PID, which implements a PID controller. 
In this implementation of 2DoF AI controller Behavioral Cloning is used to land the rocket using the two action variables (Throttle & Thrusters)
The actions are discrete.

The action_y is responsible for controlling the vertical and horizontal positioning of the rocket
using a 2D Convolutional Neural Network, by cloning the behavior of the expert system i.e. the PID controller. And hence the rocket will be drived toward the center of the platform.
As for the other DoF which is FOR NOW! the PIDtheta controller it maintained the rocket to be at equilibrium above the target
point i.e. (0,-1) and if it's not above that point it will tilt the rocket to make it move towards the target point,
the setpoint was {pi/4 * (x + Vx)}. The PIDtheta controller's gain parameters are Kp = 1,000 , Ki = 2.5, Kd = 750.


For more clear demonstrations please refer to this link => https://youtu.be/NsNVTk2JRKE

INCLUDE THE NN ARCHITECTURE

Here are the responses of the system, for multiple random landing samples;

![Figure_1](https://github.com/Hazem-Kelziye/2DoF_BehavioralCloning/assets/147067179/73152441-8a42-42c7-966e-d6fa32fa3e22)
![Figure_2](https://github.com/Hazem-Kelziye/2DoF_BehavioralCloning/assets/147067179/a7ebc58e-f021-4b02-9376-2877a23adefc)
