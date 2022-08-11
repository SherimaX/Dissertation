# This is a simplified version of the code that can be readily used inside one
# ROS node. 
# The class structure is as follows:
# 

import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

class Model:
# The controller simulates the dynamical system
    def __init__(self) -> None:
        # Parameters, check the main document
        self.a1 = 2.0
        self.a2 = 2.0
        self.b1 = 10.0
        self.b2 = 10.0
        self.sigma = 10.0
        self.epsilon = 0.1
        self.rho1 = -1.0
        self.rho2 = 1.0
        self.h = 0.01 # small time step
        # Component classes
        self.X0 = [0.2, 0.3, 1.0] # Initial state
        self.X = self.X0 # Current state, if you want to keep the time series, this should be an array
        self.time = 0.0
        self.eta1 = 0.0
        self.eta2 = 0.0

    def integrate( self, f ):
        # Perform a Runge-Kutta step
        k1 = f(self.time, self.X)
        k2 = f(self.time + self.h/2.0, self.X + self.h*k1/2.0)
        k3 = f(self.time + self.h/2.0, self.X + self.h*k2/2.0)
        k4 = f(self.time + self.h, self.X + self.h*k3)
        self.X = self.X + self.h*(k1 + 2*k2 + 2*k3 + k4)/6.0
        self.time = self.time + self.h


    def map(self, t, x, input_u = 0.0, input_v = 0.0 ):
        # The evolution map for the dynamical system
        g1 = lambda q1, q2, rho: -self.a1*q1 + self.b1*(1.0 - q1)*np.exp(-self.sigma*(rho - self.rho1)**2)*(1.0 + input_u)
        g2 = lambda q1, q2, rho: -self.a2*q2 + self.b2*(1.0 - q2)*np.exp(-self.sigma*(rho - self.rho2)**2)*(1.0 + input_v)
        f = lambda q1, q2, rho: -4.0*((rho - self.rho1)*(rho - self.rho2)*(rho - (self.rho1+self.rho2)/2.0) +
                                    (1-q1)*(rho - self.rho1)/2.0 + (1-q2)*(rho - self.rho2)/2.0)

        return np.array([self.epsilon*g1(x[0], x[1], x[2]),
                    self.epsilon*g2(x[0], x[1], x[2]),
                    f(x[0], x[1], x[2])])


    def step(self, r1, r2):
        # Performs one integration step updating the inputs and outputs of
        # the model

        f = lambda t, x: self.map( t, x, r1, r2 )
        self.integrate( f )

        b_sigma = 20.0
        eta1 = np.exp(-b_sigma*(self.X[2] - self.rho1)**2)
        eta2 = np.exp(-b_sigma*(self.X[2] - self.rho2)**2)

        return eta1, eta2
