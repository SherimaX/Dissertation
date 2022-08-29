# This is a simplified version of the code that can be readily used inside one
# ROS node. 
# The class structure is as follows:
# 

import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

class State(Enum):
# Enumeration of the possible states of the actionsystem
    SEARCH = 1
    FOLLOW = 2
    CONSUME = 3

class Percept:
# Represents a percept with useful information to locate it in egocentric space
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.depth = 0.0
        self.intensity = 0.0

class Controller:
# The controller simulates the dynamical system
    def __init__(self) -> None:
        # Parameters, check the main document
        self.a1 = 2.0
        self.a2 = 2.0
        self.b1 = 30.0
        self.b2 = 30.0
        self.sigma = 10.0
        self.epsilon = 0.01
        self.rho1 = -1.0
        self.rho2 = 1.0
        self.h = 0.005 # small time step
        # Component classes
        self.perceptualSystem = PerceptualSystem()
        self.hunger = HungerMotivationalSystem( self.perceptualSystem )
        self.thirst = ThirstMotivationalSystem( self.perceptualSystem )
        self.X0 =  [0.0, 1.0, 1.0] # Initial state
        self.X = X0 # Current state, if you want to keep the time series, this should be an array
        self.time = 0.0
        self.eta1 = 0.0
        self.eta2 = 0.0

    def integrate( self, f ):
        # Perform a Runge-Kutta step
        k1 = f(self.time, self.X)
        k2 = f(self.time + self.h/2.0, self.X + self.h*k1/2.0)
        k3 = f(self.time + self.h/2.0, X + self.h*k2/2.0)
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


    def step( self ):
    # Performs one integration step updating the inputs and outputs of
    # the model
        r1 = self.hunger.activate( self.eta1 )
        r2 = self.thirst.activate( self.eta2 )

        f = lambda t, x: self.map( t, x, r1, r2 )
        self.integrate( f )

        b_sigma = 20.0
        self.eta1 = np.exp(-b_sigma*(self.X[2] - self.rho1)**2)
        self.eta2 = np.exp(-b_sigma*(self.X[2] - self.rho2)**2)


class ActionSystem:
# The action system manages the seach/follow/consume pattern and logic
    def __init__( self ):
        self.state = State.SEARCH

    def search( self, motivationalSystem, eta ):
        # Modulate the search action by eta, you can choose to either set a hard
        # threshold or a soft activation in which the ghost of each motivational system
        # remains but feeble
        percept = motivationalSystem.perceive()
        motivationalSystem.express( eta )

        if percept is None:
            print('Doing search')
        else:
            return percept

    def follow( self, MotivationalSystem, eta ):
        motivationalSystem.express( eta )
        print('Pursuing the percept')
        percept = motivationalSystem.perceive()
        return percept

    def consume( self, motivationalSystem, eta ):
        # Consuming activates action patterns when the target has been achieved
        # In this case the function returns the consumed reward used as forcing term in
        # the dynamics
        motivationalSystem.express( eta )
        r = motivationalSystem.consume( eta )
        print('Consuming')
        return r


    def changeState( self, motivationalSystem ):
        # This function implements the finite state machine for the actions
        # The challenge is to define when something is close
        if motivationalSystem.currentPercept == None:
            self.state = State.SEARCH
        elif motivationalSystem.currentPercept.depth < 0.1: # Adjust the threshold
                self.state = State.CONSUME
        else:
            self.state = State.FOLLOW

class PerceptualSystem:
# This class is in charge of managing the raw inputs from the robot
# using visual system as an example
    def __init__(self):
        self.currentImage = None

class MotivationalSystem:
# The MotivationalSystem generates the motivational system's specific action patterns.
# This is an abstract class
    def __init__( self, perceptualSystem ):
        self.perceptualSystem = perceptualSystem
        self.actionSystem = ActionSystem()
        self.currentPercept = None

    def activate( self, eta ):
        if self.actionSystem.state == State.SEARCH:
            self.currentPercept = self.actionSystem.search( self, eta )
        elif self.actionSystem.state == State.FOLLOW:
            currentPercept = self.actionSystem.follow( self, eta )
        elif self.actionSystem.state == State.CONSUME:
            if eta > 0.5:
                r = self.actionSystem.consume( self, eta )
            else
                currentPercept = None

        self.actionSystem.changeState( self )
        return r

    def perceive( self ):
        # Override
        None

    def express( self ):
        # Override
        None

    def consume( self ):
        # Override
        r = 0
        return r

# Child classes for the specific motivational systems
class HungerMotivationalSystem(MotivationalSystem):
    def __init__( self, perceptualSystem ):
        MotivationalSystem.__init__(self, perceptualSystem )

    def perceive( self ):
        # Do the logic to detect the stimulus associated to this motivational system
        p = Percept()
        currentImage = self.perceptualSystem.currentImage

        return p # or None

    def express( self ):
        # Action patterns related to the motivational system, not directly related to movement
        None

    def consume( self ):
        # define what is the reward based on the interaction with the target
        r = 0
        return r

class ThirstMotivationalSystem(MotivationalSystem):
    def __init__( self, perceptualSystem ):
        MotivationalSystem.__init__(self, perceptualSystem )

    def perceive( self ):
        # Do the logic to detect the stimulus associated to this motivational system
        p = Percept()
        currentImage = self.perceptualSystem.currentImage

        return p # or None

    def express( self ):
        # Action patterns related to the motivational system, not directly related to movement
        None

    def consume( self ):
        # define what is the reward based on the interaction with the target
        r = 0
        return r


