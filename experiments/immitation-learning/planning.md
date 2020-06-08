# Goal

produce an NN policy that acts exactly like the current Static control agent.
it should be a swap-in replacement.

# Feature engeneering

The NN is stateless, therefore it is important that the inputs features are augmented

- setpoint and cliped setpoint
- integrated error (which the pi will produce)

# Challenges

- nested controller structure needs to be flatened, such that we only have a vector of inputs and a vector of outputs


# Training Data I

in order to train an NN, we need supervised training data.
To this end we could the state-action history of the agent with the real environment.

## Perturbations
rl1 value should change should be the primary change    

normally stable params:
setpointI, voltage, frequency


These perturbations are important, because we need to create data that captures as much of the agents behavioural reportiouar as possible.
It is also important to cover different values of the integration error. maybe random restarts could be also an option

## recording process 

most of the recording can be easily done with the stock runner, agents and environment using their history files
the perturbations and setpoint modifications can be implemented inside an agent wrapping the observe function of the staticctrl agent
no changes to runner or env are needed.

# Training Data II
instead of perturbating the environment, we coul directly generate random input data including the
 integration error and the clipped setpoint. all values could come from a very wide gaussean distribution.
 
Here the Wrapper agent must provide the internal controllers with the additional features like integral error
The NN wrapper on the otherhand must additionally calculate the correct values for that inputs before passing them to the NN

## Problem
creating such a wrapper that regenerates the PI controllers actions is hightly non-trivial