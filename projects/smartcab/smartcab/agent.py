import random
import math
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import defaultdict
from helper import calculate_safety, calculate_reliability, evaluate_results

class LearningAgent(Agent):
    """ An agent that learns to drive in the Smartcab world.
        This is the object you will be modifying. """ 

    def __init__(self, env, learning=False, epsilon=1.0, alpha=0.5):
        super(LearningAgent, self).__init__(env)     # Set the agent in the evironment 
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions

        # Set parameters of the learning agent
        self.learning = learning # Whether the agent is expected to learn
        self.Q = dict()          # Create a Q-table which will be a dictionary of tuples
        self.epsilon = epsilon   # Random exploration factor
        self.alpha = alpha       # Learning factor

        ###########
        ## TO DO ##
        ###########
        # Set any additional class parameters as needed
        self.max_trail = 800
        self.prev_state = None
        self.prev_action = None
        self.prev_reward = None
        self.def_Q = 0
        self.trial_number = 1

        
        #  We'll use defaultdict and by default it is initialized to 0
        #  Initial states include 'light status', 'oncoming traffic' 

        for i in ['green', 'red']:
            for j in [None, 'forward', 'left', 'right']:
                for k in self.env.valid_actions:
                    self.Q[(i, j, k)] = defaultdict(int)




    def reset(self, destination=None, testing=False):
        """ The reset function is called at the beginning of each trial.
            'testing' is set to True if testing trials are being used
            once training trials have completed. """

        # Select the destination as the new location to route to
        self.planner.route_to(destination)

        self.trial_number += 1
        print "trial_number is...", self.trial_number
        
        ########### 
        ## TO DO ##
        ###########
        # Update epsilon using a decay function of your choice
        # Update additional class parameters as needed
        # If 'testing' is True, set epsilon and alpha to 0
        if testing:
            self.epsilon = 0
            self.alpha = 0
        else:
            #self.epsilon -= 0.05
            self.epsilon = math.cos(self.trial_number * self.alpha)#math.exp(-0.1* self.trial_number)   # 1/k (epsilon * 1/k)

            # 1/(math.pow(self.trial_number, 2))
            # math.pow(self.alpha,self.trial_number)

        self.gamma = 0.3  # gamma determines how much future reward is valued.  0 means immediate reward
        self.prev_state = None
        self.prev_action = None
        self.prev_reward = None

        

        return None

    def build_state(self):
        """ The build_state function is called when the agent requests data from the 
            environment. The next waypoint, the intersection inputs, and the deadline 
            are all features available to the agent. """

        # Collect data about the environment
        waypoint = self.planner.next_waypoint() # The next waypoint 
        inputs = self.env.sense(self)           # Visual input - intersection light and traffic
        deadline = self.env.get_deadline(self)  # Remaining deadline

        ########### 
        ## TO DO ##
        ###########
        # Set 'state' as a tuple of relevant data for the agent
        
        
        # Set the state as a tuple and inlcude 'light status', 'oncoming traffic' and 'waypoint'
        self.state = (inputs['light'], inputs['oncoming'], waypoint)

        #print ("what is self.state (build_state)...", self.state)
        #print ("what is inputs.values (build_state)", tuple(inputs.values()))

        #state = tuple(inputs.values())

        #print ("BUILD STATE: what is state + type...", state, type(state))

        return self.state


    def get_maxQ(self, state):
        """ The get_max_Q function is called when the agent is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """

        ########### 
        ## TO DO ##
        ###########
        # Calculate the maxmimum Q-value of all actions for a given state

        maxQ_action = None
        maxQ_value = 0

        '''
        print "******** GET_MAXQ ********"
        print "what is whole self.Q..", self.Q
        print "what is state (get_maxQ)...", state
        print "what is self.state (get_maxQ)...", self.state

        print "MAX_Q: action and value" +  str(maxQ_action) + ',' + str(maxQ_value) 
        print "MAX_Q: what is self.Q[self.state]...", self.Q[self.state]
        print "length of self.Q..", len(self.Q)

        print "length of self.Q[state]", len(self.Q[state])
        #print "??? test max ???", max(self.Q[self.state].values())
        print "******** ***** ** ************"
        '''
        if len(self.Q[state]) > 0:
            print "(get_maxQ) result...", max(self.Q[state].values())
            maxQ_value = max(self.Q[state].values())
        else:
            maxQ_value = 0
        #print ("what is maxQ_value..", maxQ_value)
        return maxQ_value


    def createQ(self, state):
        """ The createQ function is called when a state is generated by the agent. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, check if the 'state' is not in the Q-table
        # If it is not, create a new dictionary for that state
        #   Then, for each action available, set the initial Q-value to 0.0


        return


    def choose_action(self, state):
        """ The choose_action function is called when the agent is asked to choose
            which action to take, based on the 'state' the smartcab is in. """

        # Set the agent state and default action
        self.state = state
        self.next_waypoint = self.planner.next_waypoint()
        action = None

        ########### 
        ## TO DO ##
        ###########
        # When not learning, choose a random action
        # When learning, choose a random action with 'epsilon' probability
        #   Otherwise, choose an action with the highest Q-value for the current state

        # a valid action is one of None, (do nothing) 'Left' (turn left), 'Right' (turn right), or 'Forward' (go forward)
        # Note that you have access to several class variables that will help you write this functionality, such as 'self.learning' and 'self.valid_actions
        if self.learning == False:
            # choose a 
            action = random.choice(self.valid_actions)

        else: # when learning
            #  choose a random action with 'ipsilon' probability
            if self.epsilon > random.random():
                action = random.choice(self.valid_actions)
            else:
                # get action with highest Q-value for the current state
                maxQ = self.get_maxQ(state)

                best_actions = []
                print "choose_action..self.Q[state]..", self.Q[state]
                for act in self.Q[state]:
                    print "choose_action..act..", act
                    if self.Q[state][act] == maxQ:
                        best_actions.append(act)
                if (len(best_actions) > 0):
                    action = random.choice(best_actions)

        print "action taken in choose_action...", str(action)
 
        return action


    def learn(self, state, action, reward):
        """ The learn function is called after the agent completes an action and
            receives an award. This function does not consider future rewards 
            when conducting learning. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, implement the value iteration update rule
        #   Use only the learning rate 'alpha' (do not use the discount factor 'gamma')

        next_state = self.build_state()


        if self.learning:

            print "(learn)what is self.state (learn)..", self.state
            print "(learn)what is state (learn)..", state
            print "(learn)what is action (learn)..", action

            currentQ = self.Q[state][action]
            print "(learn) what is currentQ....?", currentQ

            self.Q[state][action] = reward*self.alpha + currentQ*(1-self.alpha)
            print "(learn), what is after learning...", self.Q[state][action] 

        return 

    '''
    def learn(self, curr_s, curr_a, reward):
    next_s = self.build_state()
    self.createQ(next_s)
    curr_q = self.Q[curr_s][curr_a]
    delta = reward + max(self.Q[next_s][a] for a in self.valid_actions)
    self.Q[curr_s][curr_a] = (1.0 - self.alpha)*curr_q + self.alpha*delta
    '''


    def update(self):
        """ The update function is called when a time step is completed in the 
            environment for a given trial. This function will build the agent
            state, choose an action, receive a reward, and learn if enabled. """

        state = self.build_state()          # Get current state
        self.createQ(state)                 # Create 'state' in Q-table
        action = self.choose_action(state)  # Choose an action
        reward = self.env.act(self, action) # Receive a reward
        self.learn(state, action, reward)   # Q-learn

        return
        

def run():
    """ Driving function for running the simulation. 
        Press ESC to close the simulation, or [SPACE] to pause the simulation. """

    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    env = Environment(verbose=True, grid_size=(8,6))
    
    ##############
    # Create the driving agent
    # Flags:
    #   learning   - set to True to force the driving agent to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1
    #    * alpha   - continuous value for the learning rate, default is 0.5
    agent = env.create_agent(LearningAgent, learning = True, alpha = 0.6, epsilon = 1.0)
    #agent.learning = True
    #agent.epsilon = 1
    #agent.alpha = 0.5

    
    ##############
    # Follow the driving agent
    # Flags:
    #   enforce_deadline - set to True to enforce a deadline metric
    env.set_primary_agent(agent, enforce_deadline = True)
    env.testing = True
    #env.enforce_deadline = True

    ##############
    # Create the simulation
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    #   log_metrics  - set to True to log trial and simulation results to /logs
    #   optimized    - set to True to change the default log file name
    sim = Simulator(env, update_delay = 0.01, log_metrics = True, display = False, optimized = False)
    sim.testing = True
    #sim.update_delay = 0.01, 2.0
    #sim.log_metrics = True
    #sim.display = False
    #sim.optimized = True
    
    ##############
    # Run the simulator
    # Flags:
    #   tolerance  - epsilon tolerance before beginning testing, default is 0.05 
    #   n_test     - discrete number of testing trials to perform, default is 0
    #sim.n_test = agent.max_trail
    sim.run(tolerance = 0.6, n_test = agent.max_trail)

'''
def get_opt_result():
    accepted_ratings = ["A+"]
    safety_rating, reliability_rating = evaluate_results('sim_improved-learning.csv')
    # log evaluation to ratings.txt
    f = open('smartcab/ratings.txt', 'a')
    print >> f, "Rating results \n"
    print >> f, safety_rating
    print >> f, reliability_rating
    f.close()
    if safety_rating not in accepted_ratings or reliability_rating not in accepted_ratings:
        run()
        return get_opt_result()
'''
if __name__ == '__main__':
    #get_opt_result()
    run()
