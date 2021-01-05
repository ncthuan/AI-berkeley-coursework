## Q1 Value Iteration
* function: **__init_** takes a MDP on initialization and runs basic value iteration
* function: **computeQValueFromValues(state, action)** returns the Q-value of the (state, action) pair given by the value function given by self.values
```py
def computeQValueFromValues(self, state, action):
	value = 0.0
    for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
        value += prob * (self.mdp.getReward(state, action, next_state) + self.discount * self.values[next_state]) 
    return value 
```
<img src="https://latex.codecogs.com/gif.latex?Q(s,a)=\sum\limits_{s'}T(s,a,s')[R(s,a,s')+\gamma%20V(s')]"/> <br>
The function is implemented just as the equation above, T: prob, R: getReward(, gamma: discount, V: values the next state.
* function: **computeActionFromValues(state)** returns the best action according to the value function given by function computeQValueFromValues(state, action). By considering all possible actions and its q-value, select the best action with the max value.


## Q2 Bridge crossing
Noise refers to the uncertainty, to how often an agent ends up in an unintended successor state when they perform an action.<br>
So in this case of the bridge, for the agent to learn the policy straight to the goal quickly, there should be less noise/uncertainty so that the agent won't fell. So I eventually dropped it to 0.<br>
I just tune and tweak the params until the desired result is met. 


## Q4 Q-Learning
pretty much standard q-learning algorithms<br>
the update function:
```py
def update(self, state, action, nextState, reward):
	sample = reward + self.discount * self.computeValueFromQValues(nextState)
    self.q_values[(state, action)] = (1.0 - self.alpha) * self.getQValue(state, action) + self.alpha * sample
```
<img src="https://latex.codecogs.com/gif.latex?sample=R(s,a,s')+\gamma\max\limits_{a'}Q(s',a')"/> <br>
<img src="https://latex.codecogs.com/gif.latex?Q(s,a)=(1-\alpha)Q(s,a)+\alpha[sample]"/> <br>


## Q5 Epsilon greedy action selection
I used the suggested function **util.flipCoin(p)**, which returns True with probability p and False with probability 1-p, to switch between random action and best action.
```py
if util.flipCoin(self.epsilon):
    action = random.choice(legalActions)
else:
    action = self.computeActionFromQValues(state)
```


## Q6 Bridge Crossing Revisited
It's impossible. The agent was only wandering around and not heading to the goal. Eventhough the epsilon is set from 1 down to 0, which means the agent is learning with selecting the best action, the right policy is still not learned after 50 iterations.

## Q7 Q-Learning and Pacman
No further coding required


## Q8 Approximate Q-Learning
implement 2 functions: getQValue() and update()

```py
def getQValue(self, state, action):
	# returns Q(s,a)
	# __mul__ two Counter()s results in a sum of multiplications of key-matched values
    return self.weights * self.featExtractor.getFeatures(state, action)
```
<img src="https://latex.codecogs.com/gif.latex?Q(s,a)=\sum\limits_{i=1}^n%20f_i(s,a)%20w_i"/><br>
In this getQValue function, I can just multiply the weights directly because Counter() mul operation results in a sum of multiplications of key-matched values from the two Counters.<br>
In the update function, I implemented with the given formula as usual.


## Overall result:
Question q1: 6/6<br>
Question q2: 1/1<br>
Question q3: 5/5<br>
Question q4: 5/5<br>
Question q5: 4/3<br>
Question q6: 1/1<br>
Question q7: 1/1<br>
Question q8: 3/3<br>
------------------
Total: 25/25