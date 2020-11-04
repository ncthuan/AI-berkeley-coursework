
### Q1 Perceptron
Implemented just as usual and straightforward

### Q2 Perceptron Analysis
Used the provided api in the Counter class in util.py

### Q3 MIRA
MIRA differs from basic perceptron in a detail that the weights is updated by the input features scaled by τ, where τ is computed using the formula provided in the assignment. <br>
**The idea** is that, update the weights so as to fix the current mistake and to minize the changes to the weights as well. So in other words, changes as little as possible to fix the mistake. <br>
This helps the model less sensitive to noisy data and converge faster.

### Q4 Digit feature design
I used 5x5 convolution with 4 filters to add features about edges of different directions.
* The pros is that I don't have to spend much effort implementing more complex features.
* The cons is that number of features blows up (28x28 + 4x24x24), which slows the training process a little bit.
Result: improved from 78% to 84%, which is just enough to pass the grader.

### Q5
I explored the data used to train the pacman perceptron and then adapted the code so it consists with the perceptron algorithm

### Q6 
**Reuse:** I reused the code from Q1 of my previous assignment (multiAgents), which I thought would make sense because that function also extracts features from the state, and in fact, it did it pretty well.


<hr>
Overall result: Total: 25/25
