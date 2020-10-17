
### Q1 Reflex Agent
I consider a number of features of the next state: number of food eaten, distance to the closest food, capsule(power pellet), normal ghost, scared(vulnerable) ghost.<br>
Then I specify reasonable weights for each of those features.<br>
And that's pretty much it, the evaluation score is calculated by a sum of weighted features.

### Q2,3,4 Minimax, Alpha-Beta pruning, Expectimax
I programmed according to the pseudocode.
A critical thing is to handle multi MIN agents:<br>
So to deal with this, we need to check whether the current min agent is the last one or not.<br>
If it's not, we continue to the next min agent, and if it is, the max agent should take turn.

### Q5 Better evaluation function
**Reuse:** 
Since the evaluation function from Q1 is already pretty good, I could easily reuse it with some slight modification in weights<br>

<hr>
Overall result: Total: 25/25
