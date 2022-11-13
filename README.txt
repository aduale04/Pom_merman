In this project, we enhance the MCTS agent in Pommerman with three types of AMAF
heuristics. Comparisons by win rate against other agents are made among the three, and we
also tune the parameters including ɑ, C (in the UCB1 function) and rollout depth to further
improve win rate. We conclude that, using ɑ-AMAF, the plain MCTS agent can be effectively
improved in both FFA and Team mode with both full and partial observability. And tuning
parameters upon that allows us to yield a total increase in win rate of 10.4 percentage points
in FFA mode with full observability.

Load the java file into the Java project as a Java package to run the pommerman game..

The algorithms and parameters are all set, so do not set parameters again from outside of the package.

The agent can be invoked just like the built-in MCTS agent in the framework:
MCTSParams mctsParams = new MCTSParams();
player = new MCTSPlayer(seed, playerID++, mctsParams);

