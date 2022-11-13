The drop folder "groupAP" into the Java project as a Java package.

The algorithms and parameters are all set, so do not set parameters again from outside of the package.

The agent can be invoked just like the built-in MCTS agent in the framework:
MCTSParams mctsParams = new MCTSParams();
player = new MCTSPlayer(seed, playerID++, mctsParams);