package players.groupAP;

import core.GameState;
import players.heuristics.AdvancedHeuristic;
import players.heuristics.CustomHeuristic;
import players.heuristics.StateHeuristic;
import utils.ElapsedCpuTimer;
import utils.Types;
import utils.Utils;
import utils.Vector2d;

import java.util.ArrayList;
import java.util.Random;

public class SingleTreeNode_2 {
    public MCTSParams params;

    private SingleTreeNode_2 parent;
    private SingleTreeNode_2[] children;
    private double totValue;//total value
    private int nVisits;//number of visits
    private int nVisits_AMAF;//maintain AMAF score separately
    private double totValue_AMAF;//maintain AMAF score separately
    private Random m_rnd;
    private int m_depth;//depth of this node
    private double[] bounds = new double[]{Double.MAX_VALUE, -Double.MAX_VALUE};
    private double[] bounds_AMAF = new double[]{Double.MAX_VALUE, -Double.MAX_VALUE};
    private int childIdx;//the idx of this node as a child of its parent node
    private int fmCallsCount;//count of forward model calls

    private int num_actions;
    private Types.ACTIONS[] actions;

    private GameState rootState; //the true state, not a copy
    private StateHeuristic rootStateHeuristic;

    private boolean[] action_taken = new boolean[6];

    SingleTreeNode_2(MCTSParams p, Random rnd, int num_actions, Types.ACTIONS[] actions) {
        this(p, null, -1, rnd, num_actions, actions, 0, null);
    }

    private SingleTreeNode_2(MCTSParams p, SingleTreeNode_2 parent, int childIdx, Random rnd, int num_actions,
                           Types.ACTIONS[] actions, int fmCallsCount, StateHeuristic sh) {
        this.params = p;
        this.fmCallsCount = fmCallsCount;
        this.parent = parent;
        this.m_rnd = rnd;
        this.num_actions = num_actions;
        this.actions = actions;
        children = new SingleTreeNode_2[num_actions];
        totValue = 0.0;
        this.childIdx = childIdx;
        if (parent != null) {
            m_depth = parent.m_depth + 1;
            this.rootStateHeuristic = sh;
        } else
            m_depth = 0;
    }

    void setRootGameState(GameState gs) {
        this.rootState = gs;
        if (params.heuristic_method == params.CUSTOM_HEURISTIC)
            this.rootStateHeuristic = new CustomHeuristic(gs);
        else if (params.heuristic_method == params.ADVANCED_HEURISTIC) // New method: combined heuristics
            this.rootStateHeuristic = new AdvancedHeuristic(gs, m_rnd);
    }


    void mctsSearch(ElapsedCpuTimer elapsedTimer) {

        double avgTimeTaken;
        double acumTimeTaken = 0;
        long remaining;
        int numIters = 0;

        int remainingLimit = 5;// in millisecond
        boolean stop = false;

        while (!stop) {

            GameState state = rootState.copy();
            ElapsedCpuTimer elapsedTimerIteration = new ElapsedCpuTimer();
            for(int i=0; i<6; i++) {
                action_taken[i] = false;
            }
            SingleTreeNode_2 selected = treePolicy(state);//phase 1: tree selection & phase 2: expansion
            double delta = selected.rollOut(state); //phase 3: simulation
            backUp(selected, delta); //phase 4: back propagation

            //Stopping condition
            if (params.stop_type == params.STOP_TIME) {
                numIters++;
                acumTimeTaken += (elapsedTimerIteration.elapsedMillis());
                avgTimeTaken = acumTimeTaken / numIters;
                remaining = elapsedTimer.remainingTimeMillis();
                stop = remaining <= 2 * avgTimeTaken || remaining <= remainingLimit;
            } else if (params.stop_type == params.STOP_ITERATIONS) {
                numIters++;
                stop = numIters >= params.num_iterations;
            } else if (params.stop_type == params.STOP_FMCALLS) {
                fmCallsCount += params.rollout_depth;
                stop = (fmCallsCount + params.rollout_depth) > params.num_fmcalls;
            }
        }
        //System.out.println(" ITERS " + numIters);
    }

    private SingleTreeNode_2 treePolicy(GameState state) {//传入的state是copy

        SingleTreeNode_2 cur = this;

        while (!state.isTerminal() && cur.m_depth < params.rollout_depth) {
            if (cur.notFullyExpanded()) {
                return cur.expand(state); //proceed to expansion phase

            } else {
                cur = cur.uct(state); //move down the tree
            }
        }

        return cur;
    }


    private SingleTreeNode_2 expand(GameState state) {

        int bestAction = 0;
        double bestValue = -1;

        //从可以expand的坑位中随机选出一个
        for (int i = 0; i < children.length; i++) {
            double x = m_rnd.nextDouble();
            if (x > bestValue && children[i] == null) {
                bestAction = i;
                bestValue = x;
            }
        }

        //Roll the state
        roll(state, actions[bestAction]);//state更新与expansion同步

        SingleTreeNode_2 tn = new SingleTreeNode_2(params, this, bestAction, this.m_rnd, num_actions,
                actions, fmCallsCount, rootStateHeuristic);//用bestAction作为childIdx
        children[bestAction] = tn;
        return tn;
    }

    private void roll(GameState gs, Types.ACTIONS act) {
        //Simple, all random first, then my position.
        int nPlayers = 4;
        Types.ACTIONS[] actionsAll = new Types.ACTIONS[4];
        int playerId = gs.getPlayerId() - Types.TILETYPE.AGENT0.getKey();

        for (int i = 0; i < nPlayers; ++i) {
            if (playerId == i) {
                actionsAll[i] = act;
            } else {//still random opponent model
                int actionIdx = m_rnd.nextInt(gs.nActions());
                actionsAll[i] = Types.ACTIONS.all().get(actionIdx);
            }
        }

        gs.next(actionsAll);

    }

    private SingleTreeNode_2 uct(GameState state) {
        SingleTreeNode_2 selected = null;
        double bestValue = -Double.MAX_VALUE;
        for (SingleTreeNode_2 child : this.children) {
            double childValue = child.totValue / (child.nVisits + params.epsilon);
            double childValue_AMAF = child.totValue_AMAF / (child.nVisits_AMAF + params.epsilon);

            childValue = Utils.normalise(childValue, bounds[0], bounds[1]);
            childValue_AMAF = Utils.normalise(childValue_AMAF, bounds_AMAF[0], bounds_AMAF[1]);

            double childValue_integrated = params.alpha * childValue_AMAF + (1-params.alpha)* childValue;

            double uctValue = childValue_integrated +
                    params.K * Math.sqrt(Math.log(this.nVisits + 1) / (child.nVisits + params.epsilon));

            uctValue = Utils.noise(uctValue, params.epsilon, this.m_rnd.nextDouble());     //break ties randomly

            // small sampleRandom numbers: break ties in unexpanded nodes
            if (uctValue > bestValue) {
                selected = child;
                bestValue = uctValue;
            }
        }
        if (selected == null) {
            throw new RuntimeException("Warning! returning null: " + bestValue + " : " + this.children.length + " " +
                    +bounds[0] + " " + bounds[1]);
        }

        //Roll the state:
        roll(state, actions[selected.childIdx]); //state更新与tree selection同步

        return selected;
    }

    private double rollOut(GameState state) {//到了simulation阶段就没有tree node用了，只在state上操作。所以depth要手动更新
        int thisDepth = this.m_depth;

        while (!finishRollout(state, thisDepth)) {
            int action = safeRandomAction(state);
            action_taken[action] = true;
            roll(state, actions[action]);
            thisDepth++;
        }

        return rootStateHeuristic.evaluateState(state);
    }

    private int safeRandomAction(GameState state) {
        Types.TILETYPE[][] board = state.getBoard();
        ArrayList<Types.ACTIONS> actionsToTry = Types.ACTIONS.all();
        int width = board.length;
        int height = board[0].length;

        while (actionsToTry.size() > 0) {

            int nAction = m_rnd.nextInt(actionsToTry.size());
            Types.ACTIONS act = actionsToTry.get(nAction);
            Vector2d dir = act.getDirection().toVec();

            Vector2d pos = state.getPosition();
            int x = pos.x + dir.x;
            int y = pos.y + dir.y;

            if (x >= 0 && x < width && y >= 0 && y < height)
                if (board[y][x] != Types.TILETYPE.FLAMES)
                    return act.getKey();

            actionsToTry.remove(nAction);
        }

        return m_rnd.nextInt(num_actions);
    }

    @SuppressWarnings("RedundantIfStatement")
    private boolean finishRollout(GameState rollerState, int depth) {
        if (depth >= params.rollout_depth)      //rollout end condition.
            return true;

        if (rollerState.isTerminal())               //end of game
            return true;

        return false;
    }

    private void backUp(SingleTreeNode_2 node, double result)//从tree selection阶段最深的那个node开始往上
    {
        SingleTreeNode_2 n = node;
        int last_childIdx = -1; //tmp
        while (n != null) {
            update_stat(n, result);//regular update
            update_stat_AMAF(n, result); //AMAF update
            //then check children (i.e. previous node's siblings)
            if(last_childIdx != -1) {
                for(int i=0; i<6; i++){
                    if(i != last_childIdx && n.children[i] != null && action_taken[i]) {
                        update_stat_AMAF(n.children[i], result);//AMAF update for a sibling
                    }
                }
            }
            last_childIdx = n.childIdx;
            n = n.parent;
        }
    }

    void update_stat_AMAF(SingleTreeNode_2 n, double r) {
        n.nVisits_AMAF++;
        n.totValue_AMAF += r;
        if (r < n.bounds_AMAF[0]) {
            n.bounds_AMAF[0] = r;
        }
        if (r > n.bounds_AMAF[1]) {
            n.bounds_AMAF[1] = r;
        }
    }

    void update_stat(SingleTreeNode_2 n, double r){
        n.nVisits++;
        n.totValue += r;
        if (r < n.bounds[0]) {
            n.bounds[0] = r;
        }
        if (r > n.bounds[1]) {
            n.bounds[1] = r;
        }
    }

    int mostVisitedAction() {//以N(s,a)论好坏
        int selected = -1;
        double bestValue = -Double.MAX_VALUE;
        boolean allEqual = true;
        double first = -1;

        for (int i = 0; i < children.length; i++) {

            if (children[i] != null) {
                if (first == -1)
                    first = children[i].nVisits;
                else if (first != children[i].nVisits) {
                    allEqual = false;
                }

                double childValue = children[i].nVisits;
                childValue = Utils.noise(childValue, params.epsilon, this.m_rnd.nextDouble());     //break ties randomly
                if (childValue > bestValue) {
                    bestValue = childValue;
                    selected = i;
                }
            }
        }

        if (selected == -1) {
            selected = 0;
        } else if (allEqual) {
            //If all are equal, we opt to choose for the one with the best Q.
            selected = bestAction();
        }

        return selected;
    }

    private int bestAction() //以Q(s,a)论好坏
    {
        int selected = -1;
        double bestValue = -Double.MAX_VALUE;

        for (int i = 0; i < children.length; i++) {

            if (children[i] != null) {
                double childValue = children[i].totValue / (children[i].nVisits + params.epsilon);
                childValue = Utils.noise(childValue, params.epsilon, this.m_rnd.nextDouble());     //break ties randomly
                if (childValue > bestValue) {
                    bestValue = childValue;
                    selected = i;
                }
            }
        }

        if (selected == -1) {
            System.out.println("Unexpected selection!");
            selected = 0;
        }

        return selected;
    }


    private boolean notFullyExpanded() {
        for (SingleTreeNode_2 tn : children) {
            if (tn == null) {
                return true;
            }
        }

        return false;
    }
}
