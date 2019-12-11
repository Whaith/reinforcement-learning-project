from wimblepong_test.pong_testbench import PongTestbench
from agent import Agent
import argparse
import torch 

parser = argparse.ArgumentParser()
parser.add_argument("--render", "-r", action="store_true", help="Render the competition.")
parser.add_argument("--games", "-g", type=int, default=100, help="number of games.")
args = parser.parse_args()

player1 = Agent()

testbench = PongTestbench(args.render)
testbench.init_players(player1)
testbench.run_test(args.games)
