import torch
import argparse
import os
import numpy as np
from misc.utils import set_log, make_env
from tensorboardX import SummaryWriter
from trainer.train import train


def set_policy(env, tb_writer, log, args, name, i_agent):
    if name == "agent":
        from policy.agent import Agent
        policy = Agent(env=env, tb_writer=tb_writer, log=log, name=name, i_agent=i_agent, args=args)
    else:
        raise ValueError("Invalid name")

    return policy


def main(args):
    if args.central_train is True:
        raise NotImplementedError("todo")

    # Create directories
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    if not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")

    # Set logs
    tb_writer = SummaryWriter('./logs/tb_{0}'.format(args.log_name))
    log = set_log(args)

    # Create env
    env = make_env(args)

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Initialize policy
    agents = [
        set_policy(env, tb_writer, log, args, name="agent", i_agent=i_agent)
        for i_agent in range(args.n_agent)]

    # Start train
    train(agents=agents, env=env, log=log, tb_writer=tb_writer, args=args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Algorithm
    parser.add_argument(
        "--tau", default=0.01, type=float, 
        help="Target network update rate")
    parser.add_argument(
        "--batch-size", default=50, type=int, 
        help="Batch size for both actor and critic")
    parser.add_argument(
        "--policy-freq", default=2, type=int,
        help="Frequency of delayed policy updates")
    parser.add_argument(
        "--actor-lr", default=0.00005, type=float,
        help="Learning rate for actor")
    parser.add_argument(
        "--critic-lr", default=0.0005, type=float,
        help="Learning rate for critic")
    parser.add_argument(
        "--n-hidden", default=200, type=int,
        help="Number of hidden units")
    parser.add_argument(
        "--discount", default=0.99, type=float, 
        help="Discount factor")
    parser.add_argument(
        "--n-agent", default=1, type=int,
        help="Number of agent")
    parser.add_argument(
        "--central-train", action='store_true',
        help="Centralized training or not")

    # Env
    parser.add_argument(
        "--env-name", type=str, required=True,
        help="OpenAI gym environment name")
    parser.add_argument(
        "--row", type=int, required=True,
        help="# of rows in the environment")
    parser.add_argument(
        "--col", type=int, required=True,
        help="# of cols in the environment")
    parser.add_argument(
        "--ep-max-timesteps", type=int, required=True,
        help="Episode is terminated when max timestep is reached")
    parser.add_argument(
        "--n-action", type=int, default=4,
        help="# of possible actions")

    # Misc
    parser.add_argument(
        "--prefix", default="", type=str,
        help="Prefix for tb_writer and logging")
    parser.add_argument(
        "--seed", default=0, type=int, 
        help="Sets Gym, PyTorch and Numpy seeds")

    args = parser.parse_args()

    # Set log name
    args.log_name = \
        "env::%s_row::%s_col::%s_n_agent::%s_central_train::%s_prefix::%s_log" \
        % (args.env_name, args.row, args.col, args.n_agent, args.central_train, args.prefix)

    main(args=args)
