import copy
import numpy as np 

total_timesteps = 0
total_eps = 0


def eval_progress(env, agents, n_eval, log, tb_writer, args):
    global total_eps
    eval_reward = 0.

    for i_eval in range(n_eval):
        ep_timesteps = 0.
        env_observations = env.reset()

        while True:
            # Select action
            agent_actions = []
            for agent, env_observation in zip(agents, env_observations):
                agent_action = agent.select_deterministic_action(np.array(env_observation))
                agent_actions.append(agent_action)

            # Take action in env
            new_env_observations, env_rewards, done, _ = env.step(copy.deepcopy(agent_actions))

            # For next timestep
            env_observations = new_env_observations
            eval_reward += env_rewards[0]
            ep_timesteps += 1

            if done:
                break
    eval_reward /= float(n_eval)

    log[args.log_name].info("Evaluation Reward {} at episode {}".format(eval_reward, total_eps))
    tb_writer.add_scalars("reward", {"eval_reward": eval_reward}, total_eps)


def collect_one_traj(agents, env, log, args, tb_writer):
    global total_timesteps, total_eps

    ep_reward = 0
    ep_timesteps = 0
    env_observations = env.reset()

    while True:
        # Select action
        agent_actions = []
        for agent, env_observation in zip(agents, env_observations):
            agent_action = agent.select_stochastic_action(np.array(env_observation), total_timesteps)
            agent_actions.append(agent_action)

        # Take action in env
        new_env_observations, env_rewards, done, _ = env.step(copy.deepcopy(agent_actions))

        # Add experience to memory
        for i_agent, agent in enumerate(agents):
            agent.add_memory(
                obs=env_observations[i_agent],
                new_obs=new_env_observations[i_agent],
                action=agent_actions[i_agent],
                reward=env_rewards[i_agent],
                done=done)

        # For next timestep
        env_observations = new_env_observations
        ep_timesteps += 1
        total_timesteps += 1
        ep_reward += env_rewards[0]

        if done: 
            total_eps += 1
            log[args.log_name].info("Train episode reward {} at episode {}".format(ep_reward, total_eps))
            tb_writer.add_scalars("reward", {"train_reward": ep_reward}, total_eps)

            return ep_reward


def train(agents, env, log, tb_writer, args):
    while True:
        # Measure performance for reporting results in paper
        eval_progress(
            env=env, agents=agents, n_eval=1, log=log, tb_writer=tb_writer, args=args)

        # Collect one trajectory
        collect_one_traj(agents=agents, env=env, log=log, args=args, tb_writer=tb_writer)

        # Update policy
        for agent in agents:
            agent.update_policy(total_timesteps=total_timesteps)
