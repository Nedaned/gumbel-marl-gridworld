from policy.td3 import TD3


class PolicyBase(object):
    def __init__(self, env, log, tb_writer, args, name, i_agent):
        super(PolicyBase, self).__init__()

        self.env = env
        self.log = log
        self.tb_writer = tb_writer
        self.args = args
        self.name = name + str(i_agent)

    def set_dim(self):
        raise NotImplementedError()

    def set_linear_schedule(self):
        raise NotImplementedError()

    def select_stochastic_action(self):
        raise NotImplementedError()

    def clear_memory(self):
        self.memory.clear()

    def set_policy(self):
        self.policy = TD3(
            actor_input_dim=self.actor_input_dim,
            actor_output_dim=self.actor_output_dim,
            critic_input_dim=self.critic_input_dim,
            n_hidden=self.n_hidden,
            name=self.name,
            args=self.args)

    def save_weight(self, filename, directory):
        self.log[self.args.log_name].info("[{}] Saved weight".format(self.name))
        self.policy.save(filename, directory)

    def load_weight(self, filename, directory="./pytorch_models"):
        self.log[self.args.log_name].info("[{}] Loaded weight".format(self.name))
        self.policy.load(filename, directory)
