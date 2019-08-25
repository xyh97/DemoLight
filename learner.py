from script import write_summary


class Learner(object):
    """Meta-learner
    """
    def __init__(self, sampler, policy, dic_agent_conf, dic_traffic_env_conf, dic_path):
        self.sampler = sampler
        self.policy = policy
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path

        self.meta_params = self.policy.save_params()

        self.meta_target_params = self.meta_params
        self.step_cnt = 0
        self.period = self.dic_agent_conf['PERIOD']
        # todo every task's target_params

    def sample_period(self, task, batch_id, old_episodes=None):

        # set a batch_id
        task_type = None

        self.batch_id = batch_id
        tasks = [task] * self.dic_traffic_env_conf['NUM_GENERATOR']
        self.sampler.reset_task(tasks, batch_id, reset_type='learning')

        # TODO adapt to the whole logic
        self.meta_params, self.meta_target_params, episodes = \
            self.sampler.sample_period(self.policy, tasks[0], batch_id, params=self.meta_params,
                                       target_params=self.meta_target_params, old_episodes=old_episodes)
        return episodes

    def test_model(self, tasks, batch_id):
        self.sampler.reset_task(tasks, batch_id, reset_type='test')
        self.sampler.test_sample(self.policy, tasks, batch_id, params=self.meta_params)
        for task in tasks:
            task_id = self.dic_traffic_env_conf['TRAFFIC_IN_TASKS'].index(task)
            write_summary(self.dic_path, 'task_%d_%s' % (task_id, task), self.dic_traffic_env_conf["EPISODE_LEN"], batch_id)