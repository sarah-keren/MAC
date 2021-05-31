import numpy as np
from social_dilemmas.envs.cleanup import CleanupEnv

env = CleanupEnv(num_agents=5, render=True)

agents = list(env.agents.values())
print(agents)

for i in range(100):
    action_dim = agents[0].action_space.n
    rand_action = np.random.randint(action_dim, size=5)
    obs, rew, dones, info, = env.step({'agent-0': rand_action[0],
                                       'agent-1': rand_action[1],
                                       'agent-2': rand_action[2],
                                       'agent-3': rand_action[3],
                                       'agent-4': rand_action[4]})
    from pprint import pprint; pprint(obs['agent-4'].shape)
    exit()

    # print(rew, dones)
    env.render()