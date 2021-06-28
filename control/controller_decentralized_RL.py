from MAC.control.controller_decentralized import Decentralized


class DecentralizedRL(Decentralized):

    def __init__(self, env, agents):
        # initialize super class
        super().__init__(env, agents)

    def run(self, render=False, max_iteration=None, max_episode_len=1, num_episodes=1, batch_size=0):

        print("Training...")
        self.train(max_episode_len, num_episodes, batch_size=batch_size)
        print("Finished Training")

        done = False
        index = 0
        observation = self.environment.get_env().reset()
        while done is not True:
            index += 1
            if max_iteration is not None and index > max_iteration:
                break

            # display environment
            if render:
                self.environment.get_env().render()

            # get actions for each agent to perform

            joint_action = self.get_joint_action(observation)
            observation, reward, done, info = self.perform_joint_action(joint_action)
            done = all(value == True for value in done.values())
            if done:
                break

        if render:
            self.environment.get_env().render()

    def train(self, max_episode_len, num_episodes, batch_size=0):
        episode_rewards = [0.0]

        print("Starting training...")
        for i in range(num_episodes):
            obs = self.environment.get_env().reset()
            ep_reward = 0.0
            agent_rewards = {agent_name: 0.0 for agent_name in self.agents.keys()}
            for _ in range(max_episode_len):
                actions = {agent_name: self.agents[agent_name].get_train_action(obs[agent_name]) for
                               agent_name in self.agents.keys()}
                new_obs, rewards, done, info = self.environment.get_env().step(actions)
                for agent_name in self.agents.keys():
                    self.agents[agent_name].update_step(obs[agent_name], actions[agent_name],
                                                            new_obs[agent_name], rewards[agent_name], done[agent_name])
                    ep_reward += rewards[agent_name]

                if batch_size > 0:
                    for agent_name in self.agents.keys():
                        self.agents[agent_name].update_episode(batch_size)

                obs = new_obs
                terminal = False
                terminal = all(value == True for value in done.values())

                if terminal:
                    break

            episode_rewards.append(ep_reward)

            for agent_name in self.agents.keys():
                self.agents[agent_name].update_episode()
