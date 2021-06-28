from MAC.control.controller_centralized import Centralized


class CentralizedRL(Centralized):

    def __init__(self, env, agents, decision_maker):
        # initialize super class
        super().__init__(env, agents, decision_maker)

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
            for _ in range(max_episode_len):

                observations = []
                for agent_name in self.agents.keys():
                    observations.append(obs[agent_name])

                state = self.decode_state(observations, self.environment.get_needs_conv())
                joint_act = self.decision_maker.get_train_action(state)
                decoded_joint_act = self.decode_action(joint_act, self.environment.get_num_actions(),
                                               len(self.environment.get_env_agents()))

                actions = {}
                for i, agent_name in enumerate(self.environment.get_env_agents()):
                    action = decoded_joint_act[i]
                    actions[agent_name] = action

                new_obs, rewards, done, info = self.environment.get_env().step(actions)

                new_observations = []
                reward = 0
                for agent_name in self.agents.keys():
                    new_observations.append(new_obs[agent_name])
                    reward += rewards[agent_name]

                new_state = self.decode_state(observations, self.environment.get_needs_conv())
                new_done = all(value == True for value in done.values())

                self.decision_maker.update_step(state, joint_act, new_state, reward, new_done)
                ep_reward += reward

                if batch_size > 0:
                    self.decision_maker.update_episode(batch_size)

                obs = new_obs
                terminal = False
                terminal = new_done

                if terminal:
                    break

            episode_rewards.append(ep_reward)

            self.decision_maker.update_episode()