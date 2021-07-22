from control.controller_centralized import Centralized
import numpy as np

"""Class to use when using centralized controller
for RL setting
"""
class CentralizedRL(Centralized):

    def __init__(self, env, agents, central_agent):
        # initialize super class
        super().__init__(env, agents, central_agent)

    def run(self, render=False, max_iteration=None, max_episode_len=1, num_episodes=1, batch_size=0):
        """Runs the controller on the environment given in the init,
        with the agents given in the init

        Args:
            render (bool, optional): Whether to render while runngin. Defaults to False.
            max_iteration ([type], optional): Number of steps to run. Defaults to infinity.
            max_episode_len (int, optional): Upper cap on episode length. Defaults to 1.
            num_episodes (int, optional): Number of episodes. Defaults to 1.
            batch_size (int, optional): Batch size for the training algorithm. Defaults to 0.
        """
        print("Training...")
        self.train(max_episode_len, num_episodes, batch_size=batch_size)
        print("Finished Training")

        done = False
        index = 0
        observation = self.environment.get_env().reset()
        self.total_rewards = []
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
            self.total_rewards.append(reward)
            done = all(value == True for value in done.values())
            if done:
                break

        if render:
            self.environment.get_env().render()

    def get_joint_action(self, observation):
        """Returns the joint actions of all the agents

        Args:
            observation ([dict]): The agents observations

        Returns:
            dict: dict of all the actions
        """
        observations = []
        # Dict to list:
        for agent_name in self.agents.keys():
            observations.append(observation[agent_name])

        state = self.decode_state(observations, self.environment.get_needs_conv())
        joint_act = self.decision_maker.get_action(state)
        joint_act = self.decode_action(joint_act, self.environment.get_num_actions(),
                                       len(self.environment.get_env_agents()))
        joint_action = {}
        for i, agent_name in enumerate(self.environment.get_env_agents()):
            action = joint_act[i]
            joint_action[agent_name] = action

        return joint_action

    def decode_state(self, obs, needs_conv):
        """Turns the ovservation from a list to np array

        Args:
            obs (list): list of observations
            needs_conv (bool): whether we want conv layers (affects the shape)

        Returns:
            ndarray: the observations
        """
        if needs_conv:
            return np.vstack(obs)
        else:
            return np.hstack(obs)

    def decode_action(self, action, num_actions, num_agents):
        """Decodes the action from the model to RL env friendly format

        Args:
            action (int): The action from the model
            num_actions (int): number of actions avaiable to every agent
            num_agents (int): number of agents

        Returns:
            list: list of individual actions
        """
        out = []
        for ind in range(num_agents):
            out.append(action % num_actions)
            action = action // num_actions
        return list(reversed(out))

    def train(self, max_episode_len, num_episodes, batch_size=0):
        """Train the agents

        Args:
            max_episode_len (int, optional): Upper cap on episode length. Defaults to 1.
            num_episodes (int, optional): Number of episodes. Defaults to 1.
            batch_size (int, optional): Batch size for the training algorithm. Defaults to 0.
        """
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