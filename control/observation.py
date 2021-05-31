import numpy as np
class ObservationFilter:

    # init agents and their observations
    def __init__(self, obs_indices):
        self.obs_indices = obs_indices

    def get_filtered_observation(self, observation):

        # full observability
        if self.obs_indices is None:
            return observation

        filtered_observation = []
        for index in self.obs_indices:
            cur = observation[0][index]
            filtered_observation.append(cur)
        filtered_observation = np.reshape(np.vstack(filtered_observation), (1, -1))
        return filtered_observation





