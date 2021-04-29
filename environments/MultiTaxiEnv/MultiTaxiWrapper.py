from taxi_environment import TaxiEnv

class MultiTaxiWrapper:
    def __init__(self, num_taxis=2, num_passengers=1):
        self.env = TaxiEnv(
        num_taxis = num_taxis,
        num_passengers = num_passengers,
        max_fuel = None,
        domain_map = None,
        taxis_capacity = None,
        collision_sensitive_domain = True,
        fuel_type_list = None,
        option_to_stand_by = False
        )

    def reset(self):
        observation = self.env.reset()
        observation_list = [obs for obs in observation.values()]
        return observation_list
        
    def step(self, action):
        # All of these are dicts, we need them as lists:
        observation, reward, done, info = self.env.step(action)
        observation_list = [obs for obs in observation.values()]
        reward_list = [r for r in reward.values()]      
        done_list = [d for d in done.values()]      
        info_list = [i for i in info.values()]      
        return observation_list, reward_list, done_list, info_list

    def render(self):
        return self.env.render()