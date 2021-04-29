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

        # This is duplicate because __init__ also calls reset(),
        # however this is the way we can get the initial state
        self.state = self.env.reset()

    def get_total_observation(self):
        return self.state

    def get_single_observation(self, agent_name):
        return self.state[agent_name]

    def is_done(self):
        return self.env.dones['__all__']

    def step(self, action):
        self.env.step(action)

    def render(self):
        self.env.render()

    @staticmethod
    def join_actions(action_list):
        joint_action = {}
        for action in action_list:
            joint_action.update(action)
        return joint_action