from controller import controller

# a controller that decides at every iteration which action is performed by each agent
class controller_market(controller):

    # init agents and their observations
    def __init__(self, environment, agents, observations):
        # calling the super class
        super(environment, agents, observations)


    def run(self):

        done = False
        while done is not True:

            # get tasks
            tasks = self.get_current_tasks()

            # get evaluations
            task_evals = []
            for task in tasks:
                cur_task_evals = []
                for agent in self.agents():
                    cur_eval = self.get_task_eval(agent, task)
                    cur_task_evals.append(cur_eval)
                task_evals.append(cur_task_evals)

            # assign tasks
            task_assignments = self.assign_tasks(task_evals)


            for agent,task in task_assignments:
                self.perform_task(agent, task)

            if done:
                break

    def get_current_tasks(self):
        pass

    def get_task_eval(agent, task):
        pass


