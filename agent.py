import torch
from brain import Brain



class Agent:
    def __init__(self, h, w, num_actions):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.brain = Brain(h, w, num_actions, self.device)

    def update_q_function(self):
        self.brain.replay()

    def get_action(self, state, episode):
        action = self.brain.decide_action(state)
        return action

    def memorize(self, state, action, state_next, reward):
        self.brain.memory.push(state, action, state_next, reward)

    def update_target_q_function(self):
        self.brain.update_target_q_function()

    def update_epsilon(self):
        self.brain.update_epsilon()

    def model_save(self):
        self.brain.model_save()