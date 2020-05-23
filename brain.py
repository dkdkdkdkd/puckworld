from replayMemory import ReplayMemory, Transition
from dqn import Net
import torch
import torch.optim as optim
import numpy as np
import random
import torch.nn.functional as F
from config import Config


class Brain:
    def __init__(self, h, w, num_actions, device):

        self.device = device
        self.num_actions = num_actions
        self.memory = ReplayMemory(Config.CAPACITY)
        self.main_q_network = Net(h, w, num_actions).to(self.device)
        self.target_q_network = Net(h, w, num_actions).to(self.device)
        self.loss = 0.
        self.optimizer = optim.Adam(self.main_q_network.parameters(), lr=0.0001)
        self.epsilon = 1.0
        print(self.main_q_network)

    def replay(self):
        if len(self.memory) < Config.LEARNING_START:
            return

        self.batch, self.state_batch, self.action_batch, self.reward_batch, \
        self.non_final_next_states = self.make_minibatch()

        self.state_batch = self.state_batch.to(self.device)
        self.action_batch = self.action_batch.to(self.device)
        self.reward_batch = self.reward_batch.to(self.device)
        self.non_final_next_states = self.non_final_next_states.to(self.device)

        self.expected_state_action_values = self.get_expected_state_action_values()

        self.update_main_q_network()

    def decide_action(self, state):

        state = state.to(self.device)

        if self.epsilon <= np.random.uniform(0, 1):
            self.main_q_network.eval()
            with torch.no_grad():
                action = self.main_q_network(state).max(1)[1].view(1, 1)

        else:
            action = torch.LongTensor([[random.randrange(self.num_actions)]])
        return action

    def make_minibatch(self):

        transitions = self.memory.sample(Config.BATCH_SIZE)

        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        return batch, state_batch, action_batch, reward_batch, non_final_next_states

    def get_expected_state_action_values(self):

        self.main_q_network.eval()
        self.target_q_network.eval()

        self.state_action_values = self.main_q_network(self.state_batch).gather(1,
                                                                                self.action_batch)  # *********************************8

        non_final_mask = torch.BoolTensor(tuple(map(lambda s: s is not None, self.batch.next_state)))

        a_m = torch.zeros(Config.BATCH_SIZE).type(torch.LongTensor).to(self.device)

        a_m[non_final_mask] = self.main_q_network(self.non_final_next_states).detach().max(1)[1]

        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)

        next_state_values = torch.zeros(Config.BATCH_SIZE).to(self.device)

        next_state_values[non_final_mask] = self.target_q_network(
            self.non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze()

        expected_state_action_values = self.reward_batch + Config.GAMMA * next_state_values

        return expected_state_action_values

    def update_main_q_network(self):

        self.main_q_network.train()

        self.loss = F.smooth_l1_loss(self.state_action_values,
                                     self.expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def update_target_q_function(self):

        self.target_q_network.load_state_dict(self.main_q_network.state_dict())

    def update_epsilon(self):
        if self.epsilon > Config.EPSILON_MIN:
            self.epsilon -= 1 / (Config.NUM_EPISODES - Config.START_TRAIN_EP)

    def model_save(self):
        torch.save(self.main_q_network, 'puckworkd_model.pth')