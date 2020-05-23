import gym
import torch
from agent import Agent
from config import Config
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import gym_ple
from display_save import display_save
from skimage.transform import resize
import  torchvision.transforms as transforms


class Environment:
    def __init__(self):
        self.env = gym.make(Config.ENV)
        h = self.env.observation_space.shape[0]
        w = self.env.observation_space.shape[1]
        num_actions = self.env.action_space.n
        self.agent = Agent(h,w, num_actions)

        self.transforms = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

    def state_init(self):
        observation = Image.fromarray(self.env.reset())
        state = self.transforms(observation).unsqueeze(0)
        return state

    def run_step(self, state, action, step):

        # action = self.agent.get_action(state, episode)

        observation_next, reward, done, _ = self.env.step(
            action.item())
        # c1, s1, c2, s2, _, _ = state.squeeze(0).tolist()
        # height = -c1-(c1*c2-s1*s2)
        reward = torch.FloatTensor([reward/64])

        state_next = self.transforms(Image.fromarray(observation_next)).unsqueeze(0)

        return state_next, done, reward



if __name__ == '__main__':

    cartpole_env = Environment()
    frames = []
    ani_check = False
    loss_list =[]
    avg_reward_list = []

    state = cartpole_env.state_init()
    rgb_weights = [0.2989, 0.5870, 0.1140]
    for step in range(1):

        action = None
        sum_reward = 0
        # frames.append(resize(cartpole_env.env.render(mode='rgb_array'),(640, 640),anti_aliasing=True))
        action = cartpole_env.agent.get_action(state, step)

        state_next, done, reward = cartpole_env.run_step(state, action, step)
        print(state_next)
        print(reward)
        state_next = np.array([np.array(state_next.squeeze(0).squeeze(0)), np.array(state_next.squeeze(0).squeeze(0)), np.array(state_next.squeeze(0).squeeze(0))]).transpose(1, 2, 0)
        # a = np.expand_dims(np.dot(frames[-1][...,:3], rgb_weights),axis=2)
        plt.imshow(state_next)
        plt.show()
    plt.close('all')
        # plt.savefig(f'./save_image/{step}.jpg')

    # display_save(frames,200)

        #     cartpole_env.agent.memorize(state, action, state_next, reward)
        #     cartpole_env.agent.update_q_function()
        #     last_step = step
        #
        #     if step == Config.MAX_STEPS-1:
        #         avg_reward_list.append(sum_reward/(step+1))
        #
        #
        #
        # cartpole_env.agent.update_epsilon()
        # print('%d Episode: Finished after %d steps：loss = %.6lf' % (
        #     episode, step + 1, cartpole_env.agent.brain.loss))
        #
        # # if episode % 50 == 0:
        # #     loss_list.append(cartpole_env.agent.brain.loss)
        # #     plt.title("loss")
        # #     plt.plot(loss_list)
        #     # plt.show()
        #
        # if episode % 2 == 0:
        #     cartpole_env.agent.update_target_q_function()
        #
        # if episode == Config.NUM_EPISODES-1:
        #     display_save(frames, episode)
        #     frames.clear()
        #
        # if episode_final is True:
        #     break
        #
        # if last_step < 100:
        #     print('train end')
        #     episode_final = True