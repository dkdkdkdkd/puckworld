from puckworldEnv import Environment
import numpy as np
from config import Config
import matplotlib.pyplot as plt
from display_save import display_save


if __name__ == '__main__':
    puckworld_env = Environment()
    reward_sum  = 0
    reward_avg_list = []
    loss_sum = 0
    loss_avg_list = []
    paly_final  = False
    step_list = []

    state = puckworld_env.state_init()

    # for step in Config.MAX_STEPS:
    for step in range(Config.MAX_STEPS):

        action = puckworld_env.agent.get_action(state, step)
        state_next, done, reward = puckworld_env.run_step(state, action, step)
        print(step, done)
        puckworld_env.agent.memorize(state, action, state_next, reward)
        puckworld_env.agent.update_q_function()

        if step > Config.LEARNING_START:
            if step % 1000 == 0:
                print("step:"+str(step))
                reward_avg_list.append(reward_sum/1000)
                loss_avg_list.append(loss_sum/1000)
                step_list.append(step)
                reward_sum = 0
                loss_sum = 0

                plt.subplot(2, 1, 1)
                plt.plot(step_list, reward_avg_list)
                plt.title('reward_step')
                plt.xlabel('step')
                plt.ylabel('reward_avg')

                plt.subplot(2, 1, 2)
                plt.plot(step_list, loss_avg_list)
                plt.title('loss_step')
                plt.xlabel('step')
                plt.ylabel('loss_avg')

            else:
                reward_sum += reward.item()
                loss_sum += puckworld_env.agent.brain.loss

            if step % 500000 == 0:
                plt.savefig(f'result_{step}')
            else:
                if step % Config.TARGET_UPDATE_FREQ == 0 :
                    puckworld_env.agent.update_target_q_function()
                    plt.show()
                else:
                    plt.clf()


    puckworld_env.agent.model_save()

    plt.subplot(2, 1, 1)
    plt.plot(step_list, reward_avg_list)
    plt.title('reward_step')
    plt.xlabel('step')
    plt.ylabel('reward_avg')

    plt.subplot(2, 1, 2)
    plt.plot(step_list, loss_avg_list)
    plt.title('loss_step')
    plt.xlabel('step')
    plt.ylabel('loss_avg')

    plt.savefig('result.jpg')

