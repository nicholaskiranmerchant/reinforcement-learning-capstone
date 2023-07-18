'''
The point of this experiment is to show that,
if we can reliably hit the initiation set of an option,
we can learn a good initiation classifier WITHOUT fixing
it after some number of hits. 
'''

# Core imports
import numpy as np
import ipdb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch

# Local imports
from dscp.rl.MDPClass import MDP
from dscp.rl.PolicyClass import Policy
from dscp.mdps.GymMDPClass import GymMDP
from dscp.policies.sac.SACClass import SAC
from dscp.policies.ddpg.DDPGAgentClass import DDPGAgent
from dscp.policies.td3.TD3AgentClass import TD3

from dscp.rl.ReplayBufferClass import ReplayBuffer

# Typing imports
from typing import Final

# The environment for this experiment
# Maybe we should move this out-- especially if we want to
# do a number of experiments over different environments
# and reward conditions
class AntMazeEx1MDP(GymMDP):
    def __init__(self):
        super().__init__("Ant-v2")
        #MountainCarContinuous-v0
        #Ant-v2

    def get_gamma(self) -> float:
        return 0.99

    def get_reward(self):
        state = self.get_state()
        pos = state[:2]

        goal_pos = np.array([0.38, 0.2])

        return -1 * np.linalg.norm(pos - goal_pos)

def plot_results(episode_rewards : np.ndarray, episode_durations : np.ndarray) -> None:
    df = pd.DataFrame({
        "episode" : np.arange(episode_rewards.shape[0]),
        "reward" : episode_rewards,
        "duration" : episode_durations 
    })

    df.to_csv("output.csv")

    sns.relplot(
        data=pd.melt(df, id_vars=["episode"], var_name="metric"),
        kind="line",
        x="episode",
        y="value",
        hue="metric"
    )

    plt.savefig("output.png")
'''
def plot_qf(policy: Policy) -> None:
    replay_buffer : ReplayBuffer = policy.replay_buffer
    output = replay_buffer.sample_transition_batch(100000)
    plot_dict = {"xpos": [], "ypos": [], "qval": []}

    for state in output["states"]:

    for j in range(batch_size):
        i = np.random.randint(0, self.num_transitions)
        
        states[j,:] = self._states[i,:]
        actions[j,:] = self._actions[i,:]
        next_states[j,:] = self._next_states[i,:]
        rewards[j] = self._rewards[i]
        terminals[j] = self._terminals[i]
'''


def flat_rollout(mdp : MDP, policy : Policy, num_episodes : int, num_steps : int):
    episode_rewards = np.zeros((num_episodes,))
    episode_durations = np.zeros((num_episodes,))
    for i in range(num_episodes):
        mdp.reset()
        steps_executed = 0
        total_reward = 0
        for j in range(num_steps):
            state = mdp.get_state()
            action = policy.sample_action(state)
            mdp.act(action)
            next_state = mdp.get_state()
            reward = mdp.get_reward()
            terminal = mdp.get_terminal()

            policy.add_transition_to_buffer(state, action, next_state, reward, terminal)

            total_reward += reward
            steps_executed += 1

            if terminal:
                break
        
        policy.update_from_buffer()
        
        episode_rewards[i] = total_reward
        episode_durations[i] = steps_executed
        print(f"Completed episode {i} in {steps_executed} steps with reward {total_reward}")

    return episode_rewards, episode_durations

'''
def her_rollout(mdp : MDP, policy : Policy, num_episodes : int, num_steps : int, goal_state : np.ndarray, threshold : float) -> None:
    # IGNORES the underlying reward function of the MDP
    assert goal_state.shape == (mdp.get_state_dim(),)

    def gc_train(gc_state, trajectory):
        for (state, action, next_state, reward, terminal) in trajectory
            gc_state = np.hstack(state, gc_state)
            gc_next_state = np.hstack(next_state, gc_state)

            gc_distance = np.linalg.norm(next_state - gc_state)
            gc_reward = -1 * gc_distance
            gc_terminal = gc_distance < threshold 

            policy.add_transition_to_buffer(gc_state, action, gc_next_state, gc_reward, gc_terminal)
            policy.update_from_buffer()

    episode_rewards = np.zeros((num_episodes,))
    episode_durations = np.zeros((num_episodes,))
    for i in range(num_episodes):
        mdp.reset()
        steps_executed = 0
        total_reward = 0
        current_trajectory = []
        for j in range(num_steps):
            state = mdp.get_state()
            action = policy.sample_action(state)
            mdp.act(action)
            next_state = mdp.get_state()
            reward = mdp.get_reward()
            terminal = mdp.get_terminal()

            #policy.add_transition_to_buffer(state, action, next_state, reward, terminal)
            current_trajectory.append((state, action, next_state, reward, terminal))

            total_reward += reward
            steps_executed += 1

            if terminal:
                break

        (_, _, her_state, _, _) = current_trajectory[-1]
        gc_train(goal_state, current_trajectory)
        gc_train(her_state, current_trajectory)
        
        episode_rewards[i] = total_reward
        episode_durations[i] = steps_executed
        print(f"Completed episode {i} in {steps_executed} steps with reward {total_reward}")
        print(f"Current replay buffer size is {policy.replay_buffer.num_transitions}")

    return episode_rewards, episode_durations
'''


if __name__ == "__main__":
    mdp = AntMazeEx1MDP()

    '''
    policy = SAC(
        mdp.get_state_dim(),
        mdp.get_action_dim(),
        mdp.get_action_bounds(),
        mdp.get_gamma(),
        300000,
        128,
        0.001,
        0.001,
        0.2,
        0.995
    )
    '''

    '''
    policy = DDPGAgent(
        mdp.get_state_dim(),
        mdp.get_action_dim(),
        0,
        None,
    )
    '''
    policy = TD3(
        mdp.get_state_dim(),
        mdp.get_action_dim(),
        mdp.get_action_bounds()[0,1],
        device=torch.device("cpu")
        )

    rewards, durations = flat_rollout(mdp, policy, 1000, 1000)

    plot_results(rewards, durations)

    ipdb.set_trace()

    '''
    TODO:
        (i) Implement option class w/ HER
        (ii) Write option rollout
        (iii) Run rollout, pickle results
        (iv) Set up seeding and devices
    '''





