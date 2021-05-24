import gym
import compiler_gym
from dqn import Agent
import numpy as np

# envs -> compiler_gym.COMPILER_GYM_ENVS

# Compiler = LLVM | Observation Type = Autophase | Reward Signal = IR Instruction count relative to -Oz

env = gym.make("llvm-autophase-ic-v0")

# Use existing dqn to make better decisions
agent = Agent(gamma = 0.80, epsilon = 0.1, batch_size = 32,
            n_actions = env.action_space.n, eps_end = 0.08, input_dims = [56], alpha = 0.005)

# download existing dataset of programs/benchmarks
env.require_datasets(['cBench-v0', 'blas-v0'])

# action space is described by env.action_space
# the observation space (autophase) is a 56 dimensional vector

# env.reset() must be called to initialize environment and create initial observation of said env

for i in range(1,101):
    observation = env.reset()
    #observation = observation.astype(np.float32)
    done = False
    total = 0
    while not done:
        action = agent.choose_action(observation)
        new_observation, reward, done, info = env.step(action)
        total += reward
        agent.store_transition(action, observation, reward, new_observation, done)
        agent.learn()
        observation = new_observation
        print("Step " + str(i) + " Cumulative Total " + str(total) +  " Epsilon " + str(agent.epsilon) + " Action " + str(action))


# env.commandline() will write the opt command equivalent to the sequence of transformations made by agent
print(env.commandline())
# save the model for future reference
# env.write_bitcode("./cg_autotuned_program.bc")
env.close()