import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

# Based of example given in pytorch docs -> https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py

NUM_EPISODES = 10000
GAMMA = 0.99
LR = .001

class Policy(nn.Module):
	def __init__(self, input_dim, fc_dim, output_dim):
		# super allows us to call superclass methods (i.e nn.Module methods)
		super(Policy, self).__init__()
		self.input_dim = input_dim
		self.output_dim	 = output_dim
		self.fc_dim = fc_dim
		self.fc1 = nn.Linear(self.input_dim, self.fc_dim)
		self.fc2 = nn.Linear(self.fc_dim, self.output_dim)
		self.optimizer = optim.Adam(self.parameters(), LR)
		self.saved_log_probs = []
		self.rewards = []
		
	def forward(self, x):
		x = self.fc1(x)
		x = F.softmax(self.fc2(x))
		return x

# Main training loop
env = gym.make('CartPole-v0')

policy = Policy(4, 128, 2)
eps = np.finfo(np.float32).eps.item()
running_reward = 10

for i in range(NUM_EPISODES):
	state = env.reset()
	# convert to tensor
	score = 0
	rewards = []
	log_probs = []
	# Loop forever until episode terminates
	while True:
		state = torch.from_numpy(state).float().unsqueeze(0) # have to add extra dim for some reason
		probs = policy.forward(state)
		m = Categorical(probs)
		action = m.sample() # samples from probability distribution
		state, reward, done, info = env.step(action.item())
		logp = m.log_prob(action)
		log_probs.append(logp)
		rewards.append(reward)
		score += reward

		if done:
			break

	running_reward = 0.05 * score + (1 - 0.05) * running_reward
	if i % 10 == 0:
		print(f"episode: {i} average reward: {running_reward}")

    # At the end of every episode, update weights (monte-carlo method)
	# accumulate sum of discounted rewards by going backwards and inserting at start
	R = 0
	policy_loss = []
	returns = []
	for r in rewards[::-1]:
		R = r + GAMMA * R
		returns.insert(0, R)

	returns = torch.tensor(returns)
	#returns = (returns - returns.mean()) / (returns.std() + eps)

	for log_prob, R in zip(log_probs, returns):
		policy_loss.append(-log_prob * R) 
	
	policy.optimizer.zero_grad()
	policy_loss = torch.cat(policy_loss).sum()
	policy_loss.backward() # calculate gradients of the policy loss wrt each network param
	policy.optimizer.step()





