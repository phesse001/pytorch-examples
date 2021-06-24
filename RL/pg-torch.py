import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

NUM_EPISODES = 1000
GAMMA = 0.99

class Policy(nn.Module):
	def __init__(self, input_dim, output_dim, lr):
		# super allows us to call superclass methods (i.e nn.Module methods)
		super(self, Policy).__init__()
		self.input_dim = input_dim
		self.output_dim	 = fc_dim
		self.fc1 = nn.Linear(self.input_dim, self.output_dim)
		self.optimizer = optim.Adam(self.parameters(), lr)
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.to(self.device)
		
	def forward(self, x):
		x= F.softmax(self.fc1(x))
		return x

# Main training loop
env = gym.make('CartPole-v0')
action_space = env.action_space.n

policy = Policy(32, 2, .001)

for i in range(NUM_EPISODES):
	state = env.reset()
	# convert to tensor
	state = torch.from_numpy(state, dtype=np.float32)
	rewards = []
	log_probs = []
		
	score = 0

	# Loop forever until episode terminates
	while True:

		probs = policy.forward(state)
		m = Categorical(probs)
		action = m.sample() # samples from probability distribution
		next_state, reward, done, info = env.step(action)
		logp = m.log_probs(action)
		log_probs.append(logp)
		rewards.append(reward)
		score += reward

		if done:
			break

  # At the end of every episode, update weights (monte-carlo method)
	# accumulate sum of discounted rewards by going backwards and inserting at start
	R = 0
	policy_loss = []
	returns = []
	for r in rewards[::-1]:
		R = r + GAMMA * R
		returns.insert(0, R)

	returns = torch.tensor(returns)
	# in example they have normalization term, I will use it if I have to
	# returns = (returns - returns.mean() / returns.std() + eps)
	for log_prob, R in zip(log_probs, returns):
		policy_loss.append(-log_prob * R) 
	
	policy.optimizer.zero_grad()
	policy_loss = torch.cat(policy_loss).sum()
	policy_loss.backward() # calculate gradients of the policy loss wrt each network param
	policy.optimizer.step()





