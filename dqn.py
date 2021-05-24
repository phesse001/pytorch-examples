import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np

# Add the ability to replay memory -> The memory should store transitions that the agent observes.
# By sammpling from this set of memories randomly, the transitions build up a decorrelated batch
# which stabilizes the training.


# Maps the state,action to the next_state,reward
# Acts sort of like a class -> can call T = Transition(some_state, some_action, ...)

# The Network

class DQN(nn.Module):
	def __init__(self, ALPHA, input_dims, fc1_dims, fc2_dims, fc3_dims, n_actions):
		super(DQN,self).__init__()
		self.input_dims = input_dims
		self.fc1_dims = fc1_dims
		self.fc2_dims = fc2_dims
		self.fc3_dims = fc3_dims
		self.n_actions = n_actions
		self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
		self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
		self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
		self.fc4 = nn.Linear(self.fc3_dims, self.n_actions)
		self.optimizer = optim.Adam(self.parameters(), lr = ALPHA)
		self.loss = nn.MSELoss()
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.to(self.device)

	def forward(self, state):
		# first fully connected layer takes the state in as input, pass that output as input to activation function
		x = F.relu(self.fc1(state))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		actions = self.fc4(x)

		return actions

class Agent():
	# gamma is the weighting of furture rewards
	# epsilon is the amount of time the agent explores environment???
	def __init__(self, gamma, epsilon, alpha, input_dims, batch_size,
	         n_actions, max_mem_size = 100000, eps_end = 0.01, eps_dec = 5e-5):
		self.gamma = gamma
		self.epsilon = epsilon
		self.eps_end = eps_end
		self.eps_dec = eps_dec
		self.alpha = alpha
		self.n_actions = n_actions
		self.action_space = [i for i in range(n_actions)]
		self.max_mem_size = max_mem_size
		self.batch_size = batch_size
		# keep track of position of first available memory
		self.mem_cntr = 0
		self.Q_eval = DQN(alpha, input_dims, fc1_dims=1024, fc2_dims=1024, fc3_dims=1024, n_actions = self.n_actions)

		# star unpacks list into positional arguments
		self.state_mem = np.zeros((self.max_mem_size, *input_dims), dtype=np.float32)
		self.new_state_mem = np.zeros((self.max_mem_size, *input_dims), dtype=np.float32)
		self.action_mem = np.zeros(self.max_mem_size, dtype=np.int32)
		self.reward_mem = np.zeros(self.max_mem_size, dtype=np.float32)
		self.terminal_mem = np.zeros(self.max_mem_size, dtype=np.bool)

	def store_transition(self, action, state, reward, new_state, done):
		# what is the position of the first unoccupied memory
		index = self.mem_cntr % self.max_mem_size
		self.state_mem[index] = state
		self.new_state_mem[index] = new_state
		self.action_mem[index] = action
		self.reward_mem[index] = reward
		self.terminal_mem[index] = done

		self.mem_cntr += 1

	def choose_action(self, observation):
		if np.random.random() > self.epsilon:
			# sends observation as tensor to device
			# convert to float - > compiler gyms autophase vector is a long
			observation = observation.astype(np.float32)
			state = torch.tensor([observation]).to(self.Q_eval.device)
			actions = self.Q_eval.forward(state)
			action = torch.argmax(actions).item()
		else:
			# take random action
			action = np.random.choice(self.action_space)

		return action

	def learn(self):
		# start learning as soon as batch size of memory is filled
		if self.mem_cntr < self.batch_size:
			return

		# set gradients to zero
		self.Q_eval.optimizer.zero_grad()
		# select subset of memorys
		max_mem = min(self.mem_cntr, self.max_mem_size)
		# take a selection of the size of the batch size from the current pool of memory's
		# pool of memory's will be full by the time we get here
		batch = np.random.choice(max_mem, self.batch_size, replace=False)

		batch_index = np.arange(self.batch_size, dtype = np.int32)
		# sending a (random)batch of states to device
		state_batch = torch.Tensor(self.state_mem[batch]).to(self.Q_eval.device)
		new_state_batch = torch.tensor(self.new_state_mem[batch]).to(self.Q_eval.device)
		reward_batch = torch.tensor(self.reward_mem[batch]).to(self.Q_eval.device)
		terminal_batch = torch.tensor(self.terminal_mem[batch]).to(self.Q_eval.device)
		action_batch = self.action_mem[batch]
		q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
		print(self.Q_eval.forward(state_batch))
		print("size of state batch is " + str(self.Q_eval.forward(state_batch).size()))
		print("u" + str(q_eval))
		print(q_eval.size())
		# In Q-learning, the agent updates the value of executing an action
		# in the current state, using the values of executing actions in a
		# successive state. This procedure often results in an instability
		# because the values change simultaneously on both sides of the
		# update equation
		q_next = self.Q_eval.forward(new_state_batch)
		q_next[terminal_batch] = 0.0
		q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]

		loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
		loss.backward()
		self.Q_eval.optimizer.step()
		print(str(self.Q_eval))
		
		if self.epsilon > self.eps_end:
		    self.epsilon -= self.eps_dec
		else:
			self.epsilon = self.eps_end
