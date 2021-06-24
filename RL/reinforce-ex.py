# from https://medium.com/samkirkiles/reinforce-policy-gradients-from-scratch-in-numpy-6a09ae0dfe12 
import gym
import numpy as np
import matplotlib.pyplot as plt
import copy 

#Hyperparameters
NUM_EPISODES = 10000
LEARNING_RATE = 0.000025
GAMMA = 0.99

# Create gym and seed numpy
env = gym.make('CartPole-v0')
nA = env.action_space.n
np.random.seed(1)

# Init weight
w = np.random.rand(4, 2)

# Keep stats for final print of graph
episode_rewards = []

# Our policy that maps state to action parameterized by w
def policy(state,w):
	z = state.dot(w)
	exp = np.exp(z)
	return exp/np.sum(exp)

# Vectorized softmax Jacobian
def softmax_grad(softmax):
    s = softmax.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)

# Main loop 
# Make sure you update your weights AFTER each episode
for e in range(NUM_EPISODES):

	state = env.reset()[None,:]

	grads = []	
	rewards = []

	# Keep track of game score to print
	score = 0

	while True:

		# Uncomment to see your model train in real time (slower)
		#env.render()
		#print(f"state: {state}")
		#print(f"weights: \n{w}")
		# Sample from policy and take action in environment
		probs = policy(state,w)
		print(f"action probabilities: {probs}")
		#print(f"action probs: {probs}")
		# choose action with probability
		#print(nA)
		action = np.random.choice(nA,p=probs[0])
		print(f"action taken: {action}")
		next_state,reward,done,_ = env.step(action)
		next_state = next_state[None,:] # wraps into another array to make tensor

		# Compute gradient and save with reward in memory for our weight updates
		print(f"jacobian matrix: \n{softmax_grad(probs)}")
		dsoftmax = softmax_grad(probs)[action,:]
		print(f"softmax gradient: {dsoftmax}")
		dlog = dsoftmax / probs[0,action]
		print(f"dlog: {dlog}")
		grad = state.T.dot(dlog[None,:])
		print(f"grad: \n{grad}")

		grads.append(grad)
		rewards.append(reward)		

		score+=reward

		# Dont forget to update your old state to the new state
		state = next_state

		if done:
			break

	# Weight update
	for i in range(len(grads)):

		# Loop through everything that happend in the episode and update towards the log policy gradient times **FUTURE** reward
		w += LEARNING_RATE * grads[i] * sum([ r * (GAMMA ** r) for t,r in enumerate(rewards[i:])])
	
	# Append for logging and print
	episode_rewards.append(score) 
	print("EP: " + str(e) + " Score: " + str(score) + "         ",end="\r", flush=False) 

plt.plot(np.arange(NUM_EPISODES),episode_rewards)
plt.show()
env.close()