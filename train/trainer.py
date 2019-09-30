from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from modules.sample_generator import *

from train_utils import *
from model import Actor, Critic
from utils.getbatch_actor import getbatch_actor
from utils.cal_distance import cal_distance
from utils.PILloader import loader
from utils.np2tensor import np2tensor
BATCH_SIZE = 64
LEARNING_RATE = 0.001
GAMMA = 0.99
TAU = 0.001


class Trainer:

	def __init__(self, ram, action_lim=1, action_dim=3):
		"""
		:param state_dim: Dimensions of state (int)
		:param action_dim: Dimension of action (int)
		:param action_lim: Used to limit action in [-action_lim,action_lim]
		:param ram: replay memory buffer object
		:return:
		"""
		self.action_lim = action_lim
		self.ram = ram
		self.iter = 0
		self.load_episode = 0
		# self.noise = OUAction.OrnsteinUhlenbeckActionNoise()

		self.actor = Actor

		# self.actor.init_weight()
		self.actor = self.actor.cuda()
		self.target_actor = Actor
		self.target_actor = self.target_actor.cuda()
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),LEARNING_RATE)

		self.critic = Critic
		self.critic = self.critic.cuda()
		self.target_critic = Critic
		self.target_critic = self.target_critic.cuda()
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),LEARNING_RATE)
		if torch.cuda.device_count() > 1:
			print("Let's use", torch.cuda.device_count(), "GPUs!")
			self.actor = nn.DataParallel(self.actor)
			self.target_actor = nn.DataParallel(self.target_actor)
			self.critic = nn.DataParallel(self.critic)
			self.target_critic = nn.DataParallel(self.target_critic)
		hard_update(self.target_actor, self.actor)
		hard_update(self.target_critic, self.critic)

	def init_actor(self, image, gt):
		np.random.seed(123)
		torch.manual_seed(456)
		torch.cuda.manual_seed(789)

		batch_num = 64
		maxiter = 10
		self.actor.train()
		init_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.0001)
		loss_func = torch.nn.MSELoss()
		_, _, out_flag_first = getbatch_actor(np.array(image), np.array(gt).reshape([1, 4]))


		actor_samples = np.round(gen_samples(SampleGenerator('gaussian', (image.shape[1],image.shape[0]), 0.2, 1.1, None), gt, 640, [0.6, 1], None))
		idx = np.random.permutation(actor_samples.shape[0])

		batch_img_g, batch_img_l, _ = getbatch_actor(np.array(image), actor_samples)
		batch_distance = cal_distance(actor_samples, np.tile(gt, [actor_samples.shape[0], 1]))
		batch_distance = np.array(batch_distance).astype(np.float32)
		while (len(idx) < batch_num * maxiter):
			idx = np.concatenate([idx, np.random.permutation(actor_samples.shape[0])])

		pointer = 0
		# torch_image = loader(image.resize((255, 255), Image.ANTIALIAS)).unsqueeze(0).cuda()
		for iter in range(maxiter):
			next = pointer + batch_num
			cur_idx = idx[pointer: next]
			pointer = next
			feat = self.actor(batch_img_l[cur_idx], batch_img_g[cur_idx])
			loss = loss_func(feat, (torch.FloatTensor(batch_distance[cur_idx])).cuda())

			self.actor.zero_grad()
			loss.backward()
			init_optimizer.step()
			if True:
				print("init actor Iter %d, Loss %.10f" % (iter, loss.item()))
			if loss.item() < 0.0001:
				deta_flag = 0
				return deta_flag
			deta_flag = 1
		return deta_flag

	def get_exploitation_action(self, state):
		"""
		gets the action from target actor added with exploration noise
		:param state: state (Numpy array)
		:return: sampled action (Numpy array)
		"""
		state = Variable(torch.from_numpy(state))
		action = self.target_actor.forward(state).detach()
		return action.data.numpy()

	def get_exploration_action(self, state):
		"""
		gets the action from actor added with exploration noise
		:param state: state (Numpy array)
		:return: sampled action (Numpy array)
		"""
		state = Variable(torch.from_numpy(state))
		action = self.actor.forward(state).detach()
		new_action = action.data.numpy() + (self.noise.sample() * self.action_lim)
		return new_action

	def optimize(self):
		"""
		Samples a random batch from replay memory and performs optimization
		:return:
		"""
		self.actor.train()
		self.critic.train()
		s_g, s_g_, a_arr, r_arr,s_l, s_l_ = self.ram.sample(BATCH_SIZE)

		s_g = torch.from_numpy(s_g).cuda()
		s_g_ = torch.from_numpy(s_g_).cuda()
		a1 = torch.from_numpy(a_arr).cuda()
		r1 = torch.from_numpy(r_arr).cuda()
		s_l = torch.from_numpy(s_l).cuda()
		s_l_ = torch.from_numpy(s_l_).cuda()

		# ---------------------- optimize critic ----------------------
		# Use target actor exploitation policy here for loss evaluation
		a2 = self.target_actor.forward(s_l_,s_g_).detach()
		next_val = torch.squeeze(self.target_critic.forward(s_l_, a2).detach())
		# y_exp = r + gamma*Q'( s2, pi'(s2))
		y_expected = r1 + GAMMA*next_val
		# y_pred = Q( s1, a1)
		y_predicted = torch.squeeze(self.critic.forward(s_l, a1))
		# compute critic loss, and update the critic
		loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
		self.critic_optimizer.zero_grad()
		loss_critic.backward()
		self.critic_optimizer.step()

		# ---------------------- optimize actor ----------------------
		pred_a1 = self.actor.forward(s_l, s_g)
		loss_actor = -1*torch.sum(self.critic.forward(s_l, pred_a1))
		self.actor_optimizer.zero_grad()
		loss_actor.backward()
		self.actor_optimizer.step()

		soft_update(self.target_actor, self.actor, TAU)
		soft_update(self.target_critic, self.critic, TAU)

		# if self.iter % 100 == 0:
		# 	print 'Iteration :- ', self.iter, ' Loss_actor :- ', loss_actor.data.numpy(),\
		# 		' Loss_critic :- ', loss_critic.data.numpy()
		# self.iter += 1

	def save_models(self, episode_count):
		"""
		saves the target actor and critic models
		:param episode_count: the count of episodes iterated
		:return:
		"""
		torch.save(self.target_actor.state_dict(), '../models/Double_agent/' + str(episode_count + self.load_episode) + '_DA_actor.pth')
		torch.save(self.target_critic.state_dict(), '../models/Double_agent/' + str(episode_count + self.load_episode) + '_DA_critic.pth')
		print ('Models saved successfully')

	def load_models(self, load_episode):
		"""
		loads the target actor and critic models, and copies them onto actor and critic models
		:param episode: the count of episodes iterated (used to find the file name)
		:return:
		"""
		self.load_episode = load_episode
		self.actor.load_state_dict(torch.load('../models/Double_agent/' + str(load_episode) + '_DA_actor.pth'))
		self.critic.load_state_dict(torch.load('../models/Double_agent/' + str(load_episode) + '_DA_critic.pth'))
		hard_update(self.target_actor, self.actor)
		hard_update(self.target_critic, self.critic)
		print ('Models loaded succesfully')

	def show_critic_loss(self):
		self.critic.eval()
		s_g, s_g_, a_arr, r_arr, s_l, s_l_ = self.ram.sample(BATCH_SIZE)

		# s_g = torch.from_numpy(s_g).cuda()
		s_g_ = torch.from_numpy(s_g_).cuda()
		a1 = torch.from_numpy(a_arr).cuda()
		r1 = torch.from_numpy(r_arr).cuda()
		s_l = torch.from_numpy(s_l).cuda()
		s_l_ = torch.from_numpy(s_l_).cuda()

		# ---------------------- optimize critic ----------------------
		# Use target actor exploitation policy here for loss evaluation
		a2 = self.target_actor.forward(s_l_, s_g_).detach()
		next_val = torch.squeeze(self.target_critic.forward(s_l_, a2).detach())
		# y_exp = r + gamma*Q'( s2, pi'(s2))
		y_expected = r1 + GAMMA * next_val
		# y_pred = Q( s1, a1)
		y_predicted = torch.squeeze(self.critic.forward(s_l, a1))
		# compute critic loss, and update the critic
		loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
		return loss_critic
