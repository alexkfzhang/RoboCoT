import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from encoder import Encoder
import time
import copy
import random

class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class Agent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb, augment,
                 update_target_every, obs_type):
        self.device = device
        self.lr = lr
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.augment = augment
        self.update_target_every = update_target_every
        self.use_encoder = True if obs_type == 'pixels' else False


        # models
        if self.use_encoder:
            self.encoder = Encoder(obs_shape).to(device)
            self.encoder_target = Encoder(obs_shape).to(device)
            self.encoder_bc = Encoder(obs_shape).to(device)
            repr_dim = self.encoder.repr_dim
        else:
            repr_dim = obs_shape[0]

        self.trunk_target = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.LayerNorm(feature_dim), nn.Tanh()).to(device)

        self.actor = Actor(repr_dim, action_shape, feature_dim,
                           hidden_dim).to(device)
        self.actor_bc = Actor(repr_dim, action_shape, feature_dim,
                              hidden_dim).to(device)

        self.critic = Critic(repr_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(repr_dim, action_shape,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        if self.use_encoder:
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # data augmentation
        self.aug = utils.RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

    def __repr__(self):
        return "agent"

    def train(self, training=True):
        self.training = training
        if self.use_encoder:
            self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)

        obs = self.encoder(obs.unsqueeze(0)) if self.use_encoder else obs.unsqueeze(0)

        stddev = utils.schedule(self.stddev_schedule, step)

        dist = self.actor(obs, stddev)

        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)

        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        # optimize encoder and critic
        if self.use_encoder:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        if self.use_encoder:
            self.encoder_opt.step()

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        return metrics

    def update_actor(self, obs, success_obs, success_action, step, flag):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)

        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = - Q.mean()

        if flag == 1:
            stddev = 0.1
            dist_bc = self.actor(success_obs, stddev)
            log_prob_bc = dist_bc.log_prob(success_action).sum(-1, keepdim=True)
            actor_loss += - log_prob_bc.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
            metrics['actor_q'] = Q.mean().item()
            metrics['regularized_rl_loss'] = -Q.mean().item()
            metrics['rl_loss'] = -Q.mean().item()


        return metrics

    def update(self, replay_iter, expert_demo, step, flag):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        if flag == 1:
            success_obs = []
            success_action = []
            for _ in range(256):
                success_obs.append(expert_demo[random.randint(0,len(expert_demo)-1)]['obs'])
                success_action.append(expert_demo[random.randint(0,len(expert_demo)-1)]['ac'])
            success_obs = torch.Tensor(np.array(success_obs)).to("cuda")
            success_action = torch.Tensor(np.array(success_action)).to("cuda")

        else:
            success_obs = []
            success_action = []
            flag = 0

        # augment
        if self.use_encoder and self.augment:
            obs = self.aug(obs.float())
            next_obs = self.aug(next_obs.float())
            if flag == 1:
                success_obs = self.aug(success_obs)
        else:
            obs = obs.float()
            next_obs = next_obs.float()
            if flag == 1:
                success_obs = success_obs

        if self.use_encoder:
            # encode
            obs = self.encoder(obs)
            with torch.no_grad():
                next_obs = self.encoder(next_obs)

            if flag==1:
                success_obs = self.encoder(success_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        if flag == 1:
            metrics.update(self.update_actor(obs.detach(), success_obs.detach(), success_action.detach(), step, flag))
        else:
            metrics.update(self.update_actor(obs.detach(), obs.detach(), action.detach(), step, flag))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics

    def vlm_rewarder(self, observations, steps):
        vlm_reward = []

        obs = observations
        import cv2

        from mmpt.models import MMPTModel

        model, tokenizer, aligner = MMPTModel.from_pretrained(
            "projects/retri/videoclip/how2.yaml")

        model.to("cuda")
        model.eval()



        caps1, cmasks1 = aligner._build_text_seq(
            tokenizer("the robot hand moves towards soccer", add_special_tokens=False)["input_ids"]
        )

        caps1, cmasks1 = caps1[None, :], cmasks1[None, :]  # bsz=1

        caps2, cmasks2 = aligner._build_text_seq(
            tokenizer(
                "the robot hand pushes soccer into the net",
                add_special_tokens=False)["input_ids"]
        )

        caps2, cmasks2 = caps2[None, :], cmasks2[None, :]  # bsz=1

        caps3, cmasks3 = aligner._build_text_seq(
            tokenizer(
                "the soccer is in the net",
                add_special_tokens=False)["input_ids"]
        )

        caps3, cmasks3 = caps3[None, :], cmasks3[None, :]  # bsz=1

        caps4, cmasks4 = aligner._build_text_seq(
            tokenizer(
                "the robot hand opens the drawer",
                add_special_tokens=False)["input_ids"]
        )

        caps4, cmasks4 = caps4[None, :], cmasks4[None, :]  # bsz=1

        caps5, cmasks5 = aligner._build_text_seq(
            tokenizer(
                "the robot hand is at the left of the box",
                add_special_tokens=False)["input_ids"]
        )

        caps5, cmasks5 = caps5[None, :], cmasks5[None, :]  # bsz=1

        caps6, cmasks6 = aligner._build_text_seq(
            tokenizer(
                "the robot hand is moving away from soccer",
                add_special_tokens=False)["input_ids"]
        )

        caps6, cmasks6 = caps6[None, :], cmasks6[None, :]  # bsz=1

        caps7, cmasks7 = aligner._build_text_seq(
            tokenizer(
                "the robot hand is at the top of soccer",
                add_special_tokens=False)["input_ids"]
        )

        caps7, cmasks7 = caps7[None, :], cmasks7[None, :]  # bsz=1

        caps8, cmasks8 = aligner._build_text_seq(
            tokenizer(
                "the robot hand is at the left of the red button",
                add_special_tokens=False)["input_ids"]
        )

        caps8, cmasks8 = caps8[None, :], cmasks8[None, :]  # bsz=1

        caps9, cmasks9 = aligner._build_text_seq(
            tokenizer(
                "the robot hand is at the left of the box",
                add_special_tokens=False)["input_ids"]
        )

        caps9, cmasks9 = caps9[None, :], cmasks9[None, :]  # bsz=1

        caps10, cmasks10 = aligner._build_text_seq(
            tokenizer(
                "the robot hand is at the left of the box",
                add_special_tokens=False)["input_ids"]
        )

        caps10, cmasks10 = caps10[None, :], cmasks10[None, :]  # bsz=1


        video_frames = []
        for item in range(len(obs)):

            video_frames.append(np.array(np.reshape(cv2.resize(np.transpose(obs[item], (2, 1, 0)), (224, 224)), (3, 224, 224, 3)))[0])

            if len(video_frames) < 30:
                continue
            elif len(video_frames) == 30:
                with torch.no_grad():

                    output1 = model(torch.Tensor([[np.array(video_frames)]]).cuda(), torch.Tensor(caps1).cuda(), torch.Tensor(cmasks1).cuda(), return_score=True)

                    output2 = model(torch.Tensor([[np.array(video_frames)]]).cuda(), torch.Tensor(caps2).cuda(), torch.Tensor(cmasks2).cuda(), return_score=True)

                    output3 = model(torch.Tensor([[np.array(video_frames)]]).cuda(), torch.Tensor(caps3).cuda(), torch.Tensor(cmasks3).cuda(), return_score=True)

                  #  output4 = model(torch.Tensor([[np.array(video_frames)]]).cuda(), torch.Tensor(caps4).cuda(), torch.Tensor(cmasks4).cuda(), return_score=True)

                  #  output5 = model(torch.Tensor([[np.array(video_frames)]]).cuda(), torch.Tensor(caps5).cuda(), torch.Tensor(cmasks5).cuda(), return_score=True)

                    output6 = model(torch.Tensor([[np.array(video_frames)]]).cuda(), torch.Tensor(caps6).cuda(), torch.Tensor(cmasks6).cuda(), return_score=True)

                    output7 = model(torch.Tensor([[np.array(video_frames)]]).cuda(), torch.Tensor(caps7).cuda(), torch.Tensor(cmasks7).cuda(), return_score=True)

                  #  output8 = model(torch.Tensor([[np.array(video_frames)]]).cuda(), torch.Tensor(caps8).cuda(), torch.Tensor(cmasks8).cuda(), return_score=True)

                    #output9 = model(torch.Tensor([[np.array(video_frames)]]).cuda(), torch.Tensor(caps9).cuda(), torch.Tensor(cmasks9).cuda(), return_score=True)

                  #  output10 = model(torch.Tensor([[np.array(video_frames)]]).cuda(), torch.Tensor(caps10).cuda(), torch.Tensor(cmasks10).cuda(), return_score=True)

                if item <= 78:
                    tau = 1
                    maximum = np.max([np.exp(output1["score"].detach().cpu().numpy()[0]/tau), np.exp(output2["score"].detach().cpu().numpy()[0]/tau),
                                      np.exp(output3["score"].detach().cpu().numpy()[0]/tau)])

                    reward = maximum / (maximum + np.exp(output6["score"].detach().cpu().numpy()[0]/tau)+
                                        np.exp(output7["score"].detach().cpu().numpy()[0] / tau) + 1e-8)

                    for _ in range(16):
                        vlm_reward.append(reward)
                        video_frames.pop(0)
                else:
                    for _ in range(len(obs)-64):
                        vlm_reward.append(reward)

                    break

        return vlm_reward



    def save_snapshot(self):
        keys_to_save = ['actor', 'critic']
        if self.use_encoder:
            keys_to_save += ['encoder']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        return payload

    def load_snapshot(self, payload):
        for k, v in payload.items():
            self.__dict__[k] = v
        self.critic_target.load_state_dict(self.critic.state_dict())
        if self.use_encoder:
            self.encoder_target.load_state_dict(self.encoder.state_dict())
        self.trunk_target.load_state_dict(self.actor.trunk.state_dict())

        # Update optimizers
        if self.use_encoder:
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

