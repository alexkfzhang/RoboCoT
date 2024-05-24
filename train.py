#!/usr/bin/env python3

import warnings
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
from pathlib import Path

import numpy as np
import torch
from dm_env import specs

import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder
import pickle

warnings.filterwarnings('ignore', category=DeprecationWarning)
torch.backends.cudnn.benchmark = True

import torch
import torch.nn as nn
from torchvision import transforms
import cv2
from glob import glob

class WorkspaceIL:
    def __init__(self):
        self.work_dir = Path.cwd()
        self.work_dir = self.work_dir / 'ours/door_open'
        print(f'workspace: {self.work_dir}')
        import yaml
        config = "config.yaml"
        with open(config, 'r', encoding='utf-8') as fin:
            self.cfg = yaml.load(fin, Loader=yaml.FullLoader)

        self.device = torch.device(self.cfg['device'])
        self.setup()

        from agent import Agent
        self.agent = Agent(self.train_env.observation_spec()[self.cfg['obs_type']].shape, self.train_env.action_spec().shape,
                           self.cfg['device'], float(self.cfg['lr']), int(self.cfg['feature_dim']),
                           int(self.cfg['hidden_dim']), float(self.cfg['critic_target_tau']), int(self.cfg['num_expl_steps']),
                           int(self.cfg['update_every_steps']), float(self.cfg['stddev_schedule']), float(self.cfg['stddev_clip']), self.cfg['use_tb'], self.cfg['augment'],
                           int(self.cfg['update_target_every']), self.cfg['obs_type'])

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0
        self.flag=0
        self.expert_demo = []



    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg['use_tb'])
        # create envs
        from robot_suite import make
        self.train_env = make(name='door-open-v2', frame_stack=int(self.cfg['frame_stack']), action_repeat=int(self.cfg['action_repeat']), seed=int(self.cfg['seed']))
        self.eval_env = make(name='door-open-v2', frame_stack=int(self.cfg['frame_stack']), action_repeat=int(self.cfg['action_repeat']), seed=int(self.cfg['seed']))

        # create replay buffer
        data_specs = [
            self.train_env.observation_spec()[self.cfg['obs_type']],
            self.train_env.action_spec(),
            specs.Array((1,), np.float32, 'reward'),
            specs.Array((1,), np.float32, 'discount')
        ]

        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer')

        self.replay_loader = make_replay_loader(
            self.work_dir / 'buffer', int(self.cfg['replay_buffer_size']),
            int(self.cfg['batch_size']), int(self.cfg['replay_buffer_num_workers']),
            self.cfg['save_snapshot'], int(self.cfg['nstep']), float(self.cfg['discount']))

        self._replay_iter = None


        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg['save_video'] else None)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg['save_train_video'] else None)

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * int(self.cfg['action_repeat'])

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter


    def evaluate_agent(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(int(self.cfg['num_eval_episodes']))

        paths = []
        time_steps = list()
        count = 0
        while eval_until_episode(episode):
            path = []
            time_step = self.eval_env.reset()
            time_steps.append(time_step)
            self.video_recorder.init(self.eval_env, enabled=True)
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation[self.cfg['obs_type']],
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                time_steps.append(time_step)
                path.append(time_step.observation['goal_achieved'])
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f'eval_{count}.mp4')
            count += 1
            time_steps = list()
            paths.append(1 if np.sum(path) > 10 else 0)


    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(int(self.cfg['num_eval_episodes']))

        paths = []
        time_steps = list()
        while eval_until_episode(episode):
            path = []
            time_step = self.eval_env.reset()
            time_steps.append(time_step)
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation[self.cfg['obs_type']],
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                time_steps.append(time_step)
                path.append(time_step.observation['goal_achieved'])
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')


            time_steps = list()
            paths.append(1 if np.sum(path) > 10 else 0)


        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * int(self.cfg['action_repeat']) / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)
            log("success_percentage", np.mean(paths))

    def train_il(self):
        # predicates
        train_until_step = utils.Until(int(self.cfg['num_train_frames']),
                                       int(self.cfg['action_repeat']))
        seed_until_step = utils.Until(int(self.cfg['num_seed_frames']),
                                      int(self.cfg['action_repeat']))
        eval_every_step = utils.Every(int(self.cfg['eval_every_frames']),
                                      int(self.cfg['action_repeat']))

        episode_step, episode_reward = 0, 0

        time_steps = list()
        observations = list()
        actions = list()

        time_step = self.train_env.reset()
        time_steps.append(time_step)
        observations.append(time_step.observation[self.cfg['obs_type']])
        actions.append(time_step.action)

        self.train_video_recorder.init(time_step.observation[self.cfg['obs_type']])
        metrics = None

        while train_until_step(self.global_step):
            if time_step.last():

                self._global_episode += 1
                #if self._global_episode % 10 == 0:
                 #   self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                observations = np.stack(observations, 0)
                actions = np.stack(actions, 0)

                new_rewards = self.agent.vlm_rewarder(observations, self._global_step)

                summary = 0
                for i, elt in enumerate(time_steps):
                    summary += time_steps[i].observation['goal_achieved']

                if summary>10:
                    new_rewards[-1] += 100.0
                    for i, elt in enumerate(time_steps):
                        elt = elt._replace(
                            observation=time_steps[i].observation[self.cfg['obs_type']])

                        elt = elt._replace(reward=new_rewards[i])
                        self.expert_demo.append({'obs': time_steps[i].observation[self.cfg['obs_type']], 'ac': time_steps[i].action})
                      #  self.train_video_recorder.record(time_steps[i].observation[self.cfg['obs_type']])

                        self.replay_storage.add(elt)
                    self.flag = 1
                    #self.train_video_recorder.save(f'success_experience_{self.global_frame}.mp4')
                else:
                    for i, elt in enumerate(time_steps):
                        elt = elt._replace(
                            observation=time_steps[i].observation[self.cfg['obs_type']])

                        elt = elt._replace(reward=new_rewards[i])
                        self.replay_storage.add(elt)

                new_rewards_sum = np.sum(new_rewards)

                print("flag for success experience:", self.flag)
                #print(len(new_rewards))
                #print(len(time_steps))

                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * int(self.cfg['action_repeat'])
                    with self.logger.log_and_dump_ctx(self.global_frame, ty='train') as log:
                        #log('gt label for final obs', summary)
                       # log('pred label for final obs', label_im)
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        #log('success_buffer_size', len(self.success_replay_storage))
                        log('step', self.global_step)
                        log('imitation_reward', new_rewards_sum)
                        for item in range(len(new_rewards)):
                            log('window_reward', new_rewards[item])

                # reset env
                time_steps = list()
                observations = list()
                actions = list()

                time_step = self.train_env.reset()
                time_steps.append(time_step)
                observations.append(time_step.observation[self.cfg['obs_type']])
                actions.append(time_step.action)
                self.train_video_recorder.init(time_step.observation[self.cfg['obs_type']])
                # try to save snapshot
                if self.cfg['save_snapshot']:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation[self.cfg['obs_type']],
                                        self.global_step,
                                        eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step):
                # Update
                metrics = self.agent.update(self.replay_iter, self.expert_demo, self.global_step, self.flag)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward

            time_steps.append(time_step)
            observations.append(time_step.observation[self.cfg['obs_type']])
            actions.append(time_step.action)

            self.train_video_recorder.record(time_step.observation[self.cfg['obs_type']])
            episode_step += 1
            self._global_step += 1

    def save_initial_snapshot(self):
        keys_to_save = ['timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        payload.update(self.agent.save_snapshot())

        return payload

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        payload.update(self.agent.save_snapshot())
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        with open('snapshot.pt','rb') as f:
            payload = torch.load(f)
        agent_payload = {}
        for k, v in payload.items():
            if k not in self.__dict__:
                agent_payload[k] = v
        self.agent.load_snapshot(agent_payload)


def main():
    from train import WorkspaceIL as W
    workspace = W()
    workspace.train_il()

def evaluate():
    from train import WorkspaceIL as W
    workspace = W()
    workspace.load_snapshot()
    workspace.evaluate_agent()


if __name__ == '__main__':
    main()
    #evaluate()
