"""
MiniGrid-specific wrappers for CleanRL.

This module contains wrappers adapted from stable-baselines3 for use with
simple gymnasium environments in CleanRL.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TransposeImage(gym.ObservationWrapper):
    """
    Re-order image observation channels from HxWxC to CxHxW.
    
    This wrapper is required for PyTorch convolution layers which expect
    channel-first format. It handles both regular observations and FrameStack
    observations.
    
    For regular observations: (H, W, C) -> (C, H, W)
    For FrameStack observations: (num_frames, H, W, C) -> (num_frames * C, H, W)
    
    :param env: Environment to wrap
    """
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Box), (
            f"Expected Box observation space, got {type(env.observation_space)}"
        )
        
        obs_shape = env.observation_space.shape
        assert len(obs_shape) in [3, 4], (
            f"Expected 3D (H, W, C) or 4D (num_frames, H, W, C) observation space, "
            f"got shape {obs_shape}"
        )
        
        # Check if this is a FrameStack observation (4D) or regular (3D)
        if len(obs_shape) == 4:
            # FrameStack format: (num_frames, H, W, C)
            num_frames, height, width, channels = obs_shape
            new_shape = (num_frames * channels, height, width)
        else:
            # Regular format: (H, W, C)
            height, width, channels = obs_shape
            new_shape = (channels, height, width)
        
        self.observation_space = spaces.Box(
            low=env.observation_space.low.min() if hasattr(env.observation_space.low, 'min') else env.observation_space.low,
            high=env.observation_space.high.max() if hasattr(env.observation_space.high, 'max') else env.observation_space.high,
            shape=new_shape,
            dtype=env.observation_space.dtype,
        )
        self._is_framestack = len(obs_shape) == 4
    
    def observation(self, obs: np.ndarray) -> np.ndarray:
        """
        Transpose observation from channel-last to channel-first format.
        
        :param obs: Observation in channel-last format
        :return: Observation in channel-first format
        """
        if self._is_framestack:
            # FrameStack format: (num_frames, H, W, C) -> (num_frames * C, H, W)
            # Permute to (num_frames, C, H, W) then reshape to (num_frames * C, H, W)
            num_frames, height, width, channels = obs.shape
            obs = obs.transpose(0, 3, 1, 2).reshape(num_frames * channels, height, width)
        else:
            # Regular format: (H, W, C) -> (C, H, W)
            obs = obs.transpose(2, 0, 1)
        return obs

