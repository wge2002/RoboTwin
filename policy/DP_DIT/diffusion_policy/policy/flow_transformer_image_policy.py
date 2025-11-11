from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply


class FlowMatchingScheduler:
    """
    Simple Flow Matching scheduler for continuous time generative modeling.
    """
    def __init__(self, num_train_timesteps=100):
        self.num_train_timesteps = num_train_timesteps

    def set_timesteps(self, num_inference_steps):
        self.num_inference_steps = num_inference_steps
        # Create timesteps from 1 to 0 (reverse order for inference)
        self.timesteps = torch.linspace(1.0, 0.0, num_inference_steps)

    def step(self, model_output, timestep, sample):
        """
        Simplified Euler step for Flow Matching.
        In Flow Matching, model_output is the velocity field.
        """
        dt = 1.0 / self.num_inference_steps
        # Euler step: x_{t+1} = x_t + dt * v(x_t, t)
        # Since we're going backwards from t=1 to t=0, we subtract the velocity
        prev_sample = sample - dt * model_output
        return prev_sample


class FlowTransformerImagePolicy(BaseImagePolicy):

    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: FlowMatchingScheduler,
        obs_encoder: MultiImageObsEncoder,
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        obs_as_global_cond=True,
        # transformer parameters
        n_layer=12,
        n_head=12,
        n_emb=768,
        p_drop_emb=0.1,
        p_drop_attn=0.1,
        causal_attn=False,
        # parameters passed to step
        **kwargs,
    ):
        super().__init__()
        print('Initializing FlowTransformerImagePolicy')
        # parse shapes
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]

        # create flow matching model
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = TransformerForDiffusion(
            input_dim=input_dim,
            output_dim=input_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=global_cond_dim if global_cond_dim is not None else 0,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=True,
            obs_as_cond=(global_cond_dim is not None),
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        # self.kwargs = kwargs
        # Filter out parameters that are not for scheduler.step()
        scheduler_step_params = ['predict_epsilon']
        self.kwargs = {k: v for k, v in kwargs.items() if k in scheduler_step_params}


        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # ========= inference  ============
    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        # keyword arguments to scheduler.step
        **kwargs,
    ):
        model = self.model
        scheduler = self.noise_scheduler

        # Initialize with noise
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        # Set timesteps for ODE solver
        scheduler.set_timesteps(self.num_inference_steps)

        # Flow matching ODE integration using Euler method
        for i, t in enumerate(scheduler.timesteps):
            # Apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # Predict velocity (flow vector)
            velocity = model(trajectory, t, cond=global_cond)

            # Only update unconditioned parts (mask out conditioned parts)
            velocity[condition_mask] = 0.0

            # Euler step for ODE: dx/dt = v(x,t)
            # Going backwards from t=1 to t=0, so subtract velocity
            trajectory = scheduler.step(velocity, t, trajectory)

        # Final conditioning enforcement
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert "past_action" not in obs_dict  # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da + Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs,
        )

        # unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        result = {"action": action, "action_pred": action_pred}
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert "valid_mask" not in batch
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise for flow matching
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]

        # Sample random timesteps (continuous time for flow matching)
        timesteps = torch.rand(bsz, device=trajectory.device)

        # Flow matching: create noisy trajectory
        # x_t = (1-t)*noise + t*trajectory
        noisy_trajectory = (1 - timesteps.unsqueeze(-1).unsqueeze(-1)) * noise + \
                          timesteps.unsqueeze(-1).unsqueeze(-1) * trajectory

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict velocity (flow vector)
        pred_velocity = self.model(noisy_trajectory, timesteps, cond=global_cond)

        # Target velocity for flow matching: trajectory - noise
        target_velocity = trajectory - noise

        # Compute loss only on unconditioned parts
        loss = F.mse_loss(pred_velocity, target_velocity, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()
        return loss
