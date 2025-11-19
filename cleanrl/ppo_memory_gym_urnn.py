# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_urnnpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import memory_gym
from cleanrl.urnn import URNN, LegacyURNN


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 3
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    cuda_device: int = 0
    """the device to use for training"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "quantum_rl"
    """the wandb's project name"""
    wandb_entity: str = "sathyafrml"
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "MortarMayhem-Grid-v0"
    """the id of the environment"""
    total_timesteps: int = 15000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.75e-4
    """the learning rate of the optimizer"""
    min_learning_rate: float = 1e-5
    """the minimum learning rate of the optimizer"""
    complex_learning_rate: float = 8e-5
    """the learning rate of the complex parameters"""
    legacy_urnn: bool = False
    """whether to use the legacy URNN"""
    urnn_input_dense: bool = True
    """whether to add input embed to hidden state URNN"""
    urnn_units: int = 128
    """the hidden size of the URNN"""
    num_envs: int = 32
    """the number of parallel game environments"""
    num_steps: int = 512
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.995
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 8
    """the number of mini-batches"""
    update_epochs: int = 3
    """the K epochs to update the policy"""
    norm_adv: bool = False
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0001
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.25
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, video_folder_path,
            record_every=1000, frame_stack=1
        ):
    def thunk():
        env = gym.make(
            env_id,
            render_mode="rgb_array" if capture_video else None,
        )
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, video_folder_path,
                                           episode_trigger=lambda episode_id: (episode_id + 1) % record_every == 0
                                           )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if frame_stack > 1:
            env = gym.wrappers.FrameStack(env, frame_stack)
        return env
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, urnn_units=128, legacy_urnn=False, urnn_id=False):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(3, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        if legacy_urnn:
            self.urnn = LegacyURNN(512, urnn_units, norm_scale=np.sqrt(urnn_units))
        else:
            self.urnn = URNN(512, urnn_units, add_input_dense=urnn_id, norm_scale=np.sqrt(urnn_units))
        # Actor and critic take concatenated real/imaginary parts (2 * hidden_size = 256)
        self.actor = layer_init(nn.Linear(2 * urnn_units, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(2 * urnn_units, 1), std=1)

    def get_states(self, x, urnn_state, done):
        hidden = self.network(x.permute(0, 3, 1, 2) / 255.0)

        # URNN logic
        batch_size = urnn_state.shape[0]
        hidden = hidden.reshape((-1, batch_size, self.urnn.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            # Reset state where done is True
            reset_mask = (1.0 - d).view(batch_size, 1)  # (batch_size, 1)
            urnn_state = reset_mask * urnn_state + (1 - reset_mask) * self.urnn.initial_hidden(batch_size)
            
            # Process through URNN
            h_complex, urnn_state = self.urnn(h, urnn_state)
            new_hidden += [h_complex]
        
        new_hidden = torch.cat(new_hidden, dim=0)  # (seq_len * batch_size, hidden_size) complex
        
        return new_hidden, urnn_state

    def get_value(self, x, urnn_state, done):
        hidden_complex, _ = self.get_states(x, urnn_state, done)
        # Convert complex to real only when needed for critic
        hidden_real = torch.cat([hidden_complex.real, hidden_complex.imag], dim=-1)
        return self.critic(hidden_real)

    def get_action_and_value(self, x, urnn_state, done, action=None):
        hidden_complex, urnn_state = self.get_states(x, urnn_state, done)
        # Convert complex to real only when needed for actor/critic
        hidden_real = torch.cat([hidden_complex.real, hidden_complex.imag], dim=-1)
        logits = self.actor(hidden_real)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden_real), urnn_state


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    #naming section
    id = "id_" if args.urnn_input_dense else ""
    legacy = "leg" if args.legacy_urnn else ""
    run_name = f"cleanrl_ppo_memory_gym_{legacy}urnn{args.urnn_units}_{id}{args.env_id}_{args.seed}_{int(time.time())}"
    run_dir = f"./runs/cleanrl_ppo_memory_gym_{legacy}urnn{args.urnn_units}_{id}{ args.env_id}/s{args.seed}"
    os.makedirs(run_dir, exist_ok=True)
    video_folder_path = f"{run_dir}/videos"
    os.makedirs(video_folder_path, exist_ok=True)

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(run_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, video_folder_path) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs, args.urnn_units, args.legacy_urnn, args.urnn_input_dense).to(device)
    agent = torch.compile(
        agent,
        mode="max-autotune",
        dynamic=True,  # CRITICAL: handles batch size 2 vs 8 without recompilation
        fullgraph=False,  # Allow graph breaks for FFT/complex ops
    )
    optimizer = optim.Adam([
        {'params': agent.network.parameters()},
        {'params': agent.actor.parameters()},
        {'params': agent.critic.parameters()}, 
        {'params': agent.urnn.parameters(), 'lr': args.complex_learning_rate},
    ], lr=args.learning_rate, eps=1e-5)
    print([group['lr'] for group in optimizer.param_groups])

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    next_urnn_state = agent.urnn.initial_hidden(args.num_envs)  # (batch_size, hidden_size) complex

    for iteration in range(1, args.num_iterations + 1):
        initial_urnn_state = next_urnn_state.clone()
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            for group in optimizer.param_groups[:-1]:
                group['lr'] = args.min_learning_rate + frac * (args.learning_rate - args.min_learning_rate)
            optimizer.param_groups[-1]['lr'] = args.min_learning_rate + \
                frac * (args.complex_learning_rate - args.min_learning_rate)
        
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, next_urnn_state = agent.get_action_and_value(next_obs, next_urnn_state, next_done)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(
                next_obs,
                next_urnn_state,
                next_done,
            ).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        assert args.num_envs % args.num_minibatches == 0
        envsperbatch = args.num_envs // args.num_minibatches
        envinds = np.arange(args.num_envs)
        flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, args.num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    b_obs[mb_inds],
                    initial_urnn_state[mbenvinds],
                    b_dones[mb_inds],
                    b_actions.long()[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()

