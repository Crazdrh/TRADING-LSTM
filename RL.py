import os
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from StockTradingEnv import StockTradingEnv
import gc

# CRITICAL: Force garbage collection and clear cache at start
torch.cuda.empty_cache()
gc.collect()

# ========== CONFIGURATION ==========
CSV_PATH = "C:/Users/Hayden/Downloads/5mincsvma/BATS_NVDA, 5_afc69.csv"
MODEL_SAVE_PATH = "C:/Users/Hayden/Downloads/trained_lstm_model_with_ma.pth"
SEQ_LEN = 50
N_ACTIONS = 3
N_EPOCHS = 1
GAMMA = 0.99
LR = 0.0003
HIDDEN_SIZE = 128
BATCH_SIZE = 1  # Keep small

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
print(f"Using device: {device} | bf16 AMP: {use_amp}")

# ========== ENV SETUP ==========
df = pd.read_csv(CSV_PATH)
env = StockTradingEnv(df=df, window_size=SEQ_LEN)


# ========== FIXED LSTM PPO POLICY ==========
class PPO_LSTM_Policy(nn.Module):
    def __init__(self, input_dim=9, hidden_size=HIDDEN_SIZE, num_layers=2, n_actions=N_ACTIONS):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True)
        self.actor = nn.Linear(hidden_size, n_actions)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x, hc=None):
        # CRITICAL: Don't accumulate gradients in hidden states
        if hc is not None:
            h, c = hc
            h = h.detach()  # DETACH to prevent gradient accumulation
            c = c.detach()  # DETACH to prevent gradient accumulation
            hc = (h, c)

        out, hc_new = self.lstm(x, hc)
        last_out = out[:, -1, :]
        logits = self.actor(last_out)
        value = self.critic(last_out)
        return logits, value, hc_new


# Initialize model
obs = env.reset()
window = obs.T
input_dim = window.shape[1]
policy = PPO_LSTM_Policy(input_dim=input_dim, n_actions=N_ACTIONS).to(device)


# ========== FIXED PPO UTILITIES ==========
def select_action(policy, state, hc, amp=False):
    with torch.no_grad():  # CRITICAL: No gradients for action selection
        if amp:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits, value, hc_new = policy(state, hc)
        else:
            logits, value, hc_new = policy(state, hc)

        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

    return action.item(), log_prob.detach(), value.detach(), hc_new


def compute_returns(rewards, masks, gamma):
    R = 0
    returns = []
    for r, m in zip(reversed(rewards), reversed(masks)):
        R = r + gamma * R * m
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32)


# ========== MEMORY-EFFICIENT TRAINING LOOP ==========
optimizer = optim.Adam(policy.parameters(), lr=LR)
scaler = torch.cuda.amp.GradScaler() if use_amp else None

print(f"Initial VRAM: {torch.cuda.memory_allocated() // 1024 ** 2} MB")

for epoch in range(N_EPOCHS):
    # CRITICAL: Clear everything at start of epoch
    torch.cuda.empty_cache()
    gc.collect()

    state = env.reset()
    hc = None
    done = False

    # Use lists to store CPU tensors only
    episode_data = {
        'log_probs': [],
        'values': [],
        'rewards': [],
        'masks': []
    }

    step_count = 0
    while not done:
        # Process observation
        obs = state
        window = obs.T
        # CRITICAL: Normalize on CPU first
        window_mean = window.mean(axis=0)
        window_std = window.std(axis=0) + 1e-8
        window = (window - window_mean) / window_std

        # Move to GPU only when needed
        state_torch = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)

        # Get action
        action, log_prob, value, hc = select_action(policy, state_torch, hc, amp=use_amp)

        # CRITICAL: Move results to CPU immediately
        episode_data['log_probs'].append(log_prob.cpu())
        episode_data['values'].append(value.cpu())

        # Environment step
        amount = 1.0
        env_action = [float(action), float(amount)]
        next_state, reward, done, info = env.step(env_action)

        episode_data['rewards'].append(float(reward))
        episode_data['masks'].append(1 - done)

        state = next_state
        step_count += 1

        # CRITICAL: Clear GPU tensors immediately
        del state_torch
        if step_count % 50 == 0:  # Clear cache periodically
            torch.cuda.empty_cache()

    # CRITICAL: Clear hidden states before processing
    del hc
    torch.cuda.empty_cache()

    # Compute returns and advantages on CPU
    returns = compute_returns(episode_data['rewards'], episode_data['masks'], GAMMA)
    log_probs = torch.stack(episode_data['log_probs'])
    values = torch.cat(episode_data['values']).squeeze(-1)

    # Move to GPU only for training
    returns = returns.to(device)
    log_probs = log_probs.to(device)
    values = values.to(device)

    advantage = returns - values.detach()

    # PPO Updates - CRITICAL: Clear gradients properly
    for update_step in range(4):
        optimizer.zero_grad()

        if use_amp:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                actor_loss = -(log_probs * advantage).mean()
                critic_loss = (returns - values).pow(2).mean()
                loss = actor_loss + 0.5 * critic_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            actor_loss = -(log_probs * advantage).mean()
            critic_loss = (returns - values).pow(2).mean()
            loss = actor_loss + 0.5 * critic_loss

            loss.backward()
            optimizer.step()

        # CRITICAL: Clear gradients and cache after each update
        torch.cuda.empty_cache()

    print(
        f"Epoch {epoch + 1}/{N_EPOCHS} | Episode reward: {sum(episode_data['rewards']):.2f} | Loss: {loss.item():.5f}")

    # Save model
    if (epoch + 1) % 5 == 0 or (epoch + 1) == N_EPOCHS:
        torch.save(policy.state_dict(), MODEL_SAVE_PATH)
        print(f"Model saved at epoch {epoch + 1}")

    # CRITICAL: Aggressive cleanup at end of epoch
    del episode_data, returns, log_probs, values, advantage, loss
    if 'actor_loss' in locals():
        del actor_loss, critic_loss

    torch.cuda.empty_cache()
    gc.collect()

    # Print memory usage
    allocated = torch.cuda.memory_allocated() // 1024 ** 2
    reserved = torch.cuda.memory_reserved() // 1024 ** 2
    print(f"VRAM: {allocated} MB allocated, {reserved} MB reserved")

    # CRITICAL: If memory usage is growing, force reset
    if allocated > 16000:  # More than 2GB
        print("WARNING: High memory usage detected! Force clearing...")
        torch.cuda.empty_cache()
        gc.collect()

print("Training completed!")
