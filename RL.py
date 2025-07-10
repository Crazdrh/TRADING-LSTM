import os
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from stock_trading_env.env import StockTradingEnv

# ========== CONFIGURATION ==========
CSV_PATH = "C:/Users/Hayden/Downloads/your_data.csv"   # <-- Change to your CSV
MODEL_SAVE_PATH = "C:/Users/Hayden/Downloads/ppo_lstm_policy.pth"
SEQ_LEN = 50
N_ACTIONS = 3   # 0=Hold, 1=Buy, 2=Sell (per environment)
N_EPOCHS = 1000
GAMMA = 0.99
LR = 0.0003
BATCH_SIZE = 32
HIDDEN_SIZE = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
print(f"Using device: {device} | bf16 AMP: {use_amp}")

# ========== ENV SETUP ==========
df = pd.read_csv(CSV_PATH)
env = StockTradingEnv(df=df, window_size=SEQ_LEN, frame_bound=(SEQ_LEN, len(df)))

# ========== LSTM PPO POLICY ==========
class PPO_LSTM_Policy(nn.Module):
    def __init__(self, input_dim=4, hidden_size=HIDDEN_SIZE, num_layers=1, n_actions=N_ACTIONS):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True)
        self.actor = nn.Linear(hidden_size, n_actions)
        self.critic = nn.Linear(hidden_size, 1)
        
    def forward(self, x, hc=None):
        # x: (batch, seq_len, input_dim)
        if hc is None:
            out, hc = self.lstm(x)
        else:
            out, hc = self.lstm(x, hc)
        last_out = out[:, -1, :]  # Only use last output for policy/critic
        logits = self.actor(last_out)
        value = self.critic(last_out)
        return logits, value, hc

policy = PPO_LSTM_Policy(input_dim=4, n_actions=N_ACTIONS).to(device)

# ========== PPO UTILITIES ==========
def select_action(policy, state, hc, amp=False):
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=amp):
        logits, value, hc_new = policy(state, hc)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
    return action.item(), log_prob, value, hc_new

def compute_returns(rewards, masks, gamma):
    R = 0
    returns = []
    for r, m in zip(reversed(rewards), reversed(masks)):
        R = r + gamma * R * m
        returns.insert(0, R)
    return returns

# ========== TRAINING LOOP ==========
optimizer = optim.Adam(policy.parameters(), lr=LR)
scaler = torch.cuda.amp.GradScaler(enabled=False)  # Not needed for bf16, but safe

for epoch in range(N_EPOCHS):
    state = env.reset()
    hc = None
    done = False
    log_probs = []
    values = []
    rewards = []
    masks = []
    states = []
    actions = []
    
    while not done:
        obs = state
        # Only pass open, high, low, close columns (window)
        window = obs[:SEQ_LEN, :4]  # shape: (SEQ_LEN, 4)
        window = (window - window.mean(axis=0)) / (window.std(axis=0) + 1e-8)  # normalize like training
        
        action, log_prob, value, hc = select_action(policy, window[np.newaxis, :, :], hc, amp=use_amp)
        next_state, reward, done, info = env.step(action)
        
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        masks.append(1 - done)
        states.append(window)
        actions.append(action)
        state = next_state
        
    # Compute returns
    returns = compute_returns(rewards, masks, GAMMA)
    returns = torch.tensor(returns, dtype=torch.float32).to(device)
    log_probs = torch.stack(log_probs)
    values = torch.cat(values).squeeze(-1)
    advantage = returns - values.detach()
    
    # PPO Update
    for _ in range(4):  # PPO epochs per episode
        if use_amp:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # Actor loss
                actor_loss = -(log_probs * advantage).mean()
                # Critic loss
                critic_loss = (returns - values).pow(2).mean()
                loss = actor_loss + 0.5 * critic_loss
        else:
            actor_loss = -(log_probs * advantage).mean()
            critic_loss = (returns - values).pow(2).mean()
            loss = actor_loss + 0.5 * critic_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch+1}/{N_EPOCHS} | Episode reward: {sum(rewards):.2f} | Loss: {loss.item():.5f}")

    # Save model every 50 epochs (change as needed)
    if (epoch + 1) % 50 == 0 or (epoch + 1) == N_EPOCHS:
        torch.save(policy.state_dict(), MODEL_SAVE_PATH)
        print(f"Model saved at epoch {epoch+1} to {MODEL_SAVE_PATH}")
