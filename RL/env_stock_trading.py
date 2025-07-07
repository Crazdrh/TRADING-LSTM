import random
import gym
from gym import spaces
import numpy as np

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_STEPS = 20000
INITIAL_ACCOUNT_BALANCE = 10000

SIGNALS = ["strong_buy", "buy", "weak_buy", "hold", "weak_sell", "sell", "strong_sell"]

class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(StockTradingEnv, self).__init__()

        self.df = df
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        # Now, discrete action space for 7 signals
        self.action_space = spaces.Discrete(len(SIGNALS))

        # Prices contains the OHCL values for the last five prices, + 6 account stats
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(6, 6), dtype=np.float16)

    def _next_observation(self):
        frame = np.array([
            self.df.loc[self.current_step: self.current_step + 5, 'Open'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step + 5, 'High'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step + 5, 'Low'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step + 5, 'Close'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step + 5, 'Volume'].values / MAX_NUM_SHARES,
        ])

        obs = np.append(frame, [[
            self.balance / MAX_ACCOUNT_BALANCE,
            self.max_net_worth / MAX_ACCOUNT_BALANCE,
            self.shares_held / MAX_NUM_SHARES,
            self.cost_basis / MAX_SHARE_PRICE,
            self.total_shares_sold / MAX_NUM_SHARES,
            self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
        ]], axis=0)
        return obs

    def _take_action(self, action_idx):
        # Map index to signal
        signal = SIGNALS[action_idx]
        # Use same logic as before, but with signals instead of action[0]
        current_price = random.uniform(
            self.df.loc[self.current_step, "Open"], self.df.loc[self.current_step, "Close"])

        # Interpret signal
        if signal in ["strong_buy", "buy", "weak_buy"]:
            buy_strength = {"strong_buy": 1.0, "buy": 0.6, "weak_buy": 0.3}[signal]
            total_possible = int(self.balance / current_price)
            shares_bought = int(total_possible * buy_strength)
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * current_price

            self.balance -= additional_cost
            self.cost_basis = (
                prev_cost + additional_cost) / (self.shares_held + shares_bought) if (self.shares_held + shares_bought) > 0 else 0
            self.shares_held += shares_bought

        elif signal in ["strong_sell", "sell", "weak_sell"]:
            sell_strength = {"strong_sell": 1.0, "sell": 0.6, "weak_sell": 0.3}[signal]
            shares_sold = int(self.shares_held * sell_strength)
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * current_price

        # "hold" means do nothing

        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0

    def step(self, action):
        self._take_action(action)

        self.current_step += 1
        if self.current_step > len(self.df.loc[:, 'Open'].values) - 6:
            self.current_step = 0

        delay_modifier = (self.current_step / MAX_STEPS)
        reward = self.balance * delay_modifier
        done = self.net_worth <= 0

        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        self.current_step = random.randint(
            0, len(self.df.loc[:, 'Open'].values) - 6)

        return self._next_observation()

    def render(self, mode='human', close=False):
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')
