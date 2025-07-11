import random
import gym
from gym import spaces
import pandas as pd
import numpy as np

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000

INITIAL_ACCOUNT_BALANCE = 10000


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, window_size=6, use_ma=False):
        super(StockTradingEnv, self).__init__()

        # Ensure columns are lowercase for robustness
        df.columns = [c.lower() for c in df.columns]
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.use_ma = use_ma

        # Set up which features are included in observation
        self.price_cols = ['open', 'high', 'low', 'close']
        self.ma_cols = [col for col in df.columns if col.startswith('ma')] if use_ma else []

        # Compose feature list
        self.obs_features = self.price_cols + self.ma_cols

        self.feature_count = len(self.obs_features)

        # Action: [action_type, amount] (Buy, Sell, Hold)
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        # Observation: window_size days, each with feature_count features, plus 6 account features
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.feature_count + 1, self.window_size), dtype=np.float16)

    def _next_observation(self):
        # Only these 9 features (no account features!)
        features = [
            'open', 'high', 'low', 'close',
            'ma', 'ma.1', 'ma.2', 'ma.3', 'ma.4'
        ]
        # Make sure columns are lowercased in your __init__
        obs = np.array([
            self.df.loc[self.current_step: self.current_step + self.window_size - 1, feature].values / MAX_SHARE_PRICE
            for feature in features
        ])
        # obs shape: (9, window_size)
        return obs

    def _take_action(self, action):
        # Random price between open and close for this step
        open_p = self.df.loc[self.current_step, "open"]
        close_p = self.df.loc[self.current_step, "close"]
        current_price = random.uniform(open_p, close_p)

        action_type = action[0]
        amount = action[1]

        if action_type < 1:
            # Buy
            total_possible = int(self.balance / current_price)
            shares_bought = int(total_possible * amount)
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * current_price

            self.balance -= additional_cost
            self.cost_basis = (
                                      prev_cost + additional_cost) / (self.shares_held + shares_bought) if (
                                                                                                                       self.shares_held + shares_bought) > 0 else 0
            self.shares_held += shares_bought

        elif action_type < 2:
            # Sell
            shares_sold = int(self.shares_held * amount)
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * current_price

        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0

    def step(self, action):
        self._take_action(action)

        self.current_step += 1

        if self.current_step > len(self.df) - self.window_size:
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

        self.current_step = random.randint(0, len(self.df) - self.window_size)

        return self._next_observation()

    def render(self, mode='human', close=False):
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')
