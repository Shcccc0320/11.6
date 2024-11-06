import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import random

# 确保正确导入 V2XEnvironment
from env import V2XEnvironment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, num_vehicles, num_stations):
        super(Actor, self).__init__()
        self.num_vehicles = num_vehicles
        self.num_stations = num_stations

        # 公共层
        self.common_layer1 = nn.Linear(state_dim, 256)
        self.norm1 = nn.LayerNorm(256)
        self.common_layer2 = nn.Linear(256, 256)
        self.norm2 = nn.LayerNorm(256)

        # 基站选择（离散动作）
        self.base_station_layer = nn.Linear(256, num_vehicles * num_stations)

        # 带宽分配（连续动作）
        self.bandwidth_layer = nn.Linear(256, num_vehicles)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, state):
        x = torch.relu(self.norm1(self.common_layer1(state)))
        x = torch.relu(self.norm2(self.common_layer2(x)))

        # 基站选择的 logits
        bs_logits = self.base_station_layer(x)
        bs_logits = bs_logits.view(-1, self.num_vehicles, self.num_stations)

        # 减去最大值以获得数值稳定性
        bs_logits = bs_logits - bs_logits.max(dim=-1, keepdim=True)[0]

        bs_probs = torch.softmax(bs_logits, dim=-1)

        # 带宽分配，使用 sigmoid 激活函数
        bandwidth = torch.sigmoid(self.bandwidth_layer(x))

        return bs_probs, bandwidth

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, num_vehicles, num_stations):
        super(Critic, self).__init__()
        self.num_vehicles = num_vehicles
        self.num_stations = num_stations

        # 状态输入层
        self.state_layer = nn.Linear(state_dim, 256)
        self.norm1 = nn.LayerNorm(256)

        # 动作输入层
        action_input_dim = num_vehicles * (num_stations + 1)
        self.action_layer = nn.Linear(action_input_dim, 256)
        self.norm2 = nn.LayerNorm(256)

        # 合并层
        self.common_layer1 = nn.Linear(512, 256)
        self.norm3 = nn.LayerNorm(256)
        self.common_layer2 = nn.Linear(256, 256)
        self.norm4 = nn.LayerNorm(256)
        self.output_layer = nn.Linear(256, 1)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, state, bs_action_one_hot, bandwidth_action):
        # 状态特征
        state_feature = torch.relu(self.norm1(self.state_layer(state)))

        # 动作特征
        bs_action_flat = bs_action_one_hot.view(-1, self.num_vehicles * self.num_stations)
        bandwidth_action_flat = bandwidth_action.view(-1, self.num_vehicles)
        action_input = torch.cat([bs_action_flat, bandwidth_action_flat], dim=1)
        action_feature = torch.relu(self.norm2(self.action_layer(action_input)))

        # 合并特征
        x = torch.cat([state_feature, action_feature], dim=1)
        x = torch.relu(self.norm3(self.common_layer1(x)))
        x = torch.relu(self.norm4(self.common_layer2(x)))
        q_value = self.output_layer(x)
        return q_value

# DDPG Agent
class DDPGAgent:
    def __init__(self, state_dim, num_vehicles, num_stations):
        self.num_vehicles = num_vehicles
        self.num_stations = num_stations

        self.actor = Actor(state_dim, num_vehicles, num_stations).to(device)
        self.actor_target = Actor(state_dim, num_vehicles, num_stations).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-5)

        self.critic = Critic(state_dim, num_vehicles, num_stations).to(device)
        self.critic_target = Critic(state_dim, num_vehicles, num_stations).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-4)

        self.replay_buffer = deque(maxlen=100000)
        self.batch_size = 64
        self.discount = 0.99
        self.tau = 0.005

        # ε-贪心策略参数
        self.epsilon = 1.0  # 初始探索率
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def select_action(self, state, exploration=True):
        if np.isnan(state).any() or np.isinf(state).any():
            print("State contains NaN or Inf:", state)
            raise ValueError("State contains NaN or Inf")

        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        bs_probs, bandwidth = self.actor(state)

        if torch.isnan(bs_probs).any() or torch.isinf(bs_probs).any():
            print("bs_probs contains NaN or Inf:", bs_probs)
            raise ValueError("bs_probs contains NaN or Inf")

        bs_probs = bs_probs.cpu().data.numpy().squeeze()
        bandwidth = bandwidth.cpu().data.numpy().squeeze()

        # 基站选择
        if exploration and np.random.rand() < self.epsilon:
            # 随机选择基站
            base_station_selection = np.random.randint(0, self.num_stations, size=self.num_vehicles)
        else:
            # 贪心选择
            base_station_selection = np.argmax(bs_probs, axis=-1)

        # 带宽分配，添加噪声以促进探索
        if exploration:
            noise = np.random.normal(0, 0.2, size=bandwidth.shape)  # 增大噪声标准差
            bandwidth_allocations = np.clip(bandwidth + noise, 0.0, 1.0)
        else:
            bandwidth_allocations = bandwidth

        action = (base_station_selection, bandwidth_allocations)
        return action

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0, 0

        batch = random.sample(self.replay_buffer, self.batch_size)
        state, action, reward, next_state, done = zip(*batch)

        state = torch.FloatTensor(np.array(state)).to(device)
        reward = torch.FloatTensor(np.array(reward)).unsqueeze(1).to(device)
        next_state = torch.FloatTensor(np.array(next_state)).to(device)
        done = torch.FloatTensor(np.array(done)).unsqueeze(1).to(device)

        # 处理动作
        base_station_selection = np.array([a[0] for a in action])
        bandwidth_allocations = np.array([a[1] for a in action])

        # 将基站选择转换为 one-hot 编码
        bs_action_one_hot = np.zeros((self.batch_size, self.num_vehicles, self.num_stations))
        for i in range(self.batch_size):
            for v in range(self.num_vehicles):
                bs_action_one_hot[i, v, base_station_selection[i][v]] = 1

        bs_action_one_hot = torch.FloatTensor(bs_action_one_hot).to(device)
        bandwidth_action = torch.FloatTensor(bandwidth_allocations).to(device)

        # Critic 训练
        with torch.no_grad():
            next_bs_probs, next_bandwidth = self.actor_target(next_state)

            # 下一个基站的贪心选择
            next_base_station_selection = torch.argmax(next_bs_probs, dim=-1)
            next_bs_action_one_hot = nn.functional.one_hot(next_base_station_selection, num_classes=self.num_stations).float()

            next_bs_action_one_hot = next_bs_action_one_hot.to(device)
            next_bandwidth = next_bandwidth.to(device)

            target_q = self.critic_target(next_state, next_bs_action_one_hot, next_bandwidth)
            target_q = reward + ((1 - done) * self.discount * target_q)

        current_q = self.critic(state, bs_action_one_hot, bandwidth_action)
        critic_loss = nn.MSELoss()(current_q, target_q)

        if torch.isnan(critic_loss) or torch.isinf(critic_loss):
            print("Critic loss is NaN or Inf")
            raise ValueError("Critic loss is NaN or Inf")

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # Actor 训练
        bs_probs, bandwidth = self.actor(state)

        # 采样动作
        sampled_base_station_selection = torch.argmax(bs_probs, dim=-1)
        bs_action_one_hot = nn.functional.one_hot(sampled_base_station_selection, num_classes=self.num_stations).float()

        actor_loss = -self.critic(state, bs_action_one_hot, bandwidth).mean()

        if torch.isnan(actor_loss) or torch.isinf(actor_loss):
            print("Actor loss is NaN or Inf")
            raise ValueError("Actor loss is NaN or Inf")

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # 更新目标网络
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # 衰减 ε
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return actor_loss.item(), critic_loss.item()

    def add_to_replay_buffer(self, transition):
        self.replay_buffer.append(transition)

# 训练代理
env = V2XEnvironment()
state_dim = env.observation_space.shape[0]
num_vehicles = env.num_vehicles
num_stations = env.num_stations

agent = DDPGAgent(state_dim, num_vehicles, num_stations)

num_episodes = 5000
all_rewards = []
actor_losses = []
critic_losses = []

for episode in range(num_episodes):

    state = env.reset()
    if np.isnan(state).any() or np.isinf(state).any():
        print("Initial state contains NaN or Inf:", state)
        raise ValueError("Initial state contains NaN or Inf")
    episode_reward = 0
    done = False
    step = 0

    while not done:
        action = agent.select_action(state)
        # 可以选择记录动作
        # print(f"Episode {episode+1}, Step {step}, Action: {action}")

        next_state, reward, done, _ = env.step(action)

        if np.isnan(reward) or np.isinf(reward):
            print("Reward contains NaN or Inf:", reward)
            raise ValueError("Reward contains NaN or Inf")

        agent.add_to_replay_buffer((state, action, reward, next_state, float(done)))
        actor_loss, critic_loss = agent.train()
        state = next_state
        episode_reward += reward
        step += 1

    all_rewards.append(episode_reward)
    actor_losses.append(actor_loss)
    critic_losses.append(critic_loss)
    print(f"Episode: {episode + 1}, Reward: {episode_reward:.2f}, Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")

    # 每隔 100 个 episode 绘制一次结果
    if (episode + 1) % 100 == 0:
        plt.figure()
        plt.plot(all_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('DDPG on V2X Environment')
        plt.show()

        plt.figure()
        plt.plot(actor_losses, label='Actor Loss')
        plt.plot(critic_losses, label='Critic Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.title('Actor and Critic Losses')
        plt.legend()
        plt.show()
