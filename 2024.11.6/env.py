import numpy as np
import gym
from gym import spaces
import networkx as nx

class V2XEnvironment(gym.Env):
    def __init__(self):
        super(V2XEnvironment, self).__init__()

        # 初始化网格路网
        self.G = nx.Graph()
        # 添加节点和位置
        self.G.add_nodes_from([
            ('A', {'position': (1, 3)}),
            ('B', {'position': (2, 3)}),
            ('C', {'position': (0, 2)}),
            ('D', {'position': (1, 2)}),
            ('E', {'position': (2, 2)}),
            ('F', {'position': (3, 2)}),
            ('G', {'position': (0, 1)}),
            ('H', {'position': (1, 1)}),
            ('I', {'position': (2, 1)}),
            ('J', {'position': (3, 1)}),
            ('K', {'position': (1, 0)}),
            ('L', {'position': (2, 0)})
        ])

        # 添加边以构建道路结构
        self.G.add_edges_from([
            ('A', 'D'), ('D', 'H'), ('H', 'K'),
            ('B', 'E'), ('E', 'I'), ('I', 'L'),
            ('C', 'D'), ('D', 'E'), ('E', 'F'),
            ('G', 'H'), ('H', 'I'), ('I', 'J')
        ])

        # 基站配置 (0是MBS，1-4是SBS)
        self.base_stations = {
            0: {'position': (1.5, 2.5), 'total_bandwidth': 300, 'transmission_power': 60},
            1: {'position': (0.7, 2.2), 'total_bandwidth': 100, 'transmission_power': 30},
            2: {'position': (2.3, 2.7), 'total_bandwidth': 100, 'transmission_power': 30},
            3: {'position': (2.7, 1.2), 'total_bandwidth': 100, 'transmission_power': 30},
            4: {'position': (1.2, 1.1), 'total_bandwidth': 100, 'transmission_power': 30}
        }

        self.noise_power = 1e-9  # 噪声功率
        self.num_vehicles = 4    # 车辆数量
        self.num_stations = 5    # 基站数量

        # 定义QoS要求和惩罚因子
        self.max_delay = 0.01             # URLLC最大可接受时延（秒）
        self.min_rate = 10                # eMBB最小可接受数据率（Mbps）
        self.delay_penalty_factor = 0.01   # 时延惩罚因子
        self.rate_penalty_factor = 0.005   # 数据率惩罚因子

        # 定义状态空间归一化常数
        self.max_position = 3.0  # 最大坐标值
        self.max_data_requirement = 80.0  # 最大数据需求
        self.max_bandwidth = 1000.0  # 最大带宽

        # Time step and step counter
        self.delta_t = 1.0  # Time step duration
        self.step_count = 0
        self.max_steps = 1000  # Maximum number of steps per episode

        # Define observation space
        self.observation_space = self._get_observation_space()

        # Define action space
        self.action_space = self._get_action_space()

        # Initialize environment state
        self.state = self.reset()

    def _get_observation_space(self):
        # 计算状态空间维度
        vehicle_state_size = 2 + 1 + 1 + 3 + 1  # 位置(x,y)，数据需求，带宽分配，切片类型(one-hot)，活动状态标志
        base_station_state_size = self.num_stations * (2 + 1)  # 每个基站的距离、信道增益、总带宽
        total_state_size = self.num_vehicles * (vehicle_state_size + base_station_state_size)
        return spaces.Box(low=0, high=1, shape=(total_state_size,), dtype=np.float32)

    def _get_action_space(self):
        # 动作空间由基站选择（离散）和带宽分配（连续）组成
        base_station_selection_space = spaces.MultiDiscrete([self.num_stations for _ in range(self.num_vehicles)])
        bandwidth_allocation_space = spaces.Box(low=0, high=1, shape=(self.num_vehicles,), dtype=np.float32)
        return spaces.Tuple((base_station_selection_space, bandwidth_allocation_space))

    def reset(self):
        # 初始化车辆位置和状态
        routes = [
            ['A', 'D', 'E', 'B'],
            ['G', 'H', 'I', 'J'],
            ['C', 'D', 'H', 'K'],
            ['L', 'I', 'E', 'F'],
        ]

        self.state = {
            'vehicles': {
                i: {
                    'route': routes[i],
                    'route_index': 0,  # 当前所在的节点索引
                    'current_edge': (routes[i][0], routes[i][1]),  # 当前边（起点，终点）
                    'position': self.G.nodes[routes[i][0]]['position'],  # 当前位置坐标
                    'distance_on_edge': 0.0,  # 在当前边上已经移动的距离
                    'edge_length': self.calculate_edge_length(routes[i][0], routes[i][1]),  # 当前边的长度
                    'velocity': np.random.uniform(0.01, 0.05),  # 车辆速度
                    'slice_type': np.random.choice(['URLLC', 'eMBB', 'Both']),
                    'has_arrived': False,
                    'data_requirement': None,
                    'bandwidth_allocation': 0.0,
                    'base_station': None,  # 初始化时未连接基站
                }
                for i in range(self.num_vehicles)
            }
        }

        # 设置数据需求
        for vehicle in self.state['vehicles'].values():
            slice_type = vehicle['slice_type']
            if slice_type == 'URLLC':
                vehicle['data_requirement'] = np.random.uniform(20, 80)  # 数据大小 (Mbits)
            elif slice_type == 'eMBB':
                vehicle['data_requirement'] = np.random.uniform(40, 100)
            elif slice_type == 'Both':
                vehicle['data_requirement'] = np.random.uniform(20, 100)
            vehicle['bandwidth_allocation'] = 0.0

        return self.get_state()

    def calculate_edge_length(self, node1, node2):
        pos1 = np.array(self.G.nodes[node1]['position'])
        pos2 = np.array(self.G.nodes[node2]['position'])
        return np.linalg.norm(pos2 - pos1)

    def step(self, action):
        self.step_count += 1  # 增加步数计数器

        # 应用动作
        self.apply_action(action)

        # 移动车辆
        self.move_vehicles()

        # 计算奖励
        reward = self.calculate_reward()

        # 获取下一个状态
        next_state = self.get_state()

        # 检查是否达到最大步数
        if self.step_count >= self.max_steps:
            done = True
            #print(f"Reached maximum steps: {self.step_count}")
        else:
            done = self.check_if_done()

        info = {}
        return next_state, reward, done, info

    def apply_action(self, action):
        base_station_selection, bandwidth_allocations = action
        for i in range(self.num_vehicles):
            vehicle = self.state['vehicles'][i]
            if vehicle.get('has_arrived', False):
                # 确保已到达终点的车辆的动作被忽略
                vehicle['base_station'] = None
                vehicle['bandwidth_allocation'] = 0.0
                continue
            base_station_index = base_station_selection[i]
            bandwidth_fraction = np.clip(bandwidth_allocations[i], 0, 1)

            vehicle['base_station'] = base_station_index
            total_bandwidth = self.base_stations[base_station_index]['total_bandwidth']
            vehicle['bandwidth_allocation'] = bandwidth_fraction * total_bandwidth

    def move_vehicles(self):
        for vehicle_id, vehicle in self.state['vehicles'].items():
            if vehicle.get('has_arrived', False):
                continue

            # 获取当前边的信息
            from_node, to_node = vehicle['current_edge']
            edge_length = vehicle['edge_length']

            # 计算本次移动的距离
            move_distance = vehicle['velocity']

            # 更新在当前边上已经移动的距离
            vehicle['distance_on_edge'] += move_distance

            #print(
                #f"Vehicle {vehicle_id} before moving: position {vehicle['position']}, distance_on_edge {vehicle['distance_on_edge']}, edge_length {edge_length}, route_index {vehicle['route_index']}")

            while vehicle['distance_on_edge'] >= edge_length:
                # 到达当前边的终点，更新到下一个边
                vehicle['distance_on_edge'] -= edge_length
                vehicle['route_index'] += 1

                if vehicle['route_index'] >= len(vehicle['route']) - 1:
                    # 已经到达最终目的地
                    vehicle['has_arrived'] = True
                    vehicle['position'] = self.G.nodes[vehicle['route'][-1]]['position']
                    vehicle['current_edge'] = None
                    vehicle['edge_length'] = 0.0
                    vehicle['distance_on_edge'] = 0.0
                    #print(f"Vehicle {vehicle_id} has arrived at the destination at step {self.step_count}.")
                    break
                else:
                    # 更新当前边的信息
                    from_node = vehicle['route'][vehicle['route_index']]
                    to_node = vehicle['route'][vehicle['route_index'] + 1]
                    vehicle['current_edge'] = (from_node, to_node)
                    edge_length = self.calculate_edge_length(from_node, to_node)
                    vehicle['edge_length'] = edge_length
                    #print(
                        #f"Vehicle {vehicle_id} moves to next edge from {from_node} to {to_node} at step {self.step_count}.")

            if not vehicle.get('has_arrived', False):
                # 根据在当前边上的位置计算当前位置
                from_pos = np.array(self.G.nodes[from_node]['position'])
                to_pos = np.array(self.G.nodes[to_node]['position'])
                ratio = vehicle['distance_on_edge'] / edge_length if edge_length > 0 else 0.0
                ratio = np.clip(ratio, 0, 1)
                new_position = from_pos + (to_pos - from_pos) * ratio
                vehicle['position'] = new_position.tolist()
                #print(
                    #f"Vehicle {vehicle_id} after moving: position {vehicle['position']}, distance_on_edge {vehicle['distance_on_edge']}, route_index {vehicle['route_index']}")

    def get_state(self):
        state = []

        for vehicle in self.state['vehicles'].values():
            # 活动状态标志
            active_flag = 0.0 if vehicle.get('has_arrived', False) else 1.0
            # 车辆位置和数据需求归一化
            vehicle_state = [
                (vehicle['position'][0] / self.max_position) * active_flag,
                (vehicle['position'][1] / self.max_position) * active_flag,
                (vehicle['data_requirement'] / self.max_data_requirement) * active_flag,
                (vehicle['bandwidth_allocation'] / self.max_bandwidth) * active_flag,
                # 切片类型 one-hot 编码，乘以活动状态标志
                (1.0 if vehicle['slice_type'] == 'URLLC' else 0.0) * active_flag,
                (1.0 if vehicle['slice_type'] == 'eMBB' else 0.0) * active_flag,
                (1.0 if vehicle['slice_type'] == 'Both' else 0.0) * active_flag,
                active_flag  # 添加活动状态标志到状态中
            ]

            # 添加每个基站的距离、信道增益和总带宽
            for base_station_id, base_station in self.base_stations.items():
                distance = np.linalg.norm(np.array(vehicle['position']) - np.array(base_station['position'])) / self.max_position
                channel_gain = self.calculate_channel_gain(vehicle['position'], base_station['position'])
                base_station_state = [
                    distance * active_flag,
                    channel_gain * active_flag,
                    (base_station['total_bandwidth'] / self.max_bandwidth) * active_flag,
                ]
                vehicle_state.extend(base_station_state)
            state.extend(vehicle_state)

        state_array = np.array(state, dtype=np.float32)
        return state_array

    def calculate_reward(self):
        reward = 0.0
        for vehicle_id, vehicle in self.state['vehicles'].items():
            if vehicle.get('has_arrived', False):
                continue

            base_station_id = vehicle['base_station']
            slice_type = vehicle['slice_type']

            # 计算数据率和时延
            datarate = self.calculate_datarate(vehicle_id, vehicle, base_station_id)
            delay = self.calculate_delay(vehicle_id, vehicle, base_station_id)

            # 防止 NaN 或 Inf
            if np.isnan(datarate) or np.isinf(datarate):
                datarate = 0.0
            if np.isnan(delay) or np.isinf(delay):
                delay = float('inf')

            # 归一化
            normalized_delay = delay / self.max_delay if self.max_delay > 0 else 0.0
            normalized_datarate = datarate / self.min_rate if self.min_rate > 0 else 0.0

            # 根据切片类型计算奖励，提供 shaped reward
            if slice_type == 'URLLC':
                reward -= normalized_delay * self.delay_penalty_factor
                if delay <= self.max_delay:
                    reward += 1.0
            elif slice_type == 'eMBB':
                reward += normalized_datarate * self.rate_penalty_factor
                if datarate >= self.min_rate:
                    reward += 1.0
            elif slice_type == 'Both':
                reward -= normalized_delay * self.delay_penalty_factor
                reward += normalized_datarate * self.rate_penalty_factor
                if delay <= self.max_delay and datarate >= self.min_rate:
                    reward += 2.0  # 满足两个要求，奖励更高

        # 仅对奖励上限进行裁剪
        reward = np.clip(reward, -np.inf, 10)
        return reward

    def calculate_datarate(self, vehicle_id, vehicle, base_station_id):
        W_i_m = vehicle['bandwidth_allocation']  # Bandwidth allocated to the vehicle (Hz)
        if W_i_m <= 1e-6 or base_station_id is None:
            return 0.0  # No bandwidth allocated or not connected to a base station

        P = self.base_stations[base_station_id]['transmission_power']  # Transmission power (W)
        Gt_i_m = self.calculate_channel_gain(vehicle['position'], self.base_stations[base_station_id]['position'])
        sigma2 = self.noise_power
        Itm = self.calculate_interference(vehicle_id, base_station_id, vehicle['position'])

        sinr_denominator = sigma2 + Itm
        if sinr_denominator <= 0:
            sinr = 0.0
        else:
            sinr = (P * Gt_i_m) / sinr_denominator

        sinr = max(sinr, 0.0)  # Ensure SINR is non-negative

        if sinr <= 0:
            datarate = 0.0
        else:
            datarate = W_i_m * np.log2(1 + sinr)

        datarate_mbps = datarate / 1e6  # Convert to Mbps

        return datarate_mbps

    def calculate_delay(self, vehicle_id, vehicle, base_station_id):
        data_size = vehicle['data_requirement']  # Data size (Mbits)
        datarate_mbps = self.calculate_datarate(vehicle_id, vehicle, base_station_id)
        if datarate_mbps <= 1e-6:
            return float('inf')  # Zero data rate leads to infinite delay

        transmission_delay = data_size / datarate_mbps  # Delay (seconds)
        return transmission_delay




    def calculate_channel_gain(self, vehicle_position, base_station_position):
        # 频率和波长
        frequency = 2e9  # 2 GHz
        c = 3e8  # 光速
        wavelength = c / frequency

        d0 = 1.0  # 参考距离 (1 米)
        path_loss_exponent = 3.5  # 路径损耗指数
        distance = np.linalg.norm(np.array(vehicle_position) - np.array(base_station_position))
        if distance < d0:
            distance = d0

        # 路径损耗
        Lpath = (wavelength / (4 * np.pi * distance)) ** 2
        Lpath *= (d0 / distance) ** path_loss_exponent
        # 阴影衰落
        Lshadow = 10 ** (- np.random.normal(0, 2) / 10)
        # 小尺度衰落
        Lfading = np.random.rayleigh(scale=1.0)

        channel_gain = Lpath * Lshadow * Lfading
        return channel_gain

    def calculate_interference(self, vehicle_id, base_station_id, vehicle_position):
        interference = 0.0
        for other_base_station_id, other_base_station in self.base_stations.items():
            if other_base_station_id != base_station_id:
                P_other = other_base_station['transmission_power']
                G_other = self.calculate_channel_gain(vehicle_position, other_base_station['position'])
                interference += P_other * G_other
            else:
                # 同一基站其他车辆的干扰
                for other_vehicle_id, other_vehicle in self.state['vehicles'].items():
                    if other_vehicle_id != vehicle_id and other_vehicle['base_station'] == base_station_id:
                        W_other = other_vehicle['bandwidth_allocation']
                        if W_other <= 1e-6:
                            continue
                        P_other = self.base_stations[base_station_id]['transmission_power']
                        G_other = self.calculate_channel_gain(vehicle_position, other_vehicle['position'])
                        interference += P_other * G_other
        return interference

    def check_if_done(self):
        all_arrived = all(vehicle.get('has_arrived', False) for vehicle in self.state['vehicles'].values())
        #print(f"Check if done: {all_arrived}")
        return all_arrived

    def render(self, mode='human'):
        # 可视化环境
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        # 绘制基站
        for bs_id, bs in self.base_stations.items():
            if bs_id == 0:
                # MBS：红色上三角形
                plt.scatter(bs['position'][0], bs['position'][1], c='red', marker='^', s=200, label='MBS')
                plt.text(bs['position'][0] + 0.05, bs['position'][1] + 0.05, 'MBS', color='red')
            else:
                # SBS：绿色下三角形
                label = 'SBS' if bs_id == 1 else ''  # 仅为第一个 SBS 添加图例标签，避免重复
                plt.scatter(bs['position'][0], bs['position'][1], c='green', marker='v', s=200, label=label)
                plt.text(bs['position'][0] + 0.05, bs['position'][1] + 0.05, f'SBS{bs_id}', color='green')
        # 确保图例中只有一个 MBS 和一个 SBS
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        # 绘制道路网络
        pos = nx.get_node_attributes(self.G, 'position')
        nx.draw(self.G, pos, node_color='gray', node_size=50, with_labels=True)
        # 绘制车辆
        for vehicle in self.state['vehicles'].values():
            if vehicle.get('has_arrived', False):
                continue  # 不绘制已到达终点的车辆
            plt.scatter(vehicle['position'][0], vehicle['position'][1], c='blue', s=100)
            # 连接车辆和基站
            if 'base_station' in vehicle and vehicle['base_station'] is not None:
                bs_pos = self.base_stations[vehicle['base_station']]['position']
                plt.plot([vehicle['position'][0], bs_pos[0]], [vehicle['position'][1], bs_pos[1]], 'k--')
        plt.xlim(0, self.max_position)
        plt.ylim(0, self.max_position)
        plt.xlabel('X 位置')
        plt.ylabel('Y 位置')
        plt.title('V2X Environment')

        plt.show()
        plt.close()  # 关闭图形以释放资源


    def close(self):
        pass

