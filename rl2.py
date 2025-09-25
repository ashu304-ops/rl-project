
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback
import matplotlib.pyplot as plt

class ImprovedQuarterCarSuspensionEnv(gym.Env):
    """
    Improved version with curriculum learning and better reward shaping
    """

    def __init__(self, dt=0.01, max_steps=1000, difficulty='easy'):
        super(ImprovedQuarterCarSuspensionEnv, self).__init__()

        # Physical parameters
        self.mb = 300.0    
        self.mw = 60.0     
        self.ks = 16000.0  
        self.cs = 1000.0   
        self.kt = 190000.0 
        self.ct = 10.0     

        self.dt = dt
        self.max_steps = max_steps
        self.current_step = 0
        self.difficulty = difficulty

        # State and action spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1000.0, high=1000.0, shape=(1,), dtype=np.float32
        )

        self.state = np.zeros(6)
        self.road_profile = self._generate_road_profile()

    def _generate_road_profile(self):
        """Generate road profile based on difficulty level"""
        time = np.linspace(0, self.max_steps * self.dt, self.max_steps)

        if self.difficulty == 'easy':
            # Class A road (very smooth)
            psd_factor = 16e-6  # Much smoother
            bump_height = 0.01  # Smaller bumps
        elif self.difficulty == 'medium':
            # Class B road (smooth)
            psd_factor = 64e-6
            bump_height = 0.025
        else:  # hard
            # Class C road (average)
            psd_factor = 256e-6
            bump_height = 0.05

        frequencies = np.linspace(0.1, 30, 1000)
        psd_values = psd_factor / frequencies**2
        phases = np.random.uniform(0, 2*np.pi, len(frequencies))

        road_height = np.zeros(len(time))
        for i, (freq, psd, phase) in enumerate(zip(frequencies, psd_values, phases)):
            amplitude = np.sqrt(2 * psd * (frequencies[1] - frequencies[0]))
            road_height += amplitude * np.sin(2 * np.pi * freq * time + phase)

        # Add smaller bumps for easier training
        if self.difficulty != 'easy':
            bump_positions = [200, 500, 800]
            for pos in bump_positions:
                if pos < len(road_height):
                    road_height[pos:pos+20] += bump_height * np.exp(-np.linspace(0, 5, 20))

        return road_height

    def _get_road_input(self):
        if self.current_step >= len(self.road_profile):
            return 0.0, 0.0
        road_height = self.road_profile[self.current_step]
        if self.current_step > 0:
            road_velocity = (road_height - self.road_profile[self.current_step-1]) / self.dt
        else:
            road_velocity = 0.0
        return road_height, road_velocity

    def _dynamics(self, state, action, road_input):
        x_b, x_b_dot, x_w, x_w_dot, _, _ = state
        u = action[0]
        road_height, road_velocity = road_input

        f_spring = self.ks * (x_w - x_b)
        f_damper = self.cs * (x_w_dot - x_b_dot)
        f_suspension = f_spring + f_damper
        f_tire = self.kt * (road_height - x_w) + self.ct * (road_velocity - x_w_dot)

        x_b_ddot = (f_suspension + u) / self.mb
        x_w_ddot = (-f_suspension - u + f_tire) / self.mw

        return np.array([x_b_dot, x_b_ddot, x_w_dot, x_w_ddot, 
                        road_velocity, (road_input[1] if self.current_step > 0 else 0)])

    def step(self, action):
        road_input = self._get_road_input()
        self.state[4:6] = road_input

        # RK4 integration
        k1 = self._dynamics(self.state, action, road_input)
        k2 = self._dynamics(self.state + 0.5 * self.dt * k1, action, road_input)
        k3 = self._dynamics(self.state + 0.5 * self.dt * k2, action, road_input)
        k4 = self._dynamics(self.state + self.dt * k3, action, road_input)

        self.state[:4] += (self.dt / 6.0) * (k1[:4] + 2*k2[:4] + 2*k3[:4] + k4[:4])

        # Improved reward function
        reward = self._calculate_improved_reward(action)

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        # Relaxed safety constraints for easier learning
        if abs(self.state[0] - self.state[2]) > 0.15:  # Increased limit
            reward -= 100  # Reduced penalty
            terminated = True

        return self.state.astype(np.float32), reward, terminated, truncated, {}

    def _calculate_improved_reward(self, action):
        """Improved reward function with better shaping"""
        x_b, x_b_dot, x_w, x_w_dot, _, _ = self.state

        # Body acceleration (primary comfort metric)
        body_accel = abs(self._dynamics(self.state, action, (0, 0))[1])

        # Reward shaping for better learning
        if body_accel < 0.5:
            comfort_reward = 10  # Bonus for good performance
        elif body_accel < 1.0:
            comfort_reward = 5
        else:
            comfort_reward = -body_accel**2

        # Suspension travel penalty (safety)
        suspension_travel = abs(x_b - x_w)
        travel_penalty = -(suspension_travel * 50)**2

        # Control effort penalty (efficiency)
        control_penalty = -(abs(action[0]) / 1000.0) * 0.1

        # Stability reward
        stability_reward = -0.1 * (abs(x_b_dot) + abs(x_w_dot))

        # Improved weighting
        total_reward = comfort_reward + 0.1 * travel_penalty + control_penalty + stability_reward

        return total_reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.normal(0, 0.0001, 6)  # Smaller initial noise
        self.current_step = 0
        self.road_profile = self._generate_road_profile()
        return self.state.astype(np.float32), {}

    def set_difficulty(self, difficulty):
        """Change difficulty level"""
        self.difficulty = difficulty
        self.road_profile = self._generate_road_profile()


def create_improved_training():
    """Create improved training setup with curriculum learning"""

    print("Creating improved suspension RL training setup...")

    # Start with easy environment
    env = ImprovedQuarterCarSuspensionEnv(difficulty='easy')

    # Improved hyperparameters
    action_noise = NormalActionNoise(mean=np.zeros(1), sigma=0.2)

    model = TD3('MlpPolicy', env,
                action_noise=action_noise,
                learning_rate=1e-5,
                buffer_size=500000,
                learning_starts=25000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                policy_delay=2,
                target_policy_noise=0.2,
                target_noise_clip=0.5,
                policy_kwargs=dict(net_arch=[256, 256]),
                verbose=1)

    return env, model


def curriculum_training(env, model, total_timesteps=150000):
    """Implement curriculum learning"""

    print("Starting curriculum training...")

    # Stage 1: Easy road (50k timesteps)
    print("Stage 1: Training on easy road...")
    env.set_difficulty('easy')
    model.learn(total_timesteps=50000, reset_num_timesteps=False)

    # Stage 2: Medium road (50k timesteps)
    print("Stage 2: Training on medium road...")
    env.set_difficulty('medium')
    model.learn(total_timesteps=50000, reset_num_timesteps=False)

    # Stage 3: Hard road (50k timesteps)
    print("Stage 3: Training on hard road...")
    env.set_difficulty('hard')
    model.learn(total_timesteps=50000, reset_num_timesteps=False)

    print("Curriculum training completed!")
    return model


def analyze_improved_performance(env, model, episodes=10):
    """Analyze performance and log per-episode metrics"""

    logs = {
        'body_acceleration_rms': [],
        'suspension_travel_rms': [],
        'control_effort_rms': [],
        'comfort_index': [],
        'episode_rewards': [],
        'episode_lengths': []
    }

    env.set_difficulty('hard')  # Evaluate on hardest setting

    for episode in range(episodes):
        obs, _ = env.reset()
        body_accels = []
        travel_distances = []
        control_efforts = []
        episode_reward = 0
        steps = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            body_accel = abs(env._dynamics(env.state, action, (0, 0))[1])
            travel_dist = abs(env.state[0] - env.state[2])

            body_accels.append(body_accel)
            travel_distances.append(travel_dist)
            control_efforts.append(abs(action[0]))
            episode_reward += reward
            steps += 1

        # Store per-episode metrics
        if body_accels:
            logs['body_acceleration_rms'].append(np.sqrt(np.mean(np.square(body_accels))))
            logs['suspension_travel_rms'].append(np.sqrt(np.mean(np.square(travel_distances))))
            logs['control_effort_rms'].append(np.sqrt(np.mean(np.square(control_efforts))))
            logs['episode_rewards'].append(episode_reward)
            logs['episode_lengths'].append(steps)

            # Comfort index (ISO 2631-like)
            avg_accel = np.mean(body_accels)
            if avg_accel < 0.315:
                idx = 1
            elif avg_accel < 0.63:
                idx = 2
            elif avg_accel < 1.0:
                idx = 3
            else:
                idx = 4
            logs['comfort_index'].append(idx)

    return logs

def plot_training_metrics(logs):
    """Plot key performance metrics across episodes"""
    episodes = np.arange(1, len(logs['episode_rewards']) + 1)

    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    plt.plot(episodes, logs['episode_rewards'], marker='o', label='Episode Reward')
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(episodes, logs['body_acceleration_rms'], marker='o', color='orange', label='Body Accel RMS')
    plt.title("Body Acceleration RMS")
    plt.xlabel("Episode")
    plt.ylabel("Acceleration (m/s²)")
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(episodes, logs['comfort_index'], marker='o', color='green', label='Comfort Index')
    plt.title("Comfort Index")
    plt.xlabel("Episode")
    plt.ylabel("Index (1=best, 4=worst)")
    plt.yticks([1, 2, 3, 4])
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(episodes, logs['control_effort_rms'], marker='o', color='red', label='Control Effort RMS')
    plt.title("Control Effort RMS")
    plt.xlabel("Episode")
    plt.ylabel("Effort (N)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Usage example
if __name__ == "__main__":
    # Create training setup
    env, model = create_improved_training()

    # Train using curriculum or adaptive curriculum
    trained_model = curriculum_training(env, model)

    # Analyze performance
    logs = analyze_improved_performance(env, trained_model, episodes=10)

    # Print final averages
    print("\nFINAL PERFORMANCE METRICS:")
    print(f"Body Acceleration RMS: {np.mean(logs['body_acceleration_rms']):.4f} m/s²")
    print(f"Comfort Index: {np.mean(logs['comfort_index']):.1f}/4")
    print(f"Control Effort RMS: {np.mean(logs['control_effort_rms']):.1f} N")
    print(f"Average Episode Reward: {np.mean(logs['episode_rewards']):.1f}")
    print(f"Average Episode Length: {np.mean(logs['episode_lengths']):.0f} steps")

    # Plot performance
    plot_training_metrics(logs)
