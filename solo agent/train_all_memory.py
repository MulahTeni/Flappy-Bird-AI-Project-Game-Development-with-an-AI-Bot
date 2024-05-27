import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import pygame
import sys

from flappy_game import FlappyBirdEnvironment

class DQNetwork(nn.Module):
    def __init__(self, input_size, num_actions):
        super(DQNetwork, self).__init__()
        self.fc = nn.Linear(input_size, num_actions)

    def forward(self, state):
        return self.fc(state)

class DQNAgent:
    def __init__(self, state_size, action_size, screen):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1  # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DQNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.screen = screen

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            if state[0][4] - state[0][2] < random.randint(25, 60):
                return 1, 1
            return 0, 1
        state = torch.FloatTensor(state)
        act_values = self.model(state)
        return torch.argmax(act_values, dim=1).item(), 0

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0)  # Adding batch dimension
            next_state = torch.FloatTensor(next_state).unsqueeze(0)  # Adding batch dimension

            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state)).item()

            target_f = self.model(state)
            target_values = target_f.clone().detach()

            if action >= target_values.shape[1]:
                continue

            target_values[0][action] = target  # [0] since batch dimension was added

            self.optimizer.zero_grad()
            loss = self.criterion(target_f, target_values)
            loss.backward()
            self.optimizer.step()

def reset_environment(flappy_env, screen):
    # Reinitialize the environment
    flappy_env.__init__(screen)
    return flappy_env.get_states()

def main():
    pygame.init()
    screen = pygame.display.set_mode((FlappyBirdEnvironment.SCREEN_WIDTH, FlappyBirdEnvironment.SCREEN_HEIGHT))
    pygame.display.set_caption("Flappy Bird")

    flappy_env = FlappyBirdEnvironment(screen)
    state_size = flappy_env.STATE_SIZE
    action_size = flappy_env.ACTION_SIZE
    agent = DQNAgent(state_size, action_size, screen)
    episodes = 10000
    max_score = 0
    max_reward = 0
    batch_size = 128
    clock = pygame.time.Clock()

    for e in range(episodes):
        state = reset_environment(flappy_env, screen)
        state = np.reshape(state, [1, state_size])
        done = False
        total_reward = 0
        total_jump = 0
        flapping = False
        _r = 0
        while not done:
            action = 0
            if not flapping:
                action, r = agent.act(state)
                _r += r
            if action == 1:
                flapping = True
            else:
                flapping = not flapping

            next_state, reward, done = flappy_env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # Oyun ekranını güncelle
            flappy_env.render()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            pygame.display.update()
            clock.tick(30)  

            if done:
                print(f"Episode: {e}/{episodes}\tScore: {flappy_env.score}\tTotal Reward: {total_reward:.2f}\tEpsilon: {agent.epsilon:.2f}\t R: {_r}")
                if flappy_env.score > max_score:
                    max_score = flappy_env.score

                if e >= 200 and total_reward > max_reward:
                    max_reward = total_reward
                    torch.save(agent.model.state_dict(), 'flappy_bird_dqn_model.pth')
                    print(f"Model saved with Total Reward: {total_reward:.2f} at Episode: {e}")

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

    agent.model.load_state_dict(torch.load('flappy_bird_dqn_model.pth'))

if __name__ == "__main__":
    main()
