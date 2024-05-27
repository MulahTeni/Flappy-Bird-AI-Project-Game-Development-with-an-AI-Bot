import torch
import random
import numpy as np
from collections import deque
from game import FlappyBirdEnvironment, Bird
from model import Linear_QNet, QTrainer
import pygame

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
EPOCHS = 10000

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 1 # randomness
        self.decay = 0.995
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(9, 3)  # input_size = 9 (state dimension), output_size = 3 (actions)
        self.trainer = QTrainer(self.model)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return 1 if np.random.rand() < 0.15 else 0
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            return torch.argmax(prediction).item()

def train():
    agent = Agent()
    game = FlappyBirdEnvironment()
    framepersecond_clock = pygame.time.Clock()
    record_reward = 0
    
    for epoch in range(EPOCHS):
        game.reset_game()
        game.add_birds(10)
        curr_record_reward = 0
        score = 0
        while not game.game_ended:
            game.update_game_states(game.birds)
            game.render()
            for bird in game.birds:
                if bird.is_alive:
                    state_old = game.get_states_bird(bird)
                    final_move = agent.get_action(state_old)
                    reward, done, score = game.step(bird, final_move)
                    
                    curr_record_reward = max(curr_record_reward, reward)
                    
                    state_new = game.get_states_bird(bird)
                    agent.remember(state_old, final_move, reward, state_new, done)
                    agent.train_short_memory(state_old, final_move, reward, state_new, done)
            framepersecond_clock.tick(30)
        
        agent.n_games += 1
        agent.train_long_memory()
        print(f"Epoch: {epoch}\tScore: {score}\tReward: {record_reward}")
        
        if record_reward < curr_record_reward:
            record_reward = curr_record_reward
            agent.model.save()

if __name__ == '__main__':
    train()