import pygame
import sys
import os
import random

PIPE_HEIGHT = 400
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 500
BACKGROUND_SPEED = 1

class Bird:
    def __init__(self, x, y, path):
        self.x = x
        self.y = y
        self.reward = 0
        self.img = pygame.image.load(path).convert_alpha()
        self.img = pygame.transform.scale(self.img, (25, 25))
        self.horizontal_speed = 0
        self.vertical_speed = 0
        self.vertical_speed_max = 20
        self.gravity = 2
        self.after_flap_speed = -12
        self.score = 0
        self.reward = 0
        self.is_alive = True
        
    def update_state(self):
        self.vertical_speed = min(self.vertical_speed + self.gravity, self.vertical_speed_max)
        self.y += self.vertical_speed
    
    def flap(self):
        self.y = self.y - 10
        self.vertical_speed = self.after_flap_speed        
    
    def draw(self, screen):
        screen.blit(self.img, (self.x, self.y))
    
class PipePair:    
    def __init__(self, x, y, gap, path):
        self.x = x
        self.y = y
        self.gap = gap
        self.upper_img = pygame.transform.rotate(pygame.image.load(path).convert_alpha(), 180)
        self.lower_img = pygame.image.load(path).convert_alpha()  
        
    def update_state(self, speed):
        self.x -= speed
    
    def draw(self, screen):
        screen.blit(self.upper_img, (self.x, self.y))
        screen.blit(self.lower_img, (self.x, self.y + self.gap + PIPE_HEIGHT))
    
class FlappyBirdEnvironment:
    pipe_path = os.path.join("images", "pipe.png")
    background_path = os.path.join("images", "background.jpg")
    bird_path = os.path.join("images", "birds")
        
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Flappy Bird")
        self.reset_game()
    
    def reset_game(self):
        self.score = 0
        self.birds = []
        
        self.pipes = []
        self.pipes.append(PipePair(SCREEN_WIDTH // 2, random.randint(-200, -20), random.randint(60, 75), self.pipe_path))
        self.pipes.append(PipePair((SCREEN_WIDTH * 3) // 4, random.randint(-200, -20), random.randint(60, 75), self.pipe_path))
        
        self.speed = 3
        self.max_speed = 15
        
        self.background_img = pygame.image.load(self.background_path).convert()
        self.background_img = pygame.transform.scale(self.background_img, (SCREEN_WIDTH, SCREEN_HEIGHT))
        self.background_rects = [self.background_img.get_rect(), self.background_img.get_rect()]
        self.background_rects[1].x = SCREEN_WIDTH
        
        self.game_ended = False
    
    def random_bird_image(self):
        files = os.listdir(self.bird_path)
        bird_images = [file for file in files if file.endswith('.png')]
        if not bird_images:
            print("No bird images found in directory.")
            return None
        random_image = random.choice(bird_images)
        return os.path.join(self.bird_path, random_image)

    def add_birds(self, count):
        start_x = SCREEN_WIDTH // 4
        start_y = SCREEN_HEIGHT // 2
        for _ in range(count):
            path = self.random_bird_image()
            if path is None:
                continue
            new_bird = Bird(start_x, start_y, path)
            self.birds.append(new_bird)
    
    def adjust_pipes(self):
        for pipe in self.pipes[:]:
            if pipe.x <= SCREEN_WIDTH // 4 - 35:
                self.pipes.remove(pipe)
                self.score += 1
                # if self.score % 5 == 0 and self.score > 0:
                    # self.speed = min(self.speed + 1, self.max_speed)
                break
        rightmost_x = max(pipe.x for pipe in self.pipes)
        if rightmost_x <= (SCREEN_WIDTH * 3 / 4):
            self.pipes.append(PipePair(SCREEN_WIDTH, random.randint(-200, -20), random.randint(25, 45), self.pipe_path))    
    
    def render(self):
        self.screen.fill((0, 0, 0))
        for background_rect in self.background_rects:
            self.screen.blit(self.background_img, background_rect)
        for pipe in self.pipes:
            pipe.draw(self.screen)
        for bird in self.birds:
            if bird.is_alive:
                bird.draw(self.screen)
        pygame.display.update()
        
    def check_collisions(self):
        for bird in self.birds:
            if bird.is_alive:
                bird_rect = bird.img.get_rect(topleft=(bird.x, bird.y))
                if bird_rect.top <= 0 or bird_rect.bottom >= SCREEN_HEIGHT:
                    bird.is_alive = False
                for pipe in self.pipes:
                    upper_pipe_rect = pipe.upper_img.get_rect(topleft=(pipe.x, pipe.y))
                    lower_pipe_rect = pipe.lower_img.get_rect(topleft=(pipe.x, pipe.y + pipe.gap + PIPE_HEIGHT))
                    if bird_rect.colliderect(upper_pipe_rect) or bird_rect.colliderect(lower_pipe_rect):
                        bird.is_alive = False
                        break
                bird.reward += 0.001 * (self.speed - 2)
                # print(bird.reward)
        
    def update_game_states(self, birds):
        for background_rect in self.background_rects:
            background_rect.x -= BACKGROUND_SPEED
            if background_rect.right <= 0:
                background_rect.x = SCREEN_WIDTH
                
        for pipe in self.pipes:
            pipe.update_state(self.speed)
        self.adjust_pipes()
        
        for bird in birds:
            if bird.is_alive:
                bird.score = self.score
                bird.update_state()
        
        self.check_collisions()
        if not any(bird.is_alive for bird in self.birds):
            self.game_ended = True
    
    def step(self, bird, action):
        if action == 1:
            bird.flap()
        return bird.reward, not bird.is_alive, bird.score
    
    def get_states_bird(self, bird):
        next_pipes = [pipe for pipe in self.pipes if pipe.x + pipe.upper_img.get_width() + 30 > bird.x]
        if len(next_pipes) >= 2:
            next_pipe1 = next_pipes[0]
            next_pipe2 = next_pipes[1]
            
            horizontal_dist1 = next_pipe1.x - bird.x
            vertical_dist1 = (next_pipe1.y + next_pipe1.gap + PIPE_HEIGHT) - bird.y
            gap_size1 = next_pipe1.gap
            
            horizontal_dist2 = next_pipe2.x - bird.x
            vertical_dist2 = (next_pipe2.y + next_pipe2.gap + PIPE_HEIGHT) - bird.y
            gap_size2 = next_pipe2.gap
            
            return [
                bird.x, bird.y,
                self.speed,
                horizontal_dist1, vertical_dist1, gap_size1,
                horizontal_dist2, vertical_dist2, gap_size2
            ]
        return [bird.x, bird.y, self.speed] + [0]*6
        
        
    def start_game_loop(self):
        framepersecond_clock = pygame.time.Clock()
        while not self.game_ended:
            self.update_game_states(self.birds)
            self.render()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        if self.birds[0].is_alive:
                            self.birds[0].flap()
            framepersecond_clock.tick(30)
        

pygame.init()

if __name__ == "__main__":
    game = FlappyBirdEnvironment()
    game.add_birds(1)
    game.start_game_loop()
