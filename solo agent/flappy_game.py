import pygame, sys, os, random
import numpy as np

class FlappyBirdEnvironment:
    SCREEN_WIDTH = 1200
    SCREEN_HEIGHT = 500
    PIPE_HEIGHT = 400

    def __init__(self, screen):
        # Oyun ici hızlar
        self.BACKGROUND_SPEED = 1
        self.PIPE_START_SPEED = 4
        self.PIPE_MAX_SPEED = 9
        self.framepersecond = 30

        # Yer çekimi ivmesi
        self.GRAVITY = 2
        self.MAX_FALL_SPEED = 20

        # Resimleri yükle
        self.background_img = pygame.image.load(os.path.join("images", "background.jpg")).convert()
        self.background_img = pygame.transform.scale(self.background_img, (self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.bird_img = pygame.image.load(os.path.join("images", "bird.png")).convert_alpha()
        self.bird_img = pygame.transform.scale(self.bird_img, (50, 50))
        self.pipe_img = (
            pygame.transform.rotate(pygame.image.load(os.path.join("images", "pipe.png")).convert_alpha(), 180),
            pygame.image.load(os.path.join("images", "pipe.png")).convert_alpha()
        )

        # Kuşun başlangıç konumu
        self.bird_rect = self.bird_img.get_rect()
        self.bird_rect.center = (self.SCREEN_WIDTH // 4, self.SCREEN_HEIGHT // 2)

        # Arkaplan resmi ve pozisyonları
        self.background_rects = []
        for i in range(2):
            self.background_rects.append(self.background_img.get_rect())
        self.background_rects[1].x = self.SCREEN_WIDTH

        # Borular
        self.upper_pipes = []
        self.lower_pipes = []
        self.gap_sizes = []
        
        pipe_y, gap_size = self.calculate_upper_pipe_y_and_gap_size()
        self.upper_pipes.append({'x': 3 * self.SCREEN_WIDTH // 4, 'y': pipe_y})
        self.lower_pipes.append({'x': 3 * self.SCREEN_WIDTH // 4, 'y': pipe_y + gap_size + self.PIPE_HEIGHT})
        self.gap_sizes.append({'gap': gap_size})
        
        pipe_y, gap_size = self.calculate_upper_pipe_y_and_gap_size()
        self.upper_pipes.append({'x': self.SCREEN_WIDTH // 2, 'y': pipe_y})
        self.lower_pipes.append({'x': self.SCREEN_WIDTH // 2, 'y': pipe_y + gap_size + self.PIPE_HEIGHT})
        self.gap_sizes.append({'gap': gap_size})
        
        self.paused = False

        # Oyuncu bilgileri
        self.bird_velocity_y = 0
        self.bird_flap_velocity = -12
        self.pipe_speed = self.PIPE_START_SPEED
        self.score = 0
        
        # Bot için ekstra bilgiler
        self.STATE_SIZE = 5
        self.ACTION_SIZE = 2
        
        self.screen = screen

    def calculate_upper_pipe_y_and_gap_size(self):
        return random.randint(-200, -20), random.randint(65, 75)

    def draw_background(self):
        for background_rect in self.background_rects:
            self.screen.blit(self.background_img, background_rect)

    def draw_bird(self):
        self.screen.blit(self.bird_img, self.bird_rect)

    def draw_pipes(self):
        for upper_pipe, lower_pipe in zip(self.upper_pipes, self.lower_pipes):
            self.screen.blit(self.pipe_img[0], (upper_pipe['x'], upper_pipe['y']))
            self.screen.blit(self.pipe_img[1], (lower_pipe['x'], lower_pipe['y']))

    def check_add_pipe(self):
        rightmost_x = max(upper_pipe['x'] for upper_pipe in self.upper_pipes)
        if rightmost_x <= self.SCREEN_WIDTH * 3 / 4:
            pipe_y, gap_size = self.calculate_upper_pipe_y_and_gap_size()
            self.upper_pipes.append({'x': self.SCREEN_WIDTH, 'y': pipe_y})
            self.lower_pipes.append({'x': self.SCREEN_WIDTH, 'y': pipe_y + gap_size + self.PIPE_HEIGHT})
            self.gap_sizes.append({'gap': gap_size})

    def check_remove_pipe(self):
        pipes_to_remove = []
        for i, (u_pipe, l_pipe) in enumerate(zip(self.upper_pipes, self.lower_pipes)):
            if u_pipe['x'] + self.pipe_img[0].get_width() < 0:
                pipes_to_remove.append(i)

        # Remove pipes and corresponding gap sizes in reverse order
        for index in reversed(pipes_to_remove):
            del self.upper_pipes[index]
            del self.lower_pipes[index]
            del self.gap_sizes[index]

    def get_distances(self):
        distances = []
        
        for i in range(len(self.upper_pipes)):
            if self.upper_pipes[i]['x'] + self.pipe_img[0].get_width() + 30 >= self.bird_rect.left:
                distances.append((self.upper_pipes[i]['x'] - self.bird_rect.x, # kuşun ilk boruya yatay uzaklığı
                    self.upper_pipes[i]['y'] + self.pipe_img[0].get_height(), # ilk üst borunun altı
                    self.lower_pipes[i]['y'])) # ilk alt borunun üstü
        
        # Mesafeleri [i][0]'a göre sıralayalım
        distances.sort(key=lambda x: x[0])

        if distances:
            return distances[0]

        return distances
    
    def render(self):
        self.screen.fill((0, 0, 0))  # Ekranı siyahla doldur

        # Arka planı çiz
        self.draw_background()

        # Boruları çiz
        self.draw_pipes()

        # Kuşu çiz
        self.draw_bird()

        # Puanı göster
        font = pygame.font.Font(None, 36)
        score_text = font.render("Score: " + str(self.score), True, (255, 0, 0))
        self.screen.blit(score_text, (10, 10))

        # Hızı göster
        speed_text = font.render("Speed: " + str(self.pipe_speed - 3), True, (255, 0, 0))
        self.screen.blit(speed_text, (10, 46))

        # Kuşun yüksekliğini göster
        bird_height_text = font.render("Bird Coor: " + str(self.bird_rect.center), True, (255, 0, 0))
        self.screen.blit(bird_height_text, (10, 82))
        
        # Uzaklık iblgilerini göster
        x_diff, y_up, y_bot = self.get_distances()
        
        vert = font.render("Veritcal distance: " + str(x_diff), True, (255, 0, 0))
        upper = font.render("Upper Pipe: " + str(y_up), True, (255, 0, 0))
        lower = font.render("Lower Pipe: " + str(y_bot), True, (255, 0, 0))
        
        self.screen.blit(vert, (10, 118))
        self.screen.blit(upper, (10, 154))
        self.screen.blit(lower, (10, 190))

        # Ekranı güncelle
        pygame.display.update()
    
    def update_game_state(self):
        if self.bird_rect.top <= 0 or self.bird_rect.bottom >= self.SCREEN_HEIGHT:
            return 0

        bird_collision_rect = pygame.Rect(self.bird_rect.x + 5, self.bird_rect.y + 5, self.bird_rect.width - 10, self.bird_rect.height - 10)
        playerMidPos = self.SCREEN_WIDTH // 5 + self.bird_img.get_width()/2

        for u_pipe, l_pipe in zip(self.upper_pipes, self.lower_pipes):
            upper_pipe_rect = pygame.Rect(u_pipe['x'] + 10, u_pipe['y'], self.pipe_img[0].get_width() - 20, self.pipe_img[0].get_height() - 20)
            lower_pipe_rect = pygame.Rect(l_pipe['x'] + 10, l_pipe['y'] + 5, self.pipe_img[1].get_width() - 20, self.pipe_img[1].get_height())

            if bird_collision_rect.colliderect(upper_pipe_rect) or bird_collision_rect.colliderect(lower_pipe_rect):
                return 1

            pipeMidPos = u_pipe['x'] + self.pipe_img[0].get_width() / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4 + self.pipe_speed - 3:
                self.score += 1
                if self.score % 10 == 0:
                    self.pipe_speed += 1

        for background_rect in self.background_rects:
            background_rect.x -= self.BACKGROUND_SPEED
            if background_rect.right <= 0:
                background_rect.x = self.SCREEN_WIDTH

        for u_pipe, l_pipe in zip(self.upper_pipes, self.lower_pipes):
            u_pipe['x'] -= self.pipe_speed
            l_pipe['x'] -= self.pipe_speed

        self.bird_velocity_y += self.GRAVITY
        self.bird_velocity_y = min(self.bird_velocity_y, self.MAX_FALL_SPEED)
        self.bird_rect.y += self.bird_velocity_y

        self.check_add_pipe()
        self.check_remove_pipe()

        return 2

    def get_states(self):
        x_diff, y_up, y_bot = self.get_distances()
        return [
                self.pipe_speed,
                x_diff, self.bird_rect.center[1], y_up, y_bot
                ]
        
    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return 3
                elif event.key == pygame.K_SPACE or event.key == pygame.K_UP:
                    return 1
                elif event.key == pygame.K_r:
                    return 2
                elif event.key == pygame.K_q:
                    pygame.quit()
                    sys.exit()
        return 0

    def reset(self):
        main()
        return self.get_states()
        
    def step(self, action):
        done = False
        
        if action == 1:  # Flap action
            self.bird_velocity_y = self.bird_flap_velocity
        elif action == 2:
            self.reset()
        elif action == 3:
            self.paused = not self.paused

        prev_score = self.score

        # Update game state
        collision = self.update_game_state()    
        reward = 0

        x_diff, y_up_diff, y_bot_diff = self.get_distances()
        
        # Calculate reward
        if collision == 0:
            reward = -100
            done = True
        elif collision == 1:
            reward = - (abs(y_up_diff) + abs(y_bot_diff) - 70) // 10
            done = True
        else:
            if prev_score < self.score:
                reward = 100
            else:
                if y_up_diff < 5  and y_bot_diff > 5:
                    reward = 0.1

        next_state = self.get_states()

        return next_state, reward, done
            
    def game_loop(self):
        self.paused = False
        framepersecond_clock = pygame.time.Clock()
        self.score = 0

        while True:
            action = self.handle_input()
            self.step(action)

            if not self.paused:
                self.screen.fill((0, 0, 0))
                self.draw_background()
                self.draw_pipes()
                self.draw_bird()
                pygame.display.update()

            framepersecond_clock.tick(self.framepersecond)

pygame.init()
screen = pygame.display.set_mode((FlappyBirdEnvironment.SCREEN_WIDTH, FlappyBirdEnvironment.SCREEN_HEIGHT))
pygame.display.set_caption("Flappy Bird")


def main():
    flappy_env = FlappyBirdEnvironment(screen)
    flappy_env.game_loop()

if __name__ == "__main__":
    main()