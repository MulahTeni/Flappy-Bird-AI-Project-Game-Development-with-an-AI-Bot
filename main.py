import pygame
import sys
import os
import random

# Oyun ici genislik ve yukseklikler
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 500
PIPE_HEIGHT = 400


# Arkaplanın hızı
BACKGROUND_SPEED = 1

# Yer çekimi ivmesi
GRAVITY = 2
MAX_FALL_SPEED = 20

# Oyun kare hızı
framepersecond = 32

def calculate_upper_pipe_y_and_gap_size():
    return random.randint(-200, -20), random.randint(20, 40)

def draw_background(screen, background_img, background_rects):
    for background_rect in background_rects:
        screen.blit(background_img, background_rect)

def draw_bird(screen, bird_img, bird_rect):
    screen.blit(bird_img, bird_rect)

def draw_pipes(screen, pipe_img, upper_pipes, lower_pipes):
    for upper_pipe, lower_pipe in zip(upper_pipes, lower_pipes):
        screen.blit(pipe_img[0], (upper_pipe['x'], upper_pipe['y']))
        screen.blit(pipe_img[1], (lower_pipe['x'], lower_pipe['y']))
    
def handle_input():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return "toggle_pause"  # Oyun durumu değiştir
            elif event.key == pygame.K_SPACE or event.key == pygame.K_UP:
                return "flap"  # Kuşu zıplat
            elif event.key == pygame.K_q:  # "q" tuşuna basıldığında oyunu kapat
                pygame.quit()
                sys.exit()
    return None

def update_game_state(screen, bird_rect, upper_pipes, lower_pipes, pipe_img, background_rects):
    # Kuşun çarpma kontrolü
    if bird_rect.top <= 0 or bird_rect.bottom >= SCREEN_HEIGHT:
        return True  # Çarpma oldu, oyunu duraklat
    
    # Borulara çarpma kontrolü
    bird_collision_rect = pygame.Rect(bird_rect.x + 5, bird_rect.y + 5, bird_rect.width - 10, bird_rect.height - 10)
    for u_pipe, l_pipe in zip(upper_pipes, lower_pipes):
        upper_pipe_rect = pygame.Rect(u_pipe['x'], u_pipe['y'], pipe_img[0].get_width() - 20, pipe_img[0].get_height() - 10)
        lower_pipe_rect = pygame.Rect(l_pipe['x'], l_pipe['y'], pipe_img[1].get_width() - 20, pipe_img[1].get_height() - 10)
        
        if bird_collision_rect.colliderect(upper_pipe_rect) or bird_collision_rect.colliderect(lower_pipe_rect):
            return True  # Borulara çarpma oldu, oyunu duraklat

    # Arkaplanları kaydır
    for background_rect in background_rects:
        background_rect.x -= BACKGROUND_SPEED
        if background_rect.right <= 0:
            background_rect.x = SCREEN_WIDTH
    
    return False  # Çarpma yok, oyun devam ediyor

def game_loop(screen, background_img, bird_img, bird_rect, background_rects, pipe_img, upper_pipes, lower_pipes):
    paused = False
    framepersecond_clock = pygame.time.Clock()
    
    # Oyuncu hızları
    bird_velocity_y = 0  # Kuşun başlangıç düşme hızı
    bird_flap_velocity = -15  # Kuşun zıplama hızı
    
    while True:
        # Kullanıcı girişlerini işle
        action = handle_input()
        if action == "toggle_pause":
            paused = not paused
        elif action == "flap":
            bird_velocity_y = bird_flap_velocity  # Kuşu zıplat

        if not paused:
            # Oyun durumu güncelle
            collision = update_game_state(screen, bird_rect, upper_pipes, lower_pipes, pipe_img, background_rects)
            if collision:
                paused = True  # Oyunu duraklat
                # Kuşun çarptığı yerde kalmasını sağla
                if bird_rect.top <= 0:
                    bird_rect.top = 0
                elif bird_rect.bottom >= SCREEN_HEIGHT:
                    bird_rect.bottom = SCREEN_HEIGHT

            # Yer çekimi uygula
            bird_velocity_y += GRAVITY
            bird_velocity_y = min(bird_velocity_y, MAX_FALL_SPEED)  # Düşme hızını maksimum düşme hızıyla sınırla
            bird_rect.y += bird_velocity_y

            # Arkaplanları kaydır
            for background_rect in background_rects:
                background_rect.x -= BACKGROUND_SPEED
                if background_rect.right <= 0:
                    background_rect.x = SCREEN_WIDTH

            # Boruları kaydır
            for u_pipe, l_pipe in zip(upper_pipes, lower_pipes):
                u_pipe['x'] -= BACKGROUND_SPEED
                l_pipe['x'] -= BACKGROUND_SPEED

            # Ekranı temizle
            screen.fill((0, 0, 0))

            # Arkaplanları ekrana çiz
            draw_background(screen, background_img, background_rects)

            # Boruları ekrana çiz
            draw_pipes(screen, pipe_img, upper_pipes, lower_pipes)

            # Kuşu ekrana çiz
            draw_bird(screen, bird_img, bird_rect)

            # Ekranı güncelle
            pygame.display.update()

        # FPS sınırlaması
        framepersecond_clock.tick(framepersecond)

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Flappy Bird")

    # Arkaplan resmini yükle
    background_img = pygame.image.load(os.path.join("images", "background.jpg")).convert()
    background_img = pygame.transform.scale(background_img, (SCREEN_WIDTH, SCREEN_HEIGHT))

    # Kuş resmini yükle
    bird_img = pygame.image.load(os.path.join("images", "bird.png")).convert_alpha()
    bird_img = pygame.transform.scale(bird_img, (50, 50))  # Kuş resmini boyutlandır

    # Boru resmini yükle
    pipe_img = (
        pygame.transform.rotate(pygame.image.load(os.path.join("images", "pipe.png")).convert_alpha(), 180),
        pygame.image.load(os.path.join("images", "pipe.png")).convert_alpha()
    )

    # Kuşun başlangıç konumu
    bird_rect = bird_img.get_rect()
    bird_rect.center = (SCREEN_WIDTH // 4, SCREEN_HEIGHT // 2)  # Ekranın sol ortasında başlat

    bird_width = bird_rect.width
    bird_height = bird_rect.height
    
    # İki kopya arkaplan oluştur
    background_rects = []
    for i in range(2):
        background_rects.append(background_img.get_rect())

    # İkinci arkaplanı ilk arkaplanın sağ tarafına yerleştir
    background_rects[1].x = SCREEN_WIDTH

    # Başlangıç boruları oluştur
    upper_pipes = []
    lower_pipes = []
    
    pipe_y, gap_size = calculate_upper_pipe_y_and_gap_size()
    upper_pipes.append({'x': 3 * SCREEN_WIDTH // 4, 'y': pipe_y})
    lower_pipes.append({'x': 3 * SCREEN_WIDTH // 4, 'y': pipe_y + gap_size + PIPE_HEIGHT})

    pipe_y, gap_size = calculate_upper_pipe_y_and_gap_size()
    upper_pipes.append({'x': SCREEN_WIDTH // 2, 'y': pipe_y})
    lower_pipes.append({'x': SCREEN_WIDTH // 2, 'y': pipe_y + gap_size + PIPE_HEIGHT})

    # Oyun döngüsü başlat
    game_loop(screen, background_img, bird_img, bird_rect, background_rects, pipe_img, upper_pipes, lower_pipes)

if __name__ == "__main__":
    main()