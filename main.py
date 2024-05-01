import pygame
import sys
import os
import random

# Oyun ici genislik ve yukseklikler
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 500
PIPE_HEIGHT = 400


# Oyun ici hızlar
BACKGROUND_SPEED = 1
PIPE_START_SPEED = 3
PIPE_MAX_SPEED = 9
framepersecond = 32

# Yer çekimi ivmesi
GRAVITY = 2
MAX_FALL_SPEED = 20


def calculate_upper_pipe_y_and_gap_size():
    return random.randint(-200, -20), random.randint(25, 45)

def draw_background(screen, background_img, background_rects):
    for background_rect in background_rects:
        screen.blit(background_img, background_rect)

def draw_bird(screen, bird_img, bird_rect):
    screen.blit(bird_img, bird_rect)

def draw_pipes(screen, pipe_img, upper_pipes, lower_pipes):
    for upper_pipe, lower_pipe in zip(upper_pipes, lower_pipes):
        screen.blit(pipe_img[0], (upper_pipe['x'], upper_pipe['y']))
        screen.blit(pipe_img[1], (lower_pipe['x'], lower_pipe['y']))
    
def check_add_pipe(upper_pipes, lower_pipes):
    # En sağdaki borunun x konumunu bul
    rightmost_x = max(upper_pipe['x'] for upper_pipe in upper_pipes)
    if rightmost_x <= SCREEN_WIDTH * 3 / 4:  # Eğer en sağdaki boru ekranın dörtte birine ulaştıysa
        # Yeni bir boru ekle
        pipe_y, gap_size = calculate_upper_pipe_y_and_gap_size()
        upper_pipes.append({'x': SCREEN_WIDTH, 'y': pipe_y})
        lower_pipes.append({'x': SCREEN_WIDTH, 'y': pipe_y + gap_size + PIPE_HEIGHT})

def check_remove_pipe(upper_pipes, lower_pipes, pipe_img):
    # Sol ekrandan çıkan boruları listeden sil
    for u_pipe, l_pipe in zip(upper_pipes, lower_pipes):
        if u_pipe['x'] + pipe_img[0].get_width() < 0:
            upper_pipes.remove(u_pipe)
            lower_pipes.remove(l_pipe)
   
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
            elif event.key == pygame.K_r:  # "r" tuşuna basıldığında oyunu yeniden başlat
                return "restart"
            elif event.key == pygame.K_q:  # "q" tuşuna basıldığında oyunu kapat
                pygame.quit()
                sys.exit()
    return None

def update_game_state(screen, bird_rect, bird_img, upper_pipes, lower_pipes, pipe_img, background_rects, score):
    # Kuşun çarpma kontrolü
    if bird_rect.top <= 0 or bird_rect.bottom >= SCREEN_HEIGHT:
        return True, score  # Çarpma oldu, oyunu duraklat ve skoru döndür
    
    # Borulara çarpma kontrolü ve skor artırma
    bird_collision_rect = pygame.Rect(bird_rect.x + 5, bird_rect.y + 5, bird_rect.width - 10, bird_rect.height - 10)
    playerMidPos = SCREEN_WIDTH // 5 + bird_img.get_width()/2
    
    for u_pipe, l_pipe in zip(upper_pipes, lower_pipes):
        upper_pipe_rect = pygame.Rect(u_pipe['x'], u_pipe['y'], pipe_img[0].get_width() - 20, pipe_img[0].get_height() - 10)
        lower_pipe_rect = pygame.Rect(l_pipe['x'], l_pipe['y'], pipe_img[1].get_width() - 20, pipe_img[1].get_height() - 10)
        
        if bird_collision_rect.colliderect(upper_pipe_rect) or bird_collision_rect.colliderect(lower_pipe_rect):
            return True, score  # Borulara çarpma oldu, oyunu duraklat ve skoru döndür
        
        pipeMidPos = u_pipe['x'] + pipe_img[0].get_width() / 2
        if pipeMidPos <= playerMidPos < pipeMidPos + 4: 
            score += 1
        
    # Arkaplanları kaydır
    for background_rect in background_rects:
        background_rect.x -= BACKGROUND_SPEED
        if background_rect.right <= 0:
            background_rect.x = SCREEN_WIDTH
            
    return False, score  # Çarpma yok, oyun devam ediyor ve skoru döndür


def game_loop(screen, background_img, bird_img, bird_rect, background_rects, pipe_img, upper_pipes, lower_pipes):
    paused = False
    framepersecond_clock = pygame.time.Clock()
    score = 0
    
    # Oyuncu hızları
    bird_velocity_y = 0  # Kuşun başlangıç düşme hızı
    bird_flap_velocity = -15  # Kuşun zıplama hızı
    pipe_speed = PIPE_START_SPEED
    
    while True:
        # Kullanıcı girişlerini işle
        action = handle_input()
        if action == "toggle_pause":
            paused = not paused
        elif action == "flap":
            bird_velocity_y = bird_flap_velocity  # Kuşu zıplat
        elif action == "restart":
            main()
            
        if not paused:
            # Oyun durumu güncelle ve skoru al
            collision, score = update_game_state(screen, bird_rect, bird_img, upper_pipes, lower_pipes, pipe_img, background_rects, score)
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
                u_pipe['x'] -= pipe_speed
                l_pipe['x'] -= pipe_speed

            # Yeni boru ekleme kontrolü
            check_add_pipe(upper_pipes, lower_pipes)
            
            # Boru silme kontrolü
            check_remove_pipe(upper_pipes, lower_pipes, pipe_img)

            # Ekranı temizle
            screen.fill((0, 0, 0))

            # Arkaplanları ekrana çiz
            draw_background(screen, background_img, background_rects)

            # Boruları ekrana çiz
            draw_pipes(screen, pipe_img, upper_pipes, lower_pipes)

            # Kuşu ekrana çiz
            draw_bird(screen, bird_img, bird_rect)

            # Skoru ekrana yazdır
            font = pygame.font.Font(None, 36)
            text = font.render("Score: " + str(score), True, (255, 255, 255))
            screen.blit(text, (10, 10))

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
