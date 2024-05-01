import pygame
import sys
import os

# Ekran genişliği ve yüksekliği
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 500

# Arkaplanın hızı
BACKGROUND_SPEED = 1

# Yer çekimi ivmesi
GRAVITY = 2
MAX_FALL_SPEED = 20

# Oyun kare hızı
framepersecond = 32

def draw_background(screen, background_img, background_rects):
    for background_rect in background_rects:
        screen.blit(background_img, background_rect)

def draw_bird(screen, bird_img, bird_rect):
    screen.blit(bird_img, bird_rect)

def game_loop(screen, background_img, bird_img, bird_rect, background_rects):
    paused = False
    framepersecond_clock = pygame.time.Clock()
    
    # Oyuncu hızları
    bird_velocity_y = 0  # Kuşun başlangıç düşme hızı
    bird_flap_velocity = -15  # Kuşun zıplama hızı
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    paused = not paused  # Oyun durumu değiştir
                elif event.key == pygame.K_SPACE or event.key == pygame.K_UP:
                    bird_velocity_y = bird_flap_velocity  # Kuşu zıplat
                elif event.key == pygame.K_q:  # "q" tuşuna basıldığında oyunu kapat
                    pygame.quit()
                    sys.exit()

        if not paused:

            # Yer çekimi uygula
            bird_velocity_y += GRAVITY
            bird_velocity_y = min(bird_velocity_y, MAX_FALL_SPEED)  # Düşme hızını maksimum düşme hızıyla sınırla
            bird_rect.y += bird_velocity_y

            # Arkaplanları kaydır
            for background_rect in background_rects:
                background_rect.x -= BACKGROUND_SPEED
                if background_rect.right <= 0:
                    background_rect.x = SCREEN_WIDTH

        # Ekranı temizle
        screen.fill((0, 0, 0))

        # Arkaplanları ekrana çiz
        draw_background(screen, background_img, background_rects)

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

    # Kuşun başlangıç konumu
    bird_rect = bird_img.get_rect()
    bird_rect.center = (SCREEN_WIDTH // 4, SCREEN_HEIGHT // 2)  # Ekranın sol ortasında başlat

    # İki kopya arkaplan oluştur
    background_rects = []
    for i in range(2):
        background_rects.append(background_img.get_rect())

    # İkinci arkaplanı ilk arkaplanın sağ tarafına yerleştir
    background_rects[1].x = SCREEN_WIDTH

    # Oyun döngüsü başlat
    game_loop(screen, background_img, bird_img, bird_rect, background_rects)

if __name__ == "__main__":
    main()
