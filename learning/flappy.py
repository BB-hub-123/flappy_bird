
def draw_pipes(pipes):
    for pipe in pipes:
        if pipe.bottom >= 724:
            screen.blit(pipe_surface, pipe) 
        else:
            flip_pipe = pygame.transform.flip(pipe_surface, False, True)
            screen.blit(flip_pipe, pipe) 

def check_collision(pipes):
    for pipe in pipes:
        if bird_rect.colliderect(pipe):
            death_sound.play()
            return False
        
    if bird_rect.top <= - 100 or bird_rect.bottom >= 620:
         return False 
    
    return True

def rotate_bird(bird): 
    new_bird = pygame.transform.rotozoom(bird, - bird_movement * 3, 1)
    return new_bird 

def bird_animation():
    new_bird = bird_frames[bird_index]
    new_bird_rect = new_bird.get_rect(center = (100, bird_rect.centery))
    return new_bird, new_bird_rect

def score_display(game_state):
    if game_state == 'main_game':
        score_surface = game_font.render(f'Score: {int(score)}' , True, (255, 255, 255))
        score_rect = score_surface.get_rect(center = (288, 100))
        screen.blit(score_surface, score_rect)
    if game_state == 'game_over':
        high_score_surface = game_font.render(f'High Score: {int(high_score)}', True, (255, 255, 255))
        high_score_rect = high_score_surface.get_rect(center = (288, 500))
        screen.blit(high_score_surface, high_score_rect)

def update_score(score, high_score):
    if score > high_score:
        high_score = score 
    return high_score 
# spil variabler