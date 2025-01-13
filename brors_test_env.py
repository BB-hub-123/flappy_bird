import pygame
import random
import numpy as np
import sys
import os

class FlappyBirdEnv:
    def __init__(self, decision_frequency=4, speed_multiplier=50):
        """
        Initialize the environment.
        Args:
            decision_frequency: Number of decisions per second (default: 4)
            speed_multiplier: Speed up the game by this factor (default: 1)
        """
        pygame.mixer.pre_init(frequency=44100, size=16, channels=1, buffer=256)
        pygame.init()
        self.screen = pygame.display.set_mode((288, 512))
        self.clock = pygame.time.Clock()
        self.game_font = pygame.font.Font('04B_19.ttf', 20)
        
        # Control parameters
        self.decision_interval = 1000 / decision_frequency  # milliseconds between decisions
        self.last_decision_time = 0
        self.speed_multiplier = speed_multiplier
        self.base_fps = 120  # Original FPS #Ændret fra 120

        # Game variables
        self.gravity = 0.125
        self.bird_movement = 0
        self.game_active = True
        self.score = 0
        self.high_score = 0
        self.passed_pipes = set()

        # Load assets
        self.bg_surface = pygame.image.load('assets/background-day.png').convert()
        self.floor_surface = pygame.image.load('assets/base.png').convert()
        self.floor_x_pos = 0

        # Bird setup
        bird_downflap = pygame.image.load('assets/bluebird-downflap.png').convert_alpha()
        bird_midflap = pygame.image.load('assets/bluebird-midflap.png').convert_alpha()
        bird_upflap = pygame.image.load('assets/bluebird-upflap.png').convert_alpha()
        self.bird_frames = [bird_downflap, bird_midflap, bird_upflap]
        self.bird_index = 0
        self.bird_surface = self.bird_frames[self.bird_index]
        self.bird_rect = self.bird_surface.get_rect(center=(50, 206))

        # Pipe setup
        self.pipe_surface = pygame.image.load('assets/pipe-green.png').convert()
        self.pipe_list = []
        self.pipe_height = [200, 300, 400]

        # Game over surface
        self.game_over_surface = pygame.image.load('assets/message.png').convert_alpha()
        self.game_over_rect = self.game_over_surface.get_rect(center=(144, 256))

        # Sounds
        # self.flap_sound = pygame.mixer.Sound('sound/sfx_wing.wav')
        # self.death_sound = pygame.mixer.Sound('sound/sfx_hit.wav')
        # self.score_sound = pygame.mixer.Sound('sound/sfx_point.wav')

        # Event timers
        self.BIRDFLAP = pygame.USEREVENT + 1
        pygame.time.set_timer(self.BIRDFLAP, 200)
        
        # Pipe spawning variables
        self.pipe_spawn_time = 0
        self.PIPE_SPAWN_INTERVAL = 1200

    def draw_floor(self):
        self.screen.blit(self.floor_surface, (self.floor_x_pos, 450))
        self.screen.blit(self.floor_surface, (self.floor_x_pos + 288, 450))

    def create_pipe(self):
        random_pipe_pos = random.choice(self.pipe_height)
        bottom_pipe = self.pipe_surface.get_rect(midtop=(350, random_pipe_pos))
        top_pipe = self.pipe_surface.get_rect(midbottom=(350, random_pipe_pos - 150))
        return bottom_pipe, top_pipe

    def move_pipes(self, pipes):
        for pipe in pipes:
            pipe.centerx -= 2.5
        return pipes

    def draw_pipes(self, pipes):
        for pipe in pipes:
            if pipe.bottom >= 512:
                self.screen.blit(self.pipe_surface, pipe)
            else:
                flip_pipe = pygame.transform.flip(self.pipe_surface, False, True)
                self.screen.blit(flip_pipe, pipe)

    def check_collision(self, pipes):
        for pipe in pipes:
            if self.bird_rect.colliderect(pipe):
                # self.death_sound.play()
                return False

        if self.bird_rect.top <= -50 or self.bird_rect.bottom >= 450:
            return False
        return True

    def rotate_bird(self, bird):
        new_bird = pygame.transform.rotozoom(bird, -self.bird_movement * 3.5, 1)
        return new_bird

    def bird_animation(self):
        new_bird = self.bird_frames[self.bird_index]
        new_bird_rect = new_bird.get_rect(center=(50, self.bird_rect.centery))
        return new_bird, new_bird_rect

    def score_display(self, game_state):
        if game_state == 'main_game':
            score_surface = self.game_font.render(f'Score: {int(self.score)}', True, (255, 255, 255))
            score_rect = score_surface.get_rect(center=(144, 50))
            self.screen.blit(score_surface, score_rect)
        if game_state == 'game_over':
            score_surface = self.game_font.render(f'Score: {int(self.score)}', True, (255, 255, 255))
            score_rect = score_surface.get_rect(center=(144, 50))
            self.screen.blit(score_surface, score_rect)

            high_score_surface = self.game_font.render(f'High Score: {int(self.high_score)}', True, (255, 255, 255))
            high_score_rect = high_score_surface.get_rect(center=(144, 425))
            self.screen.blit(high_score_surface, high_score_rect)

    def update_score(self, pipe_list, score, passed_pipes):
        for pipe in pipe_list:
            if pipe.bottom >= 512:  # Only consider bottom pipes
                pipe_id = id(pipe)
                if pipe.centerx < 50 and pipe_id not in passed_pipes:
                    score += 1
                    passed_pipes.add(pipe_id)
                    # self.score_sound.play()
        return score
    def get_state(self):
        """Get the current state for RL."""
        if not self.pipe_list:
            next_pipe_dist_x = 288
            next_pipe_top_y = 0
            next_pipe_bottom_y = 0
            next_next_pipe_top_y = 0
            next_next_pipe_bottom_y = 0
        else:
            next_pipe = None
            next_next_pipe = None
            bottom_pipes = []
            
            # Collect bottom pipes ahead of the bird
            for pipe in self.pipe_list:
                if pipe.centerx > self.bird_rect.centerx and pipe.bottom >= 512:  # Bottom pipe
                    bottom_pipes.append(pipe)
            
            # Sort pipes by x position
            bottom_pipes.sort(key=lambda x: x.centerx)
            
            # Get next and next-next pipes if they exist
            if len(bottom_pipes) > 0:
                next_pipe = bottom_pipes[0]
                next_pipe_dist_x = next_pipe.centerx - self.bird_rect.centerx
                next_pipe_top_y = next_pipe.top - 150
                next_pipe_bottom_y = next_pipe.bottom
                
                if len(bottom_pipes) > 1:
                    next_next_pipe = bottom_pipes[1]
                    next_next_pipe_top_y = next_next_pipe.top - 150
                    next_next_pipe_bottom_y = next_next_pipe.bottom
                else:
                    next_next_pipe_top_y = 0
                    next_next_pipe_bottom_y = 0
            else:
                next_pipe_dist_x = 288
                next_pipe_top_y = 0
                next_pipe_bottom_y = 0
                next_next_pipe_top_y = 0
                next_next_pipe_bottom_y = 0

        state = np.array([
            self.bird_rect.centery,
            self.bird_movement,
            next_pipe_dist_x,
            next_pipe_top_y,
            next_pipe_bottom_y,
            next_next_pipe_top_y,
            next_next_pipe_bottom_y
        ])
        return state
    def reset(self):
        """Reset the environment for a new episode."""
        self.game_active = True  # Make sure game is active
        self.pipe_list.clear()
        self.bird_rect.center = (50, 206)
        self.bird_movement = 0
        self.score = 0
        self.passed_pipes.clear()
        
        # Reset pipe spawning timer
        self.pipe_spawn_time = pygame.time.get_ticks()
        
        # Create initial pipe
        self.pipe_list.extend(self.create_pipe())
        
        # Handle any pending events to prevent freeze
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        # Force a step to start the game
        self.bird_movement = -5  # Initial jump to start the game
        
        # Get initial state
        initial_state = self.get_state()
        
        return initial_state
    def step(self, action):
        """Execute one step in the environment."""
        current_time = pygame.time.get_ticks()
        time_since_last_decision = current_time - self.last_decision_time
        
        # Only apply the action if enough time has passed
        if time_since_last_decision < self.decision_interval:
            action = 0  # No action if not enough time has passed
        reward = 0  # No reward for surviving
        done = False

        # Create initial pipes if none exist
        if len(self.pipe_list) == 0:
            self.pipe_list.extend(self.create_pipe())

        # Handle events for bird animation only
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == self.BIRDFLAP:
                if self.bird_index < 2:
                    self.bird_index += 1
                else:
                    self.bird_index = 0
                self.bird_surface, self.bird_rect = self.bird_animation()
        
        # Time-based pipe spawning
        current_time = pygame.time.get_ticks()
        if current_time - self.pipe_spawn_time > self.PIPE_SPAWN_INTERVAL:
            self.pipe_list.extend(self.create_pipe())
            self.pipe_spawn_time = current_time

        # Apply action
        if action == 1:  # Flap
            self.bird_movement = 0
            self.bird_movement -= 5
            # self.flap_sound.play()

        # Update game state
        self.screen.blit(self.bg_surface, (0, 0))

        if self.game_active:
            # Bird
            self.bird_movement += self.gravity
            rotated_bird = self.rotate_bird(self.bird_surface)
            self.bird_rect.centery += self.bird_movement
            self.screen.blit(rotated_bird, self.bird_rect)
            self.game_active = self.check_collision(self.pipe_list)

            # Pipes
            self.pipe_list = self.move_pipes(self.pipe_list)
            self.draw_pipes(self.pipe_list)

            # Score and reward
            prev_score = self.score
            self.score = self.update_score(self.pipe_list, self.score, self.passed_pipes)
            if self.score > prev_score:
                reward = 1.0  # Reward of 1 only when passing through pipes

            self.score_display('main_game')
        else:
            self.high_score = max(self.score, self.high_score)
            self.score_display('game_over')
            self.screen.blit(self.game_over_surface, self.game_over_rect)
            done = True
            reward = -1.0  # Penalty for dying

        # Floor
        self.floor_x_pos -= 1
        self.draw_floor()
        if self.floor_x_pos <= -288:
            self.floor_x_pos = 0

        pygame.display.update()
        self.clock.tick(self.base_fps * self.speed_multiplier)
        
        # Update last decision time if action was applied
        if time_since_last_decision >= self.decision_interval:
            self.last_decision_time = current_time

        return self.get_state(), reward, done, {'score': self.score}

    def close(self):
   
        try:
            pygame.display.quit()
            pygame.quit()
        except:
            pass

if __name__ == "__main__":
    env = FlappyBirdEnv()
    done = False
    state = env.reset()
    
    while True:
        action = 0  # Default to no flap
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and env.game_active:
                    action = 1
                elif event.key == pygame.K_SPACE and not env.game_active:
                    state = env.reset()

        # Step environment
        next_state, reward, done, info = env.step(action)
        state = next_state