import pygame
import random
import numpy as np
import sys
import os

class FlappyBirdEnv:
    def __init__(self, decision_frequency=4, speed_multiplier=1000, render=False):
        """
        Initialize the environment.
        Args:
            decision_frequency: Number of decisions per second (default: 4)
            speed_multiplier: Speed up the game by this factor (default: 50)
            render: Whether to render the game (default: True)
        """
        self.render_enabled = render
        
        # Only initialize pygame if we're going to render
        if self.render_enabled:
            pygame.mixer.pre_init(frequency=44100, size=16, channels=1, buffer=256)
            pygame.init()
            self.screen = pygame.display.set_mode((288, 512))
            self.clock = pygame.time.Clock()
            self.game_font = pygame.font.Font('04B_19.ttf', 20)
            
            # Load assets
            self.bg_surface = pygame.image.load('assets/background-day.png').convert()
            self.floor_surface = pygame.image.load('assets/base.png').convert()
            self.pipe_surface = pygame.image.load('assets/pipe-green.png').convert()
            self.game_over_surface = pygame.image.load('assets/message.png').convert_alpha()
            self.game_over_rect = self.game_over_surface.get_rect(center=(144, 256))
            
            # Bird setup
            bird_downflap = pygame.image.load('assets/bluebird-downflap.png').convert_alpha()
            bird_midflap = pygame.image.load('assets/bluebird-midflap.png').convert_alpha()
            bird_upflap = pygame.image.load('assets/bluebird-upflap.png').convert_alpha()
            self.bird_frames = [bird_downflap, bird_midflap, bird_upflap]
            self.bird_index = 0
            self.bird_surface = self.bird_frames[self.bird_index]
        
        # Control parameters
        self.decision_interval = 1000 / decision_frequency
        self.last_decision_time = 0
        self.speed_multiplier = speed_multiplier
        self.base_fps = 120

        # Game variables
        self.gravity = 0.125
        self.bird_movement = 0
        self.game_active = True
        self.score = 0
        self.high_score = 0
        self.passed_pipes = set()
        self.floor_x_pos = 0
        
        # Initialize rectangles without rendering
        if self.render_enabled:
            self.bird_rect = self.bird_surface.get_rect(center=(50, 206))
        else:
            # Create a rect without pygame when not rendering
            class DummyRect:
                def __init__(self):
                    self.centerx = 50
                    self.centery = 206
                    self.center = (50, 206)
                    self.top = 206
                    self.bottom = 206
                def colliderect(self, other):
                    return (abs(self.centerx - other.centerx) < 30 and 
                           abs(self.centery - other.centery) < 30)
            self.bird_rect = DummyRect()

        # Pipe setup
        self.pipe_list = []
        self.pipe_height = [200, 300, 400]

        # Event timers
        self.BIRDFLAP = pygame.USEREVENT + 1
        if self.render_enabled:
            pygame.time.set_timer(self.BIRDFLAP, 200)
        
        # Pipe spawning variables
        self.pipe_spawn_time = 0
        self.PIPE_SPAWN_INTERVAL = 1200

    def create_pipe(self):
        """Create a new pipe pair."""
        random_pipe_pos = random.choice(self.pipe_height)
        if self.render_enabled:
            bottom_pipe = self.pipe_surface.get_rect(midtop=(350, random_pipe_pos))
            top_pipe = self.pipe_surface.get_rect(midbottom=(350, random_pipe_pos - 150))
        else:
            class DummyRect:
                def __init__(self, x, y, is_bottom):
                    self.centerx = x
                    self.centery = y
                    self.center = (x, y)
                    if is_bottom:
                        self.top = y - 160
                        self.bottom = y + 160
                    else:
                        self.top = y - 160
                        self.bottom = y
                    self.midtop = (x, self.top)
                    self.midbottom = (x, self.bottom)
                def colliderect(self, other):
                    return (abs(self.centerx - other.centerx) < 30 and 
                           abs(self.centery - other.centery) < 30)
            
            bottom_pipe = DummyRect(350, random_pipe_pos, True)
            top_pipe = DummyRect(350, random_pipe_pos - 150, False)
        return bottom_pipe, top_pipe

    def move_pipes(self, pipes):
        """Move pipes to the left and remove ones that are off screen."""
        for pipe in pipes:
            pipe.centerx -= 2.5
        # Remove pipes that are off screen
        pipes = [pipe for pipe in pipes if pipe.centerx > -50]
        return pipes

    def check_collision(self, pipes):
        """Check if bird has collided with pipes or boundaries."""
        for pipe in pipes:
            if self.bird_rect.colliderect(pipe):
                return False

        if self.bird_rect.top <= -50 or self.bird_rect.bottom >= 450:
            return False
        return True

    def rotate_bird(self, bird):
        """Rotate bird according to its movement."""
        if not self.render_enabled:
            return bird
        return pygame.transform.rotozoom(bird, -self.bird_movement * 3.5, 1)

    def bird_animation(self):
        """Update bird animation frame."""
        if not self.render_enabled:
            return self.bird_surface, self.bird_rect
        new_bird = self.bird_frames[self.bird_index]
        new_bird_rect = new_bird.get_rect(center=(50, self.bird_rect.centery))
        return new_bird, new_bird_rect

    def draw_pipes(self, pipes):
        """Draw pipes on screen."""
        if not self.render_enabled:
            return
        for pipe in pipes:
            if pipe.bottom >= 512:
                self.screen.blit(self.pipe_surface, pipe)
            else:
                flip_pipe = pygame.transform.flip(self.pipe_surface, False, True)
                self.screen.blit(flip_pipe, pipe)

    def draw_floor(self):
        """Draw the moving floor."""
        if not self.render_enabled:
            return
        self.screen.blit(self.floor_surface, (self.floor_x_pos, 450))
        self.screen.blit(self.floor_surface, (self.floor_x_pos + 288, 450))

    def score_display(self, game_state):
        """Display score on screen."""
        if not self.render_enabled:
            return
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
        """Update score when passing through pipes."""
        for pipe in pipe_list:
            if pipe.bottom >= 512:
                pipe_id = id(pipe)
                if pipe.centerx < 50 and pipe_id not in passed_pipes:
                    score += 1
                    passed_pipes.add(pipe_id)
        return score

    def render(self):
        """Render the current game state."""
        if not self.render_enabled:
            return
            
        self.screen.blit(self.bg_surface, (0, 0))

        if self.game_active:
            # Bird
            rotated_bird = self.rotate_bird(self.bird_surface)
            self.screen.blit(rotated_bird, self.bird_rect)

            # Pipes
            self.draw_pipes(self.pipe_list)
            self.score_display('main_game')
        else:
            self.screen.blit(self.game_over_surface, self.game_over_rect)
            self.score_display('game_over')

        # Floor
        self.draw_floor()
        pygame.display.update()
        self.clock.tick(self.base_fps * self.speed_multiplier)

    def step(self, action):
        """Execute one step in the environment."""
        current_time = pygame.time.get_ticks() if self.render_enabled else self.last_decision_time + self.decision_interval
        time_since_last_decision = current_time - self.last_decision_time
        
        if time_since_last_decision < self.decision_interval:
            action = 0

        reward = 0
        done = False

        # Create initial pipes if none exist
        if len(self.pipe_list) == 0:
            self.pipe_list.extend(self.create_pipe())

        # Handle events only if rendering
        if self.render_enabled:
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
        if current_time - self.pipe_spawn_time > self.PIPE_SPAWN_INTERVAL:
            self.pipe_list.extend(self.create_pipe())
            self.pipe_spawn_time = current_time

        # Apply action
        if action == 1:
            self.bird_movement = 0
            self.bird_movement -= 5

        # Update game state
        if self.game_active:
            # Bird
            self.bird_movement += self.gravity
            self.bird_rect.centery += self.bird_movement
            self.game_active = self.check_collision(self.pipe_list)

            # Pipes
            self.pipe_list = self.move_pipes(self.pipe_list)

            # Score and reward
            prev_score = self.score
            self.score = self.update_score(self.pipe_list, self.score, self.passed_pipes)
            if self.score > prev_score:
                reward = 1.0

        else:
            self.high_score = max(self.score, self.high_score)
            done = True
            reward = -1.0

        # Floor
        self.floor_x_pos -= 1
        if self.floor_x_pos <= -288:
            self.floor_x_pos = 0

        # Only render if enabled
        if self.render_enabled:
            self.render()
        
        # Update last decision time if action was applied
        if time_since_last_decision >= self.decision_interval:
            self.last_decision_time = current_time

        return self.get_state(), reward, done, {'score': self.score}

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
                if pipe.centerx > self.bird_rect.centerx and pipe.bottom >= 512:
                    bottom_pipes.append(pipe)
            
            # Sort pipes by x position
            bottom_pipes.sort(key=lambda x: x.centerx)
            
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
        self.game_active = True
        self.pipe_list.clear()
        self.bird_rect.center = (50, 206)
        self.bird_movement = 0
        self.score = 0
        self.passed_pipes.clear()
        
        self.pipe_spawn_time = pygame.time.get_ticks() if self.render_enabled else 0
        
        self.pipe_list.extend(self.create_pipe())
        
        if self.render_enabled:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
        
        self.bird_movement = -5
        
        return self.get_state()

    def close(self):
        """Clean up resources."""
        if self.render_enabled:
            try:
                pygame.display.quit()
                pygame.quit()
            except:
                pass
