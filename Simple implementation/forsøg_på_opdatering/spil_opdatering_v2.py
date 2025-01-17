import pygame
import numpy as np
import random

class FlappyBird:
    def __init__(self):
        # Initialize Pygame and game state
        pygame.init()
        
        # Game dimensions and physics constants
        self.SCREEN_WIDTH = 288
        self.SCREEN_HEIGHT = 512
        self.GROUND_HEIGHT = 450
        self.BIRD_X = 50
        self.BIRD_RADIUS = 15
        self.PIPE_WIDTH = 40
        self.GRAVITY = 0.5
        self.FLAP_STRENGTH = -7
        self.PIPE_SPEED = 3
        
        # AI training parameters
        self.decision_frequency = 4  # How often the agent makes decisions
        self.frame_count = 0
        
        # Initialize game state
        self.reset()

    def reset(self):
        # Reset bird position and physics
        self.bird_y = self.SCREEN_HEIGHT // 2
        self.bird_velocity = 0
        
        # Reset game state
        self.score = 0
        self.alive = True
        self.pipes = []
        
        # Initialize pipe spawning
        self.pipe_spawn_time = pygame.time.get_ticks()
        self.PIPE_SPAWN_INTERVAL = 1200  # Milliseconds between pipe spawns
        
        # Add first pipe
        self._add_pipe()
        
        return self.get_state()

    def _add_pipe(self):
        # Constants for pipe generation
        MIN_GAP_Y = 150
        MAX_GAP_Y = 350
        GAP_SIZE = 100
        PIPE_SPACING = 250
        
        # Generate random gap position
        gap_y = np.random.randint(MIN_GAP_Y, MAX_GAP_Y)
        
        # Create new pipe
        new_pipe = {
            'x': self.pipes[-1]['x'] + PIPE_SPACING if self.pipes else self.SCREEN_WIDTH,
            'top_y': gap_y - GAP_SIZE,
            'bottom_y': gap_y + GAP_SIZE,
            'scored': False
        }
        self.pipes.append(new_pipe)

    def get_state(self):
        # Ensure we always have pipes by adding one if empty
        if not self.pipes:
            self._add_pipe()
        
        # Find next pipe the bird needs to navigate
        next_pipe = self.pipes[0]  # Always use first pipe if no pipe ahead
        for pipe in self.pipes:
            if pipe['x'] > self.BIRD_X:
                next_pipe = pipe
                break
        
        return np.array([
            self.bird_y,
            self.bird_velocity,
            next_pipe['x'] - self.BIRD_X,
            next_pipe['top_y'],
            next_pipe['bottom_y']
        ])

    def step(self, action):
        # Initialize reward for this step
        reward = 0
        
        # Apply physics
        self.bird_velocity += self.GRAVITY
        if action == 1:  # Flap
            self.bird_velocity = self.FLAP_STRENGTH
        self.bird_y += self.bird_velocity
        
        # Ensure we have pipes
        if not self.pipes:
            self._add_pipe()
        
        # Find next pipe for reward calculation (same as in get_state)
        next_pipe = self.pipes[0]
        for pipe in self.pipes:
            if pipe['x'] > self.BIRD_X:
                next_pipe = pipe
                break
        
        # Calculate center position reward
        pipe_center_y = (next_pipe['top_y'] + next_pipe['bottom_y']) / 2
        vertical_distance = abs(self.bird_y - pipe_center_y)
        position_reward = 1.0 / (1.0 + vertical_distance/50)  # Reward being close to center
        reward += position_reward
        
        # Penalty for extreme velocities
        if abs(self.bird_velocity) > 10:
            reward -= 0.2 * (abs(self.bird_velocity) - 10)
        
        # Update pipes and check for scoring
        for pipe in self.pipes:
            pipe['x'] -= self.PIPE_SPEED
            
            if pipe['x'] < self.BIRD_X and not pipe['scored']:
                pipe['scored'] = True
                self.score += 1
                reward += 100  # Major reward for passing pipe
                
                if vertical_distance < 30:
                    reward += 50  # Bonus for precise control
        
        # Remove off-screen pipes and ensure we always have at least one pipe
        self.pipes = [p for p in self.pipes if p['x'] > -self.PIPE_WIDTH]
        if not self.pipes or (pygame.time.get_ticks() - self.pipe_spawn_time > self.PIPE_SPAWN_INTERVAL):
            self._add_pipe()
            self.pipe_spawn_time = pygame.time.get_ticks()
        
        # Check for collisions
        for pipe in self.pipes:
            if (pipe['x'] < self.BIRD_X + self.BIRD_RADIUS and 
                pipe['x'] > self.BIRD_X - self.BIRD_RADIUS - self.PIPE_WIDTH):
                if self.bird_y < pipe['top_y'] or self.bird_y > pipe['bottom_y']:
                    self.alive = False
                    collision_distance = min(
                        abs(self.bird_y - pipe['top_y']),
                        abs(self.bird_y - pipe['bottom_y'])
                    )
                    reward = -100 - collision_distance  # Bigger penalty for worse collisions
        
        # Check for boundary violations
        if self.bird_y > self.GROUND_HEIGHT or self.bird_y < 0:
            self.alive = False
            reward = -500  # Severe penalty for boundary violations
        
        return self.get_state(), reward, not self.alive

    def render(self):
        # Initialize screen if not already done
        if not hasattr(self, 'screen'):
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            pygame.display.set_caption('Flappy Bird')
            self.clock = pygame.time.Clock()

        # Control game speed
        self.clock.tick(100)  # 100 FPS for smooth animation
        
        # Draw background
        self.screen.fill((135, 206, 235))  # Sky blue background
        
        # Draw ground
        pygame.draw.rect(self.screen, 
                        (210, 180, 140),  # Sandy color
                        pygame.Rect(0, self.GROUND_HEIGHT, self.SCREEN_WIDTH, 62))
        
        # Draw bird
        pygame.draw.circle(self.screen,
                          (255, 255, 0),  # Yellow color
                          (self.BIRD_X, int(self.bird_y)),
                          self.BIRD_RADIUS)
        
        # Draw pipes
        for pipe in self.pipes:
            # Draw bottom pipe
            pygame.draw.rect(self.screen,
                            (34, 139, 34),  # Forest green
                            pygame.Rect(pipe['x'],
                                        pipe['bottom_y'],
                                        self.PIPE_WIDTH,
                                        self.SCREEN_HEIGHT - pipe['bottom_y']))
            # Draw top pipe
            pygame.draw.rect(self.screen,
                            (34, 139, 34),  # Forest green
                            pygame.Rect(pipe['x'],
                                        0,
                                        self.PIPE_WIDTH,
                                        pipe['top_y']))

        # Draw score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f'Score: {self.score}', True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        # Draw game over message if applicable
        if not self.alive:
            game_over_font = pygame.font.Font(None, 72)
            game_over_text = game_over_font.render('Game Over!', True, (255, 0, 0))
            text_x = self.SCREEN_WIDTH // 2 - game_over_text.get_width() // 2
            text_y = self.SCREEN_HEIGHT // 2 - game_over_text.get_height() // 2
            self.screen.blit(game_over_text, (text_x, text_y))

        pygame.display.flip()

    def close(self):
        pygame.quit()