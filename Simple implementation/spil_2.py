import pygame
import numpy as np

class FlappyBird:
    # Game settings
    actions = [0, 1]  # 0: do nothing, 1: flap
    fps = 300
    
    # Colors
    birdColor = (255, 255, 0)    # Yellow
    pipeColor = (34, 139, 34)    # Forest Green
    skyColor = (135, 206, 235)   # Sky Blue
    groundColor = (210, 180, 140) # Tan
    
    def __init__(self):
        pygame.init()
        self.reset()
        
    def render(self):
        if not hasattr(self, 'screen'):
            self.init_render()
            
        # Limit fps
        self.clock.tick(self.fps)
        
        # Clear screen with sky color
        self.screen.fill(self.skyColor)
        
        # Draw ground
        pygame.draw.rect(self.screen, self.groundColor, pygame.Rect(0, 450, 288, 62))
        
        # Draw bird
        pygame.draw.circle(self.screen, self.birdColor, 
                         (50, int(self.bird_y)), 15)
        
        # Draw pipes
        for pipe in self.pipes:
            # Bottom pipe
            pygame.draw.rect(self.screen, self.pipeColor,
                           pygame.Rect(pipe['x'], pipe['bottom_y'], 
                                     40, 512 - pipe['bottom_y']))
            # Top pipe
            pygame.draw.rect(self.screen, self.pipeColor,
                           pygame.Rect(pipe['x'], 0, 
                                     40, pipe['top_y']))
        
        # Draw score
        text = self.font.render(f'Score: {self.score}', True, (255, 255, 255))
        self.screen.blit(text, (10, 10))
        
        # Draw game over
        if not self.alive:
            text = self.bigfont.render('Game Over!', True, (255, 0, 0))
            text_rect = text.get_rect(center=(144, 256))
            self.screen.blit(text, text_rect)
        
        pygame.display.flip()
    
    def init_render(self):
        self.screen = pygame.display.set_mode((288, 512))
        pygame.display.set_caption('Flappy Bird')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.bigfont = pygame.font.Font(None, 72)
    
    def reset(self):
        self.bird_y = 256
        self.bird_velocity = 0
        self.score = 0
        self.alive = True
        
        # Initialize pipes
        self.pipes = []
        self.pipe_spawn_time = pygame.time.get_ticks()
        self.PIPE_SPAWN_INTERVAL = 1200
        self._add_pipe()
        
        return self.get_state()
    
    def _add_pipe(self):
        # Generer en tilfældig y-position for gap-starten
        gap_y = np.random.randint(150, 350)  # Gap-start position
        gap_size = 100  # Gap størrelse (afstand på y-aksen)
        
        # Øg afstanden mellem rørene på x-aksen
        pipe_spacing = 250  # Øg afstanden mellem rørene på x-aksen (startafstand)
        
        # Sørg for, at der er et korrekt gap mellem de to rør
        new_pipe = {
            'x': self.pipes[-1]['x'] + pipe_spacing if self.pipes else 288,  # Placer røret længere væk
            'top_y': gap_y - gap_size,  # Øvre rør skal være ovenfor gap'et
            'bottom_y': gap_y + gap_size,  # Nedre rør skal være under gap'et
            'scored': False  # Flag for om røret er blevet scoret
        }

        # Tilføj det nye rør til listen
        self.pipes.append(new_pipe)
    
    def get_state(self):
        if not self.pipes:
            next_pipe = {'x': 288, 'top_y': 0, 'bottom_y': 512}
        else:
            next_pipe = next((p for p in self.pipes if p['x'] > 50), self.pipes[0])
            
        return np.array([
            self.bird_y,
            self.bird_velocity,
            next_pipe['x'] - 50,  # distance to next pipe
            next_pipe['top_y'],   # top pipe y
            next_pipe['bottom_y'] # bottom pipe y
        ])
    
    def step(self, action):
        reward = 0.1  # small reward for staying alive
        
        # Bird physics
        self.bird_velocity += 0.5  # gravity
        if action == 1:
            self.bird_velocity = -8  # flap strength
        self.bird_y += self.bird_velocity
        
        # Update pipes
        for pipe in self.pipes:
            pipe['x'] -= 3  # pipe speed
            
            # Check if bird passed pipe
            if pipe['x'] < 50 and not pipe['scored']:
                pipe['scored'] = True
                self.score += 1
                reward = 1.0
        
        # Remove off-screen pipes (those that are completely off the left side of the screen)
        self.pipes = [p for p in self.pipes if p['x'] > -40]
        
        # Time-based pipe spawning
        current_time = pygame.time.get_ticks()
        if current_time - self.pipe_spawn_time > self.PIPE_SPAWN_INTERVAL:
            self._add_pipe()
            self.pipe_spawn_time = current_time
        
        # Check collision with pipes or boundaries
        for pipe in self.pipes:
            if (pipe['x'] < 65 and pipe['x'] > 15 and
                (self.bird_y < pipe['top_y'] or 
                 self.bird_y > pipe['bottom_y'])):
                self.alive = False
        
        # Check collision with ground or ceiling
        if self.bird_y > 450 or self.bird_y < 0:
            self.alive = False
        
        if not self.alive:
            reward = -1.0
            
        return self.get_state(), reward, not self.alive
    
    def close(self):
        pygame.quit()