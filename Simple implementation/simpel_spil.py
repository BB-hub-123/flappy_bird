import pygame
import numpy as np
import random

class FlappyBird:
    def __init__(self):
        pygame.init()
        self.reset()

        # Definer hvor ofte agenten træffer beslutninger (f.eks. 30 beslutninger per sekund)
        self.decision_frequency = 4  # Agenten træffer beslutninger 30 gange per sekund
        self.frame_count = 0  # Tæller frames

    def render(self):
        if not hasattr(self, 'screen'):
            self.init_render()

        self.clock.tick(100)  # fps sat til 100
        self.screen.fill((135, 206, 235))  # Sky Blue
        pygame.draw.rect(self.screen, (210, 180, 140), pygame.Rect(0, 450, 288, 62))  # Ground

        pygame.draw.circle(self.screen, (255, 255, 0), (50, int(self.bird_y)), 15)  # Bird

        for pipe in self.pipes:
            pygame.draw.rect(self.screen, (34, 139, 34), pygame.Rect(pipe['x'], pipe['bottom_y'], 40, 512 - pipe['bottom_y']))
            pygame.draw.rect(self.screen, (34, 139, 34), pygame.Rect(pipe['x'], 0, 40, pipe['top_y']))

        text = pygame.font.Font(None, 36).render(f'Score: {self.score}', True, (255, 255, 255))
        self.screen.blit(text, (10, 10))

        if not self.alive:
            game_over_text = pygame.font.Font(None, 72).render('Game Over!', True, (255, 0, 0))
            self.screen.blit(game_over_text, (144 - game_over_text.get_width() // 2, 256 - game_over_text.get_height() // 2))

        pygame.display.flip()

    def init_render(self):
        self.screen = pygame.display.set_mode((288, 512))
        pygame.display.set_caption('Flappy Bird')
        self.clock = pygame.time.Clock()

    def reset(self):
        self.bird_y = 256
        self.bird_velocity = 0
        self.score = 0
        self.alive = True
        self.pipes = []
        self.pipe_spawn_time = pygame.time.get_ticks()
        self.PIPE_SPAWN_INTERVAL = 1200
        self._add_pipe()
        return self.get_state()

    def _add_pipe(self):
        gap_y = np.random.randint(150, 350)
        gap_size = 100
        pipe_spacing = 250

        new_pipe = {
            'x': self.pipes[-1]['x'] + pipe_spacing if self.pipes else 288,
            'top_y': gap_y - gap_size,
            'bottom_y': gap_y + gap_size,
            'scored': False
        }
        self.pipes.append(new_pipe)

    def get_state(self):
        # Returnerer spillets aktuelle tilstand som en liste med relevante data.
        if not self.pipes:
            # Hvis der ikke er nogen rør, returneres en standardtilstand.
            next_pipe = {'x': 288, 'top_y': 0, 'bottom_y': 512}
        else:
            # Finder det næste rør, som fuglen skal passere.
            next_pipe = next((p for p in self.pipes if p['x'] > 50), self.pipes[0])

        return np.array([
            self.bird_y,
            self.bird_velocity,
            next_pipe['x'] - 50,
            next_pipe['top_y'],
            next_pipe['bottom_y']
        ])

    def step(self, action):
        reward = 0.1
        self.bird_velocity += 0.5  # gravity
        if action == 1:
            self.bird_velocity = -8  # flap strength
        self.bird_y += self.bird_velocity

        for pipe in self.pipes:
            pipe['x'] -= 3  # pipe speed
            if pipe['x'] < 50 and not pipe['scored']:
                pipe['scored'] = True
                self.score += 1  # Update score only when passing pipe
                

        self.pipes = [p for p in self.pipes if p['x'] > -40]

        if pygame.time.get_ticks() - self.pipe_spawn_time > self.PIPE_SPAWN_INTERVAL:
            self._add_pipe()
            self.pipe_spawn_time = pygame.time.get_ticks()

        for pipe in self.pipes:
            if (pipe['x'] < 65 and pipe['x'] > 15 and
                (self.bird_y < pipe['top_y'] or self.bird_y > pipe['bottom_y'])):
                self.alive = False

        if self.bird_y > 450 or self.bird_y < 0:
            self.alive = False

        if not self.alive:
            reward = -100.0

        self.frame_count += 1  # Øg frame tælleren
        if self.frame_count % (100 // self.decision_frequency) == 0:  # Beslutninger hver 30. frame
            action = 1 if random.random() < 0.5 else 0  # Random beslutning for demonstration
            self.step(action)

        return self.get_state(), reward, not self.alive

    def close(self):
        pygame.quit()

