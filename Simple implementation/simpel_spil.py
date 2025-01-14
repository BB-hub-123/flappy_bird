<<<<<<< HEAD
import pygame   #til spiludvikling
import numpy as np      #til numeriske beregninger

class FlappyBird:
    #definere classen, der gør spillet dynamisk og gør så det kan vises

    # Game settings
    actions = [0, 1]  # 0: do nothing, 1: flap
    fps = 30    #frames pr. second (styrer spilhastighed)
    
    # Colors
    birdColor = (255, 255, 0)    # Yellow
    pipeColor = (34, 139, 34)    # Forest Green
    skyColor = (135, 206, 235)   # Sky Blue
    groundColor = (210, 180, 140) # Tan
    
    def __init__(self):
        # initialiserer spillet og nulstiller til startudgangspunktet
        pygame.init()   #starter pygame
        self.reset()    #kalder reset, så spillet kan startes igen fra samme startudgangspunktet
        
    def render(self): #(render = gengive)
        # tegner spillet på skærmen

        #tjekker om der er en attribut kaldet "screen"
        if not hasattr(self, 'screen'):     #hasattr = has attribute 
            self.init_render()      #hvis ikke attributten findes, intitialiseres den her2
            
        # Limit fps
        self.clock.tick(self.fps)
        
        # Clear screen with sky color
        self.screen.fill(self.skyColor)
        
        # Draw ground 
        pygame.draw.rect(self.screen, self.groundColor, pygame.Rect(0, 450, 288, 62))
        
        # Draw bird (tegnes hvor den befinder sig)
        pygame.draw.circle(self.screen, self.birdColor, 
                         (50, int(self.bird_y)), 15)
                    #(placering på x-aksen, hvorend fuglen befinder sig i højden, cirklen/fulgens radius)
        
        # Draw pipes
        for pipe in self.pipes:
            # Bottom pipe
            pygame.draw.rect(self.screen, self.pipeColor,
                           pygame.Rect(pipe['x'], pipe['bottom_y'], 
                                     40, 512 - pipe['bottom_y']))
                    #(x, y, bredden af pipe, højden af nederste pipe)

            # Top pipe
            pygame.draw.rect(self.screen, self.pipeColor,
                           pygame.Rect(pipe['x'], 0, 
                                     40, pipe['top_y']))
                    #(x,y, breden af pipe, højden er øverste pipe)
        
        # Draw score
        text = self.font.render(f'Score: {self.score}', True, (255, 255, 255))
=======
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
>>>>>>> 6b7c2abf4ef633ad9bf03d8de11dd91a5f7bd353
        self.screen.blit(text, (10, 10))

        if not self.alive:
<<<<<<< HEAD
            text = self.bigfont.render('Game Over!', True, (255, 0, 0))
            text_rect = text.get_rect(center=(144, 256))
            self.screen.blit(text, text_rect)
        
        pygame.display.flip()   #opdaterer skærmen med de ting man lige forinden har kodet og viser dem
        #fra chatten:
        #Når du tegner elementer som figurer, billeder eller tekst i Pygame, bliver de tegnet på en buffer (en midlertidig lagringsplads i hukommelsen).
        #Bufferen opdaterer ikke skærmen direkte, hvilket betyder, at ændringer ikke umiddelbart er synlige for spilleren.

        #pygame.display.flip()
            #Denne funktion tager indholdet af bufferen og "vender" det til skærmen, så brugeren kan se de nyeste ændringer.
            #Dette skaber en dobbelt-buffering-effekt, hvor skærmbilledet kun opdateres, når det er færdiggjort. Det sikrer en glidende og fejlfri visning uden flimmer.
            
    def init_render(self):
      # Initialiserer komponenter til at tegne spillet på skærmen.
        self.screen = pygame.display.set_mode((288, 512))  # Sætter skærmstørrelsen.
        pygame.display.set_caption('Flappy Bird')  # Indstiller vinduets titel.
        self.clock = pygame.time.Clock()  # Initialiserer en clock til styring af spillets hastighed.
        self.font = pygame.font.Font(None, 36)  # Lader tilføje tekst til spillet i normal størrelse.
        self.bigfont = pygame.font.Font(None, 72)  # Lader tilføje større tekst (bruges til "Game Over!").
            
    def reset(self):
        # Nulstiller spillet til starttilstanden.
        self.bird_y = 256  # Fuglen starter midt på skærmen (vertikalt).
        self.bird_velocity = 0  # Fuglen har ingen startbevægelse.
        self.score = 0  # Scoren sættes til 0.
        self.alive = True  # Spillet sættes til aktiv tilstand.

        self.pipes = []  # Tømmer listen over rør.
        self._add_pipe()  # Tilføjer det første rør.

        return self.get_state()  # Returnerer spillets starttilstand.
    
    def _add_pipe(self):
        # Tilføjer et nyt sæt rør med en tilfældig placering af hullet.
        gap_y = np.random.randint(150, 400)  # Genererer en tilfældig placering for hullet.
        gap_size = 150  # Bredden af hullet mellem de to rør.
        new_pipe = {
            'x': 288,  # Starter røret uden for skærmen til højre.
            'top_y': gap_y - gap_size,  # Toprørets nederste kant.
            'bottom_y': gap_y + gap_size,  # Bundrørets øverste kant.
            'scored': False  # Markerer, at røret endnu ikke er passeret af fuglen.
        }
        self.pipes.append(new_pipe)  # Tilføjer røret til listen over rør.

    
=======
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

>>>>>>> 6b7c2abf4ef633ad9bf03d8de11dd91a5f7bd353
    def get_state(self):
        # Returnerer spillets aktuelle tilstand som en liste med relevante data.
        if not self.pipes:
            # Hvis der ikke er nogen rør, returneres en standardtilstand.
            next_pipe = {'x': 288, 'top_y': 0, 'bottom_y': 512}
        else:
            # Finder det næste rør, som fuglen skal passere.
            next_pipe = next((p for p in self.pipes if p['x'] > 50), self.pipes[0])

        return np.array([
<<<<<<< HEAD
            self.bird_y,       #vertikal posistion
            self.bird_velocity,     # fuglens aktuelle hastighed
            next_pipe['x'] - 50,  # distance to next pipe
            next_pipe['top_y'],   # top pipe y
            next_pipe['bottom_y'] # bottom pipe y
=======
            self.bird_y,
            self.bird_velocity,
            next_pipe['x'] - 50,
            next_pipe['top_y'],
            next_pipe['bottom_y']
>>>>>>> 6b7c2abf4ef633ad9bf03d8de11dd91a5f7bd353
        ])

    def step(self, action):
<<<<<<< HEAD
        # Udfører et skridt i spillet baseret på den givne handling.
        reward = 0.1  # small reward for staying alive
        
        # Bird physics
=======
        reward = 0.1
>>>>>>> 6b7c2abf4ef633ad9bf03d8de11dd91a5f7bd353
        self.bird_velocity += 0.5  # gravity
        if action == 1:
            self.bird_velocity = -8  # flap strength
        self.bird_y += self.bird_velocity

        for pipe in self.pipes:
<<<<<<< HEAD
            pipe['x'] -= 3  # pipe speed (hvor hurtigt de flyyter sig til venstre)
            
            # Check if bird passed pipe
            if pipe['x'] < 50 and not pipe['scored']: 
                pipe['scored'] = True   # Marker røret som passeret
                self.score += 1
=======
            pipe['x'] -= 3  # pipe speed
            if pipe['x'] < 50 and not pipe['scored']:
                pipe['scored'] = True
                self.score += 1  # Update score only when passing pipe
>>>>>>> 6b7c2abf4ef633ad9bf03d8de11dd91a5f7bd353
                reward = 1.0

        self.pipes = [p for p in self.pipes if p['x'] > -40]
<<<<<<< HEAD
        if len(self.pipes) < 3:
            self._add_pipe()    # Tilføjer et nyt rør, hvis der er færre end 3.
        
        # Check collision with pipes or boundaries
        for pipe in self.pipes:
            if (pipe['x'] < 65 and pipe['x'] > 15 and
                (self.bird_y < pipe['top_y'] or 
                 self.bird_y > pipe['bottom_y'])):
                self.alive = False    # Spillet slutter, hvis fuglen rammer røret.
        
        # Check collision with ground or ceiling
        if self.bird_y > 450 or self.bird_y < 0:
            self.alive = False    # Spillet slutter, hvis fuglen rammer jorden eller toppen
        
        if not self.alive:
            reward = -1.0    #straf for død
            
        return self.get_state(), reward, not self.alive    # Returnerer ny tilstand, belønning og spilstatus.
    
    def close(self):
        pygame.quit()   #lukker spillet
=======

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
            reward = -1.0

        self.frame_count += 1  # Øg frame tælleren
        if self.frame_count % (100 // self.decision_frequency) == 0:  # Beslutninger hver 30. frame
            action = 1 if random.random() < 0.5 else 0  # Random beslutning for demonstration
            self.step(action)

        return self.get_state(), reward, not self.alive

    def close(self):
        pygame.quit()

>>>>>>> 6b7c2abf4ef633ad9bf03d8de11dd91a5f7bd353
