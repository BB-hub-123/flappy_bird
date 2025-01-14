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
        self.screen.blit(text, (10, 10))
        
        # Draw game over
        if not self.alive:
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

    
    def get_state(self):
        # Returnerer spillets aktuelle tilstand som en liste med relevante data.
        if not self.pipes:
            # Hvis der ikke er nogen rør, returneres en standardtilstand.
            next_pipe = {'x': 288, 'top_y': 0, 'bottom_y': 512}
        else:
            # Finder det næste rør, som fuglen skal passere.
            next_pipe = next((p for p in self.pipes if p['x'] > 50), self.pipes[0])
            
        return np.array([
            self.bird_y,       #vertikal posistion
            self.bird_velocity,     # fuglens aktuelle hastighed
            next_pipe['x'] - 50,  # distance to next pipe
            next_pipe['top_y'],   # top pipe y
            next_pipe['bottom_y'] # bottom pipe y
        ])
    
    def step(self, action):
        # Udfører et skridt i spillet baseret på den givne handling.
        reward = 0.1  # small reward for staying alive
        
        # Bird physics
        self.bird_velocity += 0.5  # gravity
        if action == 1:
            self.bird_velocity = -8  # flap strength
        self.bird_y += self.bird_velocity
        
        # Update pipes
        for pipe in self.pipes:
            pipe['x'] -= 3  # pipe speed (hvor hurtigt de flyyter sig til venstre)
            
            # Check if bird passed pipe
            if pipe['x'] < 50 and not pipe['scored']: 
                pipe['scored'] = True   # Marker røret som passeret
                self.score += 1
                reward = 1.0
        
        # Remove off-screen pipes and add new ones
        self.pipes = [p for p in self.pipes if p['x'] > -40]
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