import pygame
import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time

class PenaltyKeeperEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Dimensiones del arco (2 filas x 3 columnas)
        self.goal_rows = 2  # Altura del arco
        self.goal_cols = 3  # Ancho del arco

        # Inicialización de Pygame
        pygame.init()
        self.cell_size = 100  # Tamaño de cada celda
        self.screen_width = self.goal_cols * self.cell_size
        self.screen_height = (self.goal_rows + 3) * self.cell_size  # Más espacio para el terreno
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Juego de Portero")

        # Colores
        self.field_color = (34, 139, 34)  # Verde oscuro
        self.white = (255, 255, 255)
        self.blue = (0, 0, 255)  # Azul para ganancia
        self.red = (255, 0, 0)   # Rojo para pérdida

        # Cargar y escalar sprites
        self.keeper_sprite = pygame.image.load("sprites/goalkeeper.png")
        self.keeper_sprite = pygame.transform.scale(self.keeper_sprite, (self.cell_size, int(self.cell_size * 1.2)))
        self.ball_sprite = pygame.image.load("sprites/ball.png")
        self.ball_sprite = pygame.transform.scale(self.ball_sprite, (self.cell_size // 2, self.cell_size // 2))

        # Espacios de acción y observación
        self.action_space = spaces.Discrete(self.goal_rows * self.goal_cols)  # 6 posiciones posibles (3x2)
        self.observation_space = spaces.Box(low=0, high=2, shape=(4,), dtype=np.int32)

        self.font = pygame.font.Font(None, 72)  # Fuente para resultados
        self.reset()

    def reset(self):
        # Posición inicial del portero: siempre empieza en el centro del arco
        self.keeper_position = [0, 1]  # Cambié esto para que empiece en la fila superior del arco

        # Posiciones válidas en el arco (3x2)
        self.goal_positions = [(i, j) for i in range(self.goal_rows) for j in range(self.goal_cols)]

        # Elegir una celda aleatoria para el objetivo del balón
        self.target_position = random.choice(self.goal_positions)

        # Posición inicial del balón (en la fila inferior fuera del arco)
        self.ball_position = [self.goal_rows + 2, self.target_position[1]]  # Alinear la columna con el objetivo

        self.done = False
        self.result_text = ""
        self.result_color = self.white
        self.ball_reached_goal = False  # Flag para saber si el balón llegó al arco
        self.portero_movio = False  # Flag para que el portero solo se mueva una vez
        return np.array(self.keeper_position + self.ball_position, dtype=np.int32)

    def step(self, action):
        if self.done:
            return self.reset()

        # Mover al portero según la acción solo una vez
        if not self.portero_movio and action < len(self.goal_positions):
            self.keeper_position = list(self.goal_positions[action])
            self.portero_movio = True

        # Mover el balón hacia la fila objetivo dentro del arco
        if self.ball_position[0] > self.target_position[0]:  # Hasta alcanzar la fila del objetivo
            self.ball_position[0] -= 1

        # Evaluar si el balón alcanzó el arco
        if self.ball_position[0] == self.target_position[0]:  # Cuando el balón llega a la fila del arco
            self.ball_reached_goal = True

        # Si el balón llega al arco, determinar si fue gol o atajado
        if self.ball_reached_goal:
            self.done = True
            # Verificar si la columna del portero coincide con la columna del balón cuando llega al arco
            if self.keeper_position[0] == self.target_position[0] and self.keeper_position[1] == self.ball_position[1]:
                self.result_text = "¡Atajado!"
                self.result_color = self.blue  # Azul
                reward = 1
            else:
                self.result_text = "¡Gol!"
                self.result_color = self.red  # Rojo
                reward = -2  # Se pierde $2 en caso de gol
        else:
            reward = 0  # Sin recompensa hasta que el balón llegue al arco

        return np.array(self.keeper_position + self.ball_position, dtype=np.int32), reward, self.done, {}

    def render(self, live=False):
        if not live:  # Solo renderiza si es un juego en vivo
            return
        # Dibujar fondo
        self.screen.fill(self.field_color)

        # Dibujar portería en el borde superior
        pygame.draw.line(self.screen, self.white,
                         (0, 0),
                         (self.goal_cols * self.cell_size, 0), 3)  # Línea superior
        for j in range(1, self.goal_cols):  # Líneas verticales
            pygame.draw.line(self.screen, self.white,
                             (j * self.cell_size, 0),
                             (j * self.cell_size, self.goal_rows * self.cell_size), 2)
        for i in range(1, self.goal_rows):  # Líneas horizontales (sin la inferior)
            pygame.draw.line(self.screen, self.white,
                             (0, i * self.cell_size),
                             (self.goal_cols * self.cell_size, i * self.cell_size), 2)

        # Dibujar portero
        x_keeper = self.keeper_position[1] * self.cell_size
        y_keeper = self.keeper_position[0] * self.cell_size
        self.screen.blit(self.keeper_sprite, (x_keeper, y_keeper))

        # Dibujar balón
        x_ball = self.ball_position[1] * self.cell_size + self.cell_size // 4
        y_ball = self.ball_position[0] * self.cell_size + self.cell_size // 4
        self.screen.blit(self.ball_sprite, (x_ball, y_ball))

        # Mostrar resultado si aplica
        if self.done:
            text = self.font.render(self.result_text, True, self.result_color)
            text_rect = text.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
            self.screen.blit(text, text_rect)

        pygame.display.flip()

# Q-learning
def train_agent(env, episodes=5000, alpha=0.05, gamma=0.95, epsilon=0.1, render=False):
    q_table = np.zeros((env.goal_rows, env.goal_cols, env.goal_rows + 3, env.goal_cols, env.action_space.n))  # Tabla Q
    rewards = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Epsilon-greedy policy (decrementando epsilon gradualmente)
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Exploración
            else:
                action = np.argmax(q_table[state[0], state[1], state[2], state[3]])  # Explotación

            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # Actualizar tabla Q
            q_table[state[0], state[1], state[2], state[3], action] = \
                (1 - alpha) * q_table[state[0], state[1], state[2], state[3], action] + \
                alpha * (reward + gamma * np.max(q_table[next_state[0], next_state[1], next_state[2], next_state[3]]))

            state = next_state

        rewards.append(total_reward)
        epsilon = max(0.01, epsilon * 0.995)  # Decrementar epsilon gradualmente

        if episode % 100 == 0:
            print(f"Episode {episode}: Total Reward: {total_reward}")
    
    return q_table, rewards

if __name__ == "__main__":
    env = PenaltyKeeperEnv()
    print("Entrenando al portero...")
    q_table, rewards = train_agent(env)
    print("Entrenamiento completado.")
    
    # Entrenamiento finalizado, ahora se pasa al juego en vivo
    print("¡Comienza el juego!")
    
    # Modo de juego en vivo, mientras el jugador quiera jugar
    live_game = True
    total_money = 0  # Inicializar el dinero

    while live_game:
        env.reset()
        done = False
        while not done:
            env.render(live=True)
            action = int(input("Elige una dirección para el balón (1-6): "))
            _, reward, done, _ = env.step(action)
            total_money += reward  # Actualizar el dinero basado en el resultado

            # Mostrar dinero
            print(f"Dinero acumulado: ", end="")
            if reward > 0:
                print(f"\033[34m${total_money}\033[0m")  # Azul para ganancia
            else:
                print(f"\033[31m${total_money}\033[0m")  # Rojo para pérdida

        # Preguntar si el jugador quiere continuar jugando
        continuar = input("¿Quieres seguir jugando? (s/n): ")
        if continuar.lower() != "s":
            live_game = False
            print("Juego terminado.")
    
    pygame.quit()