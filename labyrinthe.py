import random
import time
import os

# Définition du labyrinthe
maze = [
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 1, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 1, 1, 1],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 2]
]

# Paramètres d'apprentissage
discount = 0.90
learning_rate = 0.10
epsilon = 0.10
lambda_ = 0.90  # Trace decay rate

# Actions possibles
actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Droite, Gauche, Bas, Haut

# Initialisation de la table Q et des traces d'éligibilité
q_table = [[[0 for _ in range(len(actions))] for _ in range(len(maze[0]))] for _ in range(len(maze))]
e_table = [[[0 for _ in range(len(actions))] for _ in range(len(maze[0]))] for _ in range(len(maze))]

# Variables pour suivre le meilleur parcours
best_path = []
best_path_length = float('inf')

# Fonctions auxiliaires
def get_next_action(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(len(actions)))
    else:
        return q_table[state[0]][state[1]].index(max(q_table[state[0]][state[1]]))

def move(state, action):
    next_state = (state[0] + actions[action][0], state[1] + actions[action][1])
    if 0 <= next_state[0] < len(maze) and 0 <= next_state[1] < len(maze[0]) and maze[next_state[0]][next_state[1]] != 1:
        return next_state
    return state

def display_maze(maze, player_position, path=None):
    os.system('cls' if os.name == 'nt' else 'clear')
    for y in range(len(maze)):
        for x in range(len(maze[0])):
            if (y, x) == player_position:
                print('X', end=' ')
            elif path and (y, x) in path:
                print('o', end=' ')
            elif maze[y][x] == 0:
                print('.', end=' ')
            elif maze[y][x] == 1:
                print('#', end=' ')
            elif maze[y][x] == 2:
                print('G', end=' ')
        print()
    time.sleep(0.5)

# Entraînement de l'agent avec SARSA(λ)
for episode in range(5):
    state = (0, 0)
    action = get_next_action(state, epsilon)
    done = False
    path = [state]  # Pour suivre le parcours de l'agent

    while not done:
        display_maze(maze, state, path)  # Affiche le labyrinthe avec le pion

        next_state = move(state, action)
        reward = 100 if next_state == (7, 7) else -1
        next_action = get_next_action(next_state, epsilon)

        # Calcul de l'erreur TD
        td_error = reward + discount * q_table[next_state[0]][next_state[1]][next_action] - q_table[state[0]][state[1]][action]

        # Mise à jour des traces d'éligibilité
        e_table[state[0]][state[1]][action] += 1

        # Mise à jour de la table Q et des traces
        for i in range(len(maze)):
            for j in range(len(maze[0])):
                for k in range(len(actions)):
                    q_table[i][j][k] += learning_rate * td_error * e_table[i][j][k]
                    e_table[i][j][k] *= discount * lambda_

        state, action = next_state, next_action
        path.append(state)

        if next_state == (7, 7):
            done = True

            # Vérifie si le parcours actuel est le meilleur
            if len(path) < best_path_length:
                best_path = path.copy()
                best_path_length = len(path)

            # Affiche le labyrinthe final avec le pion
            display_maze(maze, state, path)
            print("Objectif atteint !")
            break  # Arrête le jeu lorsque l'objectif est atteint

   
print("Entraînement terminé.")
print("Meilleur parcours trouvé :")
display_maze(maze, best_path[-1], best_path)

