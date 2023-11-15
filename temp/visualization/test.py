import pygame
import sys

# Node class
class Node():
    def __init__(self, image, name, pos):
        self.image = pygame.image.load(image)
        self.name = name
        self.pos = pos

# Edge class
class Edge():
    def __init__(self, start_node, end_node):
        self.start_node = start_node
        self.end_node = end_node

# Initialize Pygame
pygame.init()

# Set the width, height and background color of the window
screen_width = 800
screen_height = 600
white = (255, 255, 255)

# Create a list to store nodes and edges
nodes = []
edges = []

# Adding nodes
for i in range(5):
    nodes.append(Node('e:/circle.png', str(i), (i*100+50, i*100+50)))

# Adding edges
for i in range(4):
    edges.append(Edge(nodes[i], nodes[i+1]))

# Set up the drawing window
screen = pygame.display.set_mode([screen_width, screen_height])

# Run until the user asks to quit
running = True
while running:
    # Did the user click the window close button?
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fill the background with white
    screen.fill(white)

    # Draw nodes and edges
    for node in nodes:
        screen.blit(node.image, node.pos)
    for edge in edges:
        pygame.draw.line(screen, (0,0,0), edge.start_node.pos, edge.end_node.pos, 2)

    # Flip the display
    pygame.display.flip()

# Done! Time to quit.
pygame.quit()
sys.exit()