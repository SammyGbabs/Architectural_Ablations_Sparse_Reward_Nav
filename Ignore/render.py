import pygame
import numpy as np
import moderngl
from pygame.locals import *
from custom_env import IndoorNavEnv

class HouseVisualizer:
    def __init__(self):
        self.env = IndoorNavEnv()

        pygame.init()
        self.display = (1000, 800)
        self.screen = pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)
        self.ctx = moderngl.create_context()
        self.prog = self._create_program()
        self.vao = self._create_vao()

        self.env.reset()

    def _create_program(self):
        vertex_shader = """
        #version 330
        in vec3 in_position;
        in vec3 in_color;
        out vec3 frag_color;
        uniform mat4 modelview;
        uniform mat4 projection;
        void main() {
            frag_color = in_color;
            gl_Position = projection * modelview * vec4(in_position, 1.0);
        }
        """
        fragment_shader = """
        #version 330
        in vec3 frag_color;
        out vec4 color;
        void main() {
            color = vec4(frag_color, 1.0);
        }
        """
        return self.ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)

    def draw_environment(self):
        self.ctx.clear(1.0, 1.0, 1.0)  # White background
        self._draw_grid()
        self._draw_rooms()
        self._draw_obstacles()
        self._draw_doorways()
        pygame.display.flip()

    def _draw_rooms(self):
        room_colors = {
            "living_room": (0.2, 0.2, 0.8),  # Blue
            "kitchen": (0.8, 0.8, 0.2),  # Yellow
            "bedroom": (0.2, 0.8, 0.2),  # Green
            "bathroom": (0.5, 0.0, 0.5),  # Purple
            "hallway": (0.5, 0.5, 0.5),  # Gray
        }
        
        room_positions = [
            ((0, 0), "living_room"),
            ((10, 0), "kitchen"),
            ((0, 10), "bedroom"),
            ((10, 10), "bathroom"),
            ((5, 5), "hallway"),
        ]
        
        for (x, y), room_type in room_positions:
            self._draw_floor_quad(x, y, room_colors[room_type])
    
    def _draw_obstacles(self):
        obstacle_colors = {
            "furniture": (0.6, 0.3, 0.0),  # Brown
            "appliances": (0.8, 0.2, 0.5),  # Violet-Pink
            "decorations": (0.8, 0.0, 0.0),  # Red
            "floor_items": (0.0, 0.0, 0.0),  # Black
        }
        
        obstacle_positions = [
            ((3, 3), "furniture"),
            ((12, 2), "appliances"),
            ((8, 8), "decorations"),
            ((6, 6), "floor_items"),
        ]
        
        for (x, y), obs_type in obstacle_positions:
            self._draw_cube(x, y, obstacle_colors[obs_type])
    
    def _draw_doorways(self):
        doorway_color = (1.0, 0.5, 0.0)  # Orange
        door_positions = [(5, 0), (10, 5), (5, 10), (0, 5)]
        for x, y in door_positions:
            self._draw_cube(x, y, doorway_color)
    
    def _draw_grid(self):
        grid_color = (0.0, 0.0, 0.0)  # Black
        for x in range(0, 21, 2):
            self._draw_line((x, 0), (x, 20), grid_color)
        for y in range(0, 21, 2):
            self._draw_line((0, y), (20, y), grid_color)
    
    def _draw_floor_quad(self, x, y, color):
        vertices = np.array([
            [x, y, 0], [x + 5, y, 0], [x + 5, y + 5, 0], [x, y + 5, 0]
        ], dtype='f4')
        vbo = self.ctx.buffer(vertices)
        vao = self.ctx.simple_vertex_array(self.prog, vbo, 'in_position')
        self.prog['in_color'].write(color)
        vao.render(moderngl.TRIANGLE_FAN)
    
    def _draw_cube(self, x, y, color):
        vertices = np.array([
            [x, y, 0], [x + 1, y, 0], [x + 1, y + 1, 0], [x, y + 1, 0],
        ], dtype='f4')
        vbo = self.ctx.buffer(vertices)
        vao = self.ctx.simple_vertex_array(self.prog, vbo, 'in_position')
        self.prog['in_color'].write(color)
        vao.render(moderngl.TRIANGLE_FAN)
    
    def _draw_line(self, start, end, color):
        vertices = np.array([start + (0,), end + (0,)], dtype='f4')
        vbo = self.ctx.buffer(vertices)
        vao = self.ctx.simple_vertex_array(self.prog, vbo, 'in_position')
        self.prog['in_color'].write(color)
        vao.render(moderngl.LINES)

def main():
    visualizer = HouseVisualizer()
    clock = pygame.time.Clock()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        visualizer.draw_environment()
        clock.tick(30)
    
    pygame.quit()

if __name__ == "__main__":
    main()
