import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *  # Import GLU functions
from pyrr import Matrix44
from .custom_env import IndoorNavEnv

class HouseVisualizer:
    def __init__(self, env):
        self.env = env
        self.path = []  # Add trajectory storage
        
        # Initialize GLFW with compatibility profile
        if not glfw.init():
            raise Exception("GLFW initialization failed")
        
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
        self.window = glfw.create_window(1000, 800, "Indoor Navigation", None, None)
        glfw.make_context_current(self.window)
        self.agent_color = (0.0, 1.0, 0.0)  # Add this line (green color)
        self.path = []

        
        # OpenGL setup
        glEnable(GL_DEPTH_TEST)
        glClearColor(1.0, 1.0, 1.0, 1.0)

    def _draw_floor(self):
        """Draw room floors using immediate mode"""
        room_colormap = {
            'living_room': (0.2, 0.2, 0.8),
            'kitchen': (0.8, 0.8, 0.2),
            'bedroom': (0.2, 0.8, 0.2),
            'bathroom': (0.5, 0.0, 0.5),
            'hallway': (0.5, 0.5, 0.5)
        }
        
        for name, (x1, y1, x2, y2) in self.env.rooms.items():
            glColor3f(*room_colormap[name])
            glBegin(GL_QUADS)
            glVertex3f(x1, y1, 0)
            glVertex3f(x2, y1, 0)
            glVertex3f(x2, y2, 0)
            glVertex3f(x1, y2, 0)
            glEnd()

    def _draw_cube(self, x, y, z, size):
        """Manual cube drawing"""
        s = size / 2
        glBegin(GL_QUADS)
        # Front face
        glVertex3f(x-s, y-s, z+s)
        glVertex3f(x+s, y-s, z+s)
        glVertex3f(x+s, y+s, z+s)
        glVertex3f(x-s, y+s, z+s)
        # Back face
        glVertex3f(x-s, y-s, z-s)
        glVertex3f(x+s, y-s, z-s)
        glVertex3f(x+s, y+s, z-s)
        glVertex3f(x-s, y+s, z-s)
        # Top face
        glVertex3f(x-s, y+s, z-s)
        glVertex3f(x+s, y+s, z-s)
        glVertex3f(x+s, y+s, z+s)
        glVertex3f(x-s, y+s, z+s)
        # Bottom face
        glVertex3f(x-s, y-s, z-s)
        glVertex3f(x+s, y-s, z-s)
        glVertex3f(x+s, y-s, z+s)
        glVertex3f(x-s, y-s, z+s)
        # Left face
        glVertex3f(x-s, y-s, z-s)
        glVertex3f(x-s, y-s, z+s)
        glVertex3f(x-s, y+s, z+s)
        glVertex3f(x-s, y+s, z-s)
        # Right face
        glVertex3f(x+s, y-s, z-s)
        glVertex3f(x+s, y-s, z+s)
        glVertex3f(x+s, y+s, z+s)
        glVertex3f(x+s, y+s, z-s)
        glEnd()

    def _draw_obstacles(self):
        """Draw all obstacles"""
        obstacle_map = {
            'furniture': (0.5, 0.3, 0.1),
            'appliances': (0.93, 0.51, 0.93),
            'decorations': (0.9, 0.1, 0.1),
            'floor_items': (0.1, 0.1, 0.1)
        }
        
        for cat, positions in self.env.obstacles.items():
            glColor3f(*obstacle_map[cat])
            for x, y in positions:
                self._draw_cube(x+0.5, y+0.5, 0.5, 0.5)

    def _draw_doorways(self):
        """Draw doorways as small orange cubes"""
        glColor3f(1.0, 0.5, 0.0)
        for x, y in self.env.doorways:
            self._draw_cube(x+0.5, y+0.5, 0.5, 0.3)

    def _draw_grid(self):
        """Draw grid lines"""
        glColor3f(0.0, 0.0, 0.0)
        glBegin(GL_LINES)
        for i in range(21):
            # Vertical lines
            glVertex3f(i, 0, 0.01)
            glVertex3f(i, 20, 0.01)
            # Horizontal lines
            glVertex3f(0, i, 0.01)
            glVertex3f(20, i, 0.01)
        glEnd()

    def _setup_view(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, 1000/800, 0.1, 100)  # Wider field of view
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        # Better camera angle to see agent
        gluLookAt(10, 15, 25,  # Camera position 
                10, 10, 0,   # Look-at point
                0, 0, 1)     # Up vector

    def render(self, agent_pos, trajectory=None):
        """Update to handle optional trajectory"""
        if trajectory is not None:
            self.path = trajectory.copy()
            
        while not glfw.window_should_close(self.window):
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self._setup_view()
            self._draw_floor()
            self._draw_grid()
            self._draw_obstacles()
            self._draw_doorways()
            self._draw_agent(agent_pos)
            self._draw_path()  # Add path visualization
            glfw.swap_buffers(self.window)
            glfw.poll_events()
            break

    def _draw_path(self):
        """Visualize agent's trajectory"""
        if not self.path:
            return
            
        glColor3f(0.0, 0.0, 1.0)  # Blue color for path
        glBegin(GL_LINE_STRIP)
        for x, y in self.path:
            glVertex3f(x + 0.5, y + 0.5, 0.1)  # Slightly above floor
        glEnd()

    def _draw_agent(self, position):
        x, y = position
        glColor3f(*self.agent_color)
        glPushMatrix()
        
        # Convert grid coordinates to OpenGL world space
        glTranslatef(x + 0.5, y + 0.5, 0.6)  # Center in cell and lift slightly
        
        # Draw as a sphere instead of cone
        quad = gluNewQuadric()
        gluSphere(quad, 0.3, 16, 16)  # More visible than a cone
        
        glPopMatrix()

    # def render(self, agent_pos):
    #     while not glfw.window_should_close(self.window):
    #         glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
    #         self._setup_view()
    #         self._draw_floor()
    #         self._draw_grid()
    #         self._draw_obstacles()
    #         self._draw_doorways()
            
    #         glfw.swap_buffers(self.window)
    #         glfw.poll_events()

    def close(self):
        glfw.terminate()

if __name__ == "__main__":
    env = IndoorNavEnv()
    renderer = HouseVisualizer(env)
    
    # Test agent movement visualization
    try:
        env.reset()
        while not glfw.window_should_close(renderer.window):
            action = np.random.randint(0, 5)  # Random actions for testing
            obs, reward, done, _ = env.step(action)
            renderer.render(env.agent_pos)
            if done:
                env.reset()
    finally:
        renderer.close()
