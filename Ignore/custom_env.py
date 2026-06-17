import gym
import numpy as np
from gym import spaces

class IndoorNavEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(IndoorNavEnv, self).__init__()
        # Add doorway tracking
        self.visited_doorways = set()
        self.last_room = None
        # Target room sequence for curriculum learning
        self.current_room = None  # This was missing in your implementation
        self.target_sequence = ['kitchen', 'bathroom', 'bedroom', 'hallway']
        self.current_target_idx = 0
        self.total_episodes = 0
        self.doorways_passed = 0
        self.grid_size = 20
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(16,), dtype=np.float32
        )

        # Environment layout
        self.rooms = {
            'living_room': (0,0,8,8),
            'kitchen': (0,12,8,20),
            'bedroom': (12,12,20,20),
            'bathroom': (12,0,20,8),
            'hallway': (8,8,12,12)
        }

        self.obstacles = {
            'furniture': [(3,3), (5,15), (17,4), (9,9)],
            'appliances': [(2,18), (18,17)],
            'decorations': [(4,5), (15,15)],
            'floor_items': [(10,10), (14,14)]
        }

        self.doorways = [(8,1), (1,8), (8,16), (12,16), (11,1)]
        
        # Agent state
        self.agent_pos = None
        self.max_steps = 150
        self.current_step = 0
        self.target_room = None

    def reset(self):
        """Reset environment with next target in sequence"""
        self.current_room = 'living_room'  # Set initial room
        self.doorways_passed = 0
        self.agent_pos = (1, 1)  # Always start in living room
        self.current_step = 0
        self.visited_doorways = set()
        self.last_room = None
        
        # Curriculum progression
        if self.total_episodes < len(self.target_sequence):
            # Follow sequence for initial learning
            self.target_room = self.target_sequence[self.total_episodes]
        else:
            # Randomize after completing full sequence
            self.target_room = np.random.choice(self.target_sequence)
            
        self.total_episodes += 1
        return self._get_obs()

    def _get_obs(self):
        x, y = self.agent_pos

        # Proximity Sensors (5): Obstacle/door detection (front/back/left/right/current)
        proximity_sensors = [
            self._is_obstacle(x+1, y),  # Front
            self._is_obstacle(x-1, y),  # Back
            self._is_obstacle(x, y-1),  # Left
            self._is_obstacle(x, y+1),  # Right
            self._is_doorway(x, y),     # Current position (doorway)
        ]
        
        # Target Info (4): One-hot encoded target room
        target_info = self._target_room_info()  # One-hot encoded vector for target room
        
        # Navigation State (7): Normalized position, target distance, and remaining time
        navigation_state = [
            x / self.grid_size,          # Normalized x position
            y / self.grid_size,          # Normalized y position
            self._distance_to_target(),  # Distance to target room
            self.current_step / self.max_steps,  # Normalized time step
            0.0,  # Placeholder for additional sensor data, such as velocity or orientation
            0.0,  # Another placeholder for future sensors
            0.0,  # Placeholder for additional data
        ]
        
        # Combine all parts into one observation vector
        return np.concatenate([np.array(proximity_sensors), target_info, navigation_state], dtype=np.float32)

        # Add doorway visitation status to observations
        doorway_status = [
            1.0 if (x, y) in self.visited_doorways else 0.0 
            for (x, y) in self.doorways
        ]
        
        # Update observation space shape
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(16 + len(self.doorways),),  # Added doorway status
            dtype=np.float32
        )
        
        # Combine with existing observations
        return np.concatenate([
            original_observations, 
            np.array(doorway_status)
        ], dtype=np.float32)

    def _target_room_info(self):
        # One-hot encoding for the target room
        target_rooms = ['kitchen', 'bedroom', 'bathroom', 'hallway']
        return np.array([1 if room == self.target_room else 0 for room in target_rooms], dtype=np.float32)

    def _is_obstacle(self, x, y):
        for category in self.obstacles.values():
            if (x,y) in category:
                return 1.0
        return 0.0

    def _is_doorway(self, x, y):
        return 1.0 if (x,y) in self.doorways else 0.0

    def _distance_to_target(self):
        tx1, ty1, tx2, ty2 = self.rooms[self.target_room]
        target_center = ((tx1+tx2)/2, (ty1+ty2)/2)
        return np.linalg.norm(np.array(self.agent_pos) - np.array(target_center)) / self.grid_size

    def _room_identification(self):
        return np.array([
            int(self.target_room == 'kitchen'),
            int(self.target_room == 'bedroom'),
            int(self.target_room == 'bathroom'),
            int(self.target_room == 'hallway')
        ], dtype=np.float32)

    def in_target_room(self):
        x, y = self.agent_pos
        x1, y1, x2, y2 = self.rooms[self.target_room]
        return x1 <= x <= x2 and y1 <= y <= y2

    def step(self, action):
        self.current_step += 1
        self.last_room = self.current_room  # Track previous room
        # Movement handling
        if action < 4:
            dx, dy = [(1,0), (-1,0), (0,-1), (0,1)][action]
            new_x = np.clip(self.agent_pos[0]+dx, 0,19)
            new_y = np.clip(self.agent_pos[1]+dy, 0,19)
            
            if not self._is_obstacle(new_x, new_y):
                self.agent_pos = (new_x, new_y)

        # Reward calculation
        reward = -0.1  # Base time penalty
        done = False
        collision = False

        # Collision detection
        if self._is_obstacle(*self.agent_pos):
            reward = -5
            collision = True
            done = True

        # Target reached
        elif self.in_target_room():
            time_bonus = (self.max_steps - self.current_step) * 0.2
            reward = 15 + time_bonus
            done = True

        # Step limit termination
        elif self.current_step >= self.max_steps:
            reward -= 3
            done = True

        # Doorway bonus
        if not done and self._is_doorway(*self.agent_pos):
            reward += 1.0  # Increased doorway incentive
        if self._is_doorway(*self.agent_pos):
            self.doorways_passed += 1  # Now properly initialized

        return self._get_obs(), reward, done, {
            'collision': collision,
            'timeout': self.current_step >= self.max_steps,
            'current_room': self.current_room,
            'doorways_passed': self.doorways_passed  # Add this
        }

        # New doorway validation
        current_room = self._get_current_room()
        if self.last_room != current_room:
            if not self._room_change_through_doorway():
                reward -= 2.0  # Penalize room change without doorway
                
        # Enhanced doorway reward
        if self._is_doorway(*self.agent_pos):
            if (self.agent_pos[0], self.agent_pos[1]) not in self.visited_doorways:
                reward += 3.0  # Bonus for new doorway
                self.visited_doorways.add((self.agent_pos[0], self.agent_pos[1]))
            else:
                reward += 0.5  # Small bonus for reusing doorway

        return self._get_obs(), reward, done, info
    
        def _get_current_room(self):
            x, y = self.agent_pos
            for room, (x1, y1, x2, y2) in self.rooms.items():
                if x1 <= x <= x2 and y1 <= y <= y2:
                    return room
            return 'unknown'

    def _room_change_through_doorway(self):
        """Check if last position was a doorway when changing rooms"""
        return (self.agent_pos[0], self.agent_pos[1]) in self.doorways

    def get_obstacle_types(self):
        """Return obstacle category at given position"""
        x, y = self.agent_pos
        for cat, positions in self.obstacles.items():
            if (x,y) in positions:
                return cat
        return None