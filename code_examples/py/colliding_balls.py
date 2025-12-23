import numpy as np
from numba import cuda
import math
import matplotlib.pyplot as plt
import time

# --- Configuration ---
NUM_BALLS = 10000
WIDTH, HEIGHT = 800, 800
BALL_RADIUS = 2.0  # Small radius to fit 10k balls
MAX_SPEED = 2.0
BLOCK_TPB = 128    # Threads per block

# --- CUDA Kernels ---

@cuda.jit
def move_balls_kernel(pos, vel, radius, width, height):
    """
    Updates position based on velocity and handles wall collisions.
    """
    i = cuda.grid(1)
    if i < pos.shape[0]:
        # Update Position
        pos[i, 0] += vel[i, 0]
        pos[i, 1] += vel[i, 1]

        # Wall Collisions (X-axis)
        if pos[i, 0] - radius[i] < 0:
            pos[i, 0] = radius[i]
            vel[i, 0] *= -1
        elif pos[i, 0] + radius[i] > width:
            pos[i, 0] = width - radius[i]
            vel[i, 0] *= -1

        # Wall Collisions (Y-axis)
        if pos[i, 1] - radius[i] < 0:
            pos[i, 1] = radius[i]
            vel[i, 1] *= -1
        elif pos[i, 1] + radius[i] > height:
            pos[i, 1] = height - radius[i]
            vel[i, 1] *= -1

@cuda.jit
def collide_balls_kernel(pos, vel, radius, mass):
    """
    Checks collisions between ball 'i' and all other balls 'j'.
    Performs elastic collision physics.
    """
    i = cuda.grid(1)
    n = pos.shape[0]
    
    if i >= n:
        return

    # Load data for ball i
    px_i = pos[i, 0]
    py_i = pos[i, 1]
    vx_i = vel[i, 0]
    vy_i = vel[i, 1]
    r_i = radius[i]
    m_i = mass[i]

    # Iterate over all other balls
    for j in range(n):
        if i == j:
            continue

        px_j = pos[j, 0]
        py_j = pos[j, 1]
        r_j = radius[j]

        # Distance check (squared to avoid expensive sqrt where possible)
        dx = px_j - px_i
        dy = py_j - py_i
        dist_sq = dx*dx + dy*dy
        min_dist = r_i + r_j

        # If colliding
        if dist_sq < min_dist * min_dist and dist_sq > 0:
            dist = math.sqrt(dist_sq)
            
            # Normal Vector (n)
            nx = dx / dist
            ny = dy / dist

            # Relative Velocity
            vx_j = vel[j, 0]
            vy_j = vel[j, 1]
            dvx = vx_j - vx_i
            dvy = vy_j - vy_i

            # Velocity along normal
            vel_along_normal = dvx * nx + dvy * ny

            # Do not resolve if velocities are separating
            if vel_along_normal > 0:
                continue

            # Elastic Collision Impulse
            # Assumes restitution (bounciness) = 1.0
            m_j = mass[j]
            j_impulse = -(2.0 * vel_along_normal) / (1.0/m_i + 1.0/m_j)

            # Apply impulse to ball i only (thread j handles ball j)
            # We scale by 0.5 effectively because the collision is calculated 
            # twice (once by i, once by j).
            # However, standard physics: v_new = v_old - (impulse / mass) * normal
            
            vx_i -= (j_impulse / m_i) * nx * 0.5
            vy_i -= (j_impulse / m_i) * ny * 0.5
            
            # Positional correction (prevent sticking)
            # Push ball i away from j slightly based on overlap
            overlap = min_dist - dist
            px_i -= overlap * 0.5 * nx
            py_i -= overlap * 0.5 * ny

    # Write back
    vel[i, 0] = vx_i
    vel[i, 1] = vy_i
    pos[i, 0] = px_i
    pos[i, 1] = py_i

# --- Main Simulation Class ---

class Simulation:
    def __init__(self):
        print(f"Initializing {NUM_BALLS} balls...")
        
        # Initialize Host Data
        self.pos = np.random.uniform(BALL_RADIUS, WIDTH-BALL_RADIUS, (NUM_BALLS, 2)).astype(np.float32)
        angles = np.random.uniform(0, 2*np.pi, NUM_BALLS)
        speeds = np.random.uniform(0.5, MAX_SPEED, NUM_BALLS)
        self.vel = np.column_stack((np.cos(angles)*speeds, np.sin(angles)*speeds)).astype(np.float32)
        
        self.radius = np.full(NUM_BALLS, BALL_RADIUS, dtype=np.float32)
        self.mass = np.full(NUM_BALLS, 1.0, dtype=np.float32) # Uniform mass for simplicity

        # Copy to Device (GPU)
        self.d_pos = cuda.to_device(self.pos)
        self.d_vel = cuda.to_device(self.vel)
        self.d_radius = cuda.to_device(self.radius)
        self.d_mass = cuda.to_device(self.mass)

        # Calculate grid size
        self.threadsperblock = BLOCK_TPB
        self.blockspergrid = (NUM_BALLS + (self.threadsperblock - 1)) // self.threadsperblock

    def step(self):
        # 1. Update Positions & Wall Collisions
        move_balls_kernel[self.blockspergrid, self.threadsperblock](
            self.d_pos, self.d_vel, self.d_radius, WIDTH, HEIGHT
        )
        
        # 2. Check Ball-Ball Collisions
        # Note: O(N^2) is heavy. 10,000^2 = 100,000,000 checks per frame.
        # This relies heavily on GPU massive parallelism.
        collide_balls_kernel[self.blockspergrid, self.threadsperblock](
            self.d_pos, self.d_vel, self.d_radius, self.d_mass
        )
        
        # Ensure GPU finishes before next step (optional sync for timing, implicit in data copy)
        cuda.synchronize()

    def get_positions(self):
        return self.d_pos.copy_to_host()

# --- Run & Visualize ---

def main():
    sim = Simulation()
    
    # Setup Plot
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    
    # Create Scatter Plot
    # Using a small point size (s) to render fast
    particles = ax.scatter([], [], s=2, c='cyan', alpha=0.6)
    
    text_counter = ax.text(10, HEIGHT-30, '', color='white')

    print("Starting simulation loop... (Close window to stop)")
    
    t0 = time.time()
    frame_count = 0
    
    try:
        while True:
            # Physics Step
            sim.step()
            
            # Rendering (Host-Device copy happens here)
            # To increase FPS, you might render only every n-th frame
            positions = sim.get_positions()
            particles.set_offsets(positions)
            
            # FPS Counter
            frame_count += 1
            if frame_count % 10 == 0:
                dt = time.time() - t0
                fps = frame_count / dt
                text_counter.set_text(f"Balls: {NUM_BALLS} | FPS: {fps:.2f}")
            
            plt.pause(0.001) # Small pause to allow GUI update

    except KeyboardInterrupt:
        print("Simulation stopped.")

if __name__ == "__main__":
    main()