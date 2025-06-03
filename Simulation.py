import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# Gravitational constant
G = 6.67430e-10

# Masses of the planets (kg)
mass1 = 4.867e24   # Planet 1
mass2 = 4.872e24   # Planet 2
mass3 = 8e23       # Planet 3

# Initial positions in "grid units" (scaled to be intuitive)
planet1_pos = np.array([4, 5]) * 1e8
planet2_pos = np.array([-4, -4]) * 1e8
planet3_pos = np.array([6, -3]) * 1e8

# Initial velocities (m/s)
planet1_vel = np.array([200, -100])
planet2_vel = np.array([-200, 300])
planet3_vel = np.array([50, -250])

# Initial state vector
y0 = np.concatenate([planet1_pos, planet2_pos, planet3_pos,
                     planet1_vel, planet2_vel, planet3_vel])

# Function to compute derivatives
def derivatives(t, y):
    x1, y1, x2, y2, x3, y3 = y[0:6]
    vx1, vy1, vx2, vy2, vx3, vy3 = y[6:12]
    r1 = np.array([x1, y1])
    r2 = np.array([x2, y2])
    r3 = np.array([x3, y3])

    r12 = r2 - r1
    r13 = r3 - r1
    r23 = r3 - r2

    d12 = np.linalg.norm(r12)
    d13 = np.linalg.norm(r13)
    d23 = np.linalg.norm(r23)

    a1 = G * mass2 * r12 / d12**3 + G * mass3 * r13 / d13**3
    a2 = G * mass1 * (-r12) / d12**3 + G * mass3 * r23 / d23**3
    a3 = G * mass1 * (-r13) / d13**3 + G * mass2 * (-r23) / d23**3

    return np.concatenate([[vx1, vy1, vx2, vy2, vx3, vy3], a1, a2, a3])

# Simulation time
t_span = (0, 60 * 60 * 24 * 10)  # 10 days
t_eval = np.linspace(*t_span, 3000)

# Solve the system
sol = solve_ivp(derivatives, t_span, y0, t_eval=t_eval, rtol=1e-9)
x1, y1 = sol.y[0], sol.y[1]
x2, y2 = sol.y[2], sol.y[3]
x3, y3 = sol.y[4], sol.y[5]
times = sol.t

# Set up plot
scale = 1e8  # scaling for axis to show readable grid
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)
ax.set_aspect('equal')
ax.set_title("3-Body Problem Simulation (Grid Display)")
ax.set_xlabel("x (×10⁸ m)")
ax.set_ylabel("y (×10⁸ m)")
ax.grid(True)

# Plotting elements
planet1, = ax.plot([], [], 'orange', marker='o', label='Planet 1')
planet2, = ax.plot([], [], 'blue', marker='o', label='Planet 2')
planet3, = ax.plot([], [], 'red', marker='o', label='Planet 3')
trail1, = ax.plot([], [], 'orange', linewidth=0.5)
trail2, = ax.plot([], [], 'blue', linewidth=0.5)
trail3, = ax.plot([], [], 'red', linewidth=0.5)

# Text for live data
position_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.legend()

# Update function for animation
def update(i):
    # Update planet positions (scaled to grid)
    planet1.set_data([x1[i] / scale], [y1[i] / scale])
    planet2.set_data([x2[i] / scale], [y2[i] / scale])
    planet3.set_data([x3[i] / scale], [y3[i] / scale])

    trail1.set_data(x1[:i] / scale, y1[:i] / scale)
    trail2.set_data(x2[:i] / scale, y2[:i] / scale)
    trail3.set_data(x3[:i] / scale, y3[:i] / scale)

    # Update text box with current positions
    text = (
        f"Time: {times[i] / 3600:.2f} hr\n"
        f"Planet 1: ({x1[i] / scale:.1f}, {y1[i] / scale:.1f}) ×10⁸ m\n"
        f"Planet 2: ({x2[i] / scale:.1f}, {y2[i] / scale:.1f}) ×10⁸ m\n"
        f"Planet 3: ({x3[i] / scale:.1f}, {y3[i] / scale:.1f}) ×10⁸ m"
    )
    position_text.set_text(text)

    return planet1, planet2, planet3, trail1, trail2, trail3, position_text

# Run the animation
ani = FuncAnimation(fig, update, frames=range(0, len(times), 5), interval=5, blit=True)
plt.show()
