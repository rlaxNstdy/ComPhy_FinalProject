import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

# Gravitational constant 
G = 6.67430e-11 

# Planet data: [mass (kg), color, size]
PLANETS = {
    'Mercury': [3.301e23, 'gray', 8],
    'Venus': [4.867e24, 'orange', 12],
    'Earth': [5.972e24, 'blue', 12],
    'Mars': [6.417e23, 'red', 10],
    'Jupiter': [1.898e27, 'brown', 20],
    'Saturn': [5.683e26, 'gold', 18],
    'Uranus': [8.681e25, 'cyan', 15],
    'Neptune': [1.024e26, 'darkblue', 15]
}

class PlanetSimulation:
    def __init__(self):
        self.selected_planets = []
        self.positions = []
        self.velocities = []
        self.masses = []
        self.colors = []
        self.sizes = []
        self.names = []
        self.dt = 60 * 60 * 12 # 2 hours per step
        self.scale = 1e9  # Scale for display
        self.trail_length = 600 
        self.trails = []
        
    def select_planets(self):
        """Let user select which planets to include in simulation"""
        print("Available planets:")
        planet_list = list(PLANETS.keys())
        for i, planet in enumerate(planet_list):
            print(f"{i+1}. {planet}")
        
        print("\nSelect planets for simulation (enter numbers separated by spaces):")
        print("Example: 3 4 5 (for Earth, Mars, Jupiter)")
        
        while True:
            try:
                selection = input("Your selection: ").strip().split()
                if len(selection) < 2:
                    print("Please select at least 2 planets!")
                    continue
                
                indices = [int(x) - 1 for x in selection]
                if any(i < 0 or i >= len(planet_list) for i in indices):
                    print("Invalid planet number!")
                    continue
                
                self.selected_planets = [planet_list[i] for i in indices]
                break
                
            except ValueError:
                print("Please enter valid numbers!")
        
        print(f"\nSelected planets: {', '.join(self.selected_planets)}")
    
    def initialize_planets(self):
        """Initialize planet positions, velocities, and properties"""
        n_planets = len(self.selected_planets)
        
        # Random positions in a circle to avoid overlap
        angles = np.linspace(0, 2*np.pi, n_planets, endpoint=False)
        np.random.shuffle(angles)  # Randomize angles
        
        for i, planet_name in enumerate(self.selected_planets):
            # Random distance from center
            distance = random.uniform(2, 8) * self.scale  # 2-8 * 10^9 meters
            
            # Position
            x = distance * np.cos(angles[i])
            y = distance * np.sin(angles[i])
            self.positions.append([x, y])
            
            # Much smaller initial velocities
            vel_magnitude = random.uniform(50, 200)  # m/s - very small velocities
            vx = -vel_magnitude * np.sin(angles[i]) + random.uniform(-50, 50)
            vy = vel_magnitude * np.cos(angles[i]) + random.uniform(-50, 50)
            self.velocities.append([vx, vy])
            
            # Planet properties
            self.masses.append(PLANETS[planet_name][0])
            self.colors.append(PLANETS[planet_name][1])
            self.sizes.append(PLANETS[planet_name][2])
            self.names.append(planet_name)
            
            # Initialize trails
            self.trails.append([[], []])
        
        # Convert to numpy arrays
        self.positions = np.array(self.positions)
        self.velocities = np.array(self.velocities)
        self.masses = np.array(self.masses)
    
    def calculate_forces(self):
        """Calculate gravitational forces between all planets"""
        n_planets = len(self.selected_planets)
        forces = np.zeros((n_planets, 2))
        
        for i in range(n_planets):
            for j in range(n_planets):
                if i != j:
                    # Vector from planet i to planet j
                    r_vec = self.positions[j] - self.positions[i]
                    r_mag = np.linalg.norm(r_vec)
                    
                    # Prevent close collisions while allowing interactions
                    if r_mag > 5e8:  
                        # Gravitational force magnitude
                        F_mag = G * self.masses[i] * self.masses[j] / (r_mag**2)
                        
                        # Force direction (unit vector)
                        r_unit = r_vec / r_mag
                        
                        # Add force contribution
                        forces[i] += F_mag * r_unit
        
        return forces
    
    def update_physics(self):
        """Update positions and velocities using Euler integration"""
        forces = self.calculate_forces()
        
        # Calculate accelerations
        accelerations = forces / self.masses[:, np.newaxis]
        
        # Update velocities
        self.velocities += accelerations * self.dt
        
        # Update positions
        self.positions += self.velocities * self.dt
        
        # Update trails
        for i in range(len(self.selected_planets)):
            self.trails[i][0].append(self.positions[i, 0])
            self.trails[i][1].append(self.positions[i, 1])
            
            # Limit trail length
            if len(self.trails[i][0]) > self.trail_length:
                self.trails[i][0].pop(0)
                self.trails[i][1].pop(0)
    
    def setup_plot(self):
        """Setup the matplotlib figure and axes"""
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.ax.set_xlim(-20, 20)
        self.ax.set_ylim(-20, 20)
        self.ax.set_aspect('equal')
        self.ax.set_title("Planetary Motion Simulation", fontsize=16)
        self.ax.set_xlabel("Distance (×10⁹ m)")
        self.ax.set_ylabel("Distance (×10⁹ m)")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_facecolor('black')
        
        # Create plot elements
        self.planet_plots = []
        self.trail_plots = []
        
        for i, planet_name in enumerate(self.selected_planets):
            # Planet marker
            planet_plot, = self.ax.plot([], [], 'o', 
                                      color=self.colors[i], 
                                      markersize=self.sizes[i],
                                      label=planet_name,
                                      markeredgecolor='white',
                                      markeredgewidth=1)
            self.planet_plots.append(planet_plot)
            
            # Trail
            trail_plot, = self.ax.plot([], [], '-', 
                                     color=self.colors[i], 
                                     linewidth=1.5, 
                                     alpha=0.7)
            self.trail_plots.append(trail_plot)
        
        # Info text
        self.info_text = self.ax.text(0.02, 0.98, '', 
                                    transform=self.ax.transAxes, 
                                    fontsize=10,
                                    verticalalignment='top',
                                    bbox=dict(boxstyle='round', 
                                            facecolor='white', 
                                            alpha=0.8))
        
        self.ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    def animate(self, frame):
        """Animation update function"""
        # Update physics multiple times per frame for better accuracy
        for _ in range(3):  # Run physics 3 times per animation frame
            self.update_physics()
        
        # Update planet positions
        for i, planet_plot in enumerate(self.planet_plots):
            x_scaled = self.positions[i, 0] / self.scale
            y_scaled = self.positions[i, 1] / self.scale
            planet_plot.set_data([x_scaled], [y_scaled])
        
        # Update trails
        for i, trail_plot in enumerate(self.trail_plots):
            if len(self.trails[i][0]) > 1:
                x_trail = np.array(self.trails[i][0]) / self.scale
                y_trail = np.array(self.trails[i][1]) / self.scale
                trail_plot.set_data(x_trail, y_trail)
        
        # Update info text
        info_text = f"Time: {frame * self.dt * 3 / (24 * 3600):.2f} days\n"
        info_text += f"Planets: {len(self.selected_planets)}\n"
        info_text += "Positions (×10⁹ m):\n"
        for i, name in enumerate(self.names):
            x_pos = self.positions[i, 0] / self.scale
            y_pos = self.positions[i, 1] / self.scale
            info_text += f"{name}: ({x_pos:.1f}, {y_pos:.1f})\n"
        
        self.info_text.set_text(info_text)
        
        return self.planet_plots + self.trail_plots + [self.info_text]
    
    def run_simulation(self):
        """Run the complete simulation"""
        print("=== Planetary Motion Simulation ===")
        self.select_planets()
        self.initialize_planets()
        self.setup_plot()
        
        print(f"\nStarting simulation with {len(self.selected_planets)} planets...")
        print("Close the plot window to end the simulation.")
        
        # Create and run animation
        ani = FuncAnimation(self.fig, self.animate, 
                          interval=50, blit=True, cache_frame_data=False)
        plt.show()

# Run the simulation
if __name__ == "__main__":
    sim = PlanetSimulation()
    sim.run_simulation()