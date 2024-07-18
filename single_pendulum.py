import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
g = 9.81  # Acceleration due to gravity (m/s^2)
L = 1.0   # Length of the pendulum (m)
dt = 0.02  # Time step (s)
t_max = 10  # Maximum time for the animation (s)
theta0 = np.pi / 4  # Initial angle (rad)
omega0 = 0  # Initial angular velocity (rad/s)

# Equations of motion for the single pendulum
def equations_of_motion(theta, omega, t):
    dtheta_dt = omega
    domega_dt = -(g / L) * np.sin(theta)
    return dtheta_dt, domega_dt

# Runge-Kutta 4th Order Method to solve the equations of motion
def rk4_step(f, y, t, dt):
    k1 = np.array(f(y[0], y[1], t))
    k2 = np.array(f(y[0] + dt * k1[0] / 2, y[1] + dt * k1[1] / 2, t + dt / 2))
    k3 = np.array(f(y[0] + dt * k2[0] / 2, y[1] + dt * k2[1] / 2, t + dt / 2))
    k4 = np.array(f(y[0] + dt * k3[0], y[1] + dt * k3[1], t + dt))
    y_next = y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y_next

# Initial conditions
theta = theta0
omega = omega0
y = np.array([theta, omega])

# Time vector
t = np.arange(0, t_max, dt)

# Arrays to store the trajectory for plotting
theta_vals = []
omega_vals = []

# Integrate the system using RK4
for ti in t:
    theta_vals.append(y[0])
    omega_vals.append(y[1])
    y = rk4_step(equations_of_motion, y, ti, dt)

# Animation setup
fig, ax = plt.subplots()
ax.set_xlim(-1.2 * L, 1.2 * L)
ax.set_ylim(-1.2 * L, 1.2 * L)
ax.set_aspect('equal')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('Single Pendulum Motion')

line, = ax.plot([], [], 'o-', lw=2)
point, = ax.plot([], [], 'ro')

def init():
    line.set_data([], [])
    point.set_data([], [])
    return line, point

def update(frame):
    x = L * np.sin(theta_vals[frame])
    y = -L * np.cos(theta_vals[frame])
    line.set_data([0, x], [0, y])
    point.set_data(x, y)
    return line, point

ani = animation.FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=dt*1000)

# Save the animation as a video file
ani.save('single_pendulum_animation.gif')

plt.show()
