import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
g = 9.81  # Acceleration due to gravity
m1 = 1.0  # Mass of the first pendulum
m2 = 1.0  # Mass of the second pendulum
L1 = 1.0  # Length of the first pendulum
L2 = 1.0  # Length of the second pendulum

# Equations of motion derived from the Lagrangian
def equations_of_motion(y, t, m1, m2, L1, L2, g):
    θ1, ω1, θ2, ω2 = y
    
    Δθ = θ2 - θ1
    
    denominator1 = (m1 + m2) * L1 - m2 * L1 * np.cos(Δθ) * np.cos(Δθ)
    denominator2 = (L2 / L1) * denominator1
    
    dω1 = (m2 * L1 * ω1 * ω1 * np.sin(Δθ) * np.cos(Δθ) +
           m2 * g * np.sin(θ2) * np.cos(Δθ) +
           m2 * L2 * ω2 * ω2 * np.sin(Δθ) -
           (m1 + m2) * g * np.sin(θ1)) / denominator1
    
    dω2 = (-m2 * L2 * ω2 * ω2 * np.sin(Δθ) * np.cos(Δθ) +
           (m1 + m2) * (g * np.sin(θ1) * np.cos(Δθ) -
                        L1 * ω1 * ω1 * np.sin(Δθ) -
                        g * np.sin(θ2))) / denominator2
    
    return [ω1, dω1, ω2, dω2]

# Runge-Kutta 4th Order Method
def rk4_step(f, y, t, dt, *args):
    k1 = np.array(f(y, t, *args))
    k2 = np.array(f(y + dt * k1 / 2, t + dt / 2, *args))
    k3 = np.array(f(y + dt * k2 / 2, t + dt / 2, *args))
    k4 = np.array(f(y + dt * k3, t + dt, *args))
    
    return y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

# Initial conditions
y0 = [np.pi / 2, 0, np.pi / 2, 0]  # [θ1, ω1, θ2, ω2]
y0_perturbed = [np.pi / 2, 0, np.pi / 2 + 0.01, 0]  # Slightly different θ2
t0 = 0
tf = 20
dt = 0.01

# Time vector
t = np.arange(t0, tf, dt)

# Integrate the system using RK4
y = np.zeros((len(t), 4))
y_perturbed = np.zeros((len(t), 4))
y[0] = y0
y_perturbed[0] = y0_perturbed

for i in range(1, len(t)):
    y[i] = rk4_step(equations_of_motion, y[i-1], t[i-1], dt, m1, m2, L1, L2, g)
    y_perturbed[i] = rk4_step(equations_of_motion, y_perturbed[i-1], t[i-1], dt, m1, m2, L1, L2, g)

## Animation with increased size
fig, ax = plt.subplots()
ax.set_xlim(-2.9, 2.9)  # Adjust the x-axis limits for zoom
ax.set_ylim(-2.6, 1.5)  # Adjust the y-axis limits for zoom

line1, = ax.plot([], [], 'o-', lw=2, label='Initial θ2')
line2, = ax.plot([], [], 'o-', lw=2, label='Perturbed θ2', color='r')

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    return line1, line2

def update(frame):
    θ1 = y[frame, 0]
    θ2 = y[frame, 2]
    θ1_perturbed = y_perturbed[frame, 0]
    θ2_perturbed = y_perturbed[frame, 2]
    
    x1 = L1 * np.sin(θ1)
    y1 = -L1 * np.cos(θ1)
    x2 = x1 + L2 * np.sin(θ2)
    y2 = y1 - L2 * np.cos(θ2)
    
    x1_perturbed = L1 * np.sin(θ1_perturbed)
    y1_perturbed = -L1 * np.cos(θ1_perturbed)
    x2_perturbed = x1_perturbed + L2 * np.sin(θ2_perturbed)
    y2_perturbed = y1_perturbed - L2 * np.cos(θ2_perturbed)
    
    line1.set_data([0, x1, x2], [0, y1, y2])
    line2.set_data([0, x1_perturbed, x2_perturbed], [0, y1_perturbed, y2_perturbed])
    
    return line1, line2

ani = animation.FuncAnimation(fig, update, frames=range(len(t)), init_func=init, blit=True, interval=dt*1000)
plt.title('Double Pendulums in Almost identical Situation')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
ani.save("double_pendulum.gif")




plt.show()
