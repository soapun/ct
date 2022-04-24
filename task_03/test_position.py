from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.animation as anime
from importlib import reload

import python_solvers
import cython_solver
import opencl_solver

@dataclass
class PlanetInfo:
    r_0 : float
    v_0 : float
    m_0 : float
    color : str
    name : str
    radius : float

def arr(l):
    return np.array(l, dtype=np.float64)

def planet_to_circle(planet):
    circle = plt.Circle(planet.r_0, planet.radius, fc = planet.color)
    return circle

def animate_planets(planets, R, title="some_title"):
    n_iters = 1000
    w = 5 * scale
    h = 5 * scale
    plt.style.use("dark_background")
    fig = plt.figure()
    ax = plt.axes(xlim=(-w, w + scale), ylim=(-h, h))
    ax.set_facecolor((0, 0, 0.1))
    ax.set_title(title, color="#ffffff")

    lw = 4
    legend_elements = [lines.Line2D([0], [0], color=i.color, lw=lw) for i in planets]
    ax.legend(legend_elements, [i.name for i in planets], loc='upper right')
    planet_circles = [planet_to_circle(planet) for planet in planets]


    def init():
        for pc in planet_circles:
            ax.add_patch(pc)
        return planet_circles

    def animate(i):
        j = i % n_iters
        for k in range(r_0.shape[0]):
            planet_circles[k].center = R[j, k]
        return planet_circles


    anim = anime.FuncAnimation(fig,
                            animate,
                            init_func=init,
                            frames=360,
                            interval=20,
                            blit=True)
    anim.save(f"plots\\{title}.gif")

if __name__ == "__main__":
    scale = 10 ** 12
    dt = 1.5 * 10**7
    n_iters = 1000
    G= 6.6743 * 10 ** -11

    planets = [
        PlanetInfo(
            arr([0, 0]) * scale,
            arr([0, 0]) * 10 ** 3,
            1.9 * 10**30,
            "#f9d71c",
            "Sun",
             0.3 * scale
        ),
        PlanetInfo(
            arr([0.8, 0]) * scale,
            arr([0.0, 13.07]) * 10 ** 3,
            18.986 * 10**26,
            "#d8ca9d",
            "Jupyter",
            0.2 * scale
        ),
        PlanetInfo(
            arr([1.43, 0.0]) * scale,
            arr([0.0, 9.69]) * 10 ** 3,
            5.68 * 10**26,
            "#cc7722",
            "Saturn",
            0.15 * scale
        ),
        PlanetInfo(
            arr([2.8, 0.0]) * scale,
            arr([0.0, 6.8]) * 10 ** 3,
            0.87 * 10**26,
            "#0000ff",
            "Neptune",
             0.1 * scale
        ),
        PlanetInfo(
            arr([4.5, 0.0]) * scale,
            arr([0.0, 5.4]) * 10 ** 3,
            1.0243 * 10**26,
            "#66574e",
            "Pluto",
             0.1 * scale
        ),
    ]

    m_0 = np.array([i.m_0 for i in planets]).astype(np.float64)
    r_0 = np.vstack([i.r_0 for i in planets])
    v_0 = np.vstack([i.v_0 for i in planets])

    solvers = {
        "odeint" : python_solvers.OdeintSolver(dt, n_iters, G),
        "python" : python_solvers.PythonVerletSolver(dt, n_iters, G),
        "cython" : cython_solver.CythonSolver(dt, n_iters, G),
        "multiprocessing" : python_solvers.MultiprocessingVerletSolver(dt, n_iters, G),
        "opencl" : opencl_solver.OpenCLSolver(dt, n_iters, G)
    }
    solutions = {}
    for solver in solvers:
        solutions[solver] = solvers[solver].solve(r_0, v_0, m_0)
        animate_planets(planets, solutions[solver], solver)
    plt.clf()
    plt.title("Погрешность по сравнению с odeint")
    for solver in solutions:
        diff = (solutions['odeint'] - solutions[solver]) ** 2
        diff = np.sum(diff, axis=2)
        diff = np.sum(diff, axis=1)
        diff = np.sqrt(diff)

        plt.plot(diff, label=solver, alpha=0.3)
    plt.legend()
    plt.savefig("plots\\diff.png")