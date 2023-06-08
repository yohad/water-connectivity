import numpy as np
import dedalus.public as d3
import logging
from tqdm import tqdm

from pathlib import Path

logger = logging.getLogger(__name__)


def generate_lift_function(basis):
    """Generate a function that multiplies a field by the lift operator."""
    lift_basis = basis.derivative_basis(1)

    def lift(A: d3.Field, n: int):
        return d3.Lift(A, lift_basis, n)

    return lift


def generate_derivative_function(coord: d3.Coordinate):
    """
    Generate a function that multiplies a field by the derivative operator in
    the given coordinate.
    """

    def derivative(A: d3.Field):
        return d3.Differentiate(A, coord)

    return derivative


def run_gilad_model(
    output: Path,
    length: int,
    resolution: int,
    sim_time: int,
    slope: float,
    precipitation: float,
) -> None:
    # Parameters
    m = slope
    nu = 10 / 3
    eta = 3.5
    rho = 0.95
    gamma = 50 / 3
    delta_b = 1 / 30
    delta_w = 10 / 3
    delta_h = 1e-2 / 3
    a = 33.33
    q = 0.05
    f = 0.1
    p = precipitation

    Lx, Ly = length, length
    Nx, Ny = resolution, resolution

    dealias = 2
    stop_sim_time = sim_time
    timestepper = d3.SBDF2
    dtype = np.float64

    # Bases
    coords = d3.CartesianCoordinates("x", "y")
    dist = d3.Distributor(coords, dtype=dtype)
    xbasis = d3.RealFourier(
        coords["x"], size=Nx, bounds=(0, Lx), dealias=dealias
    )
    ybasis = d3.Chebyshev(
        coords["y"], size=Ny, bounds=(0, Ly), dealias=dealias
    )

    lift = generate_lift_function(ybasis)
    dy = generate_derivative_function(coords["y"])

    # Fields
    b = dist.Field(name="b", bases=(xbasis, ybasis))
    w = dist.Field(name="w", bases=(xbasis, ybasis))
    h = dist.Field(name="h", bases=(xbasis, ybasis))

    terrain = dist.Field(name="terrain", bases=ybasis)
    x, y = dist.local_grids(xbasis, ybasis)
    terrain["g"] = m * y

    tau_b1 = dist.Field(name="tau_b1", bases=xbasis)
    tau_b2 = dist.Field(name="tau_b2", bases=xbasis)

    tau_w1 = dist.Field(name="tau_w1", bases=xbasis)
    tau_w2 = dist.Field(name="tau_w2", bases=xbasis)

    tau_h1 = dist.Field(name="tau_h1", bases=xbasis)
    tau_h2 = dist.Field(name="tau_h2", bases=xbasis)

    # Substritutions
    infiltration = a * (b + q * f) / (b + q)
    Gb = nu * w * np.power(1 + eta * b, 2)
    Gw = gamma * b * np.power(1 + eta * b, 2)

    ex, ey = coords.unit_vector_fields(dist)

    grad_b = d3.Gradient(b) + ey * lift(tau_b1, -1)
    lap_b = d3.Divergence(grad_b)

    grad_w = d3.Gradient(w) + ey * lift(tau_w1, -1)
    lap_w = d3.Divergence(grad_w)

    lap_h2 = d3.Laplacian(h * h)

    f_b = Gb * b * (1 - b)
    f_w = infiltration * h - nu * (1 - rho * b) * w - Gw * w
    f_h = p - infiltration * h + delta_h * lap_h2

    J = -2 * delta_h * h * dy(h)

    # Problem
    problem = d3.IVP(
        [b, w, h, tau_b1, tau_b2, tau_w1, tau_w2, tau_h1, tau_h2],
        namespace=locals(),
    )
    problem.add_equation("dt(b) - delta_b*lap_b + b + lift(tau_b2, -1) = f_b")
    problem.add_equation("dt(w) - delta_w*lap_w + lift(tau_w2, -1) = f_w")
    problem.add_equation(
        "dt(h) - 2 * delta_h * m * dy(h) + lift(tau_h1, -1) + lift(tau_h2, -2) = f_h"
    )

    problem.add_equation("dy(b)(y=Ly) = 0")
    problem.add_equation("dy(b)(y=0) = 0")
    problem.add_equation("dy(w)(y=Ly) = 0")
    problem.add_equation("dy(w)(y=0) = 0")

    problem.add_equation("dy(h)(y=Ly) = 0")
    # problem.add_equation("h(y=Ly) = 0")
    problem.add_equation("dy(dy(dy(h)))(y=0) = 0")

    # Solver
    solver = problem.build_solver(timestepper)
    solver.stop_sim_time = stop_sim_time

    # Initial Conditions
    b.fill_random(
        "g", seed=42, distribution="normal", scale=1e-3
    )  # Random noise
    b["g"] = (b["g"]) ** 2  # Positive noise, noise is now of order 1e-6
    w["g"] = p / nu
    h["g"] = p / a

    # Analysis
    timestep = 0.01
    snapshots_path = (f"{output}/snapshots/snapshots_p{p}_m{m}").replace(
        ".", "_"
    )
    snapshots = solver.evaluator.add_file_handler(
        snapshots_path, sim_dt=0.5, max_writes=500
    )
    snapshots.add_task(b, name="biomass")
    snapshots.add_task(w, name="soil_water")
    snapshots.add_task(h, name="surface_water")
    snapshots.add_task(J, name="flux")

    # Main loop
    logger.info("Starting main loop")
    progress_bar = tqdm(total=stop_sim_time, ncols=80)
    while solver.proceed:
        solver.step(timestep)
        progress_bar.update(timestep)
    progress_bar.close()


if __name__ == "__main__":
    run_gilad_model()
