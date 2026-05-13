# visualization.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button
from astropy.time import Time
import astropy.units as u

from constants import get_heliocentric_state, PLANET_DATA, AU, CELESTIAL_RADII
from optimizer import Optimizer


PLANET_COLORS = {
    'mercury': 'gray',
    'venus':   'orange',
    'earth':   'blue',
    'mars':    'red',
    'jupiter': 'brown',
    'saturn':  'gold',
    'uranus':  'lightblue',
    'neptune': 'darkblue',
    'sun':     'yellow'
}

PLANET_SIZES = {
    'mercury': 30,
    'venus':   40,
    'earth':   45,
    'mars':    35,
    'jupiter': 90,
    'saturn':  80,
    'uranus':  55,
    'neptune': 50,
    'sun':     150
}


def get_planet_positions_at_time(date):
    """Позиции планет в AU на момент date."""
    if isinstance(date, str):
        time = Time(date, scale='tdb')
    else:
        time = date
    positions = {}
    for name in PLANET_DATA.keys():
        pos, _ = get_heliocentric_state(name, time)
        positions[name] = pos / AU
    return positions


def plot_elliptical_orbit(ax, planet_name, time_start, duration_days=800, num_points=300):
    """Орбита планеты через astropy."""
    if planet_name == 'sun':
        return
    t0 = Time(time_start, scale='tdb')
    times = t0 + np.linspace(0, duration_days, num_points) * u.day
    orbit_points = []
    for t in times:
        pos, _ = get_heliocentric_state(planet_name, t)
        orbit_points.append(pos / AU)
    orbit_points = np.array(orbit_points).T
    ax.plot(orbit_points[0, :], orbit_points[1, :], orbit_points[2, :],
            color=PLANET_COLORS.get(planet_name, 'gray'),
            linestyle='-', linewidth=1.5, alpha=0.7,
            label=f'Orbit of {planet_name.capitalize()}')


def plot_trajectory(ax, trajectory, color='red', linewidth=2, label='Spacecraft trajectory'):
    """Траектория корабля (координаты в метрах)."""
    traj_au = trajectory / AU
    ax.plot(traj_au[0, :], traj_au[1, :], traj_au[2, :],
            color=color, linewidth=linewidth, label=label, alpha=0.9)
    ax.scatter(traj_au[0, 0], traj_au[1, 0], traj_au[2, 0],
               color='lime', s=120, marker='o', label='Start',
               edgecolors='darkgreen', linewidth=2, zorder=15)
    ax.scatter(traj_au[0, -1], traj_au[1, -1], traj_au[2, -1],
               color='red', s=120, marker='s', label='Finish',
               edgecolors='darkred', linewidth=2, zorder=15)


def visualize_mission_from_surface(start_planet, start_date_str, accel, direction,
                                   burn_time_sec, target_planet=None):
    """
    Старт с поверхности, постоянное ускорение, свободный режим.
    """
    radius_m = CELESTIAL_RADII[start_planet]
    planet_pos, planet_vel = get_heliocentric_state(start_planet, start_date_str)

    dir_norm = np.linalg.norm(direction)
    if dir_norm == 0:
        raise ValueError("Direction vector cannot be zero")
    direction = np.array(direction) / dir_norm
    start_pos = planet_pos + radius_m * direction
    start_vel = planet_vel.copy()

    opt = Optimizer(accel=accel, mode='free')
    t_array, pos_traj, vel_traj = opt.trajectory(start_pos, start_vel,
                                                 direction, burn_time_sec)

    flight_days = burn_time_sec / 86400
    end_time = Time(start_date_str, scale='tdb') + burn_time_sec * u.s
    print("=" * 60)
    print(f"Mission: from {start_planet.capitalize()} surface")
    print(f"Acceleration: {accel:.2f} m/s², direction: {direction}")
    print(f"Burn time: {burn_time_sec:.0f} s ({flight_days:.4f} days)")
    print(f"Start date: {start_date_str}")
    print(f"End date:   {end_time.iso}")

    start_au = pos_traj[:, 0] / AU
    end_au = pos_traj[:, -1] / AU
    dist_au = np.linalg.norm(end_au - start_au)
    print(f"Distance travelled: {dist_au:.5f} AU ({dist_au * 149.6:.2f} million km)")
    print("=" * 60)

    planet_pos_au = planet_pos / AU

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"Launch from {start_planet.capitalize()} surface\n"
                 f"Start: {start_date_str[:10]}   End: {end_time.iso[:10]}",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("X (AU)")
    ax.set_ylabel("Y (AU)")
    ax.set_zlabel("Z (AU)")

    ax.scatter(0, 0, 0, color='yellow', s=PLANET_SIZES['sun'],
               label='Sun', edgecolors='orange', linewidth=1, zorder=10)

    plot_elliptical_orbit(ax, start_planet, start_date_str)

    ax.scatter(planet_pos_au[0], planet_pos_au[1], planet_pos_au[2],
               color=PLANET_COLORS.get(start_planet, 'white'),
               s=PLANET_SIZES.get(start_planet, 40),
               label=f"{start_planet.capitalize()} at start",
               edgecolors='black', linewidth=0.5, zorder=5)

    plot_trajectory(ax, pos_traj)

    ax.legend(loc='upper left', fontsize='small', framealpha=0.9)

    padding = 1.0
    ax.set_xlim(planet_pos_au[0] - padding, planet_pos_au[0] + padding)
    ax.set_ylim(planet_pos_au[1] - padding, planet_pos_au[1] + padding)
    ax.set_zlim(planet_pos_au[2] - padding, planet_pos_au[2] + padding)

    def zoom_to_planet(event):
        ax.set_xlim(planet_pos_au[0] - 0.5, planet_pos_au[0] + 0.5)
        ax.set_ylim(planet_pos_au[1] - 0.5, planet_pos_au[1] + 0.5)
        ax.set_zlim(planet_pos_au[2] - 0.5, planet_pos_au[2] + 0.5)
        fig.canvas.draw_idle()

    def zoom_to_sun(event):
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_zlim(-3, 3)
        fig.canvas.draw_idle()

    plt.subplots_adjust(bottom=0.1)
    ax_button_planet = plt.axes([0.6, 0.02, 0.18, 0.06])
    btn_planet = Button(ax_button_planet, f'Zoom to {start_planet.capitalize()}')
    btn_planet.on_clicked(zoom_to_planet)

    ax_button_sun = plt.axes([0.8, 0.02, 0.15, 0.06])
    btn_sun = Button(ax_button_sun, 'Zoom to Sun')
    btn_sun.on_clicked(zoom_to_sun)

    plt.show()
    return pos_traj, t_array


if __name__ == "__main__":
    start_planet = 'earth'
    start_date = '2030-01-01 00:00:00'
    accel = 10.0
    direction = np.array([1.0, 0.0, 0.0])
    burn_time = 3600.0
    visualize_mission_from_surface(start_planet, start_date, accel, direction, burn_time)