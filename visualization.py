import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from astropy.time import Time
import astropy.units as u

from constants import get_heliocentric_state, PLANET_DATA, AU, CELESTIAL_SOI

ECLIPTIC_INCLINATION = np.radians(-23.5)
ROTATION_MATRIX = np.array([
    [1, 0, 0],
    [0, np.cos(ECLIPTIC_INCLINATION), -np.sin(ECLIPTIC_INCLINATION)],
    [0, np.sin(ECLIPTIC_INCLINATION), np.cos(ECLIPTIC_INCLINATION)]
])

def apply_rotation(points):
    points = np.asarray(points)
    if points.ndim == 1:
        return ROTATION_MATRIX @ points
    elif points.shape[0] == 3:
        return ROTATION_MATRIX @ points
    elif points.shape[1] == 3:
        return points @ ROTATION_MATRIX.T
    else:
        raise ValueError(f"Wrong shape: {points.shape}")

PLANET_COLORS = {
    'mercury': 'gray', 'venus': 'orange', 'earth': 'blue', 'mars': 'red',
    'jupiter': 'brown', 'saturn': 'gold', 'uranus': 'lightblue', 'neptune': 'darkblue', 'sun': 'yellow'
}
PLANET_SIZES = {
    'mercury': 30, 'venus': 40, 'earth': 45, 'mars': 35,
    'jupiter': 90, 'saturn': 80, 'uranus': 55, 'neptune': 50, 'sun': 150
}

def plot_orbit_from_ephem(ax, planet, start_date, end_date, color=None, linewidth=0.8, alpha=0.6, resolution=200):
    if color is None:
        color = PLANET_COLORS.get(planet, 'gray')
    
    if isinstance(start_date, str):
        t_start = Time(start_date, scale='tdb')
    else:
        t_start = start_date
    if isinstance(end_date, str):
        t_end = Time(end_date, scale='tdb')
    else:
        t_end = end_date
    
    time_points = np.linspace(t_start.jd, t_start.jd + 687, resolution)
    positions_au = []
    for jd in time_points:
        t = Time(jd, format='jd', scale='tdb')
        pos, _ = get_heliocentric_state(planet, t)
        pos_au = pos / AU
        pos_au = apply_rotation(pos_au)
        positions_au.append(pos_au)
    positions_au = np.array(positions_au).T
    ax.plot(positions_au[0, :], positions_au[1, :], positions_au[2, :],
            color=color, linewidth=linewidth, alpha=alpha, label=f'Orbit of {planet.capitalize()}')



def plot_soi_circle(ax, center_au, radius_au, color='cyan', linestyle='--', linewidth=1.5, resolution=100):
    theta = np.linspace(0, 2*np.pi, resolution)
    x = center_au[0] + radius_au * np.cos(theta)
    y = center_au[1] + radius_au * np.sin(theta)
    z = np.full_like(theta, center_au[2])
    ax.plot(x, y, z, color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.7, label=f'SOI {target_planet.capitalize()}')

def get_planet_positions_at_time(date, planets_list=None, apply_rot=True):
    if isinstance(date, str):
        time = Time(date, scale='tdb')
    else:
        time = date
    positions = {}
    if planets_list is None:
        planets_list = PLANET_DATA.keys()
    for name in planets_list:
        pos, _ = get_heliocentric_state(name, time)
        pos_au = pos / AU
        if apply_rot:
            pos_au = apply_rotation(pos_au)
        positions[name] = pos_au
    return positions

def plot_planets(ax, positions, alpha=1.0, label=None):
    for name, pos in positions.items():
        ax.scatter(pos[0], pos[1], pos[2],
                   color=PLANET_COLORS.get(name, 'white'), s=PLANET_SIZES.get(name, 30),
                   alpha=alpha, label=label if label else "")

def plot_trajectory(ax, trajectory_m, color='green', linewidth=2, label='trajectory', apply_rot=True):
    traj_m = np.asarray(trajectory_m)
    if apply_rot:
        traj_m = apply_rotation(traj_m)
    traj_au = traj_m / AU
    ax.plot(traj_au[0, :], traj_au[1, :], traj_au[2, :],
            color=color, linewidth=linewidth, label=label)
    ax.scatter(traj_au[0, 0], traj_au[1, 0], traj_au[2, 0],
               color='lime', s=80, marker='o', label='Ship start', edgecolors='darkgreen')
    ax.scatter(traj_au[0, -1], traj_au[1, -1], traj_au[2, -1],
               color='lime', s=80, marker='s', label='Ship finish', edgecolors='darkgreen')

def visualize_mission(trajectory_m, start_date_str, flight_time_sec,
                      start_planet='earth', target_planet='mars'):
    start_time = Time(start_date_str, scale='tdb')
    end_time = start_time + flight_time_sec * u.s
    end_date_str = end_time.iso

    print("=" * 60)
    print(f"Mission: {start_planet.capitalize()} -> {target_planet.capitalize()}")
    print(f"Start: {start_date_str}")
    print(f"Finish: {end_date_str}")
    print("=" * 60)

    start_positions = get_planet_positions_at_time(start_date_str, [start_planet], apply_rot=True)
    end_positions   = get_planet_positions_at_time(end_date_str, [target_planet], apply_rot=True)

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"{start_planet.capitalize()} -> {target_planet.capitalize()}\n{start_date_str[:10]} -> {end_date_str[:10]}",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("X (AU)"); ax.set_ylabel("Y (AU)"); ax.set_zlabel("Z (AU)")

    # Солнце
    ax.scatter(0, 0, 0, color='yellow', s=PLANET_SIZES['sun'], label='Sun', edgecolors='orange')
    # Орбиты
    plot_orbit_from_ephem(ax, start_planet, start_date_str, end_date_str,
                          color=PLANET_COLORS.get(start_planet, 'blue'), linewidth=1.0, alpha=0.7)
    plot_orbit_from_ephem(ax, target_planet, start_date_str, end_date_str,
                          color=PLANET_COLORS.get(target_planet, 'red'), linewidth=1.0, alpha=0.7)
    # Планеты
    plot_planets(ax, start_positions, alpha=1.0, label=f"{start_planet.capitalize()} (Start)")
    plot_planets(ax, end_positions, alpha=0.4, label=f"{target_planet.capitalize()} (Finish)")
    # Траектория
    plot_trajectory(ax, trajectory_m, apply_rot=True)
    soi_radius_m = CELESTIAL_SOI.get(target_planet, 0)
    if soi_radius_m > 0:
        soi_radius_au = soi_radius_m / AU
        target_center_au = end_positions[target_planet]
        plot_soi_circle(ax, target_center_au, soi_radius_au, color='cyan', linestyle='--', linewidth=1.5)

    # Легенда
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc='center left', bbox_to_anchor=(1.05, 0.5),
              fontsize='small', framealpha=0.9)

    ax.set_xlim(-5, 5); ax.set_ylim(-5, 5); ax.set_zlim(-5, 5)
    plt.subplots_adjust(bottom=0.1, right=0.85)

    def zoom_to_sun(event):
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_zlim(-5, 5)
        fig.canvas.draw_idle()

    def zoom_to_start(event):
        pos = start_positions[start_planet]
        ax.set_xlim(pos[0] - 0.8, pos[0] + 0.8)
        ax.set_ylim(pos[1] - 0.8, pos[1] + 0.8)
        ax.set_zlim(pos[2] - 0.8, pos[2] + 0.8)
        fig.canvas.draw_idle()


    def zoom_to_target(event):
        pos = end_positions[target_planet]
        ax.set_xlim(pos[0] - 0.8, pos[0] + 0.8)
        ax.set_ylim(pos[1] - 0.8, pos[1] + 0.8)
        ax.set_zlim(pos[2] - 0.8, pos[2] + 0.8)
        fig.canvas.draw_idle()

    ax_btn_sun = plt.axes([0.75, 0.02, 0.12, 0.06])
    btn_sun = Button(ax_btn_sun, 'Сброс к Солнцу')
    btn_sun.on_clicked(zoom_to_sun)

    ax_btn_start = plt.axes([0.45, 0.02, 0.14, 0.06])
    btn_start = Button(ax_btn_start, f'Приблизить к {start_planet.capitalize()}')
    btn_start.on_clicked(zoom_to_start)

    ax_btn_target = plt.axes([0.60, 0.02, 0.14, 0.06])
    btn_target = Button(ax_btn_target, f'Приблизить к {target_planet.capitalize()}')
    btn_target.on_clicked(zoom_to_target)

    plt.show()


if __name__ == "__main__":
    from optimizer import Optimizer
    from constants import get_heliocentric_state
    from datetime import datetime, timedelta

    start_planet = 'earth'
    target_planet = 'mars'
    start_date_str = '2030-01-01 00:00:00'
    accel = 9.8

    start_pos, start_vel = get_heliocentric_state(start_planet, start_date_str)
    opt = Optimizer(accel=accel)

    print("Оптимизация брахистохроны...")
    try:
        res = opt.find_brachistochrone(start_pos, start_vel, target_planet,
                                       date=start_date_str, w_pos=1e-4, w_vel=1e4)
        theta, phi1, phi2, t1, t2 = res.x
    except Exception as e:
        print(f"Error: {e}\nUsing test's parametrs.")
        theta, phi1, phi2 = np.radians(0.0), np.radians(0.0), np.radians(180.0)
        t1 = t2 = 100 * 86400

    total_time = t1 + t2
    _, pos_points, vel_points, _ = opt.brachistochrone_trajectory(
        start_pos, start_vel, theta, phi1, phi2, t1, t2, start_date_str, num_points=500)

    start_dt = datetime.strptime(start_date_str, '%Y-%m-%d %H:%M:%S')
    arrival_dt = start_dt + timedelta(seconds=total_time)
    arrival_date_str = arrival_dt.strftime('%Y-%m-%d %H:%M:%S')
    print(f"Параметры: theta={np.degrees(theta):.1f}°, phi1={np.degrees(phi1):.1f}°, "
          f"phi2={np.degrees(phi2):.1f}°, t1={t1/86400:.2f} дн, t2={t2/86400:.2f} дн")
   
    final_sc_pos = pos_points[:, -1]
    final_sc_vel = vel_points[:, -1]

    target_pos, target_vel = get_heliocentric_state(target_planet, arrival_date_str)

    pos_diff = final_sc_pos - target_pos
    vel_diff = final_sc_vel - target_vel

    print("=" * 60)
    print(f"Длительность: {total_time/86400:.2f} дней ({total_time:.2f} секунд)")
    print(f"Разность по положению:     {np.linalg.norm(pos_diff)/1e3:.2f} км")
    print(f"Разность по скорости:      {np.linalg.norm(vel_diff)/1e3:.2f} км/с")

    visualize_mission(pos_points, start_date_str, total_time,
                      start_planet=start_planet, target_planet=target_planet)
    
