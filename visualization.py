# visualization.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button
from astropy.time import Time
import astropy.units as u

from constants import get_heliocentric_state, PLANET_DATA, AU

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
        raise ValueError(f"Неверная форма: {points.shape}")

# ---------- Константы визуализации ----------
PLANET_COLORS = {
    'mercury': 'gray', 'venus': 'orange', 'earth': 'blue', 'mars': 'red',
    'jupiter': 'brown', 'saturn': 'gold', 'uranus': 'lightblue', 'neptune': 'darkblue', 'sun': 'yellow'
}
PLANET_SIZES = {
    'mercury': 30, 'venus': 40, 'earth': 45, 'mars': 35,
    'jupiter': 90, 'saturn': 80, 'uranus': 55, 'neptune': 50, 'sun': 150
}

# ---------- Вспомогательные функции ----------
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

def plot_orbit_circular(ax, a_au, color='gray', linestyle='--', linewidth=0.5, alpha=0.5):
    theta = np.linspace(0, 2*np.pi, 300)
    ax.plot(a_au * np.cos(theta), a_au * np.sin(theta), 0,
            color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha)

def plot_orbits(ax, planets_list):
    for name in planets_list:
        if name == 'sun': continue
        a_au = PLANET_DATA[name]['semi_major_axis'] / AU
        plot_orbit_circular(ax, a_au, color=PLANET_COLORS.get(name, 'gray'),
                            linestyle='--', linewidth=0.5, alpha=0.4)

def plot_planets(ax, positions, alpha=1.0, label=None):
    for name, pos in positions.items():
        ax.scatter(pos[0], pos[1], pos[2],
                   color=PLANET_COLORS.get(name, 'white'), s=PLANET_SIZES.get(name, 30),
                   alpha=alpha, label=label if label else "")

def plot_trajectory(ax, trajectory_m, color='red', linewidth=2, label='Траектория корабля', apply_rot=True):
    traj_m = np.asarray(trajectory_m)
    if apply_rot:
        traj_m = apply_rotation(traj_m)
    traj_au = traj_m / AU
    ax.plot(traj_au[0, :], traj_au[1, :], traj_au[2, :],
            color=color, linewidth=linewidth, label=label)
    ax.scatter(traj_au[0, 0], traj_au[1, 0], traj_au[2, 0],
               color='lime', s=80, marker='o', label='Старт корабля', edgecolors='darkgreen')
    ax.scatter(traj_au[0, -1], traj_au[1, -1], traj_au[2, -1],
               color='red', s=80, marker='s', label='Финиш корабля', edgecolors='darkred')

# ---------- Основная функция визуализации ----------
def visualize_mission(trajectory_m, start_date_str, flight_time_sec,
                      start_planet='earth', target_planet='mars'):
    start_time = Time(start_date_str, scale='tdb')
    end_time = start_time + flight_time_sec * u.s
    end_date_str = end_time.iso

    print("=" * 60)
    print(f"Миссия: {start_planet.capitalize()} → {target_planet.capitalize()}")
    print(f"Старт: {start_date_str}")
    print(f"Финиш: {end_date_str}")
    print(f"Длительность: {flight_time_sec / 86400:.2f} дней")
    print(f"Дальность: {np.linalg.norm(trajectory_m[:,-1] - trajectory_m[:,0]) / AU:.5f} AU")
    print("=" * 60)

    start_positions = get_planet_positions_at_time(start_date_str, [start_planet], apply_rot=True)
    end_positions   = get_planet_positions_at_time(end_date_str, [target_planet], apply_rot=True)

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"{start_planet.capitalize()} → {target_planet.capitalize()}\n{start_date_str[:10]} → {end_date_str[:10]}",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("X (AU)"); ax.set_ylabel("Y (AU)"); ax.set_zlabel("Z (AU)")

    # Солнце
    ax.scatter(0, 0, 0, color='yellow', s=PLANET_SIZES['sun'], label='Солнце', edgecolors='orange')
    # Орбиты
    plot_orbits(ax, [start_planet, target_planet])
    # Планеты
    plot_planets(ax, start_positions, alpha=1.0, label=f"{start_planet.capitalize()} (старт)")
    plot_planets(ax, end_positions, alpha=0.4, label=f"{target_planet.capitalize()} (финиш)")
    # Траектория
    plot_trajectory(ax, trajectory_m, apply_rot=True)

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

    def zoom_to_target(event):
        pos = end_positions[target_planet]
        ax.set_xlim(pos[0] - 0.8, pos[0] + 0.8)
        ax.set_ylim(pos[1] - 0.8, pos[1] + 0.8)
        ax.set_zlim(pos[2] - 0.8, pos[2] + 0.8)
        fig.canvas.draw_idle()

    ax_btn_sun = plt.axes([0.75, 0.02, 0.12, 0.06])
    btn_sun = Button(ax_btn_sun, 'Сброс к Солнцу')
    btn_sun.on_clicked(zoom_to_sun)

    ax_btn_target = plt.axes([0.60, 0.02, 0.14, 0.06])
    btn_target = Button(ax_btn_target, f'Приблизить к {target_planet.capitalize()}')
    btn_target.on_clicked(zoom_to_target)

    plt.show()


# ---------- Пример использования с оптимизатором брахистохроны ----------
if __name__ == "__main__":
    from optimizer import Optimizer
    from constants import get_heliocentric_state
    from datetime import datetime, timedelta

    start_planet = 'earth'
    target_planet = 'mars'
    start_date_str = '2030-01-01 00:00:00'
    accel = 0.5  # м/с²

    start_pos, start_vel = get_heliocentric_state(start_planet, start_date_str)
    opt = Optimizer(accel=accel)

    print("Оптимизация брахистохроны...")
    try:
        res = opt.find_brachistochrone(start_pos, start_vel, target_planet,
                                       date=start_date_str, w_pos=1e-4, w_vel=1e4)
        theta, phi1, phi2, t1, t2 = res.x
    except Exception as e:
        print(f"Ошибка: {e}\nИспользуем тестовые параметры.")
        theta, phi1, phi2 = np.radians(0.0), np.radians(0.0), np.radians(180.0)
        t1 = t2 = 100 * 86400  # 100 дней

    total_time = t1 + t2
    # Получаем траекторию (метры)
    _, pos_points, _, _ = opt.brachistochrone_trajectory(
        start_pos, start_vel, theta, phi1, phi2, t1, t2, start_date_str, num_points=500)

    # Вычисляем дату прибытия (только для справки)
    start_dt = datetime.strptime(start_date_str, '%Y-%m-%d %H:%M:%S')
    arrival_dt = start_dt + timedelta(seconds=total_time)
    print(f"Параметры: theta={np.degrees(theta):.1f}°, phi1={np.degrees(phi1):.1f}°, "
          f"phi2={np.degrees(phi2):.1f}°, t1={t1/86400:.2f} дн, t2={t2/86400:.2f} дн")
    print(f"Полёт: {total_time/86400:.2f} дней, финиш {arrival_dt.strftime('%Y-%m-%d %H:%M:%S')}")

    visualize_mission(pos_points, start_date_str, total_time,
                      start_planet=start_planet, target_planet=target_planet)
