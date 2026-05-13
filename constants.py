import numpy as np
from astropy.time import Time
from astropy.constants import G, M_sun, au
from astropy.coordinates import (get_body_barycentric_posvel,
                                 solar_system_ephemeris,
                                 CartesianRepresentation)
import astropy.units as u

AU = au.value
SUN_MASS = M_sun.value
SUN_RADIUS = 6.957e8

PLANET_DATA = {
    'mercury': {'mass': 3.3011e23, 'semi_major_axis': 5.7909e10, 'radius': 2.4397e6},
    'venus':   {'mass': 4.8675e24, 'semi_major_axis': 1.0821e11, 'radius': 6.0518e6},
    'earth':   {'mass': 5.97237e24, 'semi_major_axis': 1.4960e11, 'radius': 6.3710e6},
    'mars':    {'mass': 6.4171e23, 'semi_major_axis': 2.2794e11, 'radius': 3.3895e6},
    'jupiter': {'mass': 1.8982e27, 'semi_major_axis': 7.7857e11, 'radius': 6.9911e7},
    'saturn':  {'mass': 5.6834e26, 'semi_major_axis': 1.4335e12, 'radius': 5.8232e7},
    'uranus':  {'mass': 8.6810e25, 'semi_major_axis': 2.8725e12, 'radius': 2.5362e7},
    'neptune': {'mass': 1.02413e26, 'semi_major_axis': 4.4951e12, 'radius': 2.4622e7}
}

CELESTIAL_GM = {'sun': G.value * SUN_MASS}
CELESTIAL_RADII = {'sun': SUN_RADIUS}
CELESTIAL_SEMI_MAJOR_AXES = {}
CELESTIAL_SOI = {'sun': np.inf}

for name, data in PLANET_DATA.items():
    CELESTIAL_GM[name] = data['mass'] * G.value
    CELESTIAL_RADII[name] = data['radius']
    CELESTIAL_SEMI_MAJOR_AXES[name] = data['semi_major_axis']
    CELESTIAL_SOI[name] = data['semi_major_axis'] * (data['mass'] / SUN_MASS) ** (2/5)


def get_heliocentric_state(body, date, ephemeris='de432s'):
    if isinstance(date, str):
        time = Time(date, scale='tdb')
    else:
        time = date
    
    if body.lower() == 'sun':
        return np.zeros(3), np.zeros(3)
    
    with solar_system_ephemeris.set(ephemeris):
        body_pos, body_vel = get_body_barycentric_posvel(body, time)
        sun_pos, sun_vel = get_body_barycentric_posvel('sun', time)
        
        body_xyz = body_pos.represent_as(CartesianRepresentation)
        sun_xyz = sun_pos.represent_as(CartesianRepresentation)
        body_vel_xyz = body_vel.represent_as(CartesianRepresentation)
        sun_vel_xyz = sun_vel.represent_as(CartesianRepresentation)
        
        position = np.array([
            (body_xyz.x - sun_xyz.x).to(u.m).value,
            (body_xyz.y - sun_xyz.y).to(u.m).value,
            (body_xyz.z - sun_xyz.z).to(u.m).value
        ])
        
        velocity = np.array([
            (body_vel_xyz.x - sun_vel_xyz.x).to(u.m / u.s).value,
            (body_vel_xyz.y - sun_vel_xyz.y).to(u.m / u.s).value,
            (body_vel_xyz.z - sun_vel_xyz.z).to(u.m / u.s).value
        ])
        
        return position, velocity


def get_relative_state(position, velocity, body, date, ephemeris='de432s'):
    body_pos, body_vel = get_heliocentric_state(body, date, ephemeris)
    rel_pos = position - body_pos
    rel_vel = velocity - body_vel
    return rel_pos, rel_vel


def get_soi(position, date, ephemeris='de432s'):
    closest_body = 'sun'
    min_distance = np.linalg.norm(position)
    
    for planet in PLANET_DATA:
        planet_pos, _ = get_heliocentric_state(planet, date, ephemeris)
        distance = np.linalg.norm(position - planet_pos)
        
        if distance < CELESTIAL_SOI[planet] and distance < min_distance:
            closest_body = planet
            min_distance = distance
    
    return closest_body, min_distance


if __name__ == "__main__":
    date = '2030-01-01 00:00:00'
    
    print("GM, aka mu:")
    for name, gm in CELESTIAL_GM.items():
        print(f"  {name:<10} {gm:.4e}")
    
    print("\nSOI:")
    for name, soi in CELESTIAL_SOI.items():
        print(f"  {name:<10} {soi:.4e}")
    
    earth_pos, earth_vel = get_heliocentric_state('earth', date)
    test_pos = earth_pos + np.array([1e8, 0, 0])
    
    body, dist = get_soi(test_pos, date)
    print(f"\nSOI test (near Earth): {body} ({dist:.4e} m)")
    
    rel_pos, rel_vel = get_relative_state(test_pos, np.zeros(3), 'mars', date)
    print(f"Relative to Mars: {np.linalg.norm(rel_pos):.4e} m")