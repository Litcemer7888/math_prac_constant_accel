import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from constants import (CELESTIAL_GM, CELESTIAL_SOI, CELESTIAL_RADII,
                       get_heliocentric_state)

class Optimizer:
    def __init__(self, accel, integ_method='DOP853', rtol=1e-10, atol=1e-6,
                 max_step=np.inf, soi_margin=0.9):
        self.accel = accel
        self.integ_method = integ_method
        self.rtol = rtol
        self.atol = atol
        self.max_step = max_step
        self.soi_margin = soi_margin

    def _determine_soi(self, heliocentric_pos, date):
        """Return the body whose SOI contains the given heliocentric position."""
        for planet in CELESTIAL_SOI:
            if planet == 'sun':
                continue
            planet_pos, _ = get_heliocentric_state(planet, date)
            if np.linalg.norm(heliocentric_pos - planet_pos) < CELESTIAL_SOI[planet]:
                return planet
        return 'sun'

    def _convert_frame(self, pos, vel, from_body, to_body, date):
        """Convert state between body-relative frames via heliocentric."""
        if from_body == to_body:
            return pos.copy(), vel.copy()
        from_pos, from_vel = get_heliocentric_state(from_body, date)
        to_pos, to_vel = get_heliocentric_state(to_body, date)
        helio_pos = pos + from_pos
        helio_vel = vel + from_vel
        return helio_pos - to_pos, helio_vel - to_vel

    def _gravity_accel(self, pos_rel, body):
        """Gravitational acceleration in body-centered frame. Zero inside radius."""
        if body == 'neutral':
            return np.zeros(3)
        r = np.linalg.norm(pos_rel)
        if r <= CELESTIAL_RADII[body]:
            return np.zeros(3)
        return -CELESTIAL_GM[body] * pos_rel / r**3

    def _integrate_soi_segment(self, pos, vel, accel, duration, body):
        """Integrate inside a fixed SOI for a given duration. No boundary events."""
        def rhs(t, state):
            r = state[:3]
            v = state[3:]
            a_grav = self._gravity_accel(r, body)
            return np.concatenate([v, accel + a_grav])

        sol = solve_ivp(rhs, [0, duration], np.concatenate([pos, vel]),
                        method=self.integ_method, rtol=self.rtol, atol=self.atol,
                        max_step=min(self.max_step, duration/10),
                        dense_output=False)
        return sol

    def _closest_soi_crossing_time(self, helio_pos, helio_vel, date):
        """Linear estimate of minimum time to cross any planet's SOI boundary."""
        min_time = np.inf
        for planet in CELESTIAL_SOI:
            if planet == 'sun':
                continue
            p_pos, p_vel = get_heliocentric_state(planet, date)
            rel_pos = helio_pos - p_pos
            dist = np.linalg.norm(rel_pos)
            soi_r = CELESTIAL_SOI[planet]

            if dist > 0:
                v_close = -np.dot(rel_pos, helio_vel - p_vel) / dist
            else:
                v_close = 0.0

            if v_close > 0 and dist > soi_r:
                t_cross = (dist - soi_r) / v_close
                if t_cross < min_time:
                    min_time = t_cross
        return min_time

    def _safe_duration(self, pos_rel, vel_rel, body, date, remaining):
        """Maximum time step guaranteed not to cross an SOI boundary."""
        if body == 'sun':
            t_cross = self._closest_soi_crossing_time(pos_rel, vel_rel, date)
            safe = min(remaining, t_cross * self.soi_margin) if np.isfinite(t_cross) else remaining
        else:
            soi_r = CELESTIAL_SOI[body]
            dist = np.linalg.norm(pos_rel)
            if dist > 0:
                v_radial = np.dot(pos_rel, vel_rel) / dist
            else:
                v_radial = 0.0
            if v_radial > 0 and dist < soi_r:
                t_exit = (soi_r - dist) / v_radial
                safe = min(remaining, t_exit * self.soi_margin)
            else:
                safe = remaining
        return max(safe, 1e-3)

    def _integrate_with_soi_switching(self, start_pos, start_vel, accel,
                                      burn_time, date, store_trajectory=True):
        """Integrate with automatic frame switching at SOI boundaries."""
        current_pos = np.asarray(start_pos, dtype=float)
        current_vel = np.asarray(start_vel, dtype=float)
        current_body = self._determine_soi(current_pos, date)
        all_t = [0.0]
        all_pos_helio = [current_pos.copy() if current_body == 'sun' else 
                         self._convert_frame(current_pos, current_vel, current_body, 'sun', date)[0]]
        all_vel_helio = [current_vel.copy() if current_body == 'sun' else 
                         self._convert_frame(current_pos, current_vel, current_body, 'sun', date)[1]]
        all_bodies = [current_body]

        remaining = burn_time

        while remaining > 0:
            dt = self._safe_duration(current_pos, current_vel,
                                     current_body, date, remaining)

            sol = self._integrate_soi_segment(current_pos, current_vel,
                                              accel, dt, current_body)
            seg_t = sol.t[1:]
            seg_pos = sol.y[:3, 1:]
            seg_vel = sol.y[3:, 1:]

            if store_trajectory:
                for i in range(len(seg_t)):
                    t_now = all_t[-1] + seg_t[i]
                    pos_now_rel = seg_pos[:, i]
                    vel_now_rel = seg_vel[:, i]
                    if current_body == 'sun':
                        pos_helio = pos_now_rel
                        vel_helio = vel_now_rel
                    else:
                        pos_helio, vel_helio = self._convert_frame(
                            pos_now_rel, vel_now_rel, current_body, 'sun', date)
                    all_t.append(t_now)
                    all_pos_helio.append(pos_helio)
                    all_vel_helio.append(vel_helio)
                    all_bodies.append(current_body)

            last_pos_rel = sol.y[:3, -1]
            last_vel_rel = sol.y[3:, -1]
            if current_body == 'sun':
                last_pos_helio = last_pos_rel
                last_vel_helio = last_vel_rel
            else:
                last_pos_helio, last_vel_helio = self._convert_frame(
                    last_pos_rel, last_vel_rel, current_body, 'sun', date)

            remaining -= dt

            if remaining <= 0:
                break

            new_body = self._determine_soi(last_pos_helio, date)
            if new_body != current_body:
                current_pos, current_vel = self._convert_frame(
                    last_pos_helio, last_vel_helio, 'sun', new_body, date)
                current_body = new_body
            else:
                current_pos = last_pos_rel if current_body == 'sun' else last_pos_helio
                current_vel = last_vel_rel if current_body == 'sun' else last_vel_helio

        if store_trajectory:
            return (np.array(all_t), np.array(all_pos_helio).T,
                    np.array(all_vel_helio).T, all_bodies)
        else:
            return last_pos_helio, last_vel_helio

    def trajectory(self, start_pos, start_vel, direction, burn_time,
                   accel_modulus=None, date=None):
        """Return full trajectory in heliocentric coordinates."""
        if accel_modulus is None:
            accel_modulus = self.accel
        direction = np.asarray(direction, dtype=float)
        norm = np.linalg.norm(direction)
        if norm == 0:
            return (np.array([0, burn_time]),
                    np.column_stack([start_pos, start_pos]),
                    np.column_stack([start_vel, start_vel]),
                    ['sun', 'sun'])
        accel_vector = accel_modulus * direction / norm

        start_body = self._determine_soi(start_pos, date)
        pos_rel, vel_rel = self._convert_frame(start_pos, start_vel,
                                               'sun', start_body, date)
        return self._integrate_with_soi_switching(pos_rel, vel_rel,
                                                  accel_vector, burn_time, date,
                                                  store_trajectory=True)

    def maneuver(self, start_pos, start_vel, direction, burn_time,
                 accel_modulus=None, date=None):
        """Return final state (heliocentric) after burn. Stores full trajectory internally."""
        t, pos, vel, bodies = self.trajectory(start_pos, start_vel, direction,
                                              burn_time, accel_modulus, date)
        last_body = bodies[-1]
        if last_body != 'sun':
            end_pos, end_vel = self._convert_frame(pos[:, -1], vel[:, -1],
                                                   last_body, 'sun', date)
        else:
            end_pos, end_vel = pos[:, -1], vel[:, -1]
        return end_pos, end_vel

    def maneuver_fast(self, start_pos, start_vel, direction, burn_time,
                      accel_modulus=None, date=None):
        """Return final state (heliocentric) without storing trajectory."""
        if accel_modulus is None:
            accel_modulus = self.accel
        direction = np.asarray(direction, dtype=float)
        norm = np.linalg.norm(direction)
        if norm == 0:
            return start_pos.copy(), start_vel.copy()
        accel_vector = accel_modulus * direction / norm

        start_body = self._determine_soi(start_pos, date)
        pos_rel, vel_rel = self._convert_frame(start_pos, start_vel,
                                               'sun', start_body, date)
        end_pos_helio, end_vel_helio = self._integrate_with_soi_switching(
            pos_rel, vel_rel, accel_vector, burn_time, date,
            store_trajectory=False)
        return end_pos_helio, end_vel_helio

    def brachistochrone_endpoint(self, start_pos, start_vel, theta, phi1, phi2,
                                 t1, t2, date=None):
        """
        Final state after boost-brake brachistochrone.
        """
        u1 = np.array([np.cos(theta) * np.cos(phi1),
                       np.cos(theta) * np.sin(phi1),
                       np.sin(theta)])
        pos1, vel1 = self.maneuver_fast(start_pos, start_vel, u1, t1, date=date)

        u2 = np.array([np.cos(theta) * np.cos(phi2),
                       np.cos(theta) * np.sin(phi2),
                       np.sin(theta)])
        pos2, vel2 = self.maneuver_fast(pos1, vel1, -u2, t2, date=date)
        return pos2, vel2

    def guess_brachistochrone(self, start_pos, start_vel, target_pos,
                            target_vel=None, date=None):
        """
        Improved initial guess for brachistochrone parameters.
        """
        rel_pos = target_pos - start_pos
        dist = np.linalg.norm(rel_pos)

        if target_vel is None:
            target_vel = np.zeros(3)

        dv = target_vel - start_vel
        dv_mag = np.linalg.norm(dv)

        T_total = 2.0 * np.sqrt(dist / self.accel) if self.accel > 0 else 1e6

        a = self.accel
        if a > 0:
            t1 = (T_total / 2.0) + (dv_mag / (2.0 * a))
            t2 = (T_total / 2.0) - (dv_mag / (2.0 * a))
            if t2 <= 0:
                t2 = 1e-3
                t1 = (dv_mag / a) + t2
        else:
            t1 = t2 = 1e6

        if dv_mag > 0:
            u = dv / dv_mag
        else:
            if dist > 0:
                u = rel_pos / dist
            else:
                u = np.array([1.0, 0.0, 0.0])

        theta = np.arcsin(np.clip(u[2], -1.0, 1.0))
        phi1 = np.arctan2(u[1], u[0])
        phi2 = phi1 + np.pi
        if phi2 > np.pi:
            phi2 -= 2.0 * np.pi

        return theta, phi1, phi2, t1, t2

    def cost_function(self, pos, vel, target_pos, target_vel=None,
                      w_pos=1.0, w_vel=0.0):
        """
        Combined position and velocity error.
        """
        cost = w_pos * np.linalg.norm(pos - target_pos)
        if target_vel is not None:
            cost += w_vel * np.linalg.norm(vel - target_vel)
        return cost

    def find_brachistochrone(self, start_pos, start_vel, target_pos, target_vel=None,
                            date=None, initial_guess=None,
                            w_pos=1.0, w_vel=0.0,
                            method='L-BFGS-B', tol=1e-6,
                            options=None):
        """
        Optimise brachistochrone parameters to reach a target state.
        """
        if initial_guess is None:
            initial_guess = self.guess_brachistochrone(
                start_pos, start_vel, target_pos, target_vel, date)

        def objective(params):
            theta, phi1, phi2, t1, t2 = params
            if t1 <= 0 or t2 <= 0:
                return 1e20
            pos, vel = self.brachistochrone_endpoint(
                start_pos, start_vel, theta, phi1, phi2, t1, t2, date)
            return self.cost_function(pos, vel, target_pos, target_vel, w_pos, w_vel)

        bounds = [(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi),
                  (1e3, None), (1e3, None)]

        default_opts = {'maxiter': 1000, 'xatol': tol, 'fatol': tol}
        if options is not None:
            default_opts.update(options)

        result = minimize(objective, initial_guess,
                        method=method, bounds=bounds,
                        options=default_opts)
        return result


if __name__ == "__main__":
    date = '2000-01-01 00:00:00'
    earth_pos, earth_vel = get_heliocentric_state('earth', date)
    mars_pos, mars_vel = get_heliocentric_state('mars', date)

    opt = Optimizer(accel=0.5)
    res = opt.find_brachistochrone(earth_pos, earth_vel, mars_pos, mars_vel,
                                   date=date, w_pos=1.0, w_vel=0.01)

    theta, phi1, phi2, t1, t2 = res.x
    print(f"Optimised: theta={np.degrees(theta):.1f}°, phi1={np.degrees(phi1):.1f}°, "
          f"phi2={np.degrees(phi2):.1f}°, t1={t1/86400:.2f}d, t2={t2/86400:.2f}d")
    print(f"Final cost: {res.fun:.2f}")
