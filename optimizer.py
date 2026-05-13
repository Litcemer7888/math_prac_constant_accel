import numpy as np
from scipy.integrate import solve_ivp
from constants import (CELESTIAL_GM, CELESTIAL_SOI, CELESTIAL_RADII,
                       get_heliocentric_state, get_soi)

class Optimizer:
    def __init__(self, accel, mode='free', 
                 integ_method='DOP853', rtol=1e-10, atol=1e-6,
                 max_step=np.inf):
        self.accel = accel
        self.mode = mode
        self.GM_sun = CELESTIAL_GM['sun']
        
        self.integ_method = integ_method
        self.rtol = rtol
        self.atol = atol
        self.max_step = max_step
    
    def _integrate_heliocentric(self, start_pos, start_vel, accel_vector, burn_time):
        """Integrate equations of motion with Sun gravity. Cuts at solar radius."""
        def eq_motion(t, state):
            pos = state[:3]
            vel = state[3:]
            
            r = np.linalg.norm(pos)
            if r <= CELESTIAL_RADII['sun']:
                a_gravity = np.zeros(3)
            else:
                a_gravity = -self.GM_sun * pos / r**3
            
            a_total = accel_vector + a_gravity
            return np.concatenate([vel, a_total])
        
        def hit_sun(t, state):
            return np.linalg.norm(state[:3]) - CELESTIAL_RADII['sun']
        hit_sun.terminal = True
        hit_sun.direction = -1
        
        initial_state = np.concatenate([start_pos, start_vel])
        sol = solve_ivp(eq_motion, [0, burn_time], initial_state,
                       method=self.integ_method, rtol=self.rtol, atol=self.atol,
                       max_step=self.max_step, dense_output=True, events=hit_sun)
        
        return sol
    
    def _integrate_planetocentric(self, start_rel_pos, start_rel_vel, accel_vector, 
                                  burn_time, body):
        """Integrate in planet-centered frame with planet gravity only. Cuts at planet radius."""
        gm = CELESTIAL_GM[body]
        radius = CELESTIAL_RADII[body]
        
        def eq_motion(t, state):
            pos = state[:3]
            vel = state[3:]
            
            r = np.linalg.norm(pos)
            if r <= radius:
                a_gravity = np.zeros(3)
            else:
                a_gravity = -gm * pos / r**3
            
            a_total = accel_vector + a_gravity
            return np.concatenate([vel, a_total])
        
        def hit_body(t, state):
            return np.linalg.norm(state[:3]) - radius
        hit_body.terminal = True
        hit_body.direction = -1
        
        initial_state = np.concatenate([start_rel_pos, start_rel_vel])
        sol = solve_ivp(eq_motion, [0, burn_time], initial_state,
                       method=self.integ_method, rtol=self.rtol, atol=self.atol,
                       max_step=self.max_step, dense_output=True, events=hit_body)
        
        return sol
    
    def _integrate_soi(self, start_pos, start_vel, accel_vector, burn_time, date):
        """
        Patched conic integration. Uses heliocentric frame outside planet SOIs,
        switches to planetocentric inertial frame inside SOIs.
        Returns concatenated trajectory segments in heliocentric frame.
        """
        remaining_time = burn_time
        current_pos = start_pos.copy()
        current_vel = start_vel.copy()
        
        all_pos = [current_pos.copy()]
        all_vel = [current_vel.copy()]
        all_t = [0.0]
        
        while remaining_time > 0:
            body, dist_to_body = get_soi(current_pos, date)
            
            if body == 'sun':
                sol = self._integrate_heliocentric(current_pos, current_vel, 
                                                   accel_vector, remaining_time)
                
                all_t.extend(all_t[-1] + sol.t[1:])
                all_pos.extend(sol.y[:3, 1:].T)
                all_vel.extend(sol.y[3:, 1:].T)
                
                if sol.t_events and sol.t_events[0].size > 0:
                    break
                else:
                    break
            
            else:
                body_pos, body_vel = get_heliocentric_state(body, date)
                
                rel_pos = current_pos - body_pos
                rel_vel = current_vel - body_vel
                
                soi_radius = CELESTIAL_SOI[body]
                escape_speed = np.sqrt(2 * CELESTIAL_GM[body] / max(soi_radius, 1e-10))
                max_soi_time = min(remaining_time, 2 * soi_radius / max(escape_speed, 1e-10))
                
                sol = self._integrate_planetocentric(rel_pos, rel_vel, 
                                                     accel_vector, max_soi_time, body)
                
                new_rel_pos = sol.y[:3, -1]
                new_rel_vel = sol.y[3:, -1]
                
                new_pos = new_rel_pos + body_pos
                new_vel = new_rel_vel + body_vel
                
                all_t.extend(all_t[-1] + sol.t[1:])
                all_pos.extend((sol.y[:3, 1:].T + body_pos).tolist())
                all_vel.extend((sol.y[3:, 1:].T + body_vel).tolist())
                
                still_in_soi = np.linalg.norm(new_rel_pos) < CELESTIAL_SOI[body]
                
                if sol.t_events and sol.t_events[0].size > 0:
                    break
                elif still_in_soi:
                    remaining_time -= max_soi_time
                    current_pos = new_pos
                    current_vel = new_vel
                else:
                    remaining_time -= sol.t[-1]
                    current_pos = new_pos
                    current_vel = new_vel
        
        return np.array(all_t), np.array(all_pos).T, np.array(all_vel).T
    
    def maneuver(self, start_pos, start_vel, direction, burn_time,
                 accel_modulus=None, date=None):
        """
        Returns end state after constant-acceleration burn.
        Input: start_pos, start_vel, direction, burn_time [s]
        Output: end_pos, end_vel
        """
        t, pos, vel = self.trajectory(start_pos, start_vel, direction, burn_time,
                                      accel_modulus, date)
        return pos[:, -1], vel[:, -1]
    
    def trajectory(self, start_pos, start_vel, direction, burn_time,
                   accel_modulus=None, date=None):
        """
        Returns trajectory points during constant-acceleration burn.
        Input: start_pos, start_vel, direction, burn_time
        Output: t, pos, vel
        """
        if accel_modulus is None:
            accel_modulus = self.accel
        
        dir_norm = np.linalg.norm(direction)
        if dir_norm == 0:
            return (np.array([0, burn_time]),
                    np.column_stack([start_pos, start_pos]),
                    np.column_stack([start_vel, start_vel]))
        
        dir_unit = direction / dir_norm
        accel_vector = accel_modulus * dir_unit
        
        if self.mode == 'free':
            t = np.linspace(0, burn_time, 2)
            pos = start_pos[:, np.newaxis] + start_vel[:, np.newaxis] * t + \
                  0.5 * accel_vector[:, np.newaxis] * t**2
            vel = start_vel[:, np.newaxis] + accel_vector[:, np.newaxis] * t
            return t, pos, vel
        
        elif self.mode == 'sun_only':
            sol = self._integrate_heliocentric(start_pos, start_vel, 
                                               accel_vector, burn_time)
            return sol.t, sol.y[:3, :], sol.y[3:, :]
        
        elif self.mode == 'SOI':
            return self._integrate_soi(start_pos, start_vel, 
                                      accel_vector, burn_time, date)
        
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented")
    
    def set_mode(self, mode):
        self.mode = mode
    
    def set_accel(self, accel):
        self.accel = accel


if __name__ == "__main__":
    date = '2000-01-01 00:00:00'
    earth_pos, earth_vel = get_heliocentric_state('earth', date)
    
    start_pos = earth_pos + np.array([2 * CELESTIAL_RADII['earth'], 0, 0])
    start_vel = earth_vel.copy()
    
    direction = np.array([0.0, 1.0, 0.0])
    burn_time = 3600
    
    for mode in ['free', 'sun_only', 'SOI']:
        opt = Optimizer(accel=10.0, mode=mode)
        end_pos, end_vel = opt.maneuver(start_pos, start_vel, direction, burn_time, 
                                        date=date)
        
        dr = np.linalg.norm(end_pos - start_pos)
        dv = np.linalg.norm(end_vel - start_vel)
        print(f"{mode:<10} dr={dr:.4e} m  dv={dv:.4e} m/s")