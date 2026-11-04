"""Shared HKUST material speeds and source-independent TSNet grid settings.

The original CSV nominal diameters use the K9 ductile-iron wall table below.
The graph prior and hydraulic solver share these assumptions. Neither function
uses an event source label, arrival pick, or pressure trace.
"""
import numpy as np

DEFAULT_DT_S = .00104045
DEFAULT_DURATION_S = 3.0
BURST_COEFFICIENT = .001 / np.sqrt(40.)
DARCY_FRICTION_FACTOR = .02


def ductile_iron_wave_speed(diameter_mm):
    """Water K=2.1 GPa, rho=1000 kg/m3; iron E=165 GPa, nu=0.3."""
    table = np.asarray([
        [80,98,6.0],[100,118,6.0],[150,170,6.0],[200,222,6.3],
        [250,274,6.8],[300,326,7.2],[350,378,7.7],[400,429,8.1],
        [450,480,8.6],[500,532,9.0],[600,635,9.9],[700,738,10.8],
        [800,842,11.7],[900,945,12.6],[1000,1048,13.5],
        [1100,1152,14.4],[1200,1255,15.3]], dtype=float)
    dn = np.asarray(diameter_mm, dtype=float)
    if not np.isfinite(dn).all() or np.any((dn < table[0,0]) | (dn > table[-1,0])):
        raise ValueError('Nominal diameter must fall within the 80 to 1200 mm wall table.')
    outer = np.interp(dn, table[:,0], table[:,1])
    wall = np.interp(dn, table[:,0], table[:,2])
    inner = outer - 2*wall
    return (1000 * (1/2.1e9 + (1-.3**2)*(inner/wall)/165e9))**-.5


def nominal_pipe_speeds(wn, wave_speed=None):
    """Nominal m/s in the model's pipe order; optional uniform override."""
    pipes = list(wn.pipes())
    if wave_speed is not None:
        speed = float(wave_speed)
        if not np.isfinite(speed) or speed <= 0:
            raise ValueError('The uniform wave-speed override must be positive and finite.')
        return np.full(len(pipes), speed)
    return ductile_iron_wave_speed([1000*float(pipe.diameter) for _,pipe in pipes])


def discretize(model, duration=DEFAULT_DURATION_S, dt=DEFAULT_DT_S, wave_speed=None):
    """Set a CFL-safe MOC grid and return nominal/effective speed arrays.

    Apply to the unsplit network for the prior, and independently to each
    split hydraulic case. Event-specific adjusted speeds never enter training.
    """
    from tsnet.network.discretize import adjust_wavev, cal_N, max_time_step
    if not np.isfinite([duration, dt]).all() or min(duration, dt) <= 0:
        raise ValueError('Duration and requested time step must be positive and finite.')
    nominal = nominal_pipe_speeds(model, wave_speed)
    for (_, pipe), speed in zip(model.pipes(), nominal):
        pipe.wavev = float(speed)
    safe_dt = min(float(dt), .95*float(max_time_step(model)))
    segments = cal_N(model, safe_dt)
    for i, (_, pipe) in enumerate(model.pipes()):
        pipe.number_of_segments = int(np.asarray(segments[i]).item())
    model.simulation_period = float(duration)
    adjust_wavev(model)
    model.time_step = float(np.asarray(model.time_step).reshape(-1)[0])
    effective = []
    for _, pipe in model.pipes():
        pipe.wavev = float(np.asarray(pipe.wavev).reshape(-1)[0])
        effective.append(pipe.wavev)
    return {'nominal_speeds': nominal, 'effective_speeds': np.asarray(effective),
            'actual_dt': model.time_step, 'requested_dt': float(dt),
            'cfl_limited_dt': safe_dt}
