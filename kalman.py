"""
kalman_filter_fish_tracking.py
 
Smooth / de-noise 2D centroid trajectories (e.g., from TRex fish tracking)
using a Kalman filter + RTS (Rauch-Tung-Striebel) smoother.
 
Why this instead of hard thresholding on speed/accel/angle:
- Hard thresholding deletes frames outright, leaving gaps in the track.
- A Kalman filter models the fish's motion (position + velocity) and
  automatically weighs how much to trust each noisy centroid measurement
  vs. its own prediction. Noisy points get pulled toward the model instead
  of being deleted; only genuinely implausible jumps need to be flagged,
  and those get bridged smoothly by the motion model instead of leaving a
  hole in the data.
 
Requires: numpy, filterpy   (pip install filterpy)
"""
 
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
 
 
def diagnose_track(x, y, label=""):
    """Quick sanity check on a raw track before filtering. Run this first."""
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    n = len(x)
    n_nan = np.isnan(x).sum() + np.isnan(y).sum()
    n_inf = np.isinf(x).sum() + np.isinf(y).sum()
    n_finite = int((np.isfinite(x) & np.isfinite(y)).sum())
    print(f"[{label}] n={n}  finite={n_finite}  nan_vals={n_nan}  inf_vals={n_inf}")
    print(f"[{label}] first 5 x: {x[:5]}   first 5 y: {y[:5]}")
    return n_finite > 0
 
 
def build_cv_kalman(dt, r_std, q_std):
    """
    Constant-velocity Kalman filter for 2D position.
    State:       [x, vx, y, vy]
    Measurement: [x, y]
    """
    kf = KalmanFilter(dim_x=4, dim_z=2)
 
    kf.F = np.array([[1, dt, 0, 0],
                      [0, 1,  0, 0],
                      [0, 0,  1, dt],
                      [0, 0,  0, 1]])
 
    kf.H = np.array([[1, 0, 0, 0],
                      [0, 0, 1, 0]])
 
    kf.R *= r_std ** 2  # measurement noise -> how noisy the raw centroid is (pixels)
 
    q = Q_discrete_white_noise(dim=2, dt=dt, var=q_std ** 2)
    kf.Q = np.zeros((4, 4))
    kf.Q[0:2, 0:2] = q  # x, vx block
    kf.Q[2:4, 2:4] = q  # y, vy block
    # process noise -> how much real acceleration/turning you expect between
    # frames. Raise q_std for burst-swimming fish, lower it for smooth cruising.
 
    kf.P *= 500.0  # initial state uncertainty
    return kf
 
 
def smooth_track(x, y, dt=1 / 30, r_std=2.0, q_std=4.0, gate=None):
    """
    x, y  : 1D arrays, one value per frame. np.nan AND +/-inf are both
            treated as "missing" (masked-out / lost-detection) frames --
            inf is common in tracking data from a division by a near-zero
            dt or velocity/accel calc, and behaves just as destructively
            as NaN if it reaches the filter's internal matrices.
    dt    : seconds between frames (1/fps). If your frame rate is variable
            (dropped frames), pass the actual per-step dt array instead and
            rebuild kf.F / kf.Q inside the loop for each step.
    r_std : expected std of centroid measurement noise, in pixels
    q_std : expected std of "real" acceleration/turning between frames
    gate  : optional Mahalanobis-distance^2 threshold. Measurements whose
            innovation distance exceeds this are treated as outliers and
            skipped for the update step (still predicted through, not
            deleted). For a 2D measurement, 9.21 ~ 99% chi-square threshold,
            5.99 ~ 95%.
 
    Returns
    -------
    x_smooth, y_smooth : smoothed trajectory (same length as input, no gaps)
    flagged            : bool array, True where a point was gated out as an
                          implausible jump
    """
    # --- defensive input handling -------------------------------------
    # A NaN (or a wrong shape like (n,1) from a raw np.load) in the very
    # first sample will silently poison kf.x/kf.P with NaN, and *every*
    # predict()/update() after that stays NaN forever -- matrix ops never
    # "recover" from a NaN entry. This is the #1 cause of an all-NaN output.
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.shape != y.shape:
        raise ValueError(f"x and y must be the same shape, got {x.shape} vs {y.shape}")
 
    n = len(x)
    valid = np.isfinite(x) & np.isfinite(y)  # False for both NaN and +/-inf
    if not valid.any():
        raise ValueError("x/y contain no valid (finite) points to initialize the filter")
    first_valid = int(np.argmax(valid))  # index of the first usable detection
 
    x_smooth = np.full(n, np.nan)
    y_smooth = np.full(n, np.nan)
    flagged = np.zeros(n, dtype=bool)
 
    # Nothing can be estimated before the first real detection -> leave NaN.
    x_sub, y_sub = x[first_valid:], y[first_valid:]
    m = len(x_sub)
 
    kf = build_cv_kalman(dt, r_std, q_std)
    kf.x = np.array([x_sub[0], 0.0, y_sub[0], 0.0])  # seeded from a confirmed-valid point
 
    mu, cov = [], []
    for i in range(m):
        if i == 0:
            # already used this measurement to seed the state; just record it
            mu.append(kf.x.copy())
            cov.append(kf.P.copy())
            continue
 
        kf.predict()
 
        z_missing = not (np.isfinite(x_sub[i]) and np.isfinite(y_sub[i]))
 
        if not z_missing and gate is not None:
            z = np.array([x_sub[i], y_sub[i]])
            innovation = z - kf.H @ kf.x
            S = kf.H @ kf.P @ kf.H.T + kf.R
            d2 = innovation @ np.linalg.inv(S) @ innovation
            if d2 > gate:
                z_missing = True
                flagged[first_valid + i] = True
 
        if not z_missing:
            kf.update(np.array([x_sub[i], y_sub[i]]))
        # else: no update this step -> pure prediction bridges the gap
 
        mu.append(kf.x.copy())
        cov.append(kf.P.copy())
 
    mu = np.array(mu)
    cov = np.array(cov)
 
    # RTS smoother: re-runs backward through the sequence so every point's
    # estimate uses BOTH past and future data -> the best offline estimate.
    mu_smooth, _, _, _ = kf.rts_smoother(mu, cov)
 
    x_smooth[first_valid:] = mu_smooth[:, 0]
    y_smooth[first_valid:] = mu_smooth[:, 2]
    return x_smooth, y_smooth, flagged
 
 
if __name__ == "__main__":
    # --- using your own data ------------------------------------------
    import matplotlib.pyplot as plt  # needed if you call plt.plot() below
    
    noisy_x = np.load('/run/media/hck03/Vivek/conditioning_experiments_cassandre_videos/tracked/1c_4n_faulty/Clip0070_1C_4N_E3_8/data/Clip0070_1C_4N_E3_8_id0/X#wcentroid.npy')
    noisy_y = np.load('/run/media/hck03/Vivek/conditioning_experiments_cassandre_videos/tracked/1c_4n_faulty/Clip0070_1C_4N_E3_8/data/Clip0070_1C_4N_E3_8_id0/Y#wcentroid.npy')
    
    diagnose_track(noisy_x, noisy_y, label="raw")  # check nan/inf counts first
    
    xs, ys, flagged = smooth_track(
        noisy_x, noisy_y, dt=1 / 100, r_std=1.0, q_std=1.0, gate=5.0
    )
    print(f"Flagged {flagged.sum()} / {len(noisy_x)} frames as implausible jumps")
    plt.plot(noisy_x, noisy_y, color='blue', alpha=0.5)
    plt.plot(xs, ys, color='black')
    plt.show()
 