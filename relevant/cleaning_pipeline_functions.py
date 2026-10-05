import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import pandas as pd
from scipy.ndimage import uniform_filter1d
# from skimage.filters import apply_hysteresis_threshold
from IPython.display import HTML
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

# A function to plot speedy trails
def speedy_trail(x, y, ax=None, label='Speed', cmap='viridis'):
    vel_x = np.gradient(x)
    vel_y = np.gradient(y)
    vel = np.zeros((x.shape[0], 2))
    vel[:,0] = vel_x
    vel[:,1] = vel_y

    speed_array = np.linalg.norm(vel, axis=1, ord=2)
    l = 10
    if ax is None:
        fig, ax = plt.subplots(figsize = (l, l*(20/50)))
    else:
        fig = ax.get_figure()

    segment_speeds = speed_array[:-1]
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap='viridis')
    lc.set_array(segment_speeds)
    lc.set_linewidth(2.5)
    line = ax.add_collection(lc)

    fig.colorbar(line, ax=ax, label="speed")
    ax.autoscale()

    return ax

# A function to return an array of erraticness scores
def filter(x, y, k=7):
    positions = np.vstack((x, y)) # (2, 1882)
    diff = np.diff(positions, axis=1)
    d = np.linalg.norm(diff, axis=0, ord=2)
    # plt.hist(d)
    med = np.nanmedian(d)
    mad = np.nanmedian(np.abs(d - med))
    S = max(med + 1.4826*mad, 1e-9)

    # Step-size factor
    A = np.minimum(d[:-1], d[1:])/S
    A = np.minimum(A, 3)

    # reversal factor
    dot_prod = np.sum(diff[:,1:]*diff[:,:-1], axis=0)
    denominator = d[:-1]*d[1:]
    valid_step = ~np.isnan(denominator)

    safe_mask = (denominator > 0)
    cos_theta = np.zeros(dot_prod.shape[0])
    cos_theta[safe_mask] = dot_prod[safe_mask]/denominator[safe_mask]
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    R = np.full(dot_prod.shape, np.nan)
    R[valid_step] = 0.0                              # zero-length step: not reversing
    R[safe_mask] = (1 - cos_theta[safe_mask]) / 2

    # erratic score
    e = A*R
    e_full = np.full(x.shape[0], np.nan) # padding to keep array size consistent
    e_full[1:-1] = e

    # e mean
    # k = 7
    valid = ~np.isnan(e_full)
    num = uniform_filter1d(np.where(valid, e_full, 0.0), size=k, mode='constant', cval=0.0)
    den = uniform_filter1d(valid.astype(float), size=k, mode='constant', cval=0.0)

    E = np.full(x.shape[0], np.nan)
    E[den > 0] = num[den > 0] / den[den > 0]

    E_filled = np.nan_to_num(E, nan=0.0) 
    return E_filled

def dynamic_percentile_box(x, y, p_low=1, p_high=99, pad_fraction=0.05):
    # 1. Find the true edges of the track, ignoring extreme outliers (top/bottom 1%)
    x_min_core, x_max_core = np.nanpercentile(x, [p_low, p_high])
    y_min_core, y_max_core = np.nanpercentile(y, [p_low, p_high])
    
    # 2. Calculate the physical span of the track independently for X and Y
    x_span = x_max_core - x_min_core
    y_span = y_max_core - y_min_core
    
    # 3. Add a proportional buffer (e.g., 5% of the span) to safely enclose the track
    X_MIN = x_min_core - (pad_fraction * x_span)
    X_MAX = x_max_core + (pad_fraction * x_span)
    
    Y_MIN = y_min_core - (pad_fraction * y_span)
    Y_MAX = y_max_core + (pad_fraction * y_span)
    
    # 4. Mask the outliers
    out_of_bounds = (x < X_MIN) | (x > X_MAX) | (y < Y_MIN) | (y > Y_MAX)
    
    x_clean = x.copy()
    y_clean = y.copy()
    x_clean[out_of_bounds] = np.nan
    y_clean[out_of_bounds] = np.nan
    
    return x_clean, y_clean


# blanket around the trajectory to aviod plateaus
def dynamic_percentile_box(x, y, p_low=1, p_high=99, pad_fraction=0.05):
    # 1. Find the true edges of the track, ignoring extreme outliers (top/bottom 1%)
    x_min_core, x_max_core = np.nanpercentile(x, [p_low, p_high])
    y_min_core, y_max_core = np.nanpercentile(y, [p_low, p_high])
    
    # 2. Calculate the physical span of the track independently for X and Y
    x_span = x_max_core - x_min_core
    y_span = y_max_core - y_min_core
    
    # 3. Add a proportional buffer (e.g., 5% of the span) to safely enclose the track
    X_MIN = x_min_core - (pad_fraction * x_span)
    X_MAX = x_max_core + (pad_fraction * x_span)
    
    Y_MIN = y_min_core - (pad_fraction * y_span)
    Y_MAX = y_max_core + (pad_fraction * y_span)
    
    # 4. Mask the outliers
    out_of_bounds = (x < X_MIN) | (x > X_MAX) | (y < Y_MIN) | (y > Y_MAX)
    
    x_clean = x.copy()
    y_clean = y.copy()
    x_clean[out_of_bounds] = np.nan
    y_clean[out_of_bounds] = np.nan
    
    return x_clean, y_clean


# blanket around the trajectory to aviod plateaus
def blanket(x, y, method,k=None,r=None, sigma=5, min_k=31, threshold=95):
    # we need to find the largest gap in the data to determine the window size
    x_non_nan_idx = np.where(~np.isnan(x))[0]
    # bounding box
    x, y = dynamic_percentile_box(x, y, p_low=1, p_high=99, pad_fraction=0.05)

    padding_x = np.hstack(([-1], x_non_nan_idx, [len(x_non_nan_idx)]))

    d_x = np.diff(padding_x)

    lengths_of_nans_x = d_x[d_x>1]-1

    if k is None:
        if len(lengths_of_nans_x) != 0:
            k= 2*max(lengths_of_nans_x)+1
        else:
            k = 30
    k = max(k, min_k)
    if k % 2 == 0:
        k += 1

    temp_x = pd.Series(x).interpolate(method='linear').bfill().ffill().to_numpy()
    temp_y = pd.Series(y).interpolate(method='linear').bfill().ffill().to_numpy()

    if method == "median":
        smooth_x = pd.Series(x).rolling(window=k, min_periods=1, center=True).median().to_numpy()
        smooth_y = pd.Series(y).rolling(window=k, min_periods=1, center=True).median().to_numpy()
    elif method == "savgol":
        # polyorder=3 allows the filter to perfectly match sharp S-curves
        smooth_x = savgol_filter(temp_x, window_length=k, polyorder=3)
        smooth_y = savgol_filter(temp_y, window_length=k, polyorder=3)
    elif method == "gaussian":
        smooth_x = gaussian_filter1d(x, sigma=sigma)
        smooth_y = gaussian_filter1d(y, sigma=sigma)
        # smooth_x = pd.Series(x).rolling(window=k, win_type="gaussian", min_periods=1, center=True).mean(std=sigma).to_numpy()
        # smooth_y = pd.Series(y).rolling(window=k, win_type="gaussian", min_periods=1, center=True).mean(std=sigma).to_numpy()
        
    center_line = np.vstack((smooth_x, smooth_y))
    clean_pos = np.vstack((x, y))

    dist_from_cl = clean_pos - center_line
    dist_from_cl = np.linalg.norm(dist_from_cl, axis=0, ord=2)

    if r is None:
        r = np.nanpercentile(dist_from_cl, threshold)

    print(r)
    out_of_blanket_mask = (dist_from_cl >= r)
    out_of_blanket_mask[0], out_of_blanket_mask[-1] = False, False

    curr_x_anchors = x.copy()
    curr_y_anchors = y.copy()

    curr_x_anchors[out_of_blanket_mask] = np.nan
    curr_y_anchors[out_of_blanket_mask] = np.nan

    fraction_of_track_retained = 1 - out_of_blanket_mask.sum()/len(x)

    return curr_x_anchors, curr_y_anchors, smooth_x, smooth_y, fraction_of_track_retained

# function to animate the trajectory
def animate_trajectory(x, y, filename, ax=None):
    total_frames = len(x)

    valid_x = x[np.isfinite(x)]
    valid_y = y[np.isfinite(y)]

    if ax is None:
        l = 10
        fig, ax = plt.subplots(figsize = (l, l*(20/50)))
    else:
        fig = ax.get_figure()

    ax.set_xlim(np.nanmin(valid_x)-0.5, np.nanmax(valid_x)+0.5)
    ax.set_ylim(np.nanmin(valid_y)-0.5, np.nanmax(valid_y)+0.5)
    ax.invert_yaxis()

    frame_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')
    line, = ax.plot([], [], color='black', linewidth=2)

    def update(frame):
        current_x = x[:frame+1]
        current_y = y[:frame+1]
        line.set_data(current_x, current_y)
        frame_text.set_text(f'Index: {frame}')
        return line,
    ani = FuncAnimation(fig, update, frames=total_frames, interval=40, blit=False, repeat=False)
    ani.save(f'{filename}.mp4', writer='ffmpeg', fps=25)

def animate_trajectory_all(data, filename, colors, ax=None):
    total_frames = data.shape[1]
    no_fishes = data.shape[0]

    # valid_data = data[np.isfinite(data)]
    # print(valid_data.shape)
    valid_x = data[:, :, 0]
    valid_y = data[:, :, 1]

    if ax is None:
        length = 10
        fig = plt.figure(figsize=(length,(length/(50/20))))
        ax = fig.add_axes([0, 0, 1, 1])
        # ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
        mid_way = (np.nanmin(valid_x) + np.nanmax(valid_x))/2
        ax.axvline(x=mid_way-5)
        ax.axvline(x=mid_way+5)
    else:
        fig = ax.get_figure()

    ax.set_xlim(np.nanmin(valid_x)-0.5, np.nanmax(valid_x)+0.5)
    ax.set_ylim(np.nanmin(valid_y)-0.5, np.nanmax(valid_y)+0.5)
    ax.invert_yaxis()

    frame_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')
    lines = [ax.plot([], [], color=colors[i], linewidth=2, label=f'Fish {i+1}')[0] for i in range(no_fishes)]

    def update(frame):
        for i in range(no_fishes):
            current_x = valid_x[i,:frame+1]
            current_y = valid_y[i,:frame+1]
            lines[i].set_data(current_x, current_y)
        frame_text.set_text(f'Index: {frame}')
        return (*lines, frame_text)

    ani = FuncAnimation(fig, update, frames=total_frames, interval=40, blit=False, repeat=False)
    ani.save(f'{filename}.mp4', writer='ffmpeg', fps=25)


