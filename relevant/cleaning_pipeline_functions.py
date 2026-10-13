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

# seconds per frame given the frame rate is 100fps
sec_per_frame = 0.01

def speedy_trail(x, y, ax=None, cmap='viridis', label='Speed'):
    """
    This is a function to plot the trajectories showing the speed of the fish at each frame. 
    Args:
        x (array): x coordinates of the track
        y (array): y coordinates of the track
        ax (matplotlib.axes.Axes, optional): axis to plot the speedy trail on. If 'None' new axis will be created
        cmap (string, optional): colormap of choice
        label (string, optional): what the colorbar indicates
    Returns:
        ax ((matplotlib.axes.Axes): the axis on which the speedy trail is plotted
    """
    vel_x = np.gradient(x)/sec_per_frame
    vel_y = np.gradient(y)/sec_per_frame
    vel = np.zeros((x.shape[0], 2))
    vel[:, 0] = vel_x
    vel[:, 1] = vel_y

    speed_array = np.linalg.norm(vel, axis=1, ord=2)
    l = 10
    if ax is None:
        fig, ax = plt.subplots(figsize = (l, l*(20/50)))
    else:
        fig = ax.get_figure()

    segment_speeds = speed_array[:-1]
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap=cmap)
    lc.set_array(segment_speeds)
    lc.set_linewidth(1.5)
    line = ax.add_collection(lc)

    fig.colorbar(line, ax=ax, label=label)
    ax.autoscale()

    return ax

def detect_angle_spikes(vec, angle_threshold=None):
    """
    This functions finds out where there are unnatural sharp turns in the trajectory characterised by a close to -1
    cosine value. Angle at a point is the angle between the incoming vector and the outgoing vector.
    Args:
        vec (array): array of vectors from position at frame[i] to frame[i+1]
        angle_threshold (float, optional): the maximum cosine value at a point beyond which the point will be flagged 
        as a spike
    Returns:
        angle_spikes_new (array): the indices where the spikes occur
    """
    dot_prod_new = np.sum(vec[1:,:]*vec[:-1,:], axis=1)
    denominator = (np.linalg.norm(vec[1:, :], axis=1)*np.linalg.norm(vec[:-1,:], axis=1))
    safe_mask = (denominator > 0)
    cos_theta_new = np.zeros(dot_prod_new.shape[0])
    cos_theta_new[safe_mask] = dot_prod_new[safe_mask]/denominator[safe_mask]
    cos_theta_new = np.clip(cos_theta_new, -1.0, 1.0)

    if angle_threshold is None:
        angle_threshold = -0.4

    angle_spikes_mask = (cos_theta_new < angle_threshold)
    angle_spikes_new = np.flatnonzero(angle_spikes_mask) +1
    # one is added to identify the real spikes indices because the way the cos theta is calculated it points at the frame before
    return angle_spikes_new

def clean_angle_spikes(x, y, angle_spikes, vec ,cos_threshold=None, diff_threshold=None, pad_len_1=None, pad_len_2=None):
    """
    Once angle spikes have been identified this functions cleans the spikes. Its not straightforward because,
        * the spikes can extend upto few frames
        * two spikes can occur in very close proximity (absolute distance wise) and can have varying number of frames 
        between them (jitter)
    Args:
        x (array): x coordinates of the track
        y (array): y coordinates of the track
        angle_spikes (array): array of indices where the spikes occur
        vec (array): array of vectors from position at frame[i] to frame[i+1]
        cos_threshold (float, optional): the cosine value of the vector entering the spike and exiting the spike i.e., if spike 
        is at 'i' then the angle between vector from 'i-2' to 'i-1' and 'i+1' to 'i+2'
        diff_threshold (float, optional): absolute distance threshold between spikes. This is to capture spikes which are very 
        close to each other
        pad_len_1 (int, optional): number of buffer frames to be flagged for spikes which are very close to each other
        pad_len_2 (int, optional): number of buffer frames to be flagged for lonely spikes
    Returns:
        x_temp, y_temp (array): clean trajectories
    """
    # finding the difference in position between each angle spike
    x_temp, y_temp = x.copy(), y.copy()
    pos = np.vstack((x_temp, y_temp)).T
    
    pos_diff = np.diff(pos[angle_spikes], axis=0)
    diff_size = np.linalg.norm(pos_diff, ord=2, axis=1)

    # defining the optional argument values
    if diff_threshold is None:
        diff_threshold = 1
    if cos_threshold is None:   
        cos_threshold = 0.4
    if pad_len_1 is None:
        pad_len_1 = 3 
    if pad_len_2 is None:
        pad_len_2 = 2

    lonely_spikes = set(angle_spikes.tolist())
    # delting grouped angle spikes
    for i in range(len(angle_spikes) -1):
        first_spike = angle_spikes[i]
        next_spike = angle_spikes[i +1]

        idx_in = max(0, first_spike -1)
        idx_out = min(len(vec) -1, next_spike +1)

        incomming_heading = vec[idx_in]
        outgoing_heading = vec[idx_out] 
        dot_prod = np.sum(incomming_heading * outgoing_heading)
        denom = np.linalg.norm(incomming_heading, ord = 2)*np.linalg.norm(outgoing_heading, ord = 2)
        if denom == 0:
            continue   

        cos_theta = dot_prod/denom 
            
        if diff_size[i] < diff_threshold and cos_theta < cos_threshold:
            lonely_spikes.discard(first_spike)
            lonely_spikes.discard(next_spike)

            pad_len = pad_len_1
            start_idx = max(0, first_spike - pad_len)
            end_idx = min(len(x_temp), next_spike + 1 + pad_len)
            x_temp[start_idx:end_idx] = np.nan
            y_temp[start_idx:end_idx] = np.nan

    # deleting lonely angle spikes
    pad_len = pad_len_2
    for spike in lonely_spikes:
        start_idx = max(0, spike - pad_len)
        end_idx = min(len(x_temp), spike + 1 + pad_len)
        faulty_idxs = np.arange(start_idx, end_idx)
        x_temp[faulty_idxs] = np.nan
        y_temp[faulty_idxs] = np.nan

    x_temp[0], y_temp[0] = x[0], y[0]
    x_temp[-1], y_temp[-1] = x[-1], y[-1]

    return x_temp, y_temp

# blanket around the trajectory to aviod plateaus
def dynamic_percentile_box(x, y, p_low=1, p_high=99, pad_fraction=0.05):
    """
    Creates a bounding box around the data so that if there are any data points which are on the wall and very
    far from the track then its deleted.
    Args:
        x (array): x coordinates of the track
        y (array): y coordinates of the track
        p_low (int): minimum percentile of real data to consider to find min extent
        p_high (int): maximum percentile of real data to consider to find max extent
        pad_fraction (float): fraction of the expanse to pad to safely eclose the track 
    Returns:
        x_clean, y_clean (array): clean trajectories
    """
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


# blanket around the trajectory to aviod plateaus (first pass filter)
def blanket(x, y, method, k=None, r=None, sigma=5, min_k=5, threshold=95):
    """
    This is the first pass filter to weed out major deviations from the trajectory. The idea is you stitch a net along the 
    trajectory and anything which overshoots the net is a faulty reading. You create the new by creating an estimated track 
    from the raw data you have where it is smoothened over the areas where the tracking is faulty. You then measure distance
    raw data from the smooth curve and deviations beyond a threshold are flagged.
    Args:
        x (array): x coordinates of the track
        y (array): y coordinates of the track
        method (string): There are three methods available
                            * 'median' - estimated trajectory is computed by applying a sliding median filter
                            * 'savgol' - estimated trajectory is computed by applying the Savitzky-Golay filter
                            * 'gaussian' - estimated trajectory is computed by applying a 1D Gaussian filter. Independent of 
                                            window size
        k (int, optional): the size of the sliding window
        r (int, optional): the distance threshold from the estimated trajectory beyond which the data is flagged 
                            **Warning: change it only if you have a specific value in mind. Otherwise work with the threshold
                                        parameter. Don't change both together
        sigma (float, optional): standard deviation for the gaussian filter. 
        min_k (int, optional): minimum window size
        threshold (float, optional): the nth percentile you want to choose to calculate 'r', obtained by finding nth percentile
                                     on the deviations of raw data from estimated trajectory
    Returns:
        x_anchors, y_anchors (array): retained part of the raw data after cleaning
        smooth_x, smooth_y (array): estimated trajectory coordinates
        fraction_of_track_retained (float): fraction of track retained
    """
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
            k = min_k
    k = max(k, min_k)
    if k % 2 == 0:
        k += 1

    temp_x = pd.Series(x).interpolate(method='linear').bfill().ffill().to_numpy()
    temp_y = pd.Series(y).interpolate(method='linear').bfill().ffill().to_numpy()

    if method == "median":
        smooth_x = pd.Series(temp_x).rolling(window=k, min_periods=1, center=True).median().to_numpy()
        smooth_y = pd.Series(temp_y).rolling(window=k, min_periods=1, center=True).median().to_numpy()
    elif method == "savgol":
        # polyorder=3 allows the filter to perfectly match sharp S-curves
        smooth_x = savgol_filter(temp_x, window_length=k, polyorder=3)
        smooth_y = savgol_filter(temp_y, window_length=k, polyorder=3)
    elif method == "gaussian":
        smooth_x = gaussian_filter1d(temp_x, sigma=sigma)
        smooth_y = gaussian_filter1d(temp_y, sigma=sigma)
        # smooth_x = pd.Series(x).rolling(window=k, win_type="gaussian", min_periods=1, center=True).mean(std=sigma).to_numpy()
        # smooth_y = pd.Series(y).rolling(window=k, win_type="gaussian", min_periods=1, center=True).mean(std=sigma).to_numpy()
        
    center_line = np.vstack((smooth_x, smooth_y))
    clean_pos = np.vstack((x, y))

    dist_from_cl = clean_pos - center_line
    dist_from_cl = np.linalg.norm(dist_from_cl, axis=0, ord=2)

    if r is None:
        r = np.nanpercentile(dist_from_cl, threshold)

    print(f'k: {k}, r: {r}')

    out_of_blanket_mask = (dist_from_cl >= r)
    out_of_blanket_mask[0], out_of_blanket_mask[-1] = False, False

    x_anchors = x.copy()
    y_anchors = y.copy()

    x_anchors[out_of_blanket_mask] = np.nan
    y_anchors[out_of_blanket_mask] = np.nan

    fraction_of_track_retained = 1 - out_of_blanket_mask.sum()/len(x)

    return x_anchors, y_anchors, smooth_x, smooth_y, fraction_of_track_retained

# function to animate the trajectory
def animate_trajectory(x, y, filename, ax=None):
    """
    Function to animate single trajectory
    Args:
        x (array): x coordinates of the track
        y (array): y coordinates of the track
        filename (string): name of the output file
        ax (matplotlib.axes.Axes): the axis on which the trajectory is plotted
    """
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
    """
    Function to animate trajectories of all fish
    Args:
        data (array): data of all fish
        filename (string): name of the output file
        ax (matplotlib.axes.Axes): the axis on which the trajectory is plotted
    """
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

# unnecessary function
# A function to return an array of erraticness scores
# def filter(x, y, k=7):
#     positions = np.vstack((x, y)) # (2, 1882)
#     diff = np.diff(positions, axis=1)
#     d = np.linalg.norm(diff, axis=0, ord=2)
#     # plt.hist(d)
#     med = np.nanmedian(d)
#     mad = np.nanmedian(np.abs(d - med))
#     S = max(med + 1.4826*mad, 1e-9)

#     # Step-size factor
#     A = np.minimum(d[:-1], d[1:])/S
#     A = np.minimum(A, 3)

#     # reversal factor
#     dot_prod = np.sum(diff[:,1:]*diff[:,:-1], axis=0)
#     denominator = d[:-1]*d[1:]
#     valid_step = ~np.isnan(denominator)

#     safe_mask = (denominator > 0)
#     cos_theta = np.zeros(dot_prod.shape[0])
#     cos_theta[safe_mask] = dot_prod[safe_mask]/denominator[safe_mask]
#     cos_theta = np.clip(cos_theta, -1.0, 1.0)

#     R = np.full(dot_prod.shape, np.nan)
#     R[valid_step] = 0.0                              # zero-length step: not reversing
#     R[safe_mask] = (1 - cos_theta[safe_mask]) / 2

#     # erratic score
#     e = A*R
#     e_full = np.full(x.shape[0], np.nan) # padding to keep array size consistent
#     e_full[1:-1] = e

#     # e mean
#     # k = 7
#     valid = ~np.isnan(e_full)
#     num = uniform_filter1d(np.where(valid, e_full, 0.0), size=k, mode='constant', cval=0.0)
#     den = uniform_filter1d(valid.astype(float), size=k, mode='constant', cval=0.0)

#     E = np.full(x.shape[0], np.nan)
#     E[den > 0] = num[den > 0] / den[den > 0]

#     E_filled = np.nan_to_num(E, nan=0.0) 
#     return E_filled