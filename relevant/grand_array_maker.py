import numpy as np
from pathlib import Path


if __name__ == "__main__":
    N = int(input("Enter number of fishes:"))
    # dir_0_input = input("Enter path to parent directory:")
    dir_0_input  = '/home/hck03/Documents/MS_thesis/tracking_output/1c_4n_faulty'
    dir_0 = Path(dir_0_input).resolve()

    if not dir_0.exists() or not dir_0.is_dir():
        print("Error: directory not found!")
        exit()

    subfolder_name = 'data'
    x_filename = 'X#wcentroid'
    y_filename = 'Y#wcentroid'

    for expt in dir_0.iterdir():
        if not expt.is_dir():
            continue
        target_subfolder = expt / subfolder_name
        npz_files = sorted(target_subfolder.glob("*.npz"))
        no_of_fishes = len(npz_files)
        file_path = Path(target_subfolder / "grand_array.npy")
        if file_path.is_file():
            continue
        fish_list = []
        max_no_frames = 0
        for f in npz_files:
            df = np.load(f)
            frames_i = df['frame'].astype(int)
            no_of_frames = frames_i[-1]+1
            if no_of_frames > max_no_frames:
                max_no_frames = no_of_frames
        
        for fish in npz_files:
            with np.load(fish) as data_0:
                x = data_0[x_filename]
                y = data_0[y_filename]
                frames = data_0['frame'].astype(int)
                aligned_x = np.full(max_no_frames, np.nan)
                aligned_y = np.full(max_no_frames, np.nan)
                aligned_x[frames] = x
                aligned_y[frames] = y
                pos = np.column_stack((aligned_x, aligned_y))
                fish_list.append(pos)
                print(pos.shape)
        if no_of_fishes == N:
            grand_array = np.stack(fish_list, axis=0)
            np.save(file_path, grand_array)
    print("All files saved!")