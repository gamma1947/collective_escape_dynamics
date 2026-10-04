import numpy as np
from pathlib import Path


if __name__ == "__main__":
    N = int(input("Enter number of fishes:"))
    dir_0_input = input("Enter path to parent directory")
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

        fish_list = []
        for fish in npz_files:
            with np.load(fish) as data_0:
                x = data_0[x_filename]
                y = data_0[y_filename]
                pos = np.column_stack((x, y))
                fish_list.append(pos)
                # print(pos.shape)
        if no_of_fishes == N:
            grand_array = np.stack(fish_list, axis=0)
            np.save(target_subfolder / "grand_array.npy", grand_array)
    print("All files saved!")