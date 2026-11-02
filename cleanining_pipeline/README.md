# Cleaning Trex Output
<!--- cSpell:disable --->
Once you receive the tracks from TRex after tracking and correcting for ID switches, the data needs to be cleaned because there is a lot of jitter in the raw tracks. This jitter occurs when fish cross over each other or when they are moving slowly. These errors are primarily deviations in space and can be characterized by large departures from the actual trajectory of the fish.

## Preparing the data
Trex saves the tracks for individual fish in `numpy` zipped file(`.npz` ). These contain all the various arrays that Trex saves after tracking. The only files we are interested are `X#wcentroid.npy` and `Y#wcentroid.npy`, which are the coordinates of the centroid of the fish. 
1. Get all the tracked data into a "master directory" where each video has a unique session directory.
2. Then in the `grand_array_maker.py` script, enter the path to the 
"master directory" as an input to the `dir_0_input` variable. 
3. Run the script on your terminal. Once done it will have saved `grand_array.npy` of shape (`no_of_fish`, `no_of_frames`, 2) in the `data/` directory of the respective session directory.

## Running the Pipeline
`trex_cleaning_pipeline.ipynb` is the notebook to visualize cleaning your tracks and to save clean tracks.
### Step 1 - Visualizing the tracks
By executing this step you can visualize the tracks obtained by tracking also showing the speed of the fish at each frame mapped to a color map. 
### Step 2 - First pass filter - coarse filtering
In this step you can clear a lot of jitter from the data. How this works is by using the function `blanket` from `cleaning_pipeline_functions.py`. This function creates an estimated trajectory by smoothing the raw data. This smoothens over all the places where the track is erratic and has deviated a lot from the actual path taken by the animal. 
Then it calculates the distance between the raw data point and the estimated data point at each frame. Then a `r` value is computed based on the percentile `threshold`. Data points which are greater than this threshold are flagged as errors and are set to `NaN`
#### Parameters
1. `k (int, optional) | Range: [0, inf)` : ***Rolling window size***. If set to `None`, then the value is automatically calculated as `2(l) + 1` where `l` is the length of the largest gap in the raw data.
2. `threshold (float, optional) | Range [0, 100)` : The ***trajectory jitter threshold percentile*** defines the statistical cutoff within the distribution of spatial deviations; data points falling below this percentile are preserved, while those exceeding it are filtered out as noise.
3. `r (int, optional) | Range: [0, inf)` : To filter spatial noise, a ***trajectory jitter threshold*** establishes the maximum permissible distance between a raw data point and the estimated trajectory path; coordinates exceeding this limit are flagged as positioning errors. Generally set to `None` and is computed automatically as the `threshold` th percentile value of the deviations array. Advised to set a value only if you want definite control over `r` and to keep it less than 1 to 1.5cm as this is approximately the maximum dorsal width. 
4. `sigma (float, optional)` : ***Standard deviation*** for the gaussian smoothing function.
5. `min_k (int, optional)` : ***Minimum rolling window size*** if there are no gaps in data and if no `k` value is given

Once you are done filtering you interpolate between the gaps created now using **piecewise cubic hermite interpolating polynomial** `pchip` algorithm. 

### Step 3 - Second pass filter - fine cleaning
This is run on the interpolated data after cleaning all the major spikes and jumps in data. The second pass filter involving the `detect_angle_spikes` and `clean_angle_spikes` functions are to clean more finer jumps. Generally jumps within the body. These can be characterized as a large deviation in angle between the incoming vector and the outgoing vector at a point. This is exploited to identify all such faults using the `detect_angle_spikes` function. 
#### Parameter
1. `angle_threshold (float, optional) | Range: (-1,1)` : Large angles have a cosine value close to -1. Therefore any value lesser than this threshold value is flagged as an **angle spike**. Default set to -0.4

Once all the **angle spikes** have been detected the next step is to clean them. This is done using the `clean_angle_spikes` function. There are some regions where a spike occurs and the tracked path till the next spike is consistently faulty in the raw data (**long tandems**), and there are other spikes which occur in clusters or alone. 
#### Parameters
1. `cos_threshold (float, optional) | Range (-1,1)` : Cosine value of the angle between the vector entering the `first spike` and exiting the `next spike`. Should be close to 1 i.e., angle should be close to 0. So any cosine value greater than this threshold is considered as `True` for the conditional. This is to clear **long tandems**. 
2. `diff_threshold (float, optional) | Range [0, inf)` : This threshold is again to clear **long tandems**. This is the threshold for the distance between every pair of `first spike` and `next spike`. It has to be lesser than the dorsal width of the fish (generally set to < 1cm). Both this and the previous condition should be `True` for that pair of consecutive spikes to be considered as a **long tandem**
3. `pad_len_1 (int, optional) | Range [0, 10]` : Number of buffer frames to remove for **long tandems**.
4. `pad_len_2 (int, optional) | Range [0, 10]` : Number of buffer frames to remove for all other **angle spikes**.

Once the angle spikes have been cleared the trajectory is again interpolated using `pchip` algorithm and finally saved. 

You can also save the tracked video by entering the appropriate file path to the video folder and entering trial specific `cm_to_pixel` conversion factor. 
