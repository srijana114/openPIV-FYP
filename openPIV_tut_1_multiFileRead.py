from openpiv import tools, pyprocess, validation, filters, scaling
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import imageio
import importlib_resources
import pathlib

# using comma to combine data types
# with a single whitespace.

path = importlib_resources.files('openpiv')     # Use the local openPIV file for your editor
path_fig = path / 'data/test2/'     # Add the test2 folder (with multiple PIV images) to the overall path
os.chdir(path_fig)     # change the directory to those multiple images
# print(os.getcwd())
files = [f for f in os.listdir() if os.path.isfile(f)]     # Read all the files in the folder
"""
print(files[1])
frame_a = tools.imread(files[0])
frame_b = tools.imread(files[1])
fig, ax = plt.subplots(1, 2, figsize=(12, 10))
ax[0].imshow(frame_a, cmap=plt.cm.gray);
ax[1].imshow(frame_b, cmap=plt.cm.gray);
plt.show()

files2 = glob.glob("*.tif")     # Code is used to extract only the /tif files
print(files2)

print(np.size(files))
"""
N = np.size(files)     # Checks the number of files
for i in range(N-1):     # Run for loop from the first file to the second last one (since we take file pairs)
    # print(i)
    frame_a = tools.imread(files[i])     # Read the i-th file
    frame_b = tools.imread(files[i+1])     # Read the (i+1)-th file
    # fig, ax = plt.subplots(1, 2, figsize=(12, 10))
    # ax[0].imshow(frame_a, cmap=plt.cm.gray);
    # ax[1].imshow(frame_b, cmap=plt.cm.gray);
    # plt.show()

    # # DO THE PIV ANALYSIS FOR EACH OF THE COUPLED FRAME (using tutorial 1 method)
    # # YOU CAN USE THE MASKING METHOD FROM TUTORIAL 2 TO MASK OUT THE IMAGES.

    winsize = 32  # pixels, interrogation window size in frame A
    searchsize = 38  # pixels, search area size in frame B
    overlap = 17  # pixels, 50% overlap
    dt = 0.02  # sec, time interval between the two frames

    u0, v0, sig2noise = pyprocess.extended_search_area_piv(
        frame_a.astype(np.int32),
        frame_b.astype(np.int32),
        window_size=winsize,
        overlap=overlap,
        dt=dt,
        search_area_size=searchsize,
        sig2noise_method='peak2peak',
    )

    x, y = pyprocess.get_coordinates(
        image_size=frame_a.shape,
        search_area_size=searchsize,
        overlap=overlap,
    )

    invalid_mask = validation.sig2noise_val(
        sig2noise,
        threshold=1.05,
    )

    u2, v2 = filters.replace_outliers(
        u0, v0,
        invalid_mask,
        method='localmean',
        max_iter=3,
        kernel_size=3,
    )

    # convert x,y to mm
    # convert u,v to mm/sec

    x, y, u3, v3 = scaling.uniform(
        x, y, u2, v2,
        scaling_factor=96.52,  # 96.52 pixels/millimeter
    )

    # 0,0 shall be bottom left, positive rotation rate is counterclockwise
    x, y, u3, v3 = tools.transform_coordinates(x, y, u3, v3)

    tools.save('exp1_001.txt', x, y, u3, v3, invalid_mask)

    fig, ax = plt.subplots(figsize=(8, 8))
    tools.display_vector_field(
        pathlib.Path('exp1_001.txt'),
        ax=ax, scaling_factor=96.52,
        scale=50,  # scale defines here the arrow length
        width=0.0035,  # width is the thickness of the arrow
        on_img=True,  # overlay on the image
        image_name=str(path / 'data' / 'test1' / 'exp1_001_a.bmp'),
    );
    plt.show()

exit()

