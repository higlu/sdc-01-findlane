# sdc-01-findlane

This programs processes images or videos captured by a camera sensor (e.g. a dash-cam) placed in front of the car
and identify the lane the car is driving.

## usage

place your images you want to process in `input_images` sub-folder and your videos in `input_videos` sub-folder then just execute the script:

```
py main.py
```

The result will be placed in `output_images` and `output_videos` 


## notes

file `uda_utils.py` contains functions that have mostly derived from Self Driving Car Udacity course I'm attending. Please find its relative [license file](docs/UDACITY_LICENSE).

The other files contain code developed entirely by me for which [this license applies](LICENSE)
