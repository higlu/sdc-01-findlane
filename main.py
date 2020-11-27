# -*- coding: utf-8 -*-
"""
@author: Simone Diolaiuti
"""

#import the developed algorithm
from find_lane import findLane

# Import everything needed to edit/save images and video clips
import os
import errno
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

def processImages(input_dir,output_dir):
    input_imgs = os.listdir(input_dir)
    for img_name in input_imgs:
        img = mpimg.imread(input_dir + img_name)
        processed_img = findLane(img)
        mpimg.imsave(output_dir + img_name,processed_img)
        
        
def processVideos(input_dir,output_dir):
    input_vids = os.listdir(input_dir)
    for vid_name in input_vids:
        clip = VideoFileClip(input_dir + vid_name)
        try:
            processed_clip = clip.fl_image(findLane)
            processed_clip.write_videofile(output_dir + vid_name, audio=False)
        except:
            print("error while processing video", vid_name)
        finally:
            clip.reader.close()
        
def ensureDirPresence(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise        

img_input_dir = "input_images/"
img_output_dir = "output_images/"

vid_input_dir = "input_videos/"
vid_output_dir = "output_videos/"

# processing images
print("processing images in", img_input_dir, "...")
ensureDirPresence(img_output_dir)
processImages(img_input_dir,img_output_dir)

# processing videos
print("processing videos in", vid_input_dir, "...")
ensureDirPresence(vid_output_dir)
processVideos(vid_input_dir,vid_output_dir)

