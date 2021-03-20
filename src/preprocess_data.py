import skvideo.io  # pip install sk-video
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tfl
from scipy.ndimage import gaussian_filter1d
from skimage.transform import downscale_local_mean
import os


def preprocess_resp(resp_filepath,sigma,chopStart,chopEnd,save_path):
    # load response data
    resp_file = np.load(resp_filepath)
    resp_raw = resp_file['responses']
    print('Loaded data shape: ',resp_raw.shape)
    
    # filter traces through a gaussian kernel
    resp_gauss = gaussian_filter1d(resp_raw,sigma,axis=2,mode='reflect')

    # chop off pre and post movie presentation frames
    resp_chop = resp_gauss[:,:,chopStart:chopEnd]

    # normalize response between -1 and 1, keeping 0 values at 0
    min_val = resp_chop.min()
    max_val = resp_chop.max()
    abs_max = np.max((np.abs(min_val),np.abs(max_val)))
    resp_scaled = ((resp_chop+abs_max)/abs_max) - 1

    # reshape array into (nCells,nRepeats*nFrames
    resp_final = np.reshape(resp_scaled,(resp_scaled.shape[0],resp_scaled.shape[1]*resp_scaled.shape[2]))
    print('Preprocessed data shape: ',resp_final.shape)
    
    # save pre-processed response data
    save_filename = os.path.join(save_path,'response_vectors.npz')
    np.savez(save_filename,response_vectors=resp_final)
    
    
def preprocess_movie(movie_filepath,downscale_tuple,save_path):
    # load movie
    video = skvideo.io.vread(movie_filepath,as_grey=True)
    video_raw_frames = np.reshape(video,video.shape[:3])
    print('Loaded movie shape: ',video_raw_frames.shape)
    
    # black buffer of 12 pixels at top and bottom of every frame
    # crop frames to get rid of buffer and get centered square of each frame
    crop_y = 12
    yDim = video_raw_frames.shape[1]-(2*crop_y)
    crop_x = (video_raw_frames.shape[2]-yDim)//2
    frames_crop = video_raw_frames[:,
                                   crop_y:video_raw_frames.shape[1]-crop_y,
                                   crop_x:video_raw_frames.shape[2]-crop_x]
    print('Cropped movie shape: ',frames_crop.shape)

    # normalize pixel values between [-1,1]
    min_val = 0
    max_val = 255
    frames_norm = (2 * ((frames_crop-min_val)/(max_val-min_val)) ) - 1
    print('Range of values: ['+str(frames_norm.min())+','+str(frames_norm.max())+']')

    # downsample frames to size 32x32
    frames = downscale_local_mean(frames_norm,factors=downscale_tuple)
    print('Final movie shape: ',frames.shape)

    # save movie frames as .tiff file
    save_filename = os.path.join(save_path,'processed_movie.tif')
    tfl.imsave(save_filename,frames,bigtiff=True)
    
    return frames