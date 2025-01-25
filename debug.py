from PIL import Image, ImageSequence
import numpy as np
import os
import PIL

def readGif(filename, asNumpy=True):
    """ readGif(filename, asNumpy=True)
    
    Read images from an animated GIF file.  Returns a list of numpy 
    arrays, or, if asNumpy is false, a list if PIL images.
    
    """
    
    # Check PIL
    if PIL is None:
        raise RuntimeError("Need PIL to read animated gif files.")
    
    # Check Numpy
    if np is None:
        raise RuntimeError("Need Numpy to read animated gif files.")
    
    # Check whether it exists
    if not os.path.isfile(filename):
        raise IOError('File not found: '+str(filename))
    
    # Load file using PIL
    pilIm = PIL.Image.open(filename)    
    pilIm.seek(0)
    
    # Read all images inside
    images = []
    try:
        while True:
            # Get image as numpy array
            tmp = pilIm.convert() # Make without palette
            a = np.asarray(tmp)
            if len(a.shape)==0:
                raise MemoryError("Too little memory to convert PIL image to array")
            # Store, and next
            images.append(a)
            pilIm.seek(pilIm.tell()+1)
    except EOFError:
        pass
    
    # Convert to normal PIL images if needed
    if not asNumpy:
        images2 = images
        images = []
        for im in images2:            
            images.append( PIL.Image.fromarray(im) )
    
    # Done
    return images

def preprocess_human_demo(frames):
    """
    frames shape: (16, 250, 250, 3)
    frames shape: (1, 16, 250, 250, 3)
    frames shape: (1, 3, 16, 250, 250)
    """
    frames = np.array(frames)
    print(f"frames shape: {frames.shape}")
    frames = frames[None, :,:,:,:]
    print(f"frames shape: {frames.shape}")
    frames = frames.transpose(0, 4, 1, 2, 3)
    print(f"frames shape: {frames.shape}")
    return frames


video_path = "/home/aw588/git_annshin/roboclip_local/gifs/drawer-open-human2.gif"
frames = readGif(video_path)
frames = preprocess_human_demo(frames)