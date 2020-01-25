import h5py
import numpy as np
from util import read_me    
import matplotlib.pyplot
    
def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring (fig.canvas.tostring_rgb(), dtype=np.uint8 )
    buf.shape = ( w, h,3 )
 
    # canvas.tostring_rgb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf

snapnames = ('burg/snaps_0p02_0p02_1.dat',
             'burg/snaps_0p02_0p02_2p5.dat', #current validation set
             'burg/snaps_0p02_0p02_5.dat',
             'burg/snaps_0p02_0p05_1.dat',
             'burg/snaps_0p02_0p05_2p5.dat',
             'burg/snaps_0p02_0p05_5.dat',
             'burg/snaps_0p05_0p02_1.dat',
             'burg/snaps_0p05_0p02_2p5.dat',
             'burg/snaps_0p05_0p02_5.dat')

image_size = 32
num_frames = 20
stride = 5
buf3 = np.empty([1,num_frames,3, image_size,image_size])
for j in range(len(snapnames)):
#for j in (2,3):
    print(snapnames[j])
    snaps = read_me(snapnames[j]).T
    
    #reduce size for test run
    snaps = snaps[50:250,:]
    
    buf = np.zeros((snaps.shape[0],image_size,image_size,3), int)
    
    for i in range(snaps.shape[0]):
    #for i in range(50,150):
        if i % 10 == 0:
            print(i)
        #Choose a vector for plotting
        pvect=i #1050 for berg, 20 for CFD
        
        # Generate a figure with matplotlib</font>
        figure = matplotlib.pyplot.figure(figsize=(.4,.4))
        plot   = figure.add_subplot ( 111 )
        #matplotlib.pyplot.axis('off')
        plot.set_ylim([0,34])    
        
        # draw the plot
        plot.plot(np.linspace(0.0, 100.0, snaps.shape[1])[:, None], snaps[pvect,:], color='w', lw=1.5)
        plot.set_xticklabels([])
        plot.xaxis.set_ticks_position('none') 
        plot.set_yticklabels([])
        plot.yaxis.set_ticks_position('none') 
        plot.patch.set_facecolor('black')    
        
        #show the plot
        #matplotlib.pyplot.show()
        
        #convert to a numpy array of RGBA values 
        buf[i] = fig2data (figure)
    
    buf = np.swapaxes(buf,1,3)
    buf = np.swapaxes(buf,2,3)
    #buf = buf.reshape(-1,3*64*64)    
    
    #now make it gray image with blue line grayscale with black background
    #buf = 255 - buf[:,0,:,:]
    #buf = buf.reshape(-1,64*64)
    
    # Create test sets of 20 (num_frames)
    buf2 = np.ones((snaps.shape[0]-num_frames*stride,num_frames,3,image_size,image_size), int)
    for i in range(snaps.shape[0]-num_frames*stride):
        buf2[i] = buf[i::stride,:,:,:][:num_frames]
    buf3 = np.vstack([buf3,buf2])
buf3 = buf3[1:,:,:,:]
#if row == snaps.shape[0]:
#  row = 0


#  convert the numpy array to a PIL Image 
#im = fig2img ( figure )
#im.show()


#    snaps = read_me('cfd/naca0012ref0p1.snaps.mu0',
#                    'cfd/naca0012ref0p1.snaps.mu3',
#                    'cfd/naca0012ref0p1.snaps.mu5',
#                    'cfd/naca0012ref0p1.snaps.mu7',)
# buf2 = buf[:501,:,:,:]

# create array with the right formatting
#buf4 = np.ndarray.astype(buf3[:1000,:,:,:],dtype=np.uint8)
buf4 = np.ndarray.astype(buf3[:,:,:,:],dtype=np.uint8)

# split training and validation set
valid = buf4[401:802,:,:,:,:]
train = np.vstack([buf4[0:401,:,:,:,:],buf4[802:,:,:,:,:]])

# split into number divisible by 100
valid100 = valid[:400,:,:,:,:]
train100 = train[:3200,:,:,:,:]

#h = h5py.File('burg.h5', 'w')
#dset = h.create_dataset('data', data=buf4)
np.save('burg_valid.npy', valid100)
np.save('burg.npy', train100)

# Save grayscale version
#validgray = 255 - valid100[:,:,0,:,:]
#traingray = 255 - train100[:,:,0,:,:]
validgray = valid100[:,:,0,:,:]
traingray = train100[:,:,0,:,:]
np.save('burgg_valid.npy', validgray)
np.save('burgg.npy', traingray)

#dset = h.create_dataset('data2', data=snaps2)
#dset = h.create_dataset('data3', data=snaps3)


#h5file = h5py.File('burg3.h5', 'w')
#grp = h5file.create_group("train")
#dset = grp.create_dataset("data1", data = snaps1)
#dset = grp.create_dataset("data2", data = snaps2)
#dset = grp.create_dataset("data3", data = snaps3)
