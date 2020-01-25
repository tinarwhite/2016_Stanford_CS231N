import h5py
import numpy as np
from util import read_me    
import matplotlib.pyplot

ref = 0
if ref == 0:
    ns, nv, nc, nt = 3, 513, 3, 978
elif ref == 1:
    ns, nv, nc, nt = 3, 2004, 3, 3912

def post(v, which):
    v = v.reshape((ns, nc, nt), order='F')
    if which == 'r':
        Q = v[:, 0, :]
    if which == 'ru':
        Q = v[:, 1, :]
    if which == 'rv':
        Q = v[:, 2, :]
    if which == 'uabs':
        Q = np.sqrt((v[:, 1, :]/v[:, 0, :])**2+(v[:, 2, :]/v[:, 0, :])**2) 
    return Q.ravel('F')
    
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

#snapnames =  ('cfd/naca0012ref0p1.snaps.mu3',
#              'cfd/naca0012ref0p1.snaps.mu4',
#              'cfd/naca0012ref0p1.snaps.mu5',
#              'cfd/naca0012ref0p1.snaps.mu6',
#              'cfd/naca0012ref0p1.snaps.mu7')
              
snapnames =  ('cfd/naca0012ref0p1.snaps.mu2',
              'cfd/naca0012ref0p1.snaps.mu1')

image_size = 32
num_frames = 20 #20
stride = 4 #4
buf3 = np.empty([1,num_frames,3, image_size,image_size])
for j in range(len(snapnames)):
#for j in (2,3):
    print(snapnames[j])
    snaps = read_me(snapnames[j]).T
    
    #reduce size for test run
    #snaps = snaps[:16,:]
    
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
        matplotlib.pyplot.axis('off')
        #plot.set_ylim([0,34])    
        
        # draw the berg plot
        #plot.plot(np.linspace(0.0, 100.0, snaps.shape[1])[:, None], snaps[pvect,:], color='w', lw=1.5)
        #plot.set_xticklabels([])
        #plot.xaxis.set_ticks_position('none') 
        #plot.set_yticklabels([])
        #plot.yaxis.set_ticks_position('none') 
        #plot.patch.set_facecolor('black')
        
        
        which = 'uabs'
        p = np.loadtxt('cfd/naca0012ref{0:d}p1.nodes'.format(ref))
        Q = post(snaps[pvect,:], which)
        p = p.reshape((nt, 2, ns))
        t = np.arange(0, 3*nt).reshape(nt, 3, order='C')
        plot.tricontourf(p[:, 0, :].reshape(ns*nt, order='C'),
                        p[:, 1, :].reshape(ns*nt, order='C'), t, Q, np.linspace(0,2.5,26))
        plot.axis([-1.5,2.5,-1.5,1.5])
        
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
valid100 = buf4[:20,:,:,:,:]
#train100 = buf4[20:,:,:,:,:]

# split into number divisible by 100
#valid100 = valid[:400,:,:,:,:]
#train100 = train[:3200,:,:,:,:]

#h = h5py.File('burg.h5', 'w')
#dset = h.create_dataset('data', data=buf4)
#np.save('cfd_valid.npy', valid100)
#np.save('cfd.npy', train100)

np.save('cfd_valid.npy', buf4)

# Save grayscale version
#validgray = 255 - valid100[:,:,0,:,:]
#traingray = 255 - train100[:,:,0,:,:]
#validgray = valid100[:,:,0,:,:]
#traingray = train100[:,:,0,:,:]
#np.save('burgg_valid.npy', validgray)
#np.save('burgg.npy', traingray)

#dset = h.create_dataset('data2', data=snaps2)
#dset = h.create_dataset('data3', data=snaps3)


#h5file = h5py.File('burg3.h5', 'w')
#grp = h5file.create_group("train")
#dset = grp.create_dataset("data1", data = snaps1)
#dset = grp.create_dataset("data2", data = snaps2)
#dset = grp.create_dataset("data3", data = snaps3)
