from __future__ import print_function
"""
Do a mouseclick somewhere, move the mouse to some destination, release
the button.  This class gives click- and release-events and also draws
a line or a box from the click-point to the actual mouseposition
(within the same axes) until the button is released.  Within the
method 'self.ignore()' it is checked whether the button from eventpress
and eventrelease are the same.

"""

from RectangeSelect import RectangleSelect
import numpy as np
import matplotlib.pyplot as plt
import cv2, pydicom, os, pickle

im = pydicom.dcmread('EnIm1.dcm')

def rotate_back(image, center, theta, offset_1, offset_2, R):
    matrix = cv2.getRotationMatrix2D( center=center, angle=-theta, scale=1 )
    S = []
    
    for r in R:
        r_dummy = r.copy()
        r_dummy[:,0]+=offset_1
        r_dummy[:,1]+=offset_2
        S.append(np.matmul(matrix[:,0:2],r_dummy.T).T+matrix[:,2])
    return S



def subimage(image, center, theta, width, height):
    import cv2
    
    shape = ( image.shape[1], image.shape[0] )
    matrix = cv2.getRotationMatrix2D( center=center, angle=theta, scale=1 )
    image = cv2.warpAffine( src=image, M=matrix, dsize=shape )
    x = int( round(center[0] - width/2  ))
    y = int( round(center[1] - height/2 ))
    image = image[ y:y+height, x:x+width ]
    return image, matrix


def line_select_callback(eclick, erelease):
    'eclick and erelease are the press and release events'
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata


def get_two_points(image):
    global ax, fig, coords
    #import math, cv2
    
    coords =[]
    plt.ion
    def onclick(event):
        global ax, fig, coords
        #import numpy as np
        ix, iy = event.xdata, event.ydata
        coords.append((ix, iy))
        ax.plot(ix,iy,'og', linewidth=7, markersize=12)
        fig.canvas.draw()
        if len(coords)>1:
            print(coords)
            r31,r32,r33,r34 = get_ROIs(image, coords)
            fig.canvas.mpl_disconnect(cid)
            #return r31,r32,r33,r34
        

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    if len(coords)>1:
            print(coords)
            r31,r32,r33,r34 = get_ROIs(image, coords)
            fig.canvas.mpl_disconnect(cid)
            #return r31,r32,r33,r34
        

def get_ROIs(image, pts):
    global fig, ax, coords, R
    import math, cv2
    x,y = image.shape[0:2]
    centr=np.array([0.65*y,0.5*x])
    #print(center)
    len_1 = np.linalg.norm(np.array(pts[0])-centr)
    #print(len_1)
    len_2 = np.linalg.norm(np.array(pts[1])-centr)    
    offset_1 = (0.5*x)-len_1
    #print('offset: '+str(offset_1))
    offset_2 = (0.5*x)-len_2
    r11 = np.array([[0.60*y,0.70*y,0.70*y,0.60*y,0.60*y],
                    offset_1+[.05*len_1,.05*len_1,.15*len_1,.15*len_1,.05*len_1]]).T
    r12 = np.array([[0.60*y,0.70*y,0.70*y,0.60*y,0.60*y],
                    offset_1+[.35*len_1,.35*len_1,.45*len_1,.45*len_1,.35*len_1]]).T
    r13 = np.array([[0.60*y,0.70*y,0.70*y,0.60*y,0.60*y],
                    offset_2+[.05*len_2,.05*len_2,.15*len_2,.15*len_2,.05*len_2]]).T
    r14 = np.array([[0.60*y,0.70*y,0.70*y,0.60*y,0.60*y],
                    offset_2+[.35*len_2,.35*len_2,.45*len_2,.45*len_2,.35*len_2]]).T
    
    
    #print(r11)
    theta = np.rad2deg(math.atan((0.65*y)/(0.5*x)))
    matrix = cv2.getRotationMatrix2D( center=(0.65*y,0.5*x), angle=theta, scale=1 )
    
    r31=np.matmul(matrix[:,0:2],r11.T).T+matrix[:,2]
    r32=np.matmul(matrix[:,0:2],r12.T).T+matrix[:,2]
    #print(r31)
    theta_nu = 180-theta
    matrix_nu = cv2.getRotationMatrix2D( center=(0.65*y,0.5*x), angle=theta_nu, scale=1 )
    
    r33=np.matmul(matrix_nu[:,0:2],r13.T).T+matrix_nu[:,2]
    r34=np.matmul(matrix_nu[:,0:2],r14.T).T+matrix_nu[:,2]
    
    ax.plot(r31[:,0],r31[:,1],'-', linewidth=1.5, color='red')
    ax.plot(r32[:,0],r32[:,1],'-', linewidth=1.5, color='yellow')
    ax.plot(r33[:,0],r33[:,1],'-', linewidth=1.5, color='red')
    ax.plot(r34[:,0],r34[:,1],'-', linewidth=1.5, color='yellow')
    fig.canvas.draw()
    R=R+[r31,r32,r33,r34]
    
    #return r31,r32,r33,r34
    

def toggle_selector(event):
    
    if event.key in ['Q', 'q','enter'] and toggle_selector.RS.active:
        global fig, ax, R, center, ang, w, h
        print(' RectangleSelector deactivated.')
        toggle_selector.RS.set_active(False)
        x0, y0, w, h =toggle_selector.RS._rect_bbox
        ang = toggle_selector.RS.angle
        center = toggle_selector.RS.center
        image, matrix = subimage(img, center=center, 
                         theta=np.rad2deg(-ang), width=int(w), height=int(h))
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.axis('off')
        x,y = image.shape[0:2]
        ax.imshow(image, extent = [0,y,x,0], cmap='gray')
        ax.plot([0.65*y,0.65*y],[0,x], '--', linewidth=1.5, color='black')
        ax.plot([0,0.65*y],[0,0.5*x], '--', linewidth=1.5, color='black')
        ax.plot([0,0.65*y],[x,0.5*x], '--', linewidth=1.5, color='black')
        r11 = np.array([[0.55*y,0.70*y,0.70*y,0.55*y,0.55*y],[.025*x,.025*x,.075*x,.075*x,.025*x]]).T
        r12 = np.array([[0.55*y,0.70*y,0.70*y,0.55*y,0.55*y],[.175*x,.175*x,.225*x,.225*x,.175*x]]).T
        r13 = np.array([[0.55*y,0.70*y,0.70*y,0.55*y,0.55*y],[.775*x,.775*x,.825*x,.825*x,.775*x]]).T
        r14 = np.array([[0.55*y,0.70*y,0.70*y,0.55*y,0.55*y],[.975*x,.975*x,.925*x,.925*x,.975*x]]).T
        r15 = np.array([[0.50*y,0.65*y,0.65*y,0.50*y,0.50*y],[.62*x,.62*x,.38*x,.38*x,.62*x]]).T
        
        
        r21 = np.array([[0.05*y,0.15*y,0.15*y,0.05*y,0.05*y],[.45*x,.45*x,.55*x,.55*x,.45*x]]).T
        r22 = np.array([[0.25*y,0.35*y,0.35*y,0.25*y,0.25*y],[.45*x,.45*x,.55*x,.55*x,.45*x]]).T
        
        R = [r11,r12,r13,r14,r15,r21,r22]
        ax.plot(r11[:,0],r11[:,1],'-', linewidth=1.5, color='red')
        ax.plot(r12[:,0],r12[:,1],'-', linewidth=1.5, color='yellow')
        ax.plot(r13[:,0],r13[:,1],'-', linewidth=1.5, color='yellow')
        ax.plot(r14[:,0],r14[:,1],'-', linewidth=1.5, color='red')
        ax.plot(r15[:,0],r15[:,1],'-', linewidth=1.5, color='blue')
        ax.plot(r21[:,0],r21[:,1],'-', linewidth=1.5, color='red')
        ax.plot(r22[:,0],r22[:,1],'-', linewidth=1.5, color='yellow')
        
        print(get_two_points(image))
        #r31,r32,r33,r34 = get_two_points(image)
        print('now getting two points')
        
        
        
        
    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        print(' RectangleSelector activated.')
        toggle_selector.RS.set_active(True)
    if event.key=='up' and toggle_selector.RS.active:
        ang = toggle_selector.RS.angle
        toggle_selector.RS.rotate(ang+0.03)
    if event.key=='down' and toggle_selector.RS.active:
        ang = toggle_selector.RS.angle
        toggle_selector.RS.rotate(ang-0.03)
        
        
def colorbar_index(ncolors, cmap):
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    cmap = cmap_discretize(cmap, ncolors)
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors+0.5)
    colorbar = plt.colorbar(mappable)
    colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
    colorbar.set_ticklabels(range(ncolors,0,-1))

def cmap_discretize(cmap, N):
    """Return a discrete colormap from the continuous colormap cmap.
    cmap: colormap instance, eg. cm.jet. 
    N: number of colors.
    Example
    x = resize(arange(100), (5,100))
    djet = cmap_discretize(cm.jet, 5)
    imshow(x, cmap=djet)
    """
    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
        colors_i = np.concatenate((np.linspace(0, 1., N), (0.,0.,0.,0.)))
        colors_rgba = cmap(colors_i)
        indices = np.linspace(0, 1., N+1)
        cdict = {}
        for ki,key in enumerate(('red','green','blue')):
            cdict[key] = [ (indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki]) for i in range(N+1) ]
        # Return colormap object.
        return mcolors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)



global fig, ax, R, center, ang, w, h
figure, current_ax = plt.subplots()                 

img = np.mean(im.pixel_array,axis=0)

plt.imshow(img, cmap='gray')
plt.axis('off')
# drawtype is 'box' or 'line' or 'none'
toggle_selector.RS = RectangleSelect(current_ax, line_select_callback,
                                       drawtype='box', useblit=True,
                                       button=[1, 3],  # don't use middle button
                                       minspanx=5, minspany=5,
                                       spancoords='pixels',
                                       interactive=True)
plt.connect('key_press_event', toggle_selector)
plt.show()

#print(toggle_selector.RS.corners())

#%%

print(R)
        
offset_1 = int( center[0] - int(w)/2  )
offset_2 = int( center[1] - int(h)/2 )
        
S = rotate_back(img, center, np.rad2deg(-ang), offset_1, offset_2, R)
print(S)
fig_b = plt.figure()
ax_b = fig_b.add_subplot(111)
plt.imshow(img, cmap='gray')
for s in S:
    ax_b.plot(s[:,0],s[:,1],'-', linewidth=1.5, color='red')
            
        #fig_b.canvas.draw()
        
plt.axis('off')
plt.show()  

#%%

roimask = np.zeros(img.shape)
order = [1,6,10,5,11,3,8,2,7,4,9]
#R = [hor r11--1,r12--6,r13--10,r14--5,r15--11, ver r21--3,r22--8, schräg r31--2, r32--7,r33--4,r34--9]
for i in range(len(S)):
    s_dummy = np.around(S[i].copy())
    
    s_dummy = s_dummy.astype(np.int32)
    #print(s_dummy)
    cv2.fillConvexPoly(roimask, s_dummy, order[i])

#plt.legend('OM 1',)
plt.imshow(-roimask, cmap='gray')
#plt.axis('off')
      
colorbar_index(ncolors=11, cmap='gray') 

plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False, # labels along the bottom edge are off, 
    labelleft=False, # labels along the bottom edge are off, 
    right=False,
    left=False)  
plt.show()
        

#%%

#pfile = os.path.join(path,filename+'pickle')
#with open(pfile,'wb') as f:
#   pickle.dump([S, roimask], f)

#S, roimask = pickle.load(open(pfile,"rb"))
        

