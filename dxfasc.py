#  This module is meant to convert DXF drawings to the ASCII format 
#  required by the Raith ELPHY Quantum software. 
#  There are additional functions for basic proximity effect corrections
#  and ordering elements for cleaner writing

import glob, itertools
import numpy as np
import re
import dxfgrabber
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

def get_layer_names(dxf):
    """ get list of layer names. only lists layers that contain objects.
        
        this is to fix a problem with export from different versions of
        adobe illustrator and autocad. """
    layers = ['0'] # this empty layer is required
    for i, ent in enumerate(dxf.entities):
        if i==0: 
            layers.append(ent.layer)
            l = ent.layer
        elif ent.layer != l:
            layers.append(ent.layer)
            l = ent.layer
    return layers
    
def print_layer_names(filename):
    """ print the names of the layers in a dxf file """

    dxf = dxfgrabber.readfile(filename)

    #  get layer names, create ASC layer names
    layers = get_layer_names(dxf)
    for i, l in enumerate(layers):
        print('{0}:  {1}'.format(i, l))

# def map_layers_old(layers):
#     """ create ASCII layer numbers automatically based on my usual conventions """
#     new_layers = []
#     for i, l in enumerate(layers):
#         if l == '0':
#             new_layers.append(0)
#         elif 'align' in l.lower():
#             new_layers.append(63)
#         else:
#             m = re.search('q(\d+)', l.lower())
#             if (m != None):
#                 new_layers.append(int(m.group(1)))
#             else: 
#                 new_layers.append(i+50) #just give it a number that likely isn't being used
#     return new_layers

#  functions to handle creating a list of polygon vertices
#  from the dxf object

def strip_z(tuple_list):
    """ removes the unnecessary z component from tuples """
    return [t[0:2] for t in tuple_list]
    
def contains_closing_point(verts):
    # check that the polygon described by verts contains enough points to make a closed shape
    epsilon = 1e-11
    return np.all([abs(v)<epsilon for v in verts[0]-verts[-1]])

def add_closing_point(verts):
    if not contains_closing_point:
        return np.vstack((verts, verts[0]))
    else:
        return verts

def line2poly_const(ent):
    """ convert lines of constant width to filled polygons """
    
    centers = np.array(strip_z(ent.points)) # center points of line
    lower = np.zeros(centers.shape) # to hold vertices for lower parallel line
    upper = np.zeros(centers.shape) # to hold vertices for upper parallel line
    width = ent.const_width # line width

    diff = np.roll(centers,-1, axis=0)-centers # vectors representing each line segement
    phi = np.arctan2(diff[:,1],diff[:,0]) # angle each line segment makes with x-axis
    m = np.tan(phi) # slope of each line segment to avoid div by 0
    b_lower = centers[:,1]-m*centers[:,0]-0.5*width/np.cos(phi) # intercepts of lower parallel line
    b_upper = centers[:,1]-m*centers[:,0]+0.5*width/np.cos(phi) # intercepts of upper parallel lines

    # find all intersections, ignore endpoints
    eps = 1e9
    for i in range(1,ent.__len__()-1):
        if np.abs(m[i])<eps:
            a = m[i]
            bl = b_lower[i]
            bu = b_upper[i]
        elif np.abs(m[i-1])<eps:
            a = m[i-1]
            bl = b_lower[i-1]
            bu = b_upper[i-1]
        lower[i,0] = ((b_lower[i]-b_lower[i-1])/(m[i-1]-m[i]))
        lower[i,1] = a*((b_lower[i]-b_lower[i-1])/(m[i-1]-m[i]))+bl
        upper[i,0] = ((b_upper[i]-b_upper[i-1])/(m[i-1]-m[i]))
        upper[i,1] = a*((b_upper[i]-b_upper[i-1])/(m[i-1]-m[i]))+bu

    # find endpoints
    lower[0,0] = centers[0,0]+0.5*width*np.sin(phi[0])
    lower[0,1] = centers[0,1]-0.5*width*np.cos(phi[0])
    upper[0,0] = centers[0,0]-0.5*width*np.sin(phi[0])
    upper[0,1] = centers[0,1]+0.5*width*np.cos(phi[0])

    lower[-1,0] = centers[-1,0]+0.5*width*np.sin(phi[-2])
    lower[-1,1] = centers[-1,1]-0.5*width*np.cos(phi[-2])
    upper[-1,0] = centers[-1,0]-0.5*width*np.sin(phi[-2])
    upper[-1,1] = centers[-1,1]+0.5*width*np.cos(phi[-2])

    return np.vstack((lower, upper[::-1,:], [lower[0,:]]))

def same_shape(v0,v1):
    """ check if two lists of vertices contain the same points """
    # get out of here immediately if the number of points is different
    if v0.shape!=v1.shape:
        return False
    
    # sort points in some known order
    ind0 = sort_by_position(v0)
    ind1 = sort_by_position(v1)
    v0 = v0[ind0]
    v1 = v1[ind1]
    
    # check distance between points
    eps = 1e-3 # closer than 1nm is the same point
    dist = np.linalg.norm(v0-v1, axis=1)
    return np.all([d<eps for d in dist])

def remove_duplicate_polygons(poly_list):
    """ look through the list of polygons to see if any are repeated. print warning if they are. 
        
        returns: list of polygons with one of the duplicates removed """
    
    ind = []
    for i in range(len(poly_list)):
        for j in range(len(poly_list)):
            if j>=i:
                pass
            else:
                if same_shape(poly_list[i], poly_list[j]):
                    print('DUPLICATE POLYGON REMOVED ({0})'.format(i))
                    ind.append(i)
    return np.delete(poly_list, ind)

def get_vertices(dxf, layer):
    """ get list of vertices from dxf object """
    verts = []
    i = 0
    for ent in dxf.entities:
        if ent.layer == layer:
            i+=1
            
            if ent.dxftype == 'POLYLINE':
                verts.append(np.array(strip_z(ent.points)))
                
            if ent.dxftype == 'LWPOLYLINE':
                # logic to sort out what type of object ent is
                closed = ent.is_closed # closed shape
                if ent.width.__len__()==0:
                    width=False # not variable width
                else:
                    width = not all([t<0.001 for tt in ent.width for t in tt]) # maybe variable width
                cwidth = ent.const_width>0.001 # constant width
            
                if (closed and not (width or cwidth)): # closed polygons, lines have no width, easy
                    verts.append(np.array(strip_z(ent.points)))
                elif (cwidth and not (closed or width)): # lines with constant width
                    verts.append(line2poly_const(ent))
                elif (width and not (closed or cwidth)):
                    print('ENTITY ({0}). Lines of variable width not supported. DXFTYPE = LWPOLYLINE.'.format(i))
                elif (not width and not cwidth and not closed):
                    # if closed, cwidth, and width are all false it's an unclosed polygon
                    # fix it and continue, print warning about polygon being closed automatically
                    print('UNCLOSED POLYGON FIXED ({0})'.format(i))
                    v = np.array(strip_z(ent.points))
                    v = add_closing_point(v)
                    verts.append(v)
                else:
                    print('UKNOWN ENTITY ({0}). DXFTYPE = LWPOLYLINE'.format(i))
#                 
            else: 
                print('NOT A KNOWN TYPE ({0}) -- LAYER: {1}'.format(ent.dxftype, layer))
    return remove_duplicate_polygons(verts)

#  a series of functions to perform simple operations on a single set/list
#  of polygon vertices

def polyArea(verts0):
    """ find area of a polygon that has vertices in a numpy array
        verts = np.array([x0 y0], [x1 y1], ....) """
    verts1 = np.roll(verts0, -1, axis=0)
    return 0.5*np.sum(verts0[:,0]*verts1[:,1] - verts1[:,0]*verts0[:,1])

def polyCOM(verts0):
    """ find center of mass of a polygon that has vertices in a numpy array
        verts = np.array([x0 y0], [x1 y1], ....) """
    A = 1/(6*polyArea(verts0))
    verts1 = np.roll(verts0, -1, axis=0)
    C = verts0[:,0]*verts1[:,1] - verts1[:,0]*verts0[:,1]
    X = np.sum((verts0[:,0] + verts1[:,0])*C)
    Y = np.sum((verts0[:,1] + verts1[:,1])*C)
    return A*np.array([X, Y])

def polyPerimeter(verts0):
    """ find perimeter of a polygon that has vertices in a numpy array
        verts = np.array([x0 y0], [x1 y1], ....) """
    verts1 = np.roll(verts0, -1, axis=0)
    return np.sum(np.hypot(verts0[:,0] - verts1[:,0],verts0[:,1] - verts1[:,1]))

def polyUtility(verts_array, polyFunc):
    """ takes an array full of polygon vertices, as created by 
        get_vertices, and returns an array full of values returned by 
        polyFunc """
    return np.array([polyFunc(v) for v in verts_array])

#  dose calculation

def get_writefield(verts):
    """ estimate the writefield size """
    sizes = np.array([120, 500, 1000])
    corners = sizes*np.sqrt(2)/2.0
    dmax = 0
    for polygon in verts:
        for v in polygon:
            d = np.hypot(v[0],v[1])
            if (d > dmax): dmax = d
    return sizes[(dmax > corners).argmin()]

def geometry_to_dose(verts, doseMin, doseMax):
    """ takes an array of polygon vertices. returns and array of dose values calculated
        by dividing perimeter by area and using some empirical evidence to scale to the 
        proper range of doses. the total doses are scaled and limited by doseMin and doseMax. """

    data = polyUtility(verts, polyPerimeter)/abs(polyUtility(verts, polyArea))
    
    #different size scales for different writefields
    if get_writefield(verts) == 1000:
        pMin = 0.04; pMax = 1.1
    else:
        pMin = 1.0; pMax = 7.0
    
    #  split up range into 20 steps, round steps to nearest 10
    resolution =max(np.floor((doseMax-doseMin)/200)*10, 1.0)
    
    m = (doseMax-doseMin)/(pMax-pMin)
    b = doseMax - m*pMax
    
    #  clip data to within limits to make sure nothing gets a totally ridiculous dose
    #  round to nearest multiple of 'resolution' because this method can't be very accurate
    return np.clip(np.round(np.array([m*x + b for x in data])/resolution)*resolution, doseMin, doseMax)

#  sort polygons by size/location

def sort_by_dose(dose, com):
    """ takes a list of doses and centers of mass for all polygons in a layer.
        returns a list of indices that sorts those two arrays (and the vertex array)
        by dose then proximity to highest dose element. """
        
    #  sort by dose largest to smallest, then by distance from
    #  element with the largest dose (likely where the CNT is)
    
    center = com[np.argmax(dose)]
    dist = np.hypot(com[:,0]-center[0], com[:,1]-center[1])
    return np.lexsort((-dist, dose))[::-1]

def sort_by_position(com):
    """ same as sort layer, but it sorts the alignment marker layers from bottom left
        to top right. """
         
    X = -np.round(com)[:,0]
    Y = -np.round(com)[:,1]
    return np.lexsort((X, Y))[::-1]

#  two functions to write data in the proper ASCII format

def verts_block(verts):
    """ verticies to block of text """
    s = ''
    for v in verts:
        s += '{0:.4f} {1:.4f} \n'.format(v[0], v[1])
        
    # some versions of dxf give different results here....
    # add the first vertex to the end of the list, unless it is 
    # already there
    if '{0:.4f} {1:.4f} \n'.format(verts[0][0], verts[0][1]) == \
        '{0:.4f} {1:.4f} \n'.format(verts[-1][0], verts[-1][1]):
        return s
    else:
        return s + '{0:.4f} {1:.4f} \n'.format(verts[0][0], verts[0][1])

def write_layer(f, verts, dose, layer, setDose=None):
    """ Writes all vertices in a layer to an ASCII file.

        Args: f: open file object
              verts: numpy array of vertex lists
              dose: numpy array of doses
              layer: ASCII layer number
              setDose: doses will be scaled to a % of this value
                       if setDose is None doses will be written as
                       passed to this function.

        Returns: None """
              
    for i in range(len(verts)):
        if setDose:
            d = dose[i]/setDose*100.0
        else:
            d = dose[i]
        f.write('1 {0:.3f} {1:d} \n'.format(d, layer))
        f.write(verts_block(verts[i]) + '# \n')

#  one final function to rule them all...

def convert_to_asc(filename, doseMin, doseMax, map_layers):
    """ load dxf file, scale dose data using simple proximity correction, 
        order elements by size/location, export ASC file in Raith format. """

    # some stuff to handle files or filelists
    if type(filename)==type(''):
        filename = [filename]
    elif type(filename)==type([]):
        pass
    else:
        print("Enter an string or list of strings")
    
    for file in filename:
        print('working on file: {0}'.format(file))
        #  load dxf to dxfgrabber object
        dxf = dxfgrabber.readfile(file)

        #  get layer names, create ASC layer names
        layers = get_layer_names(dxf)
        # pick a map_layers function based on whatever you're current conventions are
        new_layers = map_layers(layers)

        f = open(file[:-4]+'.asc', 'w')
    
        for i in np.argsort(new_layers)[::-1]:
            l = new_layers[i]
            if l == 0:
                continue
            elif l == 63:
                verts = get_vertices(dxf, layers[i])
                com = polyUtility(verts, polyCOM)
                ind_sorted = sort_by_position(com)
                write_layer(f, verts[ind_sorted], np.ones(len(verts))*100.0, l)
            else:
                verts = get_vertices(dxf, layers[i])
                com = polyUtility(verts, polyCOM)
                dose = geometry_to_dose(verts, doseMin, doseMax)
                ind_sorted = sort_by_position(com)
                write_layer(f, verts[ind_sorted], dose[ind_sorted], l, setDose = doseMin)
        f.close()

def plot_layer(filename, layername):
    """ plot the given layer name to check the results """
       
    dxf = dxfgrabber.readfile(filename)
    verts = get_vertices(dxf, layername)   
    
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    polycol = PolyCollection(verts)
    ax.add_collection(polycol)
    ax.autoscale_view(True, True, True)
    
    lim = [ax.get_xlim(), ax.get_ylim()]
    m = round(np.abs(lim).max())
    ax.set_xlim(-m,m)
    ax.set_ylim(-m,m)
    ax.set_title('{0}'.format(layername.upper()))
    
    ax.grid()
    plt.show()
    
def plot_sample(samplename, layer_id, size, save = False):
    """ plot the entire device.  
    
        filelist -- a list of all of the relevant dxf files
        layer_id -- something to search for in the layer names 
        size -- a tuple giving (xlim, ylim) 
        
        this will save me from having to screengrab crap from Illustrator."""
    
    filelist = glob.glob(samplename+'_*.dxf')
    
    fig = plt.figure(figsize=(12,11))
    ax = fig.add_subplot(111)
    colors = itertools.cycle([plt.cm.Accent(i) for i in np.linspace(0, 1, 6)])
    
    for f in filelist:
        dxf = dxfgrabber.readfile(f)
        layers = get_layer_names(dxf)
        for l in layers:
            if layer_id.lower() in l.lower():
                verts = get_vertices(dxf, l) 
                polycol = PolyCollection(verts, facecolor=next(colors))
                ax.add_collection(polycol) 
    
    xlim = round(size[0]/2.0)
    ylim = round(size[1]/2.0)
    ax.set_xlim(-xlim,xlim)
    ax.set_ylim(-ylim,ylim)
    ax.set_title('{0} {1}'.format(samplename, layer_id))
    ax.grid()
    
    if save:
        fig.savefig('{0}_{1}.png'.format(samplename.lower(),layer_id.lower()),
                    dpi = 100)
    plt.show()