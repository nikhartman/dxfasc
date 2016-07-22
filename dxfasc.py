""" This module offers basic support for converting DXF drawings to 
    the ASCII formats supported by Raith and NPGS ebeam lithography software.
    
    The module has not been extensively tested. It may only work in a few use cases. 
    
    The package dxfgrabber is required for DXF read/write operations. """

import glob, itertools
import numpy as np
import re
import dxfgrabber
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

##############################################
### Functions for dealing with layer names ###
##############################################

def get_layer_names(dxf):

    """ Get list of layer names. Only lists layers that contain objects.
        Any artwork in layer 0 is ignored. Layer 0 however is added as an
        empty layer in this list, as this is required for some versions of the
        Raith software. 
        
        All characters are converted to caps and spaces to _. 
        
        Args:
            dxf (dxfgrabber object): dxfgrabber object refering to the drawing of interest 
        Returns:
            list (str): list of layer names """
        
    layers = ['0'] # this empty layer is required
    for i, ent in enumerate(dxf.entities):
    
        l = ent.layer.upper().replace (" ", "_")
        
        if i==0: 
            layers.append(l) # definitely add the first layer name
        elif l not in layers:
            layers.append(l) # add the layer name if it is not already included. 
    return layers

def print_layer_names(filename):
    """ Print all layers in a DXF file that contain artwork. Layer 0 is added 
        as an empty layer. 
        
        Args: 
            filename (str): name of DXF file """

    dxf = dxfgrabber.readfile(filename)

    layers = get_layer_names(dxf)
    for i, l in enumerate(layers):
        print('{0}:  {1}'.format(i, l))

##################################################################################
### Functions to normalize a list of DXF shapes into a standard polygon format ###
##################################################################################

def strip_z(tuple_list):
    """ Removes the unnecessary z component from tuples. Specifically a problem
        with Adobe Illustrator imports. 
        
        Args:
            tuple_list (list): a list of tuples in (x,y,z) or (x,y) form 

        Returns:
            list: a list of tuples in (x,y) form """
        
    return [t[0:2] for t in tuple_list]
    
def contains_closing_point(verts):
    """ Check that the polygon described by verts contains
        a closing point.
        
        Args:
            verts (list): a list of vertices in the form np.array([x,y])
        Returns:
            bool: True if verts contains a closing point. False otherwise. """
        
    eps = 1e-11 # assuming this is roughly the floating point accuracy
    return np.all([abs(v)<eps for v in verts[0]-verts[-1]])
        
def close_all_polygons(poly_list, warnings = True):
    """ Go through poly_list and look for polygons that are not closed
        (first point the same as last point). 
        
        Args:
            poly_list (list): list of 2D numpy arrays that contain x,y vertices defining polygons
        Kwargs:
            warnings (bool): True if you want warnings to print to the terminal
        
        returns: list of polygons with one of the duplicates removed """

    for i in range(len(poly_list)):
        if not contains_closing_point(poly_list[i]):
            poly_list[i] = np.vstack((poly_list[i], poly_list[i][0]))
            if warnings:
                print('POLYGON CLOSED ({0})'.format(i))

    return poly_list

def same_shape(verts0,verts1):
    """ Check if two lists of vertices contain the same points. 
    
        Args:
            verts0 (list): list of (x,y) vertices for polygon 0
            verts1 (list): list of (x,y) vertices for polygon 1
            
        Returns: 
            bool: True if verts0 and vert1 describe the same polygon """
            
    # get out of here immediately if the number of points is different
    if verts0.shape!=verts1.shape:
        return False
    
    # sort points in some known order
    ind0 = sort_by_position(verts0)
    ind1 = sort_by_position(verts1)
    verts0 = verts0[ind0]
    verts1 = verts1[ind1]
    
    # check distance between points
    eps = 1e-3 # closer than 1nm is the same point
    dist = np.linalg.norm(verts0-verts1, axis=1)
    return np.all([d<eps for d in dist])

def remove_duplicate_polygons(poly_list, warnings=True):
    """ Look through the list of polygons to see if any are repeated. Print warning if they are. 
        
        Args:
            poly_list (list): list of 2D numpy arrays that contain x,y vertices defining polygons
        Kwargs:
            warnings (bool): True if you want warnings to print to the terminal
            
        Returns: 
            list: modified poly_list with duplicates removed """
    
    ind = []
    for i in range(len(poly_list)):
        for j in range(len(poly_list)):
            if j>=i:
                pass
            else:
                if same_shape(poly_list[i], poly_list[j]):
                    if warnings:
                        print('DUPLICATE POLYGON REMOVED ({0})'.format(i))
                    ind.append(i)
    return np.delete(poly_list, ind)
    
def normalize_polygon_orientation(poly_list, warnings = True):
    """ Make sure all polygons have their vertices listed in counter-clockwise order.
    
        Args:
            poly_list (list): list of 2D numpy arrays that contain x,y vertices defining polygons
        Kwargs:
            warnings (bool): True if you want warnings to print to the terminal
            
        Returns: 
            list: modified poly_list with properly rotated polygons """
            
    for i in range(len(poly_list)):
        if polyArea(poly_list[i])<0:
            poly_list[i] = poly_list[i][::-1]
        
    return poly_list

def line2poly_const(ent):
    """ Convert lines of constant width to filled polygons. 
    
        Args:
            ent (dxfgrabber entity): an object representing a single line in the DXF pattern
        Returns:
            list (verts): a list of vertices defining the polygon that is equivalent to the 
                line of constant width. Vertices are in the form [np.array([x,y])] """
    
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

def get_vertices(dxf, layers, warnings=True):
    """ Get list of vertices from dxf object. 
    
        This is certainly full of bugs. It has only been tested with Illustrator CS5 
        and AutoCAD 2015. There are many object types that are not supported. Ideally, something
        useful will be printed to notify you about what is missing. 

        Args:
            dxf (dxfgrabber object): dxfgrabber object refering to the drawing of interest 
            layers (str, list): string or list of strings defining 
                which layers will be imported.
                
        Returns:
            list: list of polygon vertices as 2D numpy arrarys. """
        
    # make sure layers is a list
    if type(layers)==type(''):
        layers = [layers]
    elif type(layers)==type([]):
        pass
    else:
        raise TypeError("Layers should be a string or list of strings")
        
    # get all layer names in dxf
    all_layers = get_layer_names(dxf)
            
    # loop through layers to create poly_list
    poly_list = []
    i = 0
    for l in layers:
        layer = l.upper().replace(' ','_')
        
        if layer not in all_layers:
            if warnings:
                print('LAYER NOT FOUND IN DRAWING -- {0}'.format(l))
            continue
        if (layer=='0' or layer==0):
            if warnings:
                print('DO NOT USE LAYER 0 FOR DRAWINGS')
            continue
         
        for ent in dxf.entities:
            if ent.layer.upper().replace(' ', '_') == layer:
                i+=1
            
                if ent.dxftype == 'POLYLINE':
                    poly_list.append(np.array(strip_z(ent.points)))
                
                if ent.dxftype == 'LWPOLYLINE':
            
                    # logic to sort out what type of object ent is
                    closed = ent.is_closed # closed shape
                    if ent.width.__len__()==0:
                        width=False # not variable width
                    else:
                        width = not all([t<0.001 for tt in ent.width for t in tt]) # may be variable width
                    cwidth = ent.const_width>0.001 # constant width
            
                    if (closed and not (width or cwidth)): # closed polygons, lines have no width
                        poly_list.append(np.array(strip_z(ent.points)))
                    elif (cwidth and not (closed or width)): # lines with constant width
                        poly_list.append(line2poly_const(ent))
                    elif (width and not (closed or cwidth)):
                        if warnings:
                            print('ENTITY ({0}). Lines of variable width not supported. DXFTYPE = LWPOLYLINE.'.format(i))
                    elif (not width and not cwidth and not closed):
                        # if closed, cwidth, and width are all false it's an unclosed polygon
                        # add it to the list and fix it later
                        poly_list.append(np.array(strip_z(ent.points)))
                    
                    else:
                        if warnings:
                            print('UKNOWN ENTITY ({0}). DXFTYPE = LWPOLYLINE'.format(i))
                            
                # add additional dxftypes here
                    
                else:
                    if warnings:
                        print('NOT A KNOWN TYPE ({0}) -- LAYER: {1}'.format(ent.dxftype, layer))
    
    poly_list = close_all_polygons(poly_list, warnings = True) # make sure all polygons are closed
    poly_list = remove_duplicate_polygons(poly_list, warnings=True) # remove duplicates
    poly_list = normalize_polygon_orientation(poly_list, warnings=True) # orient all polygons counter-clockwise
    # order polygon lists such that the longest side, nearest to the lower left corner comes first
    return poly_list
    
####################
### Polygon math ###
####################

def polyArea(verts0):
    """ Find area of a polygon that has vertices in a numpy array
        
        Args:
            verts (array): np.array([x0 y0], [x1 y1], ....) 
        Returns:
            float: Area of polygon. Sign gives orientation (<0 clockwise). """
            
    verts1 = np.roll(verts0, -1, axis=0)
    return 0.5*np.sum(verts0[:,0]*verts1[:,1] - verts1[:,0]*verts0[:,1])

def polyCOM(verts0):
    """ Find center of mass of a polygon that has vertices in a numpy array
    
        Args:
            verts (array): np.array([x0 y0], [x1 y1], ....) 
        Returns:
            array: np.array([x_com, y_com])"""
            
    A = 1/(6*polyArea(verts0))
    verts1 = np.roll(verts0, -1, axis=0)
    C = verts0[:,0]*verts1[:,1] - verts1[:,0]*verts0[:,1]
    X = np.sum((verts0[:,0] + verts1[:,0])*C)
    Y = np.sum((verts0[:,1] + verts1[:,1])*C)
    return A*np.array([X, Y])

def polyPerimeter(verts0):
    """ Find perimeter length of a polygon that has vertices in a numpy array.
    
        Args:
            verts (array): np.array([x0 y0], [x1 y1], ....) 
        Returns:
            float: length of the polygon perimenter. """
            
    verts1 = np.roll(verts0, -1, axis=0)
    return np.sum(np.hypot(verts0[:,0] - verts1[:,0],verts0[:,1] - verts1[:,1]))

def polyUtility(poly_list, polyFunc):
    """ Takes an array full of polygon vertices, as created by 
        get_vertices, and returns an array full of values returned by 
        polyFunc
        
        Args:
            poly_list (list): list of 2D numpy arrays defining the vertices of a number of polygons
            polyFun (function): a function to apply to the list of polygons
        Returns:
            list: output of polyFunc for each polygon in poly_list """
            
    return np.array([polyFunc(v) for v in poly_list])
    
# def all_polygon_COM(dxf, layers):
#     """ get center of mass for layers """
#     
#     verts = get_vertices(layers)
#             
#     com = polyUtility(verts, polyCOM)
#     area = np.abs(polyUtility(verts, polyArea))
#     
#     return np.array([(area*com[:,0]).sum(), (area*com[:,1]).sum()])/area.sum()
    
# def bounding_box(dxf, layers, origin='ignore'): # this should be included in get_vertices
#     """ Find bounding box and proper coordinates 
#     
#         Args:
#             dxf: dxfgrabber object
#             layers -- list of layers to include in calculations
#             origin -- where the (0,0) coordinate should be located 
#             
#         Returns:
#             ll (np.array): x,y coordiates of lower left corner of drawing after shift
#             ur (np.array): x,y coordiates of upper right corner of drawing after shift
#             center (np.array): x,y coordinates of center point after shift
#             bsize (float): size of smallest bounding box (nearest micron)
#             shift (np.array): all x,y coordinates must be shifted by this vector """
#     
#     if type(layers)==type(''):
#         layers = [layers]
#     elif type(layers)==type([]):
#         pass
#     else:
#         print("Layers should be a string or list of strings")
#         
#     verts = np.vstack(get_vertices(dxf, layers, warnings=False))
# 
#     xmin = verts[:,0].min()
#     xmax = verts[:,0].max()
#     ymin = verts[:,1].min()
#     ymax = verts[:,1].max()
#     
#     ll = np.array([xmin, ymin])
#     ur = np.array([xmax, ymax])
#     center = np.array([xmin+xmax, ymin+ymax])/2.0
#     bsize = np.ceil(max(xmax-xmin, ymax-ymin))
# 
#     if origin=='lower':
#         shift = (-1)*(center-bsize/2.0)
#         return ll+shift, ur+shift, center+shift, bsize, shift
#     elif origin=='center':
#         shift = (-1)*center
#         return ll+shift, ur+shift, center+shift, bsize, shift
#     else:
#         shift = np.array([0,0])
#         return ll, ur, center, bsize, shift
        
#  dose calculation

def get_writefield(poly_list):
    """ Print the writefield size to the nearest micron.
        
        Args: 
            poly_list (list): list of 2D numpy arrays that contain x,y vertices defining polygons 
        
        Returns:
            size (float): writefield size in microns 
            center (float): center of writefield (x,y) """
            
    # put something here 

# def geometry_to_dose(verts, doseMin, doseMax):
#     """ takes an array of polygon vertices. returns and array of dose values calculated
#         by dividing perimeter by area and using some empirical evidence to scale to the 
#         proper range of doses. the total doses are scaled and limited by doseMin and doseMax. """
# 
#     data = polyUtility(verts, polyPerimeter)/abs(polyUtility(verts, polyArea))
#     
#     #different size scales for different writefields
#     if get_writefield(verts) == 1000:
#         pMin = 0.04; pMax = 1.1
#     else:
#         pMin = 1.0; pMax = 7.0
#     
#     #  split up range into 20 steps, round steps to nearest 10
#     resolution =max(np.floor((doseMax-doseMin)/200)*10, 1.0)
#     
#     m = (doseMax-doseMin)/(pMax-pMin)
#     b = doseMax - m*pMax
#     
#     #  clip data to within limits to make sure nothing gets a totally ridiculous dose
#     #  round to nearest multiple of 'resolution' because this method can't be very accurate
#     return np.clip(np.round(np.array([m*x + b for x in data])/resolution)*resolution, doseMin, doseMax)

def geometry_to_dose(verts, doseMin, doseMax):
    """ calculate approximate width of polygon. scale ebeam dose accordingly. """
    
    widths = 2*dxfasc.polyUtility(
                        polyverts, dxfasc.polyArea)/dxfasc.polyUtility(
                                            polyverts, dxfasc.polyPerimeter)
    
    # previous script gave everything over 2um doseMax
    # everything under 280nm doseMin
    # what to do now.... something simpler

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

    n = 0.2
    X = -np.round(com/n)[:,0]*n
    Y = -np.round(com/n)[:,1]*n
    return np.lexsort((X, Y))[::-1]

#####################################
### ASC output for Raith software ###
#####################################

def verts_block_asc(verts):
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

def write_layer_asc(f, verts, dose, layer, setDose=None):
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
        f.write(verts_block_asc(verts[i]) + '# \n')

####################################
### DC2 output for NPGS software ###
####################################

def write_header_dc2(f, ll, ur, layers):
    """ Write header for dc2 file. """
    header = '{0:.4f} {1:.4f} {2:.4f} {3:.4f} 0 -0.0000 0.0000\r\n'.format(
                ll[0]*8, ll[1]*8, (ur[0]-ll[0])*8, (ur[1]-ll[1])*8)
    header +=  ('42 20 0 0 0 0\r\n'
                '8.000000\r\n'
                '8.000000, 0.800000\r\n'
                '8.000000\r\n'
                '3\r\n'
                '16.000000\r\n'
                '0.000000\r\n'
                '0.000000\r\n'
                '1.000000\r\n'
                '1\r\n'
                '1\r\n'
                '1\r\n'
                'SIMPLEX2.VFN\r\n'
                '0.00000000 0.00000000 0.00000000 0.00000000\r\n'
                '0.00000000 0.00000000 0.00000000 0.00000000\r\n'
                '0.00000000 0.00000000 0.00000000 0.00000000\r\n'
                '0.00000000 0.00000000 0.00000000 0.00000000\r\n'
                '1 0 0 0 0 0 0 0\r\n'
                '0.00000000 0.00000000 0.00000000 0.00000000\r\n'
                '0.00000000 0.00000000 0.00000000 0.00000000\r\n'
                '; DesignCAD Drawing Comments /w \';\' as 1st char.\r\n')
    
    header += '23 {0} 0 0 0 0 \r\n'.format(len(layers)+1)
    header += 'DO NOT USE\r\n'
    for l in layers:
        header+='{0}\r\n'.format(l)
    
    f.write(header)
        
def verts_block_dc2(vert, color):
    """ Create block of text that defines each closed polygon. 
    
        This assumes that all objects have been converted to closed polygons. """
        
    # (type=line) (num of points in polygon) (hatching) (line width) (line type) ...
    #     (13) (0) (1) (R G B) (0) (1)
    line_hatch = 0.1 # 100nm hatching
    line_width = 0 # line width=0 for closed polygons
    line_type = 1 # 0 solid, 1 dashed (solid for wide lines, 0 for closed/filled polygons)
    block = '1 {0:d} {1:.4f} {2:.4f} {3:d} 13 0 1 0 {4:d} {5:d} {6:d} 0 1\r\n'.format(
            len(vert), line_hatch*8, line_width*8, line_type, color[0], color[1], color[2])
    for v in vert:
        block += '{0:.4f} {1:.4f} 0\r\n'.format(v[0]*8, v[1]*8)
    return block
    
##########################
### Plotting functions ###
##########################

def plot_layers(filename, layers, size):
    """ Plot the given layers. """
       
    dxf = dxfgrabber.readfile(filename)

    
# def plot_sample(samplename, layer_id, size, save = False):
#     """ plot the entire device.  
#     
#         filelist -- a list of all of the relevant dxf files
#         layer_id -- something to search for in the layer names 
#         size -- a tuple giving (xlim, ylim) 
#         
#         this will save me from having to screengrab crap from Illustrator."""
#     
#     filelist = glob.glob(samplename+'_*.dxf')
#     
#     fig = plt.figure(figsize=(12,11))
#     ax = fig.add_subplot(111)
#     colors = itertools.cycle([plt.cm.Accent(i) for i in np.linspace(0, 1, 6)])
#     
#     for f in filelist:
#         dxf = dxfgrabber.readfile(f)
#         layers = get_layer_names(dxf)
#         for l in layers:
#             if layer_id.lower() in l.lower():
#                 verts = get_vertices(dxf, l) 
#                 polycol = PolyCollection(verts, facecolor=next(colors))
#                 ax.add_collection(polycol) 
#     
#     xlim = round(size[0]/2.0)
#     ylim = round(size[1]/2.0)
#     ax.set_xlim(-xlim,xlim)
#     ax.set_ylim(-ylim,ylim)
#     ax.set_title('{0} {1}'.format(samplename, layer_id))
#     ax.grid()
#     
#     if save:
#         fig.savefig('{0}_{1}.png'.format(samplename.lower(),layer_id.lower()),
#                     dpi = 100)
#     plt.show()

#### depreciated asc functions from the markovic lab ####

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
                write_layer_asc(f, verts[ind_sorted], np.ones(len(verts))*100.0, l)
            else:
                verts = get_vertices(dxf, layers[i])
                com = polyUtility(verts, polyCOM)
                dose = geometry_to_dose(verts, doseMin, doseMax)
                ind_sorted = sort_by_position(com)
                write_layer_asc(f, verts[ind_sorted], dose[ind_sorted], l, setDose = doseMin)
        f.close()