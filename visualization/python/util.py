"""
Tools for the next steps of visualization
"""

import numpy as np
from scipy import ndimage
from scipy.io import loadmat

def read_tensor(filename, varname='voxels'):
    """ return a 4D matrix, with dimensions point, x, y, z """
    assert filename[-4:] == '.mat'
    mats = loadmat(filename)
    if varname not in mats:
        print(".mat file only has these matrices:")
        for var in mats:
            print(var)
        assert False

    voxels = mats[varname]
    dims = voxels.shape
    if len(dims) == 5:
        assert dims[1] == 1
        dims = (dims[0],) + tuple(dims[2:])
    elif len(dims) == 3:
        dims = (1) + dims
    else:
        assert len(dims) == 4
    result = np.reshape(voxels, dims)
    return result

def sigmoid(z_var, offset=0, ratio=1):
    """
    Sigmoid function
    """
    return 1.0 / (1.0 + np.exp(-1.0 * (z_var-offset) * ratio))

############################################################################
### Voxel Utility functions
############################################################################
def blocktrans_cen2side(center_size):
    """ Convert from center rep to side rep
    In center rep, the 6 numbers are center coordinates, then size in 3 dims
    In side rep, the 6 numbers are lower x, y, z, then higher x, y, z """
    center_x = float(center_size[0])
    center_y = float(center_size[1])
    center_z = float(center_size[2])
    side_x = float(center_size[3])
    side_y = float(center_size[4])
    side_z = float(center_size[5])
    lower_x, lower_y, lower_z = center_x-side_x/2., center_y-side_y/2., center_z-side_z/2.
    high_x, high_y, high_z = center_x+side_x/2., center_y+side_y/2., center_z+side_z/2.
    return [lower_x, lower_y, lower_z, high_x, high_y, high_z]

def blocktrans_side2cen6(side_size):
    """ Convert from side rep to center rep
    In center rep, the 6 numbers are center coordinates, then size in 3 dims
    In side rep, the 6 numbers are lower x, y, z, then higher x, y, z """
    lower_x, lower_y, lower_z = float(side_size[0]), float(side_size[1]), float(side_size[2])
    high_x, high_y, high_z = float(side_size[3]), float(side_size[4]), float(side_size[5])
    half_x = (lower_x+high_x)*.5
    half_y = (lower_y+high_y)*.5
    half_z = (lower_z+high_z)*.5
    abs_x = abs(high_x-lower_x)
    abs_y = abs(high_y-lower_y)
    abs_z = abs(high_z-lower_z)
    return [half_x, half_y, half_z, abs_x, abs_y, abs_z]


def center_of_mass(voxels, threshold=0.1):
    """ Calculate the center of mass for the current object.
    Voxels with occupancy less than threshold are ignored
    """
    assert voxels.ndim == 3
    center = [0]*3
    voxels_filtered = np.copy(voxels)
    voxels_filtered[voxels_filtered < threshold] = 0

    total = voxels_filtered.sum()
    if total == 0:
        print('threshold too high for current object.')
        return [length / 2 for length in voxels.shape]

    # calculate center of mass
    center[0] = np.multiply(voxels_filtered.sum(1).sum(1), np.arange(voxels.shape[0])).sum()/total
    center[1] = np.multiply(voxels_filtered.sum(0).sum(1), np.arange(voxels.shape[1])).sum()/total
    center[2] = np.multiply(voxels_filtered.sum(0).sum(0), np.arange(voxels.shape[2])).sum()/total

    return center

def downsample(voxels, step, method='max'):
    """
    downsample a voxels matrix by a factor of step.
    downsample method options: max/mean
    same as a pooling
    """
    assert step > 0
    assert voxels.ndim == 3 or voxels.ndim == 4
    assert method in ('max', 'mean')
    if step == 1:
        return voxels

    if voxels.ndim == 3:
        s_x, s_y, s_z = voxels.shape[-3:]
        r_X, r_Y, r_Z = np.ogrid[0:s_x, 0:s_y, 0:s_z]
        regions = s_z/step * s_y/step * (r_X/step) + s_z/step * (r_Y/step) + r_Z/step
        if method == 'max':
            res = ndimage.maximum(voxels, labels=regions, index=np.arange(regions.max() + 1))
        elif method == 'mean':
            res = ndimage.mean(voxels, labels=regions, index=np.arange(regions.max() + 1))
        res.shape = (s_x/step, s_y/step, s_z/step)
        return res
    else:
        res0 = downsample(voxels[0], step, method)
        res = np.zeros((voxels.shape[0],) + res0.shape)
        res[0] = res0
        for ind in range(1, voxels.shape[0]):
            res[ind] = downsample(voxels[ind], step, method)
        return res

def max_connected(voxels, distance):
    """
    Keep the max connected component of the voxels (a boolean matrix).
    distance is the distance considered as neighbors, i.e. if distance = 2,
    then two blocks are considered connected even with a hole in between
    """
    assert distance > 0
    max_component = np.zeros(voxels.shape, dtype=bool)
    voxels = np.copy(voxels)
    for start_x in range(voxels.shape[0]):
        for start_y in range(voxels.shape[1]):
            for start_z in range(voxels.shape[2]):
                if not voxels[start_x, start_y, start_z]:
                    continue
                # start a new component
                component = np.zeros(voxels.shape, dtype=bool)
                stack = [[start_x, start_y, start_z]]
                component[start_x, start_y, start_z] = True
                voxels[start_x, start_y, start_z] = False
                while len(stack) > 0:
                    coord_x, coord_y, coord_z = stack.pop()
                    for i in range(coord_x-distance, coord_x+distance + 1):
                        for j in range(coord_y-distance, coord_y+distance + 1):
                            for k in range(coord_z-distance, coord_z+distance + 1):
                                if (i-coord_x)**2+(j-coord_y)**2+(k-coord_z)**2 > distance*distance:
                                    continue
                                if voxel_exist(voxels, i, j, k):
                                    voxels[i, j, k] = False
                                    component[i, j, k] = True
                                    stack.append([i, j, k])
                if component.sum() > max_component.sum():
                    max_component = component
    return max_component


def voxel_exist(voxels, coord_x, coord_y, coord_z):
    """
    Check if voxels are in given bounds
    """
    neg_coords = coord_x < 0 or coord_y < 0 or coord_z < 0
    b_x, b_y, b_z = voxels.shape
    coords_out_of_bounds = coord_x >= b_x or coord_y >= b_y or coord_z >= b_z
    if neg_coords or coords_out_of_bounds:
        return False
    else:
        return voxels[coord_x, coord_y, coord_z]
