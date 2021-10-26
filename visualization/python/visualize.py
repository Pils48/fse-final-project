"""
Visualization module
"""

import os
import os.path
from util import *
from util_vtk import visualization

if __name__ == '__main__':
    import argparse
    CMD_PARSER = argparse.ArgumentParser(description="""Visualizing .mat voxel file. """)
    CMD_PARSER.add_argument('-t', '--threshold', metavar='threshold', type=float, default=0.1,
                            help='voxels with confidence lower than the threshold are not displayed')
    CMD_PARSER.add_argument('-i', '--index', metavar='index', type=int, default=1,
                            help='the index of objects in the inputfile that should be rendered (one based)')
    CMD_PARSER.add_argument('filename', metavar='filename', type=str,
                            help='name of .torch or .mat file to be visualized')
    CMD_PARSER.add_argument('-df', '--downsample-factor', metavar='factor', type=int, default=1,
                            help="downsample objects via a max pooling of step STEPSIZE for efficiency. Currently supporting STEPSIZE 1, 2, and 4.")
    CMD_PARSER.add_argument('-dm', '--downsample-method', metavar='downsample_method', type=str, default='max',
                            help='downsample method, where mean stands for average pooling and max for max pooling')
    CMD_PARSER.add_argument('-u', '--uniform-size', metavar='uniform_size', type=float, default=0.9,
                            help='set the size of the voxels to BLOCK_SIZE')
    CMD_PARSER.add_argument('-cm', '--colormap', action="store_true",
                            help='whether to use a colormap to represent voxel occupancy, or to use a uniform color')
    CMD_PARSER.add_argument('-mc', '--max-component', metavar='max_component', type=int, default=3,
                            help='whether to keep only the maximal connected component, where voxels of distance no larger than `DISTANCE` are considered connected. Set to 0 to disable this function.')

    ARGS = CMD_PARSER.parse_args()
    FILENAME = ARGS.filename
    MATNAME = 'voxels'
    THRESHOLD = ARGS.threshold
    IND = ARGS.index - 1 # matlab use 1 base index
    DOWNSAMPLE_FACTOR = ARGS.downsample_factor
    DOWNSAMPLE_METHOD = ARGS.downsample_method
    UNIFORM_SIZE = ARGS.uniform_size
    USE_COLORMAP = ARGS.colormap
    CONNECT = ARGS.max_component

    assert DOWNSAMPLE_METHOD in ('max', 'mean')

    # read file
    print("==> Reading input voxel file: "+FILENAME)
    voxels_raw = read_tensor(FILENAME, MATNAME)
    print("Done")

    voxels = voxels_raw[IND]

    # keep only max connected component
    print("Looking for max connected component")
    if CONNECT > 0:
        voxels_keep = (voxels >= THRESHOLD)
        voxels_keep = max_connected(voxels_keep, CONNECT)
        voxels[np.logical_not(voxels_keep)] = 0

    # downsample if needed
    if DOWNSAMPLE_FACTOR > 1:
        print("==> Performing downsample: factor: "+str(DOWNSAMPLE_FACTOR)+" method: "+DOWNSAMPLE_METHOD)
        voxels = downsample(voxels, DOWNSAMPLE_FACTOR, method=DOWNSAMPLE_METHOD)
        print("Done")

    visualization(voxels, THRESHOLD, title=str(ind+1)+'/'+str(voxels_raw.shape[0]),
                  uniform_size=UNIFORM_SIZE, use_colormap=USE_COLORMAP)
