import unittest
import numpy as np
from visualization.python.util import *
from visualization.python.util_vtk import *

class Test_sigmoid(unittest.TestCase):

    def test_valid_1(self):
        z_var = 0
        offset = 0
        ratio = 1
        a = 1.0 / (1.0 + np.exp(-1.0 * (z_var-offset) * ratio))
        self.assertEqual(a, sigmoid(z_var, offset=offset, ratio=ratio))

    def test_valid_2(self):
        z_var = 1
        offset = -5
        ratio = 2.5
        a = 1.0 / (1.0 + np.exp(-1.0 * (z_var-offset) * ratio))
        self.assertEqual(a, sigmoid(z_var, offset=offset, ratio=ratio))

    def test_valid_3(self):
        z_var = np.pi
        offset = 0
        ratio = 0
        a = 1.0 / (1.0 + np.exp(-1.0 * (z_var-offset) * ratio))
        self.assertEqual(a, sigmoid(z_var, offset=offset, ratio=ratio))
        

class Test_blocktrans_cen2side(unittest.TestCase):

    def test_valid_1(self):
        center_size = [0, 1, 2, 3, 4, 5]
        center_x = float(center_size[0])
        center_y = float(center_size[1])
        center_z = float(center_size[2])
        side_x = float(center_size[3])
        side_y = float(center_size[4])
        side_z = float(center_size[5])
        lower_x, lower_y, lower_z = center_x-side_x/2., center_y-side_y/2., center_z-side_z/2.
        high_x, high_y, high_z = center_x+side_x/2., center_y+side_y/2., center_z+side_z/2.
        a = [lower_x, lower_y, lower_z, high_x, high_y, high_z]
        self.assertEqual(a, blocktrans_cen2side(center_size))

    def test_valid_2(self):
        center_size = [0, 0, 0, 1, 0, 0]
        center_x = float(center_size[0])
        center_y = float(center_size[1])
        center_z = float(center_size[2])
        side_x = float(center_size[3])
        side_y = float(center_size[4])
        side_z = float(center_size[5])
        lower_x, lower_y, lower_z = center_x-side_x/2., center_y-side_y/2., center_z-side_z/2.
        high_x, high_y, high_z = center_x+side_x/2., center_y+side_y/2., center_z+side_z/2.
        a = [high_x, high_y, high_z, lower_z, high_y, high_z]
        self.assertNotEqual(a, blocktrans_cen2side(center_size))

    def test_valid_3(self):
        center_size = [-0, -1, -2, -np.pi, 4, 5, 0, 12, 4.5, np.pi]
        center_x = float(center_size[0])
        center_y = float(center_size[1])
        center_z = float(center_size[2])
        side_x = float(center_size[3])
        side_y = float(center_size[4])
        side_z = float(center_size[5])
        lower_x, lower_y, lower_z = center_x-side_x/2., center_y-side_y/2., center_z-side_z/2.
        high_x, high_y, high_z = center_x+side_x/2., center_y+side_y/2., center_z+side_z/2.
        a = [lower_x, lower_y, lower_z, high_x, high_y, high_z]
        self.assertEqual(a, blocktrans_cen2side(center_size))


class Test_blocktrans_side2cen6(unittest.TestCase):

    def test_valid_1(self):
        side_size = [0, 1, 2, 3, 3, 7, 5, 9]
        lower_x, lower_y, lower_z = float(side_size[0]), float(side_size[1]), float(side_size[2])
        high_x, high_y, high_z = float(side_size[3]), float(side_size[4]), float(side_size[5])
        half_x = (lower_x+high_x)*.5
        half_y = (lower_y+high_y)*.5
        half_z = (lower_z+high_z)*.5
        abs_x = abs(high_x-lower_x)
        abs_y = abs(high_y-lower_y)
        abs_z = abs(high_z-lower_z)
        a = [half_x, half_y, half_z, abs_x, abs_y, abs_z]
        self.assertEqual(a, blocktrans_side2cen6(side_size))

    def test_invalid_2(self):
        side_size = [0, 1, 2, 3, 3, 7, 5, 9]
        lower_x, lower_y, lower_z = float(side_size[0]), float(side_size[1]), float(side_size[2])
        high_x, high_y, high_z = float(side_size[3]), float(side_size[4]), float(side_size[5])
        half_x = (lower_x+high_x)*.5
        half_y = (lower_y+high_y)*.5
        half_z = (lower_z+high_z)*.5
        abs_x = abs(high_x-lower_x)
        abs_y = abs(high_y-lower_y)
        abs_z = abs(high_z-lower_z)
        a = [half_x, half_x, half_x, abs_x, abs_x, abs_x]
        self.assertNotEqual(a, blocktrans_side2cen6(side_size))


class Test_center_of_mass(unittest.TestCase):

    def test_valid_1(self):
        voxels = np.array([[[1, 2], [0.01, 0]], [[-1, -4], [np.pi, 0]]])
        threshold=0.1
        center = [0]*3
        voxels_filtered = np.copy(voxels)
        voxels_filtered[voxels_filtered < threshold] = 0
        total = voxels_filtered.sum()
        if total == 0:
            a = [length / 2 for length in voxels.shape]
        else:
            center[0] = np.multiply(voxels_filtered.sum(1).sum(1), np.arange(voxels.shape[0])).sum()/total
            center[1] = np.multiply(voxels_filtered.sum(0).sum(1), np.arange(voxels.shape[1])).sum()/total
            center[2] = np.multiply(voxels_filtered.sum(0).sum(0), np.arange(voxels.shape[2])).sum()/total
            a = center
        self.assertEqual(a, center_of_mass(voxels, threshold=threshold))

    def test_invalid_2(self):
        voxels = np.array([[[1, 2], [0.01, 0]], [[-1, -4], [np.pi, 0]]])
        threshold=0.1
        center = [0]*3
        voxels_filtered = np.copy(voxels)
        total = voxels_filtered.sum()
        if total == 0:
            a = [length / 2 for length in voxels.shape]
        else:
            center[0] = np.multiply(voxels_filtered.sum(1).sum(1), np.arange(voxels.shape[0])).sum()/total
            center[1] = np.multiply(voxels_filtered.sum(0).sum(1), np.arange(voxels.shape[1])).sum()/total
            center[2] = np.multiply(voxels_filtered.sum(0).sum(0), np.arange(voxels.shape[2])).sum()/total
            a = center
        self.assertNotEqual(a, center_of_mass(voxels, threshold=threshold))


class Test_voxel_exist(unittest.TestCase):
    
    def test_valid_1(self):
        voxels = np.array([[[1, 2], [0.01, 0]], [[-1, -4], [np.pi, 0]]])
        coord_x = 0
        coord_y = 2
        coord_z = 1.5
        neg_coords = coord_x < 0 or coord_y < 0 or coord_z < 0
        b_x, b_y, b_z = voxels.shape
        coords_out_of_bounds = coord_x >= b_x or coord_y >= b_y or coord_z >= b_z
        if neg_coords or coords_out_of_bounds:
            a = False
        else:
            a = voxels[coord_x, coord_y, coord_z]
        self.assertEqual(a, voxel_exist(voxels, coord_x, coord_y, coord_z))

    def test_invalid_1(self):
        voxels = np.array([[[1, 2], [0.01, 0]], [[-1, -4], [np.pi, 0]]])
        coord_x = 1
        coord_y = 0
        coord_z = -2
        neg_coords = coord_x < 0 or coord_y < 0 or coord_z < 0
        b_x, b_y, b_z = voxels.shape
        coords_out_of_bounds = coord_x >= b_x or coord_y >= b_y or coord_z >= b_z
        if neg_coords or coords_out_of_bounds:
            a = True
        else:
            a = voxels[coord_z, coord_y, coord_x]
        self.assertNotEqual(a, voxel_exist(voxels, coord_x, coord_y, coord_z))


class Test_block_generation(unittest.TestCase):

    def test_valid_1(self):
        z_var = 0
        offset = 0
        ratio = 1
        a = 1.0 / (1.0 + np.exp(-1.0 * (z_var-offset) * ratio))
        self.assertEqual(a, block_generation(z_var, offset=offset, ratio=ratio))


if __name__ == '__main__':
    unittest.main()
