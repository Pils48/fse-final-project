"""
VTK functions
"""
import math
import numpy as np
from util import blocktrans_cen2side, center_of_mass
import matplotlib.cm
import vtk


def block_generation(cen_size, color):
    """
    generate a block up to actor stage
    User may choose to use VTK boxsource implementation, or the polydata implementation
    """
    cube_mapper = vtk.vtkPolyDataMapper()
    cube_actor = vtk.vtkActor()

    l_x, l_y, l_z, h_x, h_y, h_z = blocktrans_cen2side(cen_size)
    vertices = [[l_x, l_y, l_z], [h_x, l_y, l_z], [h_x, h_y, l_z], [l_x, h_y, l_z],
                [l_x, l_y, h_z], [h_x, l_y, h_z], [h_x, h_y, h_z], [l_x, h_y, h_z]]

    pts = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]

    cube = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    polys = vtk.vtkCellArray()

    for i in range(0, 8):
        points.InsertPoint(i, vertices[i])

    for i in range(0, 6):
        polys.InsertNextCell(4)
        for j in range(0, 4):
            polys.InsertCellPoint(pts[i][j])
    cube.SetPoints(points)
    cube.SetPolys(polys)
    cube_mapper.SetInput(cube)
    cube_actor.SetMapper(cube_mapper)

    # set the colors
    cube_actor.GetProperty().SetColor(np.array(color[:3]))
    cube_actor.GetProperty().SetAmbient(0.5)
    cube_actor.GetProperty().SetDiffuse(.5)
    cube_actor.GetProperty().SetSpecular(0.1)
    cube_actor.GetProperty().SetSpecularColor(1, 1, 1)
    cube_actor.GetProperty().SetDiffuseColor(color[:3])
    # cube_actor.GetProperty().SetAmbientColor(1, 1, 1)
    # cube_actor.GetProperty().ShadingOn()
    return cube_actor

def generate_all_blocks(voxels, threshold=0.1, uniform_size=-1, use_colormap=False):
    """
    Generate one block per voxel, with block size and color dependent on probability.
    Performance is desirable if number of blocks is below 20,000.
    """
    assert voxels.ndim == 3
    actors = []
    counter = 0
    dims = voxels.shape

    cmap = matplotlib.cm.get_cmap('jet')
    default_color = [0.9, 0, 0]

    for k in range(dims[2]):
        for j in range(dims[1]):
            for i in range(dims[0]):
                occupancy = voxels[i][j][k]
                if occupancy < threshold:
                    continue

                if use_colormap:
                    color = cmap(float(occupancy))
                else:    # use default color
                    color = default_color

                if 0 < uniform_size <= 1:
                    block_size = uniform_size
                else:
                    block_size = occupancy
                block = [i+0.5, j+0.5, k+0.5, block_size, block_size, block_size]
                actors.append(block_generation(block, color=(color)))
                counter = counter + 1

    print(counter, "blocks filled")
    return actors

def display(actors, cam_pos, cam_vocal, cam_up, title=None):
    """ Display the scene from actors.
    cam_pos: list of positions of cameras.
    cam_vocal: vocal point of cameras
    cam_up: view up direction of cameras
    title: display window title
    """

    ren_win = vtk.vtkRenderWindow()
    window_size = 1024

    renderer = vtk.vtkRenderer()
    for actor in actors:
        renderer.AddActor(actor)
    renderer.SetBackground(1, 1, 1)
    renWin.AddRenderer(renderer)

    camera = vtk.vtkCamera()
    renderer.SetActiveCamera(camera)
    renderer.ResetCamera()

    # the object is located at 0 <= x, y, z <= dims[i]
    camera.SetFocalPoint(*cam_vocal)
    camera.SetViewUp(*cam_up)
    camera.SetPosition(*cam_pos)

    renWin.SetSize(window_size, window_size)

    iren = vtk.vtkRenderWindowInteractor()
    style = vtk.vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(style)
    iren.SetRenderWindow(ren_win)
    if title is not None:
        renWin.SetWindowName(title)

    renderer.ResetCameraClippingRange()
    renWin.Render()

    iren.Initialize()
    iren.Start()

def visualization(voxels, threshold, title=None, uniform_size=-1, use_colormap=False):
    """
    Given a voxel matrix, plot all occupied blocks (defined by voxels[x][y][z] > threshold)
    if size_change is set to true, block size will be proportional to voxels[x][y][z]
    otherwise voxel matrix is transfered to {0, 1} matrix,
    where consecutive blocks are merged for performance.

    The function saves an image at address ofilename, with form jpg/png.
    If form is empty string, no image is saved.

    """
    actors = generate_all_blocks(voxels, threshold,
                                 uniform_size=uniform_size, use_colormap=use_colormap)

    center = center_of_mass(voxels)
    distance = voxels.shape[0] * 2.8
    height = voxels.shape[2] * 0.85
    rad = math.pi * 0.43 #+ math.pi
    cam_pos = [center[0]+distance*math.cos(rad), center[1]+distance*math.sin(rad), center[2]+height]

    display(actors, cam_pos, center, (0, 0, 1), title=title)
