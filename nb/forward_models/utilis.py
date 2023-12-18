import firedrake as fd
import numpy as np
import vtk
import pyvista as pv
import gmsh
from datetime import datetime
import os


def get_vtk_unstructured_tet_grid(nodes, tetrahedra):

    number_of_cells = tetrahedra.shape[0]
    nodes_per_cell = 4

    # Each cell in the cell array needs to include the size of the cell
    # and the points belonging to the cell
    cells_size_array = np.full(
        (number_of_cells, 1), nodes_per_cell, dtype=np.int64)
    cells_for_vtk = np.append(
        cells_size_array, tetrahedra, axis=1).ravel().astype(int)
    # each cell is a VTK_TETRA
    celltypes = np.full(number_of_cells, vtk.VTK_TETRA, dtype=np.uint8)

    return pv.UnstructuredGrid(cells_for_vtk, celltypes, nodes)

def get_vtk_unstructured_tri_grid(nodes, triangles):

    number_of_cells = triangles.shape[0]
    nodes_per_cell = 3

    # Each cell in the cell array needs to include the size of the cell
    # and the points belonging to the cell
    cells_size_array = np.full(
        (number_of_cells, 1), nodes_per_cell, dtype=np.int64)
    cells_for_vtk = np.append(
        cells_size_array, triangles, axis=1).ravel().astype(int)
    # each cell is a VTK_TETRA
    celltypes = np.full(number_of_cells, vtk.VTK_TRIANGLE, dtype=np.uint8)

    return pv.UnstructuredGrid(cells_for_vtk, celltypes, nodes)

def cell_project(v, V, solver_parameters={'ksp_type': 'cg', 'pc_type': 'sor', 'ksp_converged_reason': None}):
    """
    Project an expression `v` into a cell-based function space `V`.
    
    If `v` is a dictionary then we can project different expressions
    on regions with different cell tags.
    
    The default solver parameters apply conjugate gradients as an
    iterative linear solver, with successive over-relaxation as a
    preconditioner.
    """
    v_, dv = fd.TestFunction(V), fd.TrialFunction(V)
    a = fd.inner(v_, dv)*fd.dx
    if isinstance(v, dict):
        L = sum([fd.inner(v_, val)*fd.dx(reg) for (reg, val) in v.items()])
    else:
        L = fd.inner(v_, v)*fd.dx
    u = fd.Function(V)
    fd.solve(a == L, u, solver_parameters=solver_parameters)
    return u

def facet_project(v, V, solver_parameters={'ksp_type': 'cg', 'pc_type': 'jacobi', 'ksp_converged_reason': None}):
    """
    Project an expression `v` into a facet-based function space `V`.
    
    If `v` is a dictionary then we can project different expressions
    on regions with different facet tags.
    
    The default solver parameters apply conjugate gradients as an
    iterative linear solver, with Jacobi as a preconditioner.
    """
    v_, dv = fd.TestFunction(V), fd.TrialFunction(V)
    a = fd.inner(fd.avg(v_), fd.avg(dv))*fd.dS
    if isinstance(v, dict):
        L = sum([fd.inner(fd.avg(v_), val)*fd.dS(reg) for (reg, val) in v.items()])
    else:
        L = fd.inner(fd.avg(v_), v)*fd.dS
    u = fd.Function(V)
    fd.solve(a == L, u, solver_parameters=solver_parameters)
    return u

def write_displaced_mesh_vtk(mesh, u, filename, fileds_list=[], scaling_factor=1):
    S1s_e = fd.FunctionSpace(mesh, "DG", 1)  # space for piecewise constant material properties
    x, y = fd.SpatialCoordinate(mesh)
    eicx, eicy = fd.Function(S1s_e).interpolate(x).dat.data, fd.Function(S1s_e).interpolate(y).dat.data
    eic = np.vstack([eicx,eicy]).T
    eux, euy = fd.Function(S1s_e).assign(u.sub(0)).dat.data, fd.Function(S1s_e).assign(u.sub(1)).dat.data
    eu = np.vstack([eux,euy]).T
    # Displaced coordinates
    ecc = eic + eu * scaling_factor
    ecc = np.hstack([ecc, np.zeros((ecc.shape[0],1))])
    triangles = np.arange(eu.shape[0]).reshape(eu.shape[0]//3,3)
    vtk_object = get_vtk_unstructured_tri_grid(ecc, triangles)
    # Adding fields
    vtk_object.point_data['Displacement'] = eu
    for field in fileds_list:
        vtk_object.point_data[field.name()] = fd.Function(S1s_e).interpolate(field).dat.data
    vtk_object.save(f'{filename}.vtu')
    
def cylinder_mesh(sample_radius=.25, sample_length=1, mesh_size=.075, model_name="3d_cylinder"):
    gmsh.initialize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add(model_name)
    # Cylindrical volume that represents the intact sample
    cylinder_sample_tag = gmsh.model.occ.addCylinder(x=.0, y=.0, z=.0,
                                                    dx=.0, dy=.0, dz=sample_length,
                                                    r=sample_radius)
    gmsh.model.occ.synchronize()
    # 3 Bottom, 2 Top, 1 Side
    boundaries = gmsh.model.getBoundary([(3, cylinder_sample_tag)])
    # Volume: 301
    gmsh.model.addPhysicalGroup(3, [cylinder_sample_tag], 301)
    # Side: 201
    gmsh.model.addPhysicalGroup(2, [1], 201)
    # Top: 202
    gmsh.model.addPhysicalGroup(2, [2], 202)
    # Bottom: 203
    gmsh.model.addPhysicalGroup(2, [3], 203)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.write(model_name + '.msh')
    mesh = fd.Mesh(model_name + '.msh')
    return mesh


def mesh2d_rectangle(width, height, mesh_size=None, filename = None, delete_mesh_file = True):
    """
    Create 2d mesh of a rectangle with given width and height.

    Lines:
    (1) Left;
    (2) top;
    (3) right;
    (4) bottom.

    Surfaces:
    (1) Rectangle surface.

    Parameters:
    -----------
    filename (str): the name of the mesh files;
    width (float): the rectangle width(unit:m);
    height (float): the rectangle height(unit:m).
    """

    if mesh_size == None:
        mesh_size = height/20

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    # add a new model
    gmsh.model.add("rectangle")

    # bottom left vertex
    gmsh.model.geo.addPoint(-width/2, -height/2, 0, mesh_size, 0)
    # top left vertex
    gmsh.model.geo.addPoint(-width/2, height/2, 0, mesh_size, 1)
    # top right vertex
    gmsh.model.geo.addPoint(width/2, height/2, 0, mesh_size, 2)
    # bottom right vertex
    gmsh.model.geo.addPoint(width/2, -height/2, 0, mesh_size, 3)

    # left line
    gmsh.model.geo.addLine(0, 1, 1)
    # top line
    gmsh.model.geo.addLine(1, 2, 2)
    # right line
    gmsh.model.geo.addLine(2, 3, 3)
    # right line
    gmsh.model.geo.addLine(3, 0, 4)

    gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
    # complete domain
    gmsh.model.geo.add_plane_surface([1], 1)

    # construct the plane surface

    gmsh.model.geo.synchronize()

    # Labelling the edges
    gmsh.model.addPhysicalGroup(1, [1], 1) # Left
    gmsh.model.addPhysicalGroup(1, [2], 4) # Top
    gmsh.model.addPhysicalGroup(1, [3], 2) # Right
    gmsh.model.addPhysicalGroup(1, [4], 3) # Bottom
    # Labelling the surface
    gmsh.model.addPhysicalGroup(2, [1], 1)

    gmsh.option.setNumber("Mesh.Algorithm", 2)

    # generate a 2D mesh
    gmsh.model.mesh.generate(2)

    # create provisional mesh file name
    if filename is None:
        filename = datetime.now().strftime("%Y%m%d%H%M%S")

    # genarate mesh files
    gmsh.write(filename + '.geo_unrolled')
    os.rename(filename + '.geo_unrolled', filename + '.geo')
    gmsh.write(filename + '.msh')
    gmsh.finalize()

    # create Firedrake (PETsC) mesh
    mesh = fd.Mesh(filename + '.msh')

    # remove mesh files
    if delete_mesh_file is True:
        os.remove(filename+ '.msh')
        os.remove(filename+ '.geo')
    
    return mesh