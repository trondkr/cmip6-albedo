import ESMF
import xarray as xr
import numpy as np
import xesmf as xe
from cmip6_preprocessing.preprocessing import combined_preprocessing

ESMF.Manager(debug=True)


class CMIP6_regrid:

    def regrid_variable(self, varname, ds_in, ds_out, interpolation_method="bilinear"):

        regridder = xe.Regridder(ds_in, ds_out, interpolation_method,
                                 ignore_degenerate=True,
                                 periodic=True,
                                 extrap="inverse_dist")
        regridder._grid_in = None
        regridder._grid_out = None
        print("[CMIP6_regrid] regridding {}".format(varname))

        return regridder(ds_in[varname])

    def setup_source_and_destination_2D_grids(self, current_grid, out_grid_anom):
        print("In grid {} vs {}".format(np.shape(current_grid["lats"]),np.shape(current_grid["lons"])))
        print("Out grid {} vs {}".format(np.shape(out_grid_anom["lats"]),np.shape(out_grid_anom["lons"])))

        in_grid = self.grid_create_from_coordinates(current_grid["lats"], current_grid["lons"])

        out_grid = self.grid_create_from_coordinates(out_grid_anom["lats"], out_grid_anom["lons"])

        srcfield = ESMF.Field(in_grid, name="srcfield", staggerloc=ESMF.StaggerLoc.CENTER)
        dstfield = ESMF.Field(out_grid, name="dstfield", staggerloc=ESMF.StaggerLoc.CENTER)

        return in_grid, out_grid, srcfield, dstfield

    def grid_create_from_coordinates(self, xcoords, ycoords, xcorners=False, ycorners=False, corners=False, domask=False,
                                     doarea=False, ctk=ESMF.TypeKind.R8):
        """
        Create a 2 dimensional Grid using the bounds of the x and y coordiantes.
        :param xcoords: The 1st dimension or 'x' coordinates at cell centers, as a Python list or numpy Array
        :param ycoords: The 2nd dimension or 'y' coordinates at cell centers, as a Python list or numpy Array
        :param xcorners: The 1st dimension or 'x' coordinates at cell corners, as a Python list or numpy Array
        :param ycorners: The 2nd dimension or 'y' coordinates at cell corners, as a Python list or numpy Array
        :param domask: boolean to determine whether to set an arbitrary mask or not
        :param doarea: boolean to determine whether to set an arbitrary area values or not
        :param ctk: the coordinate typekind
        :return: grid
        """
        [x, y] = [0, 1]

        # create a grid given the number of grid cells in each dimension, the center stagger location is allocated, the
        # Cartesian coordinate system and type of the coordinates are specified
        max_index = np.array([len(xcoords), len(ycoords)])
        grid = ESMF.Grid(max_index, staggerloc=[ESMF.StaggerLoc.CENTER], coord_sys=ESMF.CoordSys.CART, coord_typekind=ctk)

        # set the grid coordinates using numpy arrays, parallel case is handled using grid bounds
        gridXCenter = grid.get_coords(x)
        x_par = xcoords[grid.lower_bounds[ESMF.StaggerLoc.CENTER][x]:grid.upper_bounds[ESMF.StaggerLoc.CENTER][x]]
        gridXCenter[...] = x_par.reshape((x_par.size, 1))

        gridYCenter = grid.get_coords(y)
        y_par = ycoords[grid.lower_bounds[ESMF.StaggerLoc.CENTER][y]:grid.upper_bounds[ESMF.StaggerLoc.CENTER][y]]
        gridYCenter[...] = y_par.reshape((1, y_par.size))

        # create grid corners in a slightly different manner to account for the bounds format common in CF-like files
        if corners:
            grid.add_coords([ESMF.StaggerLoc.CORNER])
            lbx = grid.lower_bounds[ESMF.StaggerLoc.CORNER][x]
            ubx = grid.upper_bounds[ESMF.StaggerLoc.CORNER][x]
            lby = grid.lower_bounds[ESMF.StaggerLoc.CORNER][y]
            uby = grid.upper_bounds[ESMF.StaggerLoc.CORNER][y]

            gridXCorner = grid.get_coords(x, staggerloc=ESMF.StaggerLoc.CORNER)
            for i0 in range(ubx - lbx - 1):
                gridXCorner[i0, :] = xcorners[i0+lbx, 0]
            gridXCorner[i0 + 1, :] = xcorners[i0+lbx, 1]

            gridYCorner = grid.get_coords(y, staggerloc=ESMF.StaggerLoc.CORNER)
            for i1 in range(uby - lby - 1):
                gridYCorner[:, i1] = ycorners[i1+lby, 0]
            gridYCorner[:, i1 + 1] = ycorners[i1+lby, 1]

        # add an arbitrary mask
        if domask:
            mask = grid.add_item(ESMF.GridItem.MASK)
            mask[:] = 1
            mask[np.where((1.75 <= gridXCenter.any() < 2.25) &
                          (1.75 <= gridYCenter.any() < 2.25))] = 0

        # add arbitrary areas values
        if doarea:
            area = grid.add_item(ESMF.GridItem.AREA)
            area[:] = 5.0

        return grid

    def setup_ESMF_weights(self, srcfield, dstfield, mg):

        # write regridding weights to file
        filename = "esmpy_weight_file.nc"
        if ESMF.local_pet() == 0:
            import os
            if os.path.isfile(
                    os.path.join(os.getcwd(), filename)):
                os.remove(os.path.join(os.getcwd(), filename))

        mg.barrier()
        regrid = ESMF.Regrid(srcfield, dstfield, filename=filename,
                             regrid_method=ESMF.RegridMethod.BILINEAR,
                             unmapped_action=ESMF.UnmappedAction.IGNORE)

        # # create a regrid object from file
        regrid = ESMF.RegridFromFile(srcfield, dstfield, filename)
        return regrid

    def setup_ESMF_regridding(self, current_grid, out_grid_anom):
        mg = ESMF.Manager(debug=True)
        in_grid, out_grid, srcfield, dstfield = self.setup_source_and_destination_2D_grids(current_grid, out_grid_anom)
        regrid = self.setup_ESMF_weights(srcfield, dstfield, mg)
        dstfield.data[:] = 1e20
        return srcfield, dstfield, regrid


