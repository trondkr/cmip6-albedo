import xesmf as xe
import ESMF

ESMF.Manager(debug=True)


class CMIP6_regrid:

    def regrid_variable(self, varname, ds_in, ds_out, transpose=True):
        regridder = xe.Regridder(ds_in, ds_out, 'bilinear', reuse_weights=True, ignore_degenerate=True)
        regridder._grid_in = None
        regridder._grid_out = None
        if transpose:
            return regridder(ds_in[varname].T)
        else:
            return regridder(ds_in[varname])
