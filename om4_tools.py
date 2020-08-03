import xarray as xr
import dask.array
import numpy as np

def get_iaf_variable(exp_dir, cycle, nc_file, variable, lat_north = -29, year = False):
    if year == False:
        nc = xr.open_mfdataset(exp_dir+'/cycle'+str(cycle)+'/'+nc_file+'/*.'+nc_file+'.nc')
    else:
        nc = xr.open_mfdataset(exp_dir+'/cycle'+str(cycle)+'/'+nc_file+'/'+str(year)+'0101.'+nc_file+'.nc')
    v = nc[variable]
    v = v.sel(yh = slice(-90, lat_north))
    return v

def get_pp_variable(exp_dir, nc_name, var_name, ts = 'ts', annual = 'annual', yr = '20yr'):
    nc = xr.open_mfdataset(exp_dir+'/'+nc_name+'/'+ts+'/'+annual+'/'+yr+'/'+nc_name+'.*.'+var_name+'.nc')