# -*- coding: utf-8 -*- 
from datetime import datetime
import xarray as xr
import dask.array
import numpy as np
import cv2
import warnings
warnings.filterwarnings('ignore')
import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)
import sys
sys.path.append('/home/Ruth.Moorman/Southern_ACC_boundary_dynamics')
### import required python functions
from psig_contour import *
from om4_tools import *





name = 'annual_SUC_psig_C1.nc'

exp_dir = '/archive/oar.gfdl.ogrp-account/CMIP6/OMIP/xanadu_mom6_20181101/OM4p125_IAF_csf_JRA55do1-3_r5_cycle2/gfdl.ncrc4-intel16f2-prod/pp'
static = xr.open_dataset(exp_dir+'/ocean_daily/ocean_daily.static.nc')
bathy = static.deptho.load()
land_mask = static.wet_u.load()
geolon = static.geolon_u.load()
geolat = static.geolat_u.load()

umo_C1 = xr.open_mfdataset('/archive/oar.gfdl.ogrp-account/CMIP6/OMIP/xanadu_mom6_20181101/OM4p125_IAF_csf_JRA55do1-3_r5_cycle1/gfdl.ncrc4-intel16f2-prod/pp/ocean_annual_z/ts/annual/20yr/ocean_annual_z.*.umo.nc', chunks = {'time':1})
umo_C1 = umo_C1.umo

umo_2d_C1 = umo_C1.sel(yh = slice(-90,-29)).sum(dim = 'z_l')

ntime = len(umo_2d_C1.time.values)
nlat = len(umo_2d_C1.yh.values)
nlon = len(umo_2d_C1.xq.values)


time = np.arange('1838','1898', dtype='datetime64[Y]')
psig_array = xr.DataArray(np.empty((ntime, nlat, nlon)), coords = [time, umo_2d_C1.yh, umo_2d_C1.xq], dims = ['time', 'yh', 'xq'])
psig_array[:,:,:] = np.nan
psig_array.attrs['long_name'] = 'Barotropic Steamfunction'
psig_array.attrs['units'] = 'Sv'
SUCpsig_mask_array = xr.DataArray(np.empty((ntime, nlat, nlon)), coords = [time, umo_2d_C1.yh, umo_2d_C1.xq], dims = ['time', 'yh', 'xq'])
SUCpsig_mask_array[:,:,:] = np.nan
SUCpsig_mask_array.attrs['description'] = 'masks for the region south of the southernmost unblocked contour of the monthly mean barotropic streamfunction'
SUCpsig_numbered_array = xr.DataArray(np.empty((ntime, nlat, nlon)), coords = [time, umo_2d_C1.yh, umo_2d_C1.xq], dims = ['time', 'yh', 'xq'])
SUCpsig_numbered_array[:,:,:] = np.nan
SUCpsig_numbered_array.attrs['description'] = 'numbered index along the southernmost unblocked contour of the monthly mean barotropic streamfunction'
SUCpsig_array = xr.DataArray(np.empty((ntime)), coords = [time], dims = ['time'])
SUCpsig_array[:] = np.nan
SUCpsig_array.attrs['description'] = 'value for the southermost unblocked contour of the monthly mean barotropic streamfunction (also the ACC transport)'

start = datetime.now()
for i in range(ntime):
    contour_masked_south,contour_mask_numbered,psi_g,contour, _= OM4_ACCS_psig_contour(umo_2d_C1.isel(time = i), land_mask.sel(yh = slice(-90,-29)))
    psig_array[i,:,:] = psi_g.values
    SUCpsig_mask_array[i,:,:] = contour_masked_south.values
    SUCpsig_numbered_array[i,:,:] = contour_mask_numbered.values
    SUCpsig_array[i] = contour
#     print(contour)
end = datetime.now()
print("Time (done loop) p125 annual =", end-start)

cycletime = xr.DataArray(np.arange('1958','2018', dtype='datetime64[Y]'), coords = [time], dims=['time'])
cycle = xr.DataArray(np.ones(ntime), coords = [time], dims=['time'])

psig_array.coords['SUC_psig'] = SUCpsig_array
psig_array.coords['geolon'] = geolon
psig_array.coords['geolat'] = geolat
psig_array.coords['cycletime'] = cycletime
psig_array.coords['cycle'] = cycle

SUCpsig_mask_array.coords['SUC_psig'] = SUCpsig_array
SUCpsig_mask_array.coords['geolon'] = geolon
SUCpsig_mask_array.coords['geolat'] = geolat
SUCpsig_mask_array.coords['cycletime'] = cycletime
SUCpsig_mask_array.coords['cycle'] = cycle

SUCpsig_numbered_array.coords['SUC_psig'] = SUCpsig_array
SUCpsig_numbered_array.coords['geolon'] = geolon
SUCpsig_numbered_array.coords['geolat'] = geolat
SUCpsig_numbered_array.coords['cycletime'] = cycletime
SUCpsig_numbered_array.coords['cycle'] = cycle

ds = xr.Dataset({'SUC_psig_mask':SUCpsig_mask_array, 'SUC_psig_numbered':SUCpsig_numbered_array,'psig':psig_array})
ds.to_netcdf('/work/Ruth.Moorman/masks_and_contours/OM4p125_IAF_csf_JRA55do1-3_r5/'+name)