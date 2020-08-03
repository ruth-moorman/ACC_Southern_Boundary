import xarray as xr
import dask.array
import numpy as np
import cv2

def OM4_ACCS_psig_contour(umo_2D,land_mask,lat_north = -29):
    ρ = 1036 # kg/m^3
    psi = -umo_2D.cumsum('yh')/(1e6*ρ) # divide by 1e6 to convert m^3/s
    psi_acc = np.nanmin(psi.sel(xq = slice(-69, -67), yh = slice(-80, -55)))
    print('Max value of streamfunction south of 55S and within 69W-67W (ACC transport) = ', -psi_acc, 'Sv')
    psi_g = psi-psi_acc
    psi_g = psi_g.load()
    
    adjust = 0
    for k in range(200):
        
        contour = -psi_acc-(adjust * 0.1)
        temp = psi_g.where(psi_g>=contour) * 0 + 1
        temp_mask = psi_g.copy().fillna(1)
        xh = temp_mask.xq
        yh = temp_mask.yh
        temp_mask = temp_mask.where(temp_mask >= contour) * 0 + 1
        temp_mask = temp_mask.fillna(0)
        temp_mask = temp_mask.values # extract numpy file
        # # this spreads the isobath contour so all points are connected adjacently (not just 
        # # diagonally):
        kernel = np.ones((3,3),np.uint8)
        contour_mask0 = cv2.dilate(temp_mask,kernel,iterations=1) - temp_mask
        # start at western edge of domain, at y point closest to correct depth contour:
        contour_mask = np.zeros_like(temp_mask) 
        contour_lat_index_start = np.where(contour_mask0[:,0]>0)[0][-1]
        contour_mask[contour_lat_index_start,0] = 1
        # loop through to find adjacent point closest to contour depth (not inc previous point):
        last_index_i = np.array([0])
        last_index_j = contour_lat_index_start
        count = np.array([0])


        while last_index_i<(len(xh)-1): # 1440 is xh dimension length
            # first time don't go backwards:
            if last_index_i == 0:
                points_to_compare = np.array([contour_mask0[last_index_j,last_index_i+1],
                    0,
                    contour_mask0[last_index_j+1,last_index_i],
                    contour_mask0[last_index_j-1,last_index_i]])
            else:
                points_to_compare = np.array([contour_mask0[last_index_j,last_index_i+1],
                    contour_mask0[last_index_j,last_index_i-1],
                    contour_mask0[last_index_j+1,last_index_i],
                    contour_mask0[last_index_j-1,last_index_i]])
            new_loc = np.where(points_to_compare==1)[0]
            # this gives each point along contour a unique index number:
            if len(new_loc)==0:
                contour_mask0[last_index_j,last_index_i] = 0
                count = count - 1
                last_index_i = np.where(contour_mask==count+1)[1]
                last_index_j = np.where(contour_mask==count+1)[0]
                if len(last_index_i)>1:
                    last_index_i = last_index_i[0]
                    last_index_j = last_index_j[0]
                else:
                    continue
            elif len(new_loc)>0:
                if new_loc[0] == 0:
                    new_index_i = last_index_i+1
                    new_index_j = last_index_j
                elif new_loc[0] == 1:
                    new_index_i = last_index_i-1
                    new_index_j = last_index_j
                elif new_loc[0] == 2:
                    new_index_i = last_index_i
                    new_index_j = last_index_j+1            
                elif new_loc[0] == 3:
                    new_index_i = last_index_i
                    new_index_j = last_index_j-1

                contour_mask[new_index_j,new_index_i] = count + 2
                contour_mask0[last_index_j,last_index_i] = 2
                last_index_j = new_index_j
                last_index_i = new_index_i
                count += 1
                if len(np.array([last_index_i])) >1:
                    last_index_i = last_index_i[0]
                    last_index_j = last_index_j[0]
        
        if (contour_mask * land_mask).sum(dim = 'yh').min(dim = 'xq') == 1: 
            if len(np.unique(contour_mask))==len(np.unique(contour_mask* land_mask.fillna(0))):
            
#             print('adjusted:',adjust)
                break
            else:
                adjust += 1 
        else:
            adjust += 1 
    # # this is the sequentially numbered isobath, it provides an along isobath index
#     print(adjust)
    print(contour)
    contour_mask_numbered = contour_mask
    # sometimes there's a discontinuoity at 0
    if last_index_j > contour_lat_index_start:
        a = 0
        for m in np.arange(last_index_j,contour_lat_index_start-1, -1):
            contour_mask_numbered[m,-1] = contour_mask_numbered[last_index_j,-1] +a
            a += 1
    elif last_index_j < contour_lat_index_start:
        a = 0
        for m in np.arange(last_index_j, contour_lat_index_start+1, 1):
            contour_mask_numbered[m,-1] = contour_mask_numbered[last_index_j,-1] +a
            a += 1
    # fill in points to north of contour:
    contour_masked_above = np.copy(contour_mask_numbered)
    contour_masked_above[-1,0] = -100

    # from top left:
    for ii in range(len(xh)-1):
        for jj in range(len(yh))[::-1][:-1]:
            if contour_masked_above[jj,ii] == -100:
                if contour_masked_above[jj-1,ii] == 0:
                    contour_masked_above[jj-1,ii] = -100
                if contour_masked_above[jj,ii+1] == 0:
                    contour_masked_above[jj,ii+1] = -100
                if contour_masked_above[jj-1,ii+1] == 0:
                    contour_masked_above[jj-1,ii+1] = -100
    # from top right:
    for ii in range(len(xh))[::-1][:-1]:
        for jj in range(len(yh))[::-1][:-1]:
            if contour_masked_above[jj,ii] == -100:
                if contour_masked_above[jj-1,ii] == 0:
                    contour_masked_above[jj-1,ii] = -100
                if contour_masked_above[jj,ii-1] == 0:
                    contour_masked_above[jj,ii-1] = -100
                if contour_masked_above[jj-1,ii-1] == 0:
                    contour_masked_above[jj-1,ii-1] = -100
    # from bottom right:
    for ii in range(len(xh))[::-1][:-1]:
        for jj in range(len(yh)-1):
            if contour_masked_above[jj,ii] == -100:
                if contour_masked_above[jj+1,ii] == 0:
                    contour_masked_above[jj+1,ii] = -100
                if contour_masked_above[jj,ii-1] == 0:
                    contour_masked_above[jj,ii-1] = -100
                if contour_masked_above[jj+1,ii-1] == 0:
                    contour_masked_above[jj+1,ii-1] = -100
    # from bottom left:
    for ii in range(len(xh)-1):
        for jj in range(len(yh)-1):
            if contour_masked_above[jj,ii] == -100:
                if contour_masked_above[jj+1,ii] == 0:
                    contour_masked_above[jj+1,ii] = -100
                if contour_masked_above[jj,ii+1] == 0:
                    contour_masked_above[jj,ii+1] = -100
                if contour_masked_above[jj+1,ii+1] == 0:
                    contour_masked_above[jj+1,ii+1] = -100
    contour_masked_south = xr.DataArray(contour_masked_above, coords = [yh, xh], dims = ['yh', 'xq'])
    contour_mask_numbered = xr.DataArray(contour_mask_numbered, coords = [yh, xh], dims = ['yh', 'xq'])
    contour_masked_south = contour_masked_south.where(contour_masked_south >= 0) * 0 + 1 
    
    return contour_masked_south,contour_mask_numbered,psi_g,contour, temp_mask