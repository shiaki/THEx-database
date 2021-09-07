#!/usr/bin/env python

'''
    Calculates extinction correction for a position.
'''

from collections import OrderedDict
import numpy as np

import sfdmap

dustmap = sfdmap.SFDMap()

ext_factor = OrderedDict(

    # FUV, NUV: Peek & Schiminovich. 2013
    FUV_PS=lambda e: 10.47 * e + 8.59 * (e ** 2) - 82.8 * (e ** 3),
    FUV_PS_err=lambda e: np.sqrt((0.43 * e) ** 2 + (6.9 * (e ** 2)) ** 2 \
                              + (26.0 * (e ** 3)) ** 2),
    NUV_PS=lambda e: 8.36  * e + 14.3 * (e ** 2) - 82.8 * (e ** 3),
    NUV_PS_err=lambda e: np.sqrt((0.43 * e) ** 2 + (6.9 * (e ** 2)) ** 2 \
                            + (26.0 * (e ** 3)) ** 2),

    # FUV, NUV: Yuan et al. 2013, column a
    FUV_Y=lambda e: 4.89 * e, FUV_Y_err=lambda e: 0.60 * e,
    NUV_Y=lambda e: 7.24 * e, NUV_Y_err=lambda e: 0.08 * e,

    # optical and NIR: Yuan et al. 2013, column a
    u=lambda e: 4.39 * e, u_err=lambda e: 0.04 * e,
    g=lambda e: 3.30 * e, g_err=lambda e: 0.03 * e,
    r=lambda e: 2.31 * e, r_err=lambda e: 0.03 * e,
    i=lambda e: 1.71 * e, i_err=lambda e: 0.02 * e,
    z=lambda e: 1.29 * e, z_err=lambda e: 0.02 * e,

    # y-band **Interpolated** using PS1 effective wl.
    y=lambda e: 1.13 * e, y_err=lambda e: 0.02 * e,

    J=lambda e: 0.72 * e, J_err=lambda e: 0.01 * e,
    H=lambda e: 0.46 * e, H_err=lambda e: 0.01 * e,
    Ks=lambda e: 0.306 * e, Ks_err=lambda e: 0.01 * e, # error estimated.

    # MIR W1, W2: Yuan et al. 2013, column a
    W1=lambda e: 0.18 * e, W1_err=lambda e: 0.01 * e,
    W2=lambda e: 0.16 * e, W2_err=lambda e: 0.01 * e,

    # MIR W3, W4: My own calculation using alpha=-1.95 (Wang & Jiang 2014)
    W3=lambda e: 0.0247 * e, W3_err=lambda e: 0.01 * e, # err estimated.
    W4=lambda e: 0.00756 * e, W4_err= lambda e: 0.01 * e, # err estimated
)

def extinction_correction(ra, deg):

    # read local reddening
    ebv_t = dustmap.ebv(ra, deg)

    # calculate extinction in each band
    ext_val_t = OrderedDict()
    for col_i, fc_i in ext_factor.items():
        a_val_i = fc_i(ebv_t)
        a_val_i = a_val_i if (a_val_i >= 0.) else np.nan
        ext_val_t['MWExt_' + col_i] = a_val_i

    return ext_val_t