#!/usr/bin/env python

import os, sys
import json

import numpy as np
from astropy.coordinates import SkyCoord

from dl import authClient as ac, queryClient as qc
from dl.helpers.utils import convert
from getpass import getpass

sqlstr = '''
    SELECT {cols} FROM {tab}
    WHERE q3c_radial_query({ra_col}, {dec_col}, {ra_c}, {dec_c}, {rad})
'''

_debug_mode = False

def datalab_query(crd, radius, cat_info, token):

    # status
    crd_repr = 'coord:{:.6f},{:.6f}/radius:{:.3f}'.format(
                crd.ra.deg, crd.dec.deg, radius)
    stat = dict(is_complete=True,
                queried_with=crd_repr,)

    # table and columns to search.
    tab_t = cat_info['table_name']
    col_t = ', '.join(['"' + w[0] + '"' for w in cat_info['columns']]) \
            if isinstance(cat_info['columns'], (list, tuple)) \
            else cat_info['columns'] # '*' or comma-separated list of cols
    ra_col, dec_col = cat_info['crd_cols']

    # generate query string.
    if 'sql_fmt' not in cat_info: # use default sql string.
        sqlstr_t = sqlstr.format(cols=col_t, tab=tab_t,
                                ra_col=ra_col, dec_col=dec_col,
                                ra_c=crd.ra.deg, dec_c=crd.dec.deg,
                                rad=radius / 60. / 60.)
        # switch to Q3C routines, YJ, 081019
    else:
        sqlstr_t = cat_info['sql_fmt'].format(
                cols=col_t, tab=tab_t, ra_col=ra_col, dec_col=dec_col,
                ra_c=crd.ra.deg, dec_c=crd.dec.deg, rad=radius / 60. / 60.)
        if _debug_mode: print('pre-defined format used.')

    # set shorter wait time
    qc.set_timeout_request(8)

    # query, parse results,
    try:
        cat_qr = qc.query(token, sql=sqlstr_t)
    except Exception as err:
        stat['is_complete'] = False
        print(sqlstr_t)
        print(err)

    # failed: return and mark as incomplete.
    if not stat['is_complete']:
        return stat

    # parse table
    restab_i = [w.split(',') for w in cat_qr.split('\n') if len(w) > 1]
    colnames, tab_recs = restab_i[0], list()
    dtp_cvt = {
        'f': lambda x: float(x) if x else np.nan,
        'i': lambda x: int(x) if x else -1,
        'S': lambda x: x.strip(),
    }

    # check columns defined in the catalog.
    if _debug_mode:
        catdef_cols = [w[0] for w in cat_info['data_model']]
        undef_cols = list(set(colnames) - set(catdef_cols))
        if undef_cols: raise RuntimeError('undefined: ' + repr(undef_cols))
        extra_cols = list(set(catdef_cols) - set(colnames))
        if extra_cols: raise RuntimeError('extra cols: ' + repr(extra_cols))

    # pack into dict and convert data types.
    for row_i in restab_i[1:]:
        row_dict_i = dict(zip(colnames, row_i))
        for col_i, dtp_i in cat_info['data_model']:
            '''
            if col_i not in row_dict_i: # skip columns that are ignored
                raise RuntimeError('undefined col: {}'.format(col_i))
            '''
            row_dict_i[col_i] = dtp_cvt[dtp_i[0]](row_dict_i[col_i])
        tab_recs.append(row_dict_i)
    stat['srcs'] = tab_recs

    return stat

# previous version
'''
# calculate box of query
cos_delta = np.cos(crd.dec.radian)
box_radius = radius / 60. / 60. # asec -> deg

ra_llim, ra_ulim, dec_llim, dec_ulim =      \
    crd.ra.deg  - box_radius / cos_delta,   \
    crd.ra.deg  + box_radius / cos_delta,   \
    crd.dec.deg - box_radius,               \
    crd.dec.deg + box_radius

# polar case (unlikely)
if (dec_llim - box_radius < -90.) \
        or (dec_ulim + box_radius > 90.):
    sqlstr_t = sqlstr_polar.format(cols=col_t, tab=tab_t,
            dec_llim=dec_llim, dec_ulim=dec_ulim,
            ra_col=ra_col, dec_col=dec_col)

# meridian-crossing case,
elif ra_llim <= 0. or ra_ulim > 360.:
    sqlstr_t = sqlstr_meridian.format(cols=col_t, tab=tab_t,
            ra_llim=np.fmod(ra_llim + 360., 360.),
            ra_ulim=np.fmod(ra_ulim, 360.),
            dec_llim=dec_llim, dec_ulim=dec_ulim,
            ra_col=ra_col, dec_col=dec_col)

# general case.
else:
    sqlstr_t = sqlstr.format(cols=col_t, tab=tab_t,
            ra_llim=ra_llim, ra_ulim=ra_ulim,
            dec_llim=dec_llim, dec_ulim=dec_ulim,
            ra_col=ra_col, dec_col=dec_col)
'''
# written before DL supports Q3C. YJ, 080819

# OBSOLETE
__sqlstr = '''
    SELECT {cols} FROM {tab}
    WHERE {ra_col} BETWEEN {ra_llim} AND {ra_ulim}
        AND {dec_col} BETWEEN {dec_llim} AND {dec_ulim}
    '''

# OBSOLETE
__sqlstr_meridian = '''
    SELECT {cols} FROM {tab}
    WHERE ({ra_col} < {ra_ulim} OR {ra_col} > {ra_llim})
        AND {dec_col} BETWEEN {dec_llim} AND {dec_ulim}
    '''

# OBSOLETE
__sqlstr_polar = '''
    SELECT {cols} FROM {tab}
    WHERE {dec_col} BETWEEN {dec_llim} AND {dec_ulim}
    '''
