#!/usr/bin/env python

import os, sys
import json

import requests

import numpy as np
from astropy.coordinates import SkyCoord

_debug_mode = False
_ps1_baseurl = "https://catalogs.mast.stsci.edu/api/v0.1/panstarrs"

def ps1search(ra, dec, radius, dr='dr2', table_name='stack',
              columns=[], constr={}):

    # radius in degrees here.

    url = _ps1_baseurl + '/{:}/{:}.csv'.format(dr, table_name)
    req_st = dict(ra=ra, dec=dec, radius=radius)
    if columns: req_st['columns'] = '[{}]'.format(','.join(columns))
    if constr: req_st.update(constr)
    r = requests.post(url, data=req_st)
    r.raise_for_status()

    return r.text

def mastps1_query(crd, radius, cat_info,):

    # radius in arcseconds.

    # status
    crd_repr = 'coord:{:.6f},{:.6f}/radius:{:.3f}'.format(
                crd.ra.deg, crd.dec.deg, radius)
    stat = dict(is_complete=True,
                queried_with=crd_repr,)
                # search_radius=radius)

    # table and columns to search.
    tab_t = cat_info['table_name']
    col_t = [w[0] for w in cat_info['columns']]

    # query, parse results,
    try:
        cat_qr = ps1search(crd.ra.deg, crd.dec.deg, radius / 3.6e3,
                table_name=tab_t, columns=col_t,
                constr={'nDetections.gt': 1})
    except Exception:
        stat['is_complete'] = False

    # no source: return empty list.
    if stat['is_complete'] and isinstance(cat_qr, str) and (not cat_qr):
        stat['srcs'] = list()
        return stat

    # failed: return and mark as incomplete.
    if not stat['is_complete']:
        return stat

    # parse table
    restab_i = [w.split(',') for w in cat_qr.split('\r\n') if len(w) > 1]
    colnames, tab_recs = restab_i[0], list()
    dtp_cvt = {
        'f': lambda x: float(x) if x else np.nan,
        'i': lambda x: int(x) if x else -1, # <--
        'S': lambda x: x.strip(),
    } # FIXME: overwrites the default value

    # check columns defined in the catalog.
    if _debug_mode:
        catdef_cols = [w[0] for w in cat_info['columns']]
        undef_cols = list(set(colnames) - set(catdef_cols))
        if undef_cols: raise RuntimeError('undefined: ' + repr(undef_cols))
        extra_cols = list(set(catdef_cols) - set(colnames))
        if extra_cols: raise RuntimeError('extra cols: ' + repr(extra_cols))

    # pack into dict and convert data types.
    for row_i in restab_i[1:]:
        row_dict_i = dict(zip(colnames, row_i))
        for col_i, dtp_i in cat_info['columns']:
            if col_i not in row_dict_i: continue
            row_dict_i[col_i] = dtp_cvt[dtp_i[0]](row_dict_i[col_i])
        tab_recs.append(row_dict_i)
    stat['srcs'] = tab_recs

    return stat
