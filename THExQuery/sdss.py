#!/usr/bin/env python

import os, sys
import json

import numpy as np
from astropy.coordinates import SkyCoord
from astroquery.sdss import SDSS

from .utils import row_to_list

sqlstr = '''
    SELECT *
    FROM PhotoPrimary AS p,
    dbo.fGetNearbyObjEq({ra_c}, {dec_c}, {rad}) AS n
    LEFT JOIN Photoz AS z ON z.objID=n.objID
    WHERE p.objID = n.objID
'''

sqlstr_xv = '''
    SELECT p.objid, p.mode, p.type,
        p.ra, p.dec, p.u, p.g, p.r, p.i, p.z,
        p.petroRad_r, p.petroRadErr_r,
        p.petroR50_r, p.petroR50Err_r,
        p.petroR90_r, p.petroR90Err_r
    FROM PhotoPrimary AS p,
    dbo.fGetNearbyObjEq({ra_c}, {dec_c}, {rad}) AS n
    WHERE p.objID = n.objID
'''

_debug_mode = False

# convert table column names to db keys
as_key = lambda w: w.lower().replace(' ', '_')

def sdss_query(crd, radius, cat_id, is_xvalid):

    assert cat_id == 'SDSS16pz'

    # status
    crd_repr = 'coord:{:.6f},{:.6f}/radius:{:.3f}'.format(
                crd.ra.deg, crd.dec.deg, radius)
    stat = dict(is_complete=True,
                queried_with=crd_repr,)

    # generate query string.
    sqlstr_t = (sqlstr if not is_xvalid else sqlstr_xv).format(
            ra_c=crd.ra.deg, dec_c=crd.dec.deg, rad=radius / 60.)

    # query, parse results,
    try:
        sdss_tab = SDSS.query_sql(sqlstr_t, data_release=16)

    except Exception as err:
        stat['is_complete'] = False
        print(sqlstr_t)
        print(err)

    # failed: return and mark as incomplete.
    if not stat['is_complete']:
        return stat

    # parse vizier tab,
    if sdss_tab:
        sdss_src_list = list()
        colnames = [as_key(w) for w in sdss_tab.colnames]
        # if 'titleskyserver_errortitle' in colnames: # FIX THIS
        for rec_i in sdss_tab:
            rec_i = row_to_list(rec_i)
            sdss_src_list.append({k: v for k, v in zip(colnames, rec_i)})
        stat['srcs'] = sdss_src_list

    else: # no source detected, put an empty list.
        stat['srcs'] = list()

    return stat

'''
"SDSS16pz" : {
    "is_complete" : true,
    "queried_with" : "coord:9.103531,38.060712/radius:45.000",
    "srcs" : [
        {
            "titleskyserver_errortitle" : "</head><body bgcolor=white>"
        },
        {
            "titleskyserver_errortitle" : "<h2>An error occured</h2><H3 BGCOLOR=pink><font color=red><br><html><body><h1>408 Request Time-out</h1>"
        },
        {
            "titleskyserver_errortitle" : "Your browser didn't send a complete request in time."
        },
        {
            "titleskyserver_errortitle" : "</body></html>"
        },
        {
            "titleskyserver_errortitle" : "</font></H3></BODY></HTML>"
        }
    ]
},
'''

if None and __name__ == '__main__':

    # generate query string.
    sqlstr_t = sqlstr.format(ra_c=185, dec_c=0, rad=3.)

    # query, parse results,
    sdss_tab = SDSS.query_sql(sqlstr_t)
