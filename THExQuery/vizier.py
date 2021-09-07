#!/usr/bin/env python

'''
    Vizier utilities.
'''

import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier

# switch to Harvard
Vizier.VIZIER_SERVER = 'vizier.cfa.harvard.edu'

from .utils import row_to_list

# convert table column names to db keys
as_key = lambda w: w.lower().replace(' ', '_')

def vizier_query(crd, radius, cat_vid, extra_cols=None):

    # status
    crd_repr = 'coord:{:.6f},{:.6f}/radius:{:.3f}'.format(
                crd.ra.deg, crd.dec.deg, radius)
    stat = dict(is_complete=True,
                queried_with=crd_repr,)

    # make vizier instance
    viz_cols = ['*'] # default: use vizier cols
    if isinstance(extra_cols, (list, tuple)): # extra cols provided
        viz_cols = ['*'] + list(extra_cols)
    if (extra_cols is True) or (extra_cols == 'full'): # all columns
        viz_cols = ['**']
    viz = Vizier(columns=viz_cols)

    # search vizier.
    try:
        viz_tab = viz.query_region(crd,
                                   radius=radius * u.arcsec,
                                   catalog=cat_vid)
        viz_tab = viz_tab[0] # get the first table.

    except IndexError: # returns no result.
        viz_tab = list()

    except Exception as err: # other
        viz_tab = list()
        stat['is_complete'] = False

    # failed for any reasons: stop.
    if not stat['is_complete']:
        return stat

    # parse vizier tab,
    if viz_tab:
        viz_src_list = list()
        colnames = [as_key(w) for w in viz_tab.colnames]
        for rec_i in viz_tab:
            rec_i = row_to_list(rec_i)
            viz_src_list.append( \
                    {k: v for k, v in zip(colnames, rec_i) if k != 'all'}
            )
        stat['srcs'] = viz_src_list

    else: # no source detected, put an empty list.
        stat['srcs'] = list()

    return stat
