#!/usr/bin/env python

'''
    Search local catalogs
'''

import itertools as itt
import warnings

import numpy as np
import healpy as hp

from tables.table import Table as pytables_table
from astropy.io.fits.fitsrec import FITS_rec as astropy_fits

def query_kdtree(cat, crd, radius, tree, col_subset=None):

    # status
    crd_repr = 'coord:{:.6f},{:.6f}/radius:{:.3f}'.format(
                crd.ra.deg, crd.dec.deg, radius)
    stat = dict(is_complete=True,
                queried_with=crd_repr,
                indexed_with='kdtree')

    # convert to cartesian
    ra_rad, dec_rad = crd.ra.radian, crd.dec.radian
    cos_ra, sin_ra = np.cos(ra_rad), np.sin(ra_rad)
    cos_dec, sin_dec = np.cos(dec_rad), np.sin(dec_rad)
    cx, cy, cz = cos_ra * cos_dec, sin_ra * cos_dec, sin_dec

    # search radius
    rad_i = radius * 4.848e-6 # arcsec -> radian

    # search tree.
    rowid = tree.query_ball_point((cx, cy, cz), rad_i)

    # no result: stop and return
    if not rowid:
        stat['srcs'] = list()
        return stat

    # has result: get records in original table.
    if isinstance(cat, pytables_table): # pytables
        rows = tab.read_coordinates(rowid)
    elif isinstance(cat, astropy_fits): # astropy fits
        rows = cat[rowid]
    else: # neither astropy fits nor hdf table: stop.
        raise TypeError("Invalid type for 'tab'.")

    # parse results.
    cat_dtype, cat_colnames = rows.dtype.descr, rows.dtype.names
    rows_list, tab_recs = rows.tolist(), list()
    for row_i in rows_list:
        rdic_i = dict(zip(cat_colnames, row_i))
        for rdtp_i in cat_dtype:
            if col_subset and (rdtp_i[0] not in col_subset):
                rdic_i.pop(rdtp_i[0]) # remove this column.
            if len(rdtp_i) == 2: # normal column,
                col_i, dtp_i = rdtp_i
                if 'S' in rdtp_i[1]: # bytes to str
                    rdic_i[col_i] = rdic_i[col_i].decode().strip()
            elif len(rdtp_i) == 3: # nested column,
                col_i, dtp_i, dim_i = rdtp_i
                rdic_i[col_i] = rdic_i[col_i].tolist()
            else: raise RuntimeError('!')
        tab_recs.append(rdic_i)

    # back to results.
    stat['srcs'] = tab_recs

    return stat

def query_healpix(cat, crd, radius, hpid_tab, hpid_cols, col_subset=None):

    # hpid_cols : [(colname, N_side, nested/ring)]

    # sanity check
    assert isinstance(cat, pytables_table) \
            and isinstance(hpid_tab, pytables_table)

    # status
    crd_repr = 'coord:{:.6f},{:.6f}/radius:{:.3f}'.format(
                crd.ra.deg, crd.dec.deg, radius)
    stat = dict(is_complete=True,
                queried_with=crd_repr,
                indexed_with='healpix')

    if len(hpid_cols) > 1: # having more than one healpix columns,
        hp_res = [60. * hp.pixelfunc.nside2resol(nside_i, arcmin=True) \
                for col_i, nside_i, scheme_i in hpid_cols] # in arcsec.
        # select the best hpid resolution using search radius,
        hpcol_id = np.argmin(np.abs(np.log(hp_res) - np.log(radius)))
    else: # otherwise: use the only healpix column
        hpcol_id = 0
    hp_col, hp_Nside, hp_scheme = hpid_cols[hpcol_id]

    # determine pixel order.
    assert hp_scheme.lower() in ['nested', 'ring']
    is_nest = hp_scheme.lower() == 'nested'

    # write hp cols into results.
    stat['indexed_with'] = \
            'healpix:{:}{:}'.format(hp_Nside, 'n' if is_nest else 'r')

    # convert to cartesian
    ra_rad, dec_rad = crd.ra.radian, crd.dec.radian
    cos_ra, sin_ra = np.cos(ra_rad), np.sin(ra_rad)
    cos_dec, sin_dec = np.cos(dec_rad), np.sin(dec_rad)
    cx, cy, cz = cos_ra * cos_dec, sin_ra * cos_dec, sin_dec
    vec_i = (cx, cy, cz)

    # search radius
    rad_i = radius * 4.848e-6 # arcsec -> radian

    # which pixels are inside the circle?
    hpid_i = hp.query_disc(hp_Nside, vec_i, rad_i, \
            inclusive=True, nest=is_nest)
    stat['hpid'] = hpid_i.tolist()

    # warn if there are too many pixels
    if len(hpid_i) > 32:
        warnings.warn('** Too many healpix pixels returned.')

    # find rows indices,
    row_idc = list()
    for id_t in hpid_i:
        cond_str = '''{col:} == {val:} '''.format(col=hp_col, val=id_t)
        row_idc.append(hpid_tab.get_where_list(cond_str))
    row_idc = list(itt.chain(*row_idc))

    # check returned length, empty -> return empty
    if not row_idc:
        stat['srcs'] = list()
        return stat

    # have sources returned: read from main table,
    rows = cat.read_coordinates(row_idc)

    # pack into dict.
    cat_dtype, cat_colnames = rows.dtype.descr, rows.dtype.names
    rows_list, tab_srcs = rows.tolist(), list()

    for row_i in rows_list:
        rdic_i = dict(zip(cat_colnames, row_i))
        for rdtp_i in cat_dtype:
            if col_subset and (rdtp_i[0] not in col_subset):
                rdic_i.pop(rdtp_i[0]) # remove this column.
            if len(rdtp_i) == 2: # normal column,
                col_i, dtp_i = rdtp_i
                if 'S' in rdtp_i[1]: # bytes to str
                    rdic_i[col_i] = rdic_i[col_i].decode().strip()
            elif len(rdtp_i) == 3: # nested column,
                col_i, dtp_i, dim_i = rdtp_i
                rdic_i[col_i] = rdic_i[col_i].to_list()
            else: raise RuntimeError('!')
        tab_srcs.append(rdic_i)

    # back to results.
    stat['srcs'] = tab_srcs

    return stat
