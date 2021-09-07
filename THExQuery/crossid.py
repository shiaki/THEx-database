#!/usr/bin/env python

'''
    Cross-identify sources across multiple catalogs.
'''

import os, sys, json
import itertools as itt
import hashlib

from pprint import pprint

import numpy as np
from numpy.linalg import norm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import requests

from astropy.coordinates import SkyCoord

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from .utils import *
from .vac import *
from .getimage import retrieve_image

# a handy function.
plain_radec = lambda crd: [crd.ra.deg, crd.dec.deg]

def _skycoord_to_cvec(crd,):
    ra_r, dec_r = crd.ra.radian, crd.dec.radian
    cos_ra, sin_ra = np.cos(ra_r), np.sin(ra_r)
    cos_dec, sin_dec = np.cos(dec_r), np.sin(dec_r)
    return cos_ra * cos_dec, sin_ra * cos_dec, sin_dec

def _delta_cvec_to_2d(dcart, vn_ra, vn_dec):
    return  np.sum(dcart * vn_ra)  / 4.8481368e-6, \
            np.sum(dcart * vn_dec) / 4.8481368e-6

def _isvalid_crd(c):
    if isinstance(c, np.float): return np.isfinite(c)
    elif isinstance(c, str): return not (not c)
    else: return False

_is_valid_mag = lambda x: (x is not None) and np.isfinite(x) and (-5. < x < 36.)
def _get_mag(src, mag_cols):
    for col_i in mag_cols:
        val_i = src[col_i]
        if _is_valid_mag(val_i): return val_i
    return 99.

def best_hostmeta(cp_dict, cats):

    # assemble a list of catalogs,
    cat_names = ['NED', 'SIMBAD', 'HyperLEDA', 'NSA101',
                 'TwoMASSXSC', 'SDSS16pz', 'GALEXAIS67', 'TwoMASSPSC',
                 'DES2', 'ls8pz', 'PS1v2']
    for cat_i in cats.keys():
        if cat_i not in cat_names: # set list of preference
            cat_names.append(cat_i)

    stat = dict(host_ra=np.nan, host_dec=np.nan, host_srcid=None)
    for cat_i in cat_names:
        if cat_i in cp_dict:

            # set ra, dec, source id
            stat['host_ra'], stat['host_dec'] = \
                    plain_radec(cp_dict[cat_i]['srcs'][0]['_crd'])
            id_col_t = cats[cat_i]['srcid_col']
            stat['host_srcid'] = \
                [cat_i, id_col_t, cp_dict[cat_i]['srcs'][0][id_col_t]]
            # try: # DEBUG TEMP FIX
            #     stat['host_srcid'] = \
            #             [cat_i, id_col_t, cp_dict[cat_i]['srcs'][0][id_col_t]]
            # except: stat['host_srcid'] = ['NA', 'NA', 'NA']

            '''
            try:
                stat['host_srcid'] = \
                        cp_dict[cat_i]['srcs'][0][id_col_t] \
                        if id_col_t else ''
            except:
                stat['host_srcid'] = ''
            '''

            # set source of host coord.
            # stat['host_coord_src'] = cat_i
            break

    # finally, set dxy shift.
    stat['host_dxy'] = cp_dict['_avr_dxy'] if '_avr_dxy' in cp_dict \
                           else [np.nan, np.nan]

    return stat

def cross_identify(event, sources, catalogs,
                   source_filter=None, stellar_filter=None, astrom_tol=None,
                   always_seek_secondary=True, rank_func=None,
                   var_thres=False,
                   thres_axis=None,
                   do_plot=False, fig_basename=None, fig_style=None,
                   fig_bg_image=False, fig_stamps=None, fig_legend_style=None,
                   fig_fname_ext='pdf',):

    '''
    Cross-identify sources across multiple catalogs.

    Parameters
    ----------
    event : dict
        Mongodb document (in dict form) from the event collection.

    sources : dict
        Combined list of sources from different data sources.

    catalogs : dict
        Catalog description file.

    do_plot : bool
        Switch to make figures for cross-matching results.

    fig_basename : str or None
        File for output figures.

    fig_style : dict
        Style of output figures.

    Returns
    -------
    '''

    # check arguments.
    assert ((not do_plot) or fig_basename), 'Must provide basename for figures.'
    assert ((not do_plot) or fig_style), 'No style definition for figures.'

    # set default params.
    if not stellar_filter: stellar_filter = dict()
    if not source_filter: source_filter = dict()
    if not astrom_tol: astrom_tol = dict()
    # TODO: Must explicitly provide filters.

    # initial status (some keys added later!)
    stat = {

        'host_coord_known': False,
        'host_coord_type': None,
        'host_coord_matched': False,

        'N_groups': 0,
        'groups': [], # cross-identified sources here.

        'vcc': event['vcc'],
        'last_update': date_stamp(),
    }

    #----------------------------------------------------------
    # set reference frame

    # find center of query, calculate cartesian coord.
    crd_0, crd_src, rad_c = best_available_coord(event, sources)
    cvec_0 = np.array(_skycoord_to_cvec(crd_0))
    # 'sources' includes VAC results.

    # project onto 2d plane,
    cos_theta_0, sin_theta_0 = \
            np.cos(crd_0.dec.radian), np.sin(crd_0.dec.radian)
    cos_phi_0, sin_phi_0 = np.cos(crd_0.ra.radian), np.sin(crd_0.ra.radian)

    # (this is used to project things on a local 2-d plane)
    vn_ra =  np.array([-sin_phi_0, cos_phi_0, 0.])
    vn_dec = np.array([-sin_theta_0 * cos_phi_0, \
                       -sin_theta_0 * sin_phi_0, \
                        cos_theta_0])
    # can be done using a 3x3 rotation matrix, but let's do inner product.

    # 2-d position of event coord. (can be zero.)
    ra_ec, dec_ec = event['ra_deg'], event['dec_deg']
    crd_ec = SkyCoord(ra=ra_ec, dec=dec_ec, unit=('deg', 'deg'))
    delta_cvec_ec = np.subtract(_skycoord_to_cvec(crd_ec), cvec_0)
    dpos_ec_2d = _delta_cvec_to_2d(delta_cvec_ec, vn_ra, vn_dec)

    # redshift of the event.
    z_ec = event['redshift'] \
           if ('redshift' in event) and np.isfinite(event['redshift']) \
           else np.nan

    #----------------------------------------------------------
    # update info on positions and referencce frames.

    # event meta info.
    stat['event_name'] = event['name']
    stat['event_alias'] = event['alias']
    stat['revised_type'] = event['claimedtype']

    # update event absolute and relative coord,
    stat['event_ra'], stat['event_dec'] = ra_ec, dec_ec
    stat['event_radec_err'] = event['radec_err'] if 'radec_err' in event \
                              else np.nan
    stat['event_dxy'] = float(dpos_ec_2d[0]), float(dpos_ec_2d[1])
    stat['event_z'] = z_ec

    # existing host info.
    stat['reported_host'] = dict(name=event['host'] if 'host' in event else '',
                                 ra=event['host_ra_deg'],
                                 dec=event['host_dec_deg'],)

    # names resolved.
    keys_t = ['valid_names', 'resolved_names',
              'resolved_coord_inconsistent', 'inconsistent_pairs',
              'resolved_coord_best']
    stat['resolved_host_coord'] = {k: sources[k] for k in keys_t}

    # best-available host coord.
    stat['host_coord_known'] = crd_src['coord_src'] != 'event_coord'
    stat['host_coord_type'] = crd_src['coord_src'] \
                              if stat['host_coord_known'] else ''

    # update field info.
    field_coord_t = dict(ra=crd_0.ra.deg, dec=crd_0.dec.deg,
                         vn_ra=vn_ra.tolist(), vn_dec=vn_dec.tolist(),
                         cvec=cvec_0.tolist(),) # radius=rad_c,
    stat['field_coord'] = {**crd_src, **field_coord_t}

    # other flags
    stat['flags'] = list()

    #----------------------------------------------------------
    # put sources info list, do selection.

    # put catalog sources into a single list.
    src_list, excl_src_list = list(), list()
    for cat_i, cat_info_i in catalogs.items():

        if cat_i not in sources:
            continue # skip if catalog does not exist.

        # source table is 'basic' in VAC results.
        src_key_i = 'srcs' if ('srcs' in sources[cat_i]) else 'basic'
        if src_key_i not in sources[cat_i]:
            continue # Not resolved in VACs: skip.

        # use '_cat' key to indicate data source.
        for src_j in sources[cat_i][src_key_i]:
            src_list.append({**src_j, **{'_cat': cat_i}})

    # calculate positions in cartesian coord,
    cvec_delta, id_badsrc = list(), list()
    for i_src, src_i in enumerate(src_list):

        # find key names for coordinates.
        for ra_col_i, dec_col_i in catalogs[src_i['_cat']]['coord_keys']:
            if (ra_col_i in src_i) and (dec_col_i in src_i):
                break # got a valid key pair.
        assert (ra_col_i in src_i) and (dec_col_i in src_i), \
                "Missing RA/Dec keys. Check 'coord_keys'."
        #       + '\n' + repr(src_i) # DEBUG

        # bad coordinates: skip. (it happens.)
        ra_i, dec_i = src_i[ra_col_i], src_i[dec_col_i]
        if not (_isvalid_crd(ra_i) and _isvalid_crd(dec_i)):
            id_badsrc.append((i_src, 'BAD COORD C1'))
            continue

        # invalid source: skip.
        assert src_i['_cat'] in source_filter # PATCH
        if (src_i['_cat'] in source_filter) \
                and (not source_filter[src_i['_cat']](src_i)):
            id_badsrc.append((i_src, 'FILTER FUNC'))
            continue

        # remove sources beyond the search radius.
        if not (is_valid_decimal_coord(ra_i, dec_i) \
                or is_valid_sexagesimal_coord(ra_i, dec_i)):
            warnings.warn('Invalid coord detected!')
            id_badsrc.append((i_src, 'BAD COORD C2'))
            continue
        crd_i = SkyCoord(ra=ra_i, dec=dec_i, \
                unit=tuple(catalogs[src_i['_cat']]['coord_unit']))
        if crd_i.separation(crd_0).arcsec > rad_c:
            id_badsrc.append((i_src, 'BEYOND RADIUS'))
            continue

        # (accepted the source)

        # convert to cartesian.
        cvec_delta.append(np.array(_skycoord_to_cvec(crd_i)) - cvec_0)

        # also put this skycoord into the list.
        src_list[i_src]['_crd'] = crd_i

    # drop rejected sources in the list.
    for i_src, reason_i in id_badsrc[::-1]:
        excl_src_i = src_list.pop(i_src)
        if reason_i != 'BEYOND RADIUS':
            excl_src_list.append((excl_src_i, reason_i))

    #----------------------------------------------------------
    # calculate source positions,

    # 2-d position of catalog sources.
    dpos_2d = [_delta_cvec_to_2d(w, vn_ra, vn_dec) for w in cvec_delta]
    dpos_2d.append((0., 0.)) # a special point, the origin. (crd_0)

    # DEBUG: PLOT POS.
    '''
    dpos_2d_x, dpos_2d_y = np.array(dpos_2d).T
    plt.scatter(-dpos_2d_x, dpos_2d_y, s=2)
    plt.show()
    sys.exit()
    '''

    # calculate pairwise distance of catalog sources.
    N_srcs = len(src_list)
    w_srcs = np.zeros((N_srcs + 1, N_srcs + 1),) + np.inf # normed distance
    # d_srcs = np.zeros((N_srcs + 1, N_srcs + 1), dtype='i4') # conn matrix

    for i_src, j_src in itt.combinations(range(N_srcs + 1), 2):
        # +1: here we also include (0., 0.) as a "virtual source".

        # ignore sources from the same catalog.
        if (i_src < N_srcs) and (j_src < N_srcs): # (valid source)
            if src_list[i_src]['_cat'] == src_list[j_src]['_cat']:
                continue # (same catalog) - > skip

        # calculate distance
        d_ij = norm(np.subtract(dpos_2d[i_src], dpos_2d[j_src]))

        # calculate tolerance, zero for crd_0
        tol_i, tol_j = 0., 0.
        cat_i, cat_j = '__None', '__None'
        if i_src < N_srcs: # is a normal source, not the origin.
            cat_i, src_i = src_list[i_src]['_cat'], src_list[i_src]
            tol_i = astrom_tol[cat_i]
        if j_src < N_srcs: # normal source.
            cat_j, src_j = src_list[j_src]['_cat'], src_list[j_src]
            tol_j = astrom_tol[cat_j]
        tol_ij = np.sqrt(tol_i ** 2 + tol_j ** 2)

        assert cat_i != cat_j # should be true << New

        # normalized distance << New
        w_srcs[i_src, j_src] = w_srcs[j_src, i_src] = d_ij / tol_ij

    #----------------------------------------------------------
    # identify connected components.

    if var_thres: # use per-field threshold

        # original catalog of each source, threshold axis,
        src_cats = [w['_cat'] for w in src_list] + ['_crd_0']
        thres_axis_p = np.flip(np.logspace(-0.5, 0.5, 31)) \
                       if thres_axis is None else thres_axis

        # results
        conn_score_th = np.zeros_like(thres_axis_p)
        conn_results_th = list() # d_srcs, N_cps, cps_label, cps_0,

        # try values and calculate "goodness-of-matching" score
        for k_th, th_k in enumerate(thres_axis_p):

            # group corss-matched components.
            d_srcs_k = csr_matrix((w_srcs < th_k).astype('i4'))
            N_cps_k, cps_label_k = connected_components(d_srcs_k)
            cps_0_k = cps_label_k[-1] # last point (crd_0), queried coord.

            # put into array
            conn_results_th.append((d_srcs_k, N_cps_k, cps_label_k, cps_0_k))

            # count fraction of properly matched events.
            cats_incl_k = [list() for _ in range(N_cps_k)]
            for w, v in zip(cps_label_k, src_cats):
                cats_incl_k[w].append(v)

            # find properly matched sources in each catalog
            for u, cats_incl_u in enumerate(cats_incl_k):
                Np_u = [w for w in cats_incl_u \
                        if (cats_incl_u.count(w) == 1)].__len__()
                conn_score_th[k_th] += Np_u * (Np_u - 1) / 2

        # find the best matching,
        '''
        i_th_best = np.argmax(conn_score_th)                    \
                    if np.unique(conn_score_th).size > 1        \
                    else np.argmin(np.abs(thres_axis_p - 1.))
        '''

        # find the best matching.
        is_max_th = np.max(conn_score_th) == conn_score_th
        assert np.any(is_max_th) # must be at least one
        dlog_th = np.where(is_max_th, np.log10(thres_axis_p) ** 2, np.inf)
        i_th_best = np.argmin(dlog_th)

        # use the best matching.
        d_srcs, N_cps, cps_label, cps_0 = conn_results_th[i_th_best]
        N_cps_p = [w[1] for w in conn_results_th]
        thres = thres_axis_p[i_th_best]

        # DEBUG: plot score
        '''
        plt.plot(thres_axis_p, conn_score_th)
        plt.axvline(x=thres_axis_p[i_th_best])
        plt.show()
        '''

    else: # use fixed threshold (thres = 1),
        d_srcs = csr_matrix((w_srcs < 1.).astype('i4'))
        N_cps, cps_label = connected_components(d_srcs)
        cps_0 = cps_label[-1] # the last point, crd_0, our queried coord.
        thres = None

    # DEBUG: PLOT POS.
    '''
    dpos_2d_x, dpos_2d_y = np.array(dpos_2d).T
    ax = plt.gca()
    ax.scatter(-dpos_2d_x, dpos_2d_y, s=2)
    for x, y, l in zip(dpos_2d_x, dpos_2d_y, cps_label):
        ax.annotate(str(l), xy=(-x, y))
    plt.show()
    sys.exit()
    '''

    #----------------------------------------------------------
    # assign source properties.

    # group catalog sources into groups by component id.,
    xid_cps = [dict(_confusion=False,
                    _confusion_cats=list(),
                    _includes_queried_coord=(cps_0 == k).item()) \
               for k in np.unique(cps_label)]
    for i_src, (src_i, cpid_i) in enumerate(zip(src_list, cps_label)):
        #                        here the last one (queried coord) is ignored!

        cat_i = src_i.pop('_cat')
        if cat_i not in xid_cps[cpid_i]: # not in component: create new list.
            xid_cps[cpid_i][cat_i] = dict(_confusion=False, srcs=list())
        xid_cps[cpid_i][cat_i]['srcs'].append( \
                {**src_i, **{'_dxy': dpos_2d[i_src]}})
        # also put d_xy position into the dict.

        '''
        # check for confusion.
        xid_cps[cps_i][cat_i]['_confusion'] = \
                xid_cps[cps_i][cat_i]['srcs'].__len__() > 1
        if xid_cps[cps_i][cat_i]['_confusion'] \
                and (cat_i not in xid_cps[cps_i]['_confusion_cats']):
            xid_cps[cps_i]['_confusion'] = True
            xid_cps[cps_i]['_confusion_cats'].append(cat_i)
        '''

    # mark confusion sources.
    for cp_i, cat_j in itt.product(xid_cps, catalogs.keys()):

        # skip if the catalog is not in the component.
        if not (cat_j in cp_i): continue

        # check for confusion.
        cp_i[cat_j]['_confusion'] = cp_i[cat_j]['srcs'].__len__() > 1

        if cp_i[cat_j]['_confusion']:
            if not (cat_j in cp_i['_confusion_cats']):
                cp_i['_confusion'] = True
                cp_i['_confusion_cats'].append(cat_j)

    # DEBUG
    # pprint(xid_cps[26])

    # find mean dx, dy, ra, dec of each cross-matched component.
    _fc = lambda x: float(x)
    for i_cp, cp_i in enumerate(xid_cps):

        # for this component: get ra/dec, dx/dy of sources in each catalog
        radec_i = [ [(src_k['_crd'].ra.deg, src_k['_crd'].dec.deg) \
                     for src_k in catsrc_j['srcs']] \
                   for cat_j, catsrc_j in cp_i.items() if cat_j in catalogs]
        dxy_i = [[src_k['_dxy'] for src_k in catsrc_j['srcs']] \
                 for cat_j, catsrc_j in cp_i.items() if cat_j in catalogs]
        cat_nsrc_i = [(cat_j, len(catsrc_j['srcs'])) \
                      for cat_j, catsrc_j in cp_i.items() if cat_j in catalogs]

        # flatten the list.
        radec_i = list(itt.chain(*radec_i))
        dxy_i = list(itt.chain(*dxy_i))

        # if there are sources: calculate mean
        if radec_i and dxy_i:

            # position statistics.
            cp_i['_avr_radec'] = np.mean(radec_i, axis=0).tolist()
            cp_i['_avr_dxy'] = np.mean(dxy_i, axis=0).tolist()
            cp_i['_std_dxy'] = np.std(dxy_i, axis=0).tolist()
            cp_i['_cov_dxy'] = np.cov(dxy_i, rowvar=False, bias=False).tolist()

            # calculate shape stats for multiply-matched components.
            if len(dxy_i) > 1:

                cxx_t, cyy_t = cp_i['_cov_dxy'][0][0], cp_i['_cov_dxy'][1][1]
                cxy_t, cyx_t = cp_i['_cov_dxy'][0][1], cp_i['_cov_dxy'][1][0]
                pr_t = np.abs(cxy_t) / np.sqrt(cxx_t * cyy_t)
                cp_i['_shape_r'] = _fc(pr_t)
                # C_10 and C_01 should be the same.

                # shape of cov matrix.
                u_t = (cxx_t + cyy_t) / 2
                v_t = np.sqrt(4. * cxy_t * cyx_t + (cxx_t - cyy_t) ** 2) / 2
                w_t = np.clip((u_t - v_t) / (u_t + v_t), 0., 1.)
                q_t, e_t = np.sqrt(w_t), np.sqrt(1. - w_t)
                p_t = (1. - q_t) / (1. + q_t)
                cp_i['_shape_p'], cp_i['_shape_q'] = _fc(p_t), _fc(q_t)
                cp_i['_shape_e'] = _fc(e_t)

            else: # only one element, fill default values.
                cp_i['_shape_r'] = 1.
                cp_i['_shape_p'], cp_i['_shape_q'] = 1., 0.
                cp_i['_shape_e'] = 1.

            # mean distance to centroid
            dist_t = norm(np.subtract(dxy_i, cp_i['_avr_dxy']), axis=1)
            cp_i['_avr_dist'] = np.mean(dist_t).tolist()

        # calculate number of possible connections.
        if cat_nsrc_i:

            # find combinations.
            cat_nsrc_sum_i = np.sum([w[1] for w in cat_nsrc_i])
            cp_i['_N_max_conn'] = _fc( \
                    cat_nsrc_sum_i * (cat_nsrc_sum_i - 1) / 2)

            # subtract same-catalog cases.
            for cat_k, nsrc_j in cat_nsrc_i:
                if nsrc_j < 2: continue
                cp_i['_N_max_conn'] -= _fc(nsrc_j * (nsrc_j - 1) / 2)

            # count actual number of combinations.
            sidx_t = np.arange(cps_label.size - 1)[cps_label[:-1] == i_cp]
            d_srcs_sub_i = d_srcs[np.ix_(sidx_t, sidx_t)]
            cp_i['_N_conn'] = _fc(np.sum(d_srcs_sub_i) / 2)

            # "connectivity"
            cp_i['_F_conn'] = _fc(cp_i['_N_conn'] / (cp_i['_N_max_conn'] + 1))

    # DEBUG
    '''
    ax = plt.gca()
    for i_cp, cp_i in enumerate(xid_cps):
        ax.annotate(str(i_cp), xy=(-cp_i['_avr_dxy'][0], cp_i['_avr_dxy'][1]),
                    ha='center', va='center')
        ax.scatter(-cp_i['_avr_dxy'][0], cp_i['_avr_dxy'][1], s=36, alpha=0.2)
    plt.show()
    sys.exit()
    '''

    #----------------------------------------------------------
    # label stars, sort confusing sources.

    # sanity check
    assert all([w in stellar_filter for w in catalogs.keys()])

    # label potential stellar objects.
    for cp_i in xid_cps:
        for cat_j, filter_j in stellar_filter.items():
            if cat_j not in cp_i:
                continue # not matched in this group -> skip
            for src_k in cp_i[cat_j]['srcs']: # label stellar sources.
                # if filter_j(src_k): src_k['__is_stellar'] = True
                src_k['__is_stellar'] = filter_j(src_k)

    # sort confusion sources by magnitude, distance and stellarity.
    _val_to_rank = lambda a: np.argsort(np.argsort(a))
    _magc = lambda x: 99. if (x is None) or \
            (not np.isfinite(x)) or (not (0. < x < 32.)) else x
    for cp_i in xid_cps:
        for cat_j in cp_i['_confusion_cats']: # only for cats with confusion

            # num of confusing cat srcs in this grp
            ccat_srcs_j = cp_i[cat_j]['srcs']
            N_ccat_src_j = len(ccat_srcs_j)

            # rank by stellarity
            is_stellar_j = [int(('__is_stellar' in w) \
                    and bool(w['__is_stellar'])) for w in ccat_srcs_j]
            is_stellar_rank_j = np.array(is_stellar_j)

            # source magnitude rank
            mag_col_t = catalogs[cat_j]['mag_col'] \
                        if ('mag_col' in catalogs[cat_j]) else None
            mag_val_i = [_get_mag(w, mag_col_t) for w in ccat_srcs_j] \
                         if mag_col_t else None
            # if (not (mag_val_i is None)) and (cat_j == 'PS1v2'): print(mag_val_i)
            mag_val_rank_i = _val_to_rank(mag_val_i) if mag_col_t \
                             else np.zeros(N_ccat_src_j)

            # rank srcs by distance
            src_dist_j = [norm(w['_dxy']) for w in ccat_srcs_j]
            src_dist_rank_j = _val_to_rank(src_dist_j) # always available!

            # the overall rank
            rank_sum_t = is_stellar_rank_j * N_ccat_src_j \
                       + mag_val_rank_i \
                       + src_dist_rank_j / N_ccat_src_j
            id_rank_j = np.argsort(rank_sum_t)

            # re-order sources.
            cp_i[cat_j]['srcs'] = [ccat_srcs_j[i] for i in id_rank_j]

    # label stellar sources (and remove inner labels)
    _st_frac = lambda x, y: 0. if (x == 0 and y == 0) else (x / y)
    for cp_i in xid_cps:

        # init: list of star src. stat flag.
        cp_i['_stellar_srcs'] = list()
        cp_i['_is_stellar'] = False

        # count votes
        N_st_i, N_nonst_i = 0, 0
        repr_N_st_i, repr_N_nonst_i = 0, 0

        for cat_j in stellar_filter.keys(): # only cats with stellar filter.

            if not (cat_j in cp_i):
                continue # not in this group -> skip.

            # iterate over potential stellar srcs.
            for k_src, src_k in enumerate(cp_i[cat_j]['srcs']):

                is_ste_k = src_k.pop('__is_stellar')
                if is_ste_k is None: # do not have S/G separation.
                    continue # should not happen!

                if not is_ste_k: # not a star.
                    N_nonst_i += 1
                    repr_N_nonst_i += 1 if (k_src == 0) else 0
                    continue

                # this is a star!
                N_st_i += 1
                repr_N_st_i += 1 if (k_src == 0) else 0

                # label star in list. drop label.
                cp_i['_stellar_srcs'].append((cat_j, k_src))
                # src_k.pop('__is_stellar')

                if k_src == 0: # repr src is a star -> entire grp is a star
                    cp_i['_is_stellar'] = True

        # count vote.
        cp_i['_stellar_frac_all'] = _st_frac(N_st_i, N_st_i + N_nonst_i)
        cp_i['_stellar_frac_repr'] = _st_frac( \
                repr_N_st_i, repr_N_st_i + repr_N_nonst_i)

    #----------------------------------------------------------
    # summarize rejected sources.

    excl_src_stat = dict()
    for cat_i, catsrc_i in sources.items():
        if cat_i not in catalogs: continue
        srcs_key_t = 'basic' if ('basic' in catsrc_i) else 'srcs' # VAC/survey
        if srcs_key_t not in catsrc_i: continue
        excl_src_stat[cat_i] = dict(N_src=len(catsrc_i[srcs_key_t]),
                                    N_excl=0,
                                    excl_srcs=list())

    for src_i, reason_i in excl_src_list:
        cat_k = src_i.pop('_cat')
        excl_src_stat[cat_k]['N_excl'] += 1
        excl_src_stat[cat_k]['excl_srcs'].append(
                dict(data=src_i, reason=reason_i))

    # calculate total number
    N_src_t, N_excl_t = 0, 0
    for cat_i, rej_stat_i in excl_src_stat.items():
        N_src_t += rej_stat_i['N_src']
        N_excl_t += rej_stat_i['N_excl']

    excl_src_stat = {k: v for k, v in excl_src_stat.items() if v['N_excl']}
    stat['excluded_srcs'] = {'_N_src': N_src_t, '_N_excl': N_excl_t,
                             **excl_src_stat}

    #----------------------------------------------------------
    # generate unique group ID.

    # DEBUG
    _make_gsrcid = lambda x: hashlib.md5(json.dumps(x, \
            sort_keys=True, default=str).encode('utf-8')).hexdigest()

    for cp_i in xid_cps:
        srcid_i = dict()
        for cat_j, catinfo_j in catalogs.items():
            if not (cat_j in cp_i): continue
            srcid_col_t = catinfo_j['srcid_col']
            srcid_i[cat_j] = [w[srcid_col_t] for w in cp_i[cat_j]['srcs']]

            # DEBUG (TEMP FIX)
            # try: srcid_i[cat_j] = [w[srcid_col_t] for w in cp_i[cat_j]['srcs']]
            # except: srcid_i[cat_j] = [_make_gsrcid(w) for w in cp_i[cat_j]['srcs']]

        cp_i['_group_srcid'] = srcid_i
        cp_i['_group_uid'] = hashlib.md5(json.dumps( \
                srcid_i, sort_keys=True).encode()).hexdigest()

    # DEBUG
    '''
    ax = plt.gca()
    for i_cp, cp_i in enumerate(xid_cps):
        ax.annotate(str(i_cp), xy=(-cp_i['_avr_dxy'][0], cp_i['_avr_dxy'][1]),
                    ha='center', va='center')
        ax.scatter(-cp_i['_avr_dxy'][0], cp_i['_avr_dxy'][1], s=36, alpha=0.2)
    plt.show()
    sys.exit()
    '''

    #----------------------------------------------------------
    # rank cross-matched groups to find host info

    # and sort by distance to the center of query.
    cps_dist = [(norm(cp_i['_avr_dxy']) if '_avr_dxy' in cp_i else 0.) \
                for cp_i in xid_cps]
    cps_order = np.argsort(cps_dist) # <- rank by distance at the moment.

    # DEBUG
    '''
    ax = plt.gca()
    cps_rank = np.argsort(cps_order)
    for (i_cp, cp_i), rank_i in zip(enumerate(xid_cps), cps_rank):
        ax.annotate(str(i_cp) + '/' + str(rank_i),
                    xy=(-cp_i['_avr_dxy'][0], cp_i['_avr_dxy'][1]),
                    ha='center', va='center')
        ax.scatter(-cp_i['_avr_dxy'][0], cp_i['_avr_dxy'][1], s=36, alpha=0.2)
    plt.show()
    sys.exit()
    #'''

    '''
    pprint(xid_cps[26])
    sys.exit()
    '''

    # indices of components already included.
    cpid_used = list()

    # True when known host coord matches a star.
    hostcrd_match_star = False

    # True when a primary host candidate is set.
    has_primary = False

    # identified (by host name or host crd) and matches the center of query,
    if stat['host_coord_known']: # and (cps_order[0] == cps_0):
        #       the second condition is almost always guranteed,

        '''
        try:
            assert cps_order[0] == cps_0
        except:
            pprint({k: v for k, v in xid_cps[cps_0].items() if '_' in k})
            pprint({k: v for k, v in xid_cps[cps_order[0]].items() if '_' in k})
            print(cps_dist[cps_0], cps_dist[cps_order[0]])
            raise
        '''
        # it fails when queried coord matches a diffused large group. It's distance
        # could be further than normal objects.

        if cps_order[0] != cps_0:
            stat['flags'].append('QCRD_DID_MOT_MATCH_NEAREST')

        # if any([(w in xid_cps[cps_0]) for w in catalogs.keys()]):

        # valid cross-matching in any catalogs, (** should be real!)
        if xid_cps[cps_0]['_group_srcid']:

            # RULE 0: identified by host name or coord, the center of query
            # matches some catalogs -> confirmed (or primary, if stellar.)

            # input host coord cross-matched.
            stat['host_coord_matched'] = True

            # matches a galaxy-like object: mark confirmed (default case)
            stat_t = {'_xid_flag': 'confirmed'}

            if xid_cps[cps_0]['_is_stellar']: # matched a stellar object
                stat_t['_xid_flag'] = 'primary' # mark as primary candidate.
                has_primary = True
                hostcrd_match_star = True

            # put into list, mark as confirmed (or primary).
            stat['groups'].append({**xid_cps[cps_0], **stat_t})
            stat['N_groups'] += 1
            cpid_used.append(cps_0)

            # find the best host ra, dec, and identifier,
            stat.update(best_hostmeta(xid_cps[cps_0], catalogs))
            # either confirmed or primary

        else: # host coord known but not matched.
            pass

    # DEBUG
    # print(cpid_used, has_primary)
    # sys.exit()

    # Nothing found in previous step, or object identified as prim candidate:
    if (not stat['groups']) or hostcrd_match_star or always_seek_secondary:

        # 201108: Modified, if `always_seek_secondary` is True, always seek secondary.

        # RULE 1:   Not searching host coord / host coord matched nothing,
        #           host coord matches stellar objects,
        #           or always seeking secondary objects -->
        #  -->  Find and rank candidates.

        # use external regressor to rank components.
        if rank_func:
            rank_func.annotate(xid_cps, z_ec, dpos_ec_2d)
            cps_rank_score = np.array([ \
                    w['_rank_score']['_default']['score'] for w in xid_cps \
            ])

        else: # no regressor provided: use distance directly (and complain).
            print('*** Need trained regressor to rank candidates! ' \
                  '(Naive nearest ranker used.)')
            cps_rank_score = -np.array(cps_dist)

        # already have primary component? update ranking socres.
        if stat['groups']:
            primary_t = stat['groups'][0]
            primary_t['_rank_score'] = xid_cps[cps_0]['_rank_score']
            primary_t['_rank_features'] = xid_cps[cps_0]['_rank_features']
            assert len(stat['groups']) == 1

        # re-order sources.
        cps_order = np.argsort(-cps_rank_score) # higher -> more likely.

        # object matching the center of query is the best object:
        '''
        if (cps_order[0] == cps_0) \
                and any([(w in xid_cps[cps_0]) for w in catalogs.keys()]):
            # transient coord matches the best-ranking object.
            pass
            # stat['host_coord_matched'] = True # mark as valid cross-match.
            # stat['_central_coord_matched_best'] = True # mark as valid cross-match.
            # TODO: should eliminate this branch!
        '''

        # DEBUG
        # for i, j in zip(cps_order, cps_rank_score): print(i, j)

        # now put them into the list of candidate hosts.
        for cpid_i in cps_order:

            # DEBUG
            '''
            ax = plt.gca()
            cps_rank = np.argsort(cps_order)
            for (i_cp, cp_i), rank_i in zip(enumerate(stat['groups']), cps_rank):
                ax.annotate(str(i_cp) + '/' + str(rank_i),
                            xy=(-cp_i['_avr_dxy'][0], cp_i['_avr_dxy'][1]),
                            ha='center', va='center')
                ax.scatter(-cp_i['_avr_dxy'][0], cp_i['_avr_dxy'][1], s=36, alpha=0.2)
            ax.set_xlim(-60, 60)
            ax.set_ylim(-60, 60)
            ax.set_title('Just added ' + str(cpid_i))
            plt.show()
            # sys.exit()
            '''

            # this component is already marked as primary candidate -> skip.
            '''
            if hostcrd_match_star and (cpid_i == cps_0):
                continue
            '''

            # this component is already included as primary host
            '''
            if always_seek_secondary and (cpid_i in cpid_used):
                continue
            # 201108+ To avoid duplicates when always_seek_secondary and host_coord_known
            '''

            # skip if this component is already in-place.
            if cpid_i in cpid_used:

                # DEBUG
                # print('Component', cpid_i, 'skipped!')
                # sys.exit()

                continue

            # assign xidstat flag: confirmed/other or primary/secondary
            if stat['host_coord_matched'] and (not hostcrd_match_star):
                xidst_i = 'other' # known host coord matched a galaxy

            else: # no host coord, or host coord matches a star
                xidst_i = 'secondary' if has_primary else 'primary'

            '''
            if not any([w in xid_cps[cpid_i] for w in catalogs.keys()]):
            '''

            # group contains nothing, mark as event.
            if not xid_cps[cpid_i]['_group_srcid']: xidst_i = 'event'

            # update when a real object (instead of event crd) is used
            # if xidst_i == 'primary': has_primary = True

            # put into the list of candidates.
            stat['groups'].append({**xid_cps[cpid_i], **{'_xid_flag': xidst_i}})
            stat['N_groups'] += 1
            # cpid_used.append(cps_0) # <- BUG
            cpid_used.append(cpid_i)

            # update coord if this is a primary source.
            if xidst_i == 'primary':
                stat.update(best_hostmeta(xid_cps[cpid_i], catalogs))
                has_primary = True

            # debug
            '''
            print('Added: ', cpid_i, 'current list', cpid_used)
            # pprint(stat['groups'])
            ax = plt.gca()
            cps_rank = np.argsort(cps_order)
            for (i_cp, cp_i), rank_i in zip(enumerate(stat['groups']), cps_rank):
                ax.annotate(str(i_cp) + '/' + str(rank_i),
                            xy=(-cp_i['_avr_dxy'][0], cp_i['_avr_dxy'][1]),
                            ha='center', va='center')
                ax.scatter(-cp_i['_avr_dxy'][0], cp_i['_avr_dxy'][1], s=36, alpha=0.2)
            ax.set_xlim(-60, 60)
            ax.set_ylim(-60, 60)
            ax.set_title('Just added ' + str(cpid_i))
            plt.show()
            # sys.exit()
            # '''

    # convert '_crd' object into plain ra/dec.
    for cp_i in stat['groups']:
        for cat_j, catsrc_j in cp_i.items(): # for every catalog object
            # mammoths
            if cat_j in catalogs: # skip non-catalog keys
                for k_src, src_k in enumerate(catsrc_j['srcs']):
                    # dinosaurs
                    if '_crd' in src_k:
                        src_k['_crd'] = plain_radec(src_k['_crd'])
                        # trilobites

    # DEBUG
    '''
    pprint(cpid_used)
    ax = plt.gca()
    cps_rank = np.argsort(cps_order)
    for (i_cp, cp_i), rank_i in zip(enumerate(stat['groups']), cps_rank):
        ax.annotate(str(i_cp) + '/' + str(rank_i),
                    xy=(-cp_i['_avr_dxy'][0], cp_i['_avr_dxy'][1]),
                    ha='center', va='center')
        ax.scatter(-cp_i['_avr_dxy'][0], cp_i['_avr_dxy'][1], s=36, alpha=0.2)
    plt.show()
    sys.exit()
    '''

    # calculate offset using the best host.
    if 'host_dxy' not in stat:
        stat.update(dict(host_ra=np.nan, host_dec=np.nan,
                         host_srcid=None, # xid_host_coord_src=None,
                         host_dxy=[np.nan, np.nan]))

    host_offset_t = np.subtract(stat['host_dxy'], stat['event_dxy'])
    stat['host_offset'] = host_offset_t.tolist()
    stat['host_dist'] = norm(host_offset_t)

    #----------------------------------------------------------
    # write other info

    # cross-matching threshold and score
    stat['xid_thres'] = {'N_cps': N_cps,
                         'thres_axis': thres_axis_p.tolist(),
                         'conn_score': conn_score_th.tolist(),
                         'thres': thres,} \
                        if var_thres else {'thres': thres}

    # and finally, assign a random id.
    stat['rand_id'] = np.random.rand()

    #----------------------------------------------------------
    # make figure if necessary

    # make a figure.
    if do_plot:
        fig_fname = '.'.join([fig_basename, \
                event['name'], fig_fname_ext]).replace(' ', '_')
        _crossid_plot(xid_cps, crd_0, event['name'], fig_fname, fig_style,
                      fig_bg_image, fig_stamps, fig_legend_style)

    return stat

def _crossid_plot(xid_cps, crd_0, name, fig_fname,
                  fig_style, fig_bg_image, fig_stamps,
                  fig_legend_style):

    # determine figure extension.
    cp_avr = np.array([w['_avr_dxy'] for w in xid_cps if ('_avr_dxy' in w)])
    if cp_avr.size < 2:
        return # terminate early if it doesn't work.
    ax_rg = np.abs(cp_avr).max() * 1.2

    # figure
    fig = plt.figure(figsize=(6., 6.))
    ax = fig.add_subplot(1, 1, 1)

    # download and display background image?
    if fig_bg_image:
        bg_img, bg_src = retrieve_image(crd_0.ra.deg, crd_0.dec.deg, ax_rg)
        if bg_src:
            ax.imshow(bg_img, origin='upper',
                      extent=(ax_rg, -ax_rg, -ax_rg, ax_rg),
                      alpha=0.7)

    # catalogs for which we create legend.
    legend_cats = list()

    # draw components onto the figure.
    for cp_i in xid_cps:

        if ('_avr_dxy' not in cp_i):
            continue

        # mean pos and std of srcs.
        xy_i, xystd_i = cp_i['_avr_dxy'], norm(cp_i['_std_dxy'])

        # first: circle of cross-matched object.
        if fig_style['_misc']['circle']:# and (xystd_i > 1.e-6):
            circ_rad_i = xystd_i * fig_style['_misc']['circle_factor']
            circ_rad_i = max(circ_rad_i, fig_style['_misc']['circle_minrad'])
            style_key_i = '_xid_star' if cp_i['_is_stellar'] else '_xid_galaxy'
            circ_i = Circle(xy_i, radius=circ_rad_i, **fig_style[style_key_i])
            ax.add_artist(circ_i)

        # a dict for stellar objects.
        stars_i = dict()
        for cat_j, src_id_j in cp_i['_is_stellar']:
            stars_i.setdefault(cat_j, []).append(src_id_j)

        # draw objects in each component.
        for cat_j, src_stat_j in cp_i.items():

            if cat_j[0] == '_': # not a catalog key: skip
                continue

            # put this catalog into legend list.
            if cat_j not in legend_cats:
                legend_cats.append(cat_j)

            # read list of sources, get rel coord, plot.
            dxy_j = [w['_dxy'] for w in src_stat_j['srcs']]

            # check: has stellar sources? --> use stellar style
            if cat_j in stars_i:
                for k_dxy, dxy_k in enumerate(dxy_j):
                    st_k = fig_style[cat_j] if (k_dxy not in stars_i[cat_j]) \
                           else fig_style[cat_j + '_st'] # select style by type.
                    ax.scatter(dxy_k[0], dxy_k[1], **st_k)
            else: # plot in a unified style
                dxy_j = np.array(dxy_j)
                ax.scatter(dxy_j[:, 0], dxy_j[:, 1], **fig_style[cat_j])

    # zoom-in cross-matched groups.
    if fig_stamps: # expect a dict/list.

        # single zoom-in panel: convert to list.
        assert isinstance(fig_stamps, (dict, list))
        zoom_cps = [fig_stamps] \
                   if isinstance(fig_stamps, dict) \
                   else fig_stamps

        # make inset axis for each component.
        for zoom_cp_i in zoom_cps:

            # make axis.
            ax_ins_i = zoomed_inset_axes(ax, zoom_cp_i['zoom_lv'],
                                         loc=zoom_cp_i['ax_loc'])

            # get cross-matched component.
            zcp_i = xid_cps[zoom_cp_i['id_cp']]

            # draw circle.
            if fig_style['_misc']['circle_in_zoom']:
                crad_i = norm(zcp_i['_std_dxy']) \
                       * fig_style['_misc']['circle_factor']
                crad_i = max(crad_i, fig_style['_misc']['circle_minrad'])
                style_key_i = '_xid_star' \
                              if zcp_i['_is_stellar'] \
                              else '_xid_galaxy'
                circ_i = Circle(zcp_i['_avr_dxy'], radius=crad_i, \
                                **fig_style[style_key_i])
                ax_ins_i.add_artist(circ_i)

            # count stellar objects, (cat, src_id)
            stars_i = dict()
            for cat_j, src_id_j in zcp_i['_is_stellar']:
                stars_i.setdefault(cat_j, []).append(src_id_j)

            # draw individual catalogs.
            for cat_j, src_stat_j in zcp_i.items():

                if cat_j[0] == '_': # skip non-catalog keys
                    continue

                # read list of sources, get rel coord, plot.
                dxy_j = [w['_dxy'] for w in src_stat_j['srcs']]

                # use stellar style if contains stellar sources.
                if cat_j in stars_i:
                    for k_dxy, dxy_k in enumerate(dxy_j):

                        # select '_st' (stellar) type for stars
                        st_k = fig_style[cat_j] \
                               if (k_dxy not in stars_i[cat_j]) \
                               else fig_style[cat_j + '_st']
                        ax_ins_i.scatter(dxy_k[0], dxy_k[1], **st_k)

                else: # or plot in a unified style
                    dxy_j = np.array(dxy_j)
                    ax_ins_i.scatter(dxy_j[:, 0], dxy_j[:, 1],
                                     **fig_style[cat_j])

            # adjust stamp size.
            xll_i, xul_i = ax_ins_i.get_xlim()
            yll_i, yul_i = ax_ins_i.get_xlim()
            ax_ext_i = max(yul_i - yll_i, xul_i - xll_i) \
                     * fig_style['_misc']['axins_ext_factor'] / 2.
            ax_ins_i.set_xlim(zcp_i['_avr_dxy'][0] - ax_ext_i,
                              zcp_i['_avr_dxy'][0] + ax_ext_i)
            ax_ins_i.set_ylim(zcp_i['_avr_dxy'][1] - ax_ext_i,
                              zcp_i['_avr_dxy'][1] + ax_ext_i)

            # hide labels and ticks.
            ax_ins_i.set_xticks([])
            ax_ins_i.set_yticks([])

            # indicate zoom scale.
            ax_ins_i.annotate('{:}x Scale'.format(zoom_cp_i['zoom_lv']),
                    xy=(0.05, 0.05), xycoords='axes fraction',
                    ha='left', va='bottom',)

            # and connect to the original region,
            # mark_inset(ax, ax_ins_i, loc1=2, loc2=4, fc='none', ec='0.5')

    # make legend for catalogs.
    if fig_legend_style is not None: # Not NULL --> plot legend.
        legend_elem, legend_label = list(), list()
        for elem_i, label_i in zip(*ax.get_legend_handles_labels()):
            if label_i not in legend_label: # remove duplicates.
                legend_elem.append(elem_i)
                legend_label.append(label_i)
        legend = ax.legend(legend_elem, legend_label, **fig_legend_style)
        for hd_i in legend.legendHandles:
            hd_i._sizes = [30]

        '''
        for cat_i, cat_style_i in fig_style.items():
            if (cat_i in legend_cats) \
                    or (cat_i.replace('_st', '') in legend_cats):
                # lab_t = cat_i.replace('_st', ' (Star)').replace('_', ' ')
                elm_i = ax.scatter(ax_rg * 9., ax_rg * 9., **cat_style_i,)
                legend_elem.append(elm_i)
        '''

    ax.set_xlim(ax_rg, -ax_rg)
    ax.set_ylim(-ax_rg, ax_rg)
    ax.set_xlabel(r'$\Delta$ RA [arcsec]')
    ax.set_ylabel(r'$\Delta$ Dec [arcsec]')
    ax.set_title(name)

    plt.savefig(fig_fname)
    plt.close()

    return None

# did I really write this? I cannot even understand. YJ, 080719

# now sort other cross-matched groups.
'''
cps_rank = np.zeros_like(cps_order,)
cps_rank[cps_order] = np.arange(cps_order.size).astype(int)
'''

# RULE 1: objects identified in value-added catalogs have higher rank.
'''
vacs_t = ['NED', 'SIMBAD', 'HyperLEDA', 'NSA101']
vac_count = [np.sum([w in cp_i for w in vacs_t]) for cp_i in xid_cps]
vac_count = np.clip(vac_count, 0, 1)
# Not counting numbers of VACs (Y. 191030)
cps_rank -= np.array(vac_count) * (N_cps + 1)
'''

# RULE 2: stellar objects have lower rank, especially when multiple
# catalogs indicate that the object is star-like.
'''
stellar_count = [len(cp_i['_is_stellar']) for cp_i in xid_cps]
cps_rank += np.array(stellar_count) * (N_cps + 1)
'''

# RULE 3: LS DR7 special.
# Rule 3 is no longer needed. Using LS DR8 now.
# Using ridge classifier now. 200117
# Previous approach, pretty inefficient.

'''
# distance below threshold, different catalog -> connected.
if (cat_i != cat_j) and (d_ij < tol_ij * 1.):
    d_srcs[i_src, j_src] = d_srcs[j_src, i_src] = 1
'''

'''
d_srcs = csr_matrix(d_srcs)
N_cps, cps_label = connected_components(d_srcs)
cps_0 = cps_label[-1] # the last point, crd_0, our queried coord.
'''
