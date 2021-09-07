#!/usr/bin/env python

'''
    Rank cross-matched groups with trained Ridge Regressor
'''

import pickle
import numpy as np
from numpy.linalg import norm
from astropy.cosmology import WMAP9 as cosmo

def _radec_to_cvec(crd,):
    ra_r, dec_r = crd.ra.radian, crd.dec.radian
    cos_ra, sin_ra = np.cos(ra_r), np.sin(ra_r)
    cos_dec, sin_dec = np.cos(dec_r), np.sin(dec_r)
    return cos_ra * cos_dec, sin_ra * cos_dec, sin_dec

# how to scale distance?
_dist_scale_func = [
    lambda x: 0.0167 * x,           # linear scale dist in arcmin.
    lambda x: np.arcsinh(x),        # arcsinh-scaled dist (arcsec)
    lambda x: 1. / (1. + x),        # inverse distance.
]
_dist_z_scale_func = [
    lambda x, s: x * s / 15.,               # proper dist in 15 kpc.
    lambda x, s: np.arcsinh(x * s),         # arcsinh-scaled dist (kpc).
    lambda x, s: 1. / (1. + x * s / 15.),   # inverse distance (kpc).
]

_val_to_rank = lambda x: np.argsort(np.argsort(x)) / len(x)

class RidgeRegressorRanker(object):

    ''' Ridge Regressor '''

    def __init__(self, model_file):

        ''' Load trained model. '''

        with open(model_file, 'rb') as f: self._model = pickle.load(f)

        self._cl, self._cl_z = self._model['cl'], self._model['cl_z']
        self._cat_to_index = self._model['cat_to_index'] # dict.

    def annotate(self, xid_groups, z_ec, dpos_2d_ec):

        if np.isfinite(z_ec): # with redshift: use redshift scaling
            self._annotate_z(xid_groups, z_ec, dpos_2d_ec)

        else: # no redshift: use angular distance.
            self._annotate(xid_groups, z_ec, dpos_2d_ec)

    def _annotate_z(self, xid_groups, z_ec, dpos_2d_ec):

        ''' Predict using the model. '''

        # which objects should be ranked
        is_valid = [any([w in cp_i for w in self._cat_to_index.keys()])
                    for cp_i in xid_groups]
        N_valid = np.sum(is_valid)

        # nothing to do: stop.
        if not np.any(is_valid): return

        _val_to_rank = lambda x: np.argsort(np.argsort(x)) / x.size

        X = list()

        # calculate input variables for components.
        for cp_i, valid_i in zip(xid_groups, is_valid):

            # ignore invalid sources.
            if not valid_i: continue

            # count catalogs.
            cat_count_i = np.zeros(22)
            for k, v in self._cat_to_index.items():
                if k in cp_i: cat_count_i[v] = 1.
            cat_sum_i = np.sum(cat_count_i)

            # scaled dist.
            dist_i = norm(np.subtract(cp_i['_avr_dxy'], dpos_2d_ec))

            # distance.
            dist_scale_i = [fc(dist_i) for fc in _dist_scale_func]

            # redshift-scaled distance.
            kpc_scal = cosmo.kpc_proper_per_arcmin( \
                    np.sqrt(z_ec ** 2 + 1.e-5)).value / 60.
            dist_z_scale_i = [fc(dist_i, kpc_scal) \
                              for fc in _dist_z_scale_func]
            dist_rank_i = dist_i # dummy value.

            # shape param
            shape_i = [
                cp_i['_shape_r'],
                cp_i['_shape_p'],
                cp_i['_shape_q'],
                cp_i['_shape_e'],
            ]
            if np.any(~np.isfinite(shape_i)):
                shape_i = [1., 1., 0., 1.]

            # connectivity.
            conn_i = [
                np.arcsinh(cp_i['_N_max_conn']),
                np.arcsinh(cp_i['_N_conn']),
                cp_i['_F_conn']
            ]

            # stellarity.
            is_stellar_i = int(cp_i['_is_stellar'])

            X.append(np.concatenate([
                cat_count_i,    # 0 ... 21
                [cat_sum_i],    # 22 **             -> rank
                dist_scale_i,   # 23, 24, 25
                dist_z_scale_i, # 26, 27, 28
                [dist_rank_i],  # 29 **             -> rank
                shape_i,        # 30, 31, 32, 33
                conn_i,         # 34, 35, 36
                [is_stellar_i]  # 37
            ]))

        # merge X into a single list.
        X = np.vstack(X)
        # print(X.shape, len(xid_groups))

        # calcualte contextual variables.
        X[:, 22] = _val_to_rank(X[:, 22])
        X[:, 29] = _val_to_rank(X[:, 29])

        # call regressor.
        v_sc = self._cl_z.decision_function(X)
        v_sc_rank = np.argsort(np.argsort(-v_sc))

        # calculate input variables for components.
        i_X = 0
        for cp_i, valid_i in zip(xid_groups, is_valid):

            # not included: skip.
            if not valid_i: continue

            cp_i['_rank_score'] = float(v_sc[i_X])
            cp_i['_rank_var'] = X[i_X].tolist()
            cp_i['_rank_model'] = 'RRz_v1'
            cp_i['_rank_in_field'] = int(v_sc_rank[i_X])

            i_X += 1

    def _annotate(self, xid_groups, z_ec, dpos_2d_ec):

        ''' Predict using the model. '''

        # which objects should be ranked
        is_valid = [any([w in cp_i for w in self._cat_to_index.keys()])
                    for cp_i in xid_groups]
        N_valid = np.sum(is_valid)

        # nothing to do: stop.
        if not np.any(is_valid): return

        _val_to_rank = lambda x: np.argsort(np.argsort(x)) / x.size

        X = list()

        # calculate input variables for components.
        for cp_i, valid_i in zip(xid_groups, is_valid):

            # ignore invalid sources.
            if not valid_i: continue

            # count catalogs.
            cat_count_i = np.zeros(22)
            for k, v in self._cat_to_index.items():
                if k in cp_i: cat_count_i[v] = 1.
            cat_sum_i = np.sum(cat_count_i)

            # scaled dist.
            dist_i = norm(np.subtract(cp_i['_avr_dxy'], dpos_2d_ec))

            # distance.
            dist_scale_i = [fc(dist_i) for fc in _dist_scale_func]

            # rank of distance
            dist_rank_i = dist_i # dummy value.

            # shape param
            shape_i = [
                cp_i['_shape_r'],
                cp_i['_shape_p'],
                cp_i['_shape_q'],
                cp_i['_shape_e'],
            ]
            if np.any(~np.isfinite(shape_i)):
                shape_i = [1., 1., 0., 1.]

            # connectivity.
            conn_i = [
                np.arcsinh(cp_i['_N_max_conn']),
                np.arcsinh(cp_i['_N_conn']),
                cp_i['_F_conn']
            ]

            # stellarity.
            is_stellar_i = int(cp_i['_is_stellar'])

            X.append(np.concatenate([
                cat_count_i,    # 0 ... 21
                [cat_sum_i],    # 22 **             -> rank
                dist_scale_i,   # 23, 24, 25
                [dist_rank_i],  # 26 **             -> rank
                shape_i,        # 27, 28, 29, 30
                conn_i,         # 31, 32, 33
                [is_stellar_i]  # 34
            ]))

        # merge X into a single list.
        X = np.vstack(X)
        # print(X.shape, len(xid_groups))

        # calcualte contextual variables.
        X[:, 22] = _val_to_rank(X[:, 22])
        X[:, 26] = _val_to_rank(X[:, 26])

        # call regressor.
        v_sc = self._cl.decision_function(X)
        v_sc_rank = np.argsort(np.argsort(-v_sc))

        # calculate input variables for components.
        i_X = 0
        for cp_i, valid_i in zip(xid_groups, is_valid):

            # not included: skip.
            if not valid_i: continue

            cp_i['_rank_score'] = float(v_sc[i_X])
            cp_i['_rank_var'] = X[i_X].tolist()
            cp_i['_rank_model'] = 'RR_v1'
            cp_i['_rank_in_field'] = int(v_sc_rank[i_X])

            i_X += 1
