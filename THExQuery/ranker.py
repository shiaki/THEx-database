#!/usr/bin/env python

'''
    Rank cross-matched groups with trained Ridge Regressor
'''

import pickle
import itertools as itt
from pprint import pprint
from collections import defaultdict

import numpy as np
np.seterr(all='ignore')
from numpy.linalg import norm
from astropy.cosmology import WMAP9 as cosmo

def _radec_to_cvec(crd,):
    ra_r, dec_r = crd.ra.radian, crd.dec.radian
    cos_ra, sin_ra = np.cos(ra_r), np.sin(ra_r)
    cos_dec, sin_dec = np.cos(dec_r), np.sin(dec_r)
    return cos_ra * cos_dec, sin_ra * cos_dec, sin_dec

_is_valid_mag = lambda x: (x is not None) and np.isfinite(x) and (0. < x < 30.)
_opt_mag_cols = [
    ('LS8pz',       ['mag_r', 'mag_g']),
    ('PS1v2',       ['rKronMag', 'iKronMag', 'gKronMag']),
    ('DES2',        ['mag_auto_r', 'mag_auto_i', 'mag_auto_g']), # really?
    ('SDSS16pz',    ['r', 'i', 'g']),
    ('VSTATLAS3',   ['rpmag', 'ipmag', 'gpmag']),
    ('SkyMapper2',  ['r_petro', 'i_petro', 'g_petro']),
]
def _optical_mag(cp, survey=None):
    ''' Find a representative optical magnitude. '''
    for cat_i, cols_i in _opt_mag_cols:
        if survey and (cat_i not in survey): continue
        if cat_i not in cp: continue
        for col_k in cols_i:
            mag_k = cp[cat_i]['srcs'][0][col_k]
            if _is_valid_mag(mag_k):
                return (cat_i, col_k, mag_k)
    return '', '', np.nan

_is_valid_rKron = lambda x: np.isfinite(x) and (x > 0.)
_dist = lambda a, b: np.sqrt(np.sum(np.power(np.subtract(a, b), 2)))
def _PS1_dKron(cp, sn_dxy):
    if ('PS1v2' not in cp) or (not cp['PS1v2']['srcs']):
        return np.nan, np.nan
    d_theta = _dist(cp['_avr_dxy'], sn_dxy)
    r_Kron = np.nan
    for col_t in ['rKronRad', 'iKronRad', 'gKronRad']:
        if _is_valid_rKron(cp['PS1v2']['srcs'][0][col_t]):
            r_Kron = cp['PS1v2']['srcs'][0][col_t]
            break
    return [d_theta / r_Kron, r_Kron]

_exp_Nr = 3.524045080558021
_dev_Nr = 5.579448327961108
def _LS_dRe(cp, sn_dxy):

    ''' Find d_DLR using LS DR8 data. '''

    if ('LS8pz' not in cp) or (not cp['LS8pz']['srcs']):
        return [np.nan, np.nan, np.nan, np.nan]

    ls_rsrc = cp['LS8pz']['srcs'][0]

    if ls_rsrc['type'] in ['PSF', 'DUP']:
        eps2_k, eps1_k = 0., 1.
        q_k = 1.
        a_k, b_k = 0., 0.
        scale_frac = 1.

    elif ls_rsrc['type'] in ['EXP', 'REX']: # exponential-like

        # axis ratio
        eps1_k, eps2_k = ls_rsrc['shapeexp_e1'], ls_rsrc['shapeexp_e2']
        epsn_k = np.sqrt(eps1_k ** 2 + eps2_k ** 2)
        q_k = (1. - epsn_k) / (1. + epsn_k)

        # size and scale factor
        a_k = ls_rsrc['shapeexp_r']
        b_k = a_k * q_k
        scale_frac = _exp_Nr

    elif ls_rsrc['type'] == 'DEV':

        # axis ratio
        eps1_k, eps2_k = ls_rsrc['shapedev_e1'], ls_rsrc['shapedev_e2']
        epsn_k = np.sqrt(eps1_k ** 2 + eps2_k ** 2)
        q_k = (1. - epsn_k) / (1. + epsn_k)

        # size and scale factor
        a_k = ls_rsrc['shapedev_r']
        b_k = a_k * q_k
        scale_frac = _dev_Nr

    elif ls_rsrc['type'] == 'COMP':

        _dfrac = ls_rsrc['fracdev']
        _dfrac_interp = lambda x1, x2: x1 * (1. - _dfrac) + x2 * _dfrac

        # axis ratio
        eps1_k = _dfrac_interp(ls_rsrc['shapeexp_e1'], ls_rsrc['shapedev_e1'])
        eps2_k = _dfrac_interp(ls_rsrc['shapeexp_e2'], ls_rsrc['shapedev_e2'])
        epsn_k = np.sqrt(eps1_k ** 2 + eps2_k ** 2)
        q_k = (1. - epsn_k) / (1. + epsn_k)

        # size and scale factor
        a_k = _dfrac_interp(ls_rsrc['shapeexp_r'], ls_rsrc['shapedev_r'])
        b_k = a_k * q_k
        scale_frac = _dfrac_interp(_exp_Nr, _dev_Nr)

    else: raise RuntimeError('!')

    # pos angle
    phi_k = np.pi / 2. - np.arctan2(eps2_k, eps1_k) / 2. # phi_k: North-of-east.

    # FIX BAD VALUES.
    if (ls_rsrc['type'] not in ['PSF', 'DUP']) and \
            ((a_k * b_k == 0.) or (not np.isfinite(phi_k + a_k + b_k))):
        a_k, b_k, q_k, phi_k, scale_frac = 0., 0., 1., 0., 1.
        # print('Values fixed.')

    # expand radius a little bit.
    a_k = np.sqrt(a_k ** 2 + 0.01)
    b_k = np.sqrt(b_k ** 2 + 0.01)

    # avoid div by zero error.
    inv_a_k = np.divide(1., a_k)
    inv_b_k = np.divide(1., b_k)

    # factors.
    cos_phi_k, sin_phi_k = np.cos(phi_k), np.sin(phi_k)
    cxx_k = (cos_phi_k * inv_a_k) ** 2 + (sin_phi_k * inv_b_k) ** 2
    cyy_k = (sin_phi_k * inv_a_k) ** 2 + (cos_phi_k * inv_b_k) ** 2
    cxy_k = cos_phi_k * sin_phi_k * (2. * inv_a_k ** 2 - 2. * inv_b_k ** 2)

    # dra, ddec = SN - Galaxy, from galaxy to SN
    dra_k  = sn_dxy[0] - cp['_avr_dxy'][0]
    ddec_k = sn_dxy[1] - cp['_avr_dxy'][1]
    ddlr_k = np.sqrt( cxx_k * dra_k  ** 2 + cyy_k * ddec_k ** 2
                    + cxy_k * dra_k * ddec_k)

    iso_r50 = np.sqrt(a_k * b_k)
    rval = [ddlr_k / scale_frac, ddlr_k, iso_r50 * scale_frac, iso_r50]
    assert np.all(np.isfinite(rval))
    return rval

_is_valid_photoz = lambda x: np.isfinite(x) and (-0.1 < x < 9.)
def _LS_delta_z(cp, sn_z):

    ''' Find d_DLR using LS DR8 data. '''

    if ('LS8pz' not in cp) or (not cp['LS8pz']['srcs']):
        return [np.nan, np.nan, np.nan]

    ls_rsrc = cp['LS8pz']['srcs'][0]
    zph_mean, zph_std = ls_rsrc['z_phot_mean'], ls_rsrc['z_phot_std']
    if not (_is_valid_photoz(zph_mean) and _is_valid_photoz(zph_std)):
        return [np.nan, np.nan, np.nan]

    delta_zph = zph_mean - sn_z
    nsigma_zph = delta_zph / zph_std
    return [zph_mean, delta_zph ** 2, np.sqrt(nsigma_zph ** 2 + 1.) - 1.]

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

_empty_group_features = {

    # object count.
    'object_count': [0.] * 21,
    'object_count_rank': np.nan, # convert!

    # distances.
    'angular_distance': [np.nan] * 3,
    'zscaled_distance': [np.nan] * 3,
    'angular_distance_rank': np.nan,

    # shape..
    'shape_param': [np.nan] * 4,
    'connectivity_param': [np.nan] * 3,

    # stellarity
    'stellarity': [np.nan],
    'stellarity_frac': [np.nan],

    # magnitudes.
    'optical_mags': [np.nan],
    'ps1_mags': [np.nan],
    'ls8_mags': [np.nan],

    # size and shape params.
    'ps1_size_dKron': [np.nan] * 2,
    'ls8_size_dRe': [np.nan] * 4,

    # photo-z
    'ls8_delta_zph': [np.nan] * 3,
}

# assign features to a group.
def _featurize(xid_groups, sn_z, sn_dxy, cat_to_index):

    # scaling factor from asec to kpc
    kpc_scal = cosmo.kpc_proper_per_arcmin( \
            np.sqrt(sn_z ** 2 + 1.e-5)).value / 60. \
            if np.isfinite(sn_z) else np.nan

    # features.
    group_features = [dict() for _ in xid_groups]
    for i_group, group_i in enumerate(xid_groups):

        # check: empty group?
        if len(group_i['_group_srcid']) == 0:
            group_features[i_group].update(_empty_group_features)
            continue

        # catalog count.
        cat_count_i = np.zeros(len(cat_to_index)).tolist()
        for k, v in cat_to_index.items():
            if (k in group_i) and group_i[k]['srcs']:
                cat_count_i[v] = 1.
        cat_sum_i = np.sum(cat_count_i) # convert to rank!

        # scaled dist.
        dist_i = np.linalg.norm(np.subtract(group_i['_avr_dxy'], sn_dxy)) # \
        #        if has_source_i else np.nan
        dist_scale_i = [fc(dist_i) for fc in _dist_scale_func]
        dist_rank_i = dist_i # convert to rank!

        # z-scaled dist.
        dist_z_scale_i = [fc(dist_i, kpc_scal) for fc in _dist_z_scale_func]

        # group shape.
        shape_i = [
            group_i['_shape_r'],
            group_i['_shape_p'],
            group_i['_shape_q'],
            group_i['_shape_e'],
        ]

        if np.any(~np.isfinite(shape_i)):
            shape_i = [1., 1., 0., 1.]

        # connectivity.
        conn_i = [
            np.arcsinh(group_i['_N_max_conn']),
            np.arcsinh(group_i['_N_conn']),
            group_i['_F_conn']
        ]

        # stellarity.
        is_stellar_i = float(group_i['_is_stellar'])
        stellar_frac_i = group_i['_stellar_frac_repr']

        # magnitude.
        mag_i = _optical_mag(group_i)[2]

        # PS1 features.
        PS1_mag_i = _optical_mag(group_i, 'PS1v2')[2]
        PS1_dKron_i = _PS1_dKron(group_i, sn_dxy)

        # LS features.
        LS_mag_i = _optical_mag(group_i, 'LS8pz')[2]
        LS_dRe_i = _LS_dRe(group_i, sn_dxy)
        LS_delta_z_i = _LS_delta_z(group_i, sn_z)

        # fill results.
        group_dict_i = group_features[i_group]

        # object count.
        group_dict_i['object_count'] = cat_count_i
        group_dict_i['object_count_rank'] = cat_sum_i # convert!

        # distances.
        group_dict_i['angular_distance'] = dist_scale_i
        group_dict_i['zscaled_distance'] = dist_z_scale_i
        group_dict_i['angular_distance_rank'] = dist_rank_i

        # shape..
        group_dict_i['shape_param'] = shape_i
        group_dict_i['connectivity_param'] = conn_i

        # stellarity
        group_dict_i['stellarity'] = [is_stellar_i]
        group_dict_i['stellarity_frac'] = [stellar_frac_i]

        # magnitudes.
        group_dict_i['optical_mags'] = [mag_i]
        group_dict_i['ps1_mags'] = [PS1_mag_i]
        group_dict_i['ls8_mags'] = [LS_mag_i]

        # size and shape params.
        group_dict_i['ps1_size_dKron'] = PS1_dKron_i
        group_dict_i['ls8_size_dRe'] = LS_dRe_i

        # photo-z
        group_dict_i['ls8_delta_zph'] = LS_delta_z_i

    # calculate the rank.
    Nsrc_rank = np.array([w['object_count_rank'] for w in group_features])
    dist_rank = np.array([w['angular_distance_rank'] for w in group_features])
    _is_valid = np.greater(Nsrc_rank, 0.)
    Nsrc_rank = robust_val_to_rank(Nsrc_rank, _is_valid)
    dist_rank = robust_val_to_rank(dist_rank, _is_valid)
    assert np.all(np.isfinite(dist_rank[_is_valid]))
    assert np.all(np.isfinite(Nsrc_rank[_is_valid]))

    # write back.
    for group_dict_i, Nsrc_rank_i, dist_rank_i \
            in zip(group_features, Nsrc_rank, dist_rank):
        group_dict_i['object_count_rank'] = [Nsrc_rank_i]
        group_dict_i['angular_distance_rank'] = [dist_rank_i]

    # done
    return group_features

# assign features to a group.
def _featurize_lazy(xid_groups, sn_z, sn_dxy, cat_to_index):

    ''' Calc features using existing data, only update z-related info. '''

    # scaling factor from asec to kpc
    kpc_scal = cosmo.kpc_proper_per_arcmin( \
            np.sqrt(sn_z ** 2 + 1.e-5)).value / 60. \
            if np.isfinite(sn_z) else np.nan

    # copy existing.
    _ddc = lambda d: {k: np.atleast_1d(v) for k, v in d.items()}
    group_features = [_ddc(w['_rank_features']) for w in xid_groups]

    # update new features.
    for i_group, group_i in enumerate(xid_groups):

        # check: empty group?
        if len(group_i['_group_srcid']) == 0: continue

        # calc z-related info.
        dist_i = np.linalg.norm(np.subtract(group_i['_avr_dxy'], sn_dxy))
        dist_z_scale_i = [fc(dist_i, kpc_scal) for fc in _dist_z_scale_func]
        LS_delta_z_i = _LS_delta_z(group_i, sn_z)

        # update z-related info.
        group_features[i_group]['zscaled_distance'] = dist_z_scale_i
        group_features[i_group]['ls8_delta_zph'] = LS_delta_z_i

    return group_features

# rank function.
_val_to_rank = lambda x: np.argsort(np.argsort(x)) / len(x)
def robust_val_to_rank(X, is_valid):
    _rank = np.zeros_like(X) + np.nan
    _rank[is_valid] = _val_to_rank(X[is_valid])
    return _rank

# check if a group has certain catalog matched.
_has_catalog = lambda G, c: (c in G) and bool(G[c]['srcs'])

class Ranker(object):

    ''' Ranking functions. '''

    def __init__(self, model_file):

        ''' Load trained model. '''

        with open(model_file, 'rb') as f:
            self._model = pickle.load(f)

    def annotate(self, xid_groups, z_ec, dpos_2d_ec, lazy=False):

        # do we have redshift?
        field_has_z = np.isfinite(z_ec)

        # optical magnitudes.
        group_mag = [_optical_mag(g)[2] for g in xid_groups]
        group_has_mag = np.isfinite(group_mag).tolist()
        field_has_mag = any(group_has_mag)

        # check coverage of LS and PS1
        group_has_PS1 = [_has_catalog(g, 'PS1v2') for g in xid_groups]
        group_has_LS  = [_has_catalog(g, 'LS8pz') for g in xid_groups]
        field_has_PS1, field_has_LS = any(group_has_PS1), any(group_has_LS)

        # determine which models to use.
        use_models = list()
        _suffix = 'Z' if field_has_z else ''
        use_models.append('Basic8' + _suffix)
        if field_has_LS:  use_models.append('LS' + _suffix)
        if field_has_PS1: use_models.append('PS1' + _suffix)
        if field_has_mag: use_models.append('AnyMag' + _suffix)

        '''
        use_models = list()

        use_models.append('Basic8') # Basic
        if field_has_z: use_models.append('Basic8Z')

        if field_has_mag: # AnyMag
            use_models.append('AnyMag')
            if field_has_z: use_models.append('AnyMagZ')

        if field_has_PS1: # PS1
            use_models.append('PS1')
            if field_has_z: use_models.append('PS1Z')

        if field_has_LS: # LS
            use_models.append('LS')
            if field_has_z: use_models.append('LS')
        '''

        # featurize
        group_features = _featurize_lazy(xid_groups, z_ec, dpos_2d_ec,
                self._model['cat_to_index']) if lazy else _featurize( \
                xid_groups, z_ec, dpos_2d_ec, self._model['cat_to_index'])
        # sn_z, sn_dxy, cat_to_index

        # calculate scores.
        for i_group, grp_feat_i in enumerate(group_features):

            # calc rank scores.
            rank_scores_i = dict()
            for model_j in use_models:

                # pack features.
                features_j = self._model[model_j + '_features']
                X_j = np.concatenate([grp_feat_i[t] for t in features_j])

                # exclude groups with bad features.
                if np.any(~np.isfinite(X_j)): continue

                # scale and calculate score.
                Xavr_j = self._model[model_j + '_Xavr']
                Xstd_j = self._model[model_j + '_Xstd']
                X_j = ((X_j - Xavr_j) / Xstd_j).reshape((1, -1))

                # write dictionary
                rank_scores_i[model_j] = dict(_X=X_j.ravel().tolist(),)

                # use classifiers.
                for cl_name_k in self._model['classifiers']:
                    cl_k = self._model[model_j + '_' + cl_name_k]
                    if hasattr(cl_k, 'decision_function'):
                        score_k = cl_k.decision_function(X_j)
                    elif hasattr(cl_k, 'predict_log_proba'):
                        score_k = cl_k.predict_log_proba(X_j)
                    else: raise RuntimeError('!')
                    score_k = score_k.ravel()[-1] # take the last one.
                    rank_scores_i[model_j][cl_name_k] = dict(score=score_k)

            # write.
            xid_groups[i_group]['_rank_score'] = rank_scores_i
            xid_groups[i_group]['_rank_features'] = { \
                    k: (v if len(v) > 1 else v[0]) \
                    for k, v in group_features[i_group].items() \
                    if k[0] != '_'}

        # calculate rank in the field.
        for (model_i, classifier_j) in \
                itt.product(use_models, self._model['classifiers']):

            # collect scores first.
            scores_ik = list()
            for group_k in xid_groups:
                rank_scores_k = group_k['_rank_score']
                if (model_i in rank_scores_k) \
                        and (classifier_j in rank_scores_k[model_i]):
                    score_t = rank_scores_k[model_i][classifier_j]['score']
                else: score_t = np.nan
                scores_ik.append(score_t)
            scores_ik = np.array(scores_ik)

            # skip if nothing is here.
            is_valid_ik = np.isfinite(scores_ik)
            if not np.any(is_valid_ik): continue
            _ranks_ik = np.argsort(np.argsort(-scores_ik[is_valid_ik]))
            ranks_ik = np.zeros(scores_ik.size) + np.nan
            ranks_ik[is_valid_ik] = _ranks_ik

            # write back.
            for group_k, rank_k in zip(xid_groups, ranks_ik):
                rank_scores_k = group_k['_rank_score']
                if not np.isfinite(rank_k): continue
                rank_scores_k[model_i][classifier_j]['rank'] = int(rank_k)

        # choose default rank score.
        def_set = 'Basic8Z' if field_has_z else 'Basic8'
        def_cl = 'Logistic_v3'
        for group_i in xid_groups:
            if not (def_set in group_i['_rank_score']):
                group_i['_rank_score']['_default'] = dict( \
                        score=-1.e99, rank=int(1.e9),
                        features=None, ranker=None,)
                continue
            rank_scores_i = group_i['_rank_score'][def_set][def_cl]
            def_dict_i = dict(score=rank_scores_i['score'],
                              rank=rank_scores_i['rank'],
                              features=def_set,
                              ranker=def_cl)
            group_i['_rank_score']['_default'] = def_dict_i

        # done!

    def extract_ranking_input(self, xid_groups, z_ec, dpos_2d_ec,
            lazy=False, skip_invalid=True, return_group_id=False):

        ''' Extract the input variables of ranking functions. '''

        # do we have redshift?
        field_has_z = np.isfinite(z_ec)

        # optical magnitudes.
        group_mag = [_optical_mag(g)[2] for g in xid_groups]
        group_has_mag = np.isfinite(group_mag).tolist()
        field_has_mag = any(group_has_mag)

        # check coverage of LS and PS1
        group_has_PS1 = [_has_catalog(g, 'PS1v2') for g in xid_groups]
        group_has_LS  = [_has_catalog(g, 'LS8pz') for g in xid_groups]
        field_has_PS1, field_has_LS = any(group_has_PS1), any(group_has_LS)

        # determine which models to use.
        use_models = list()
        _suffix = 'Z' if field_has_z else ''
        use_models.append('Basic8' + _suffix)
        if field_has_LS:  use_models.append('LS' + _suffix)
        if field_has_PS1: use_models.append('PS1' + _suffix)
        if field_has_mag: use_models.append('AnyMag' + _suffix)
        # these "models" are called "feature sets" in the paper.

        # featurize groups
        group_features = _featurize_lazy(xid_groups, z_ec, dpos_2d_ec,
                self._model['cat_to_index']) if lazy else _featurize( \
                xid_groups, z_ec, dpos_2d_ec, self._model['cat_to_index'])
        # sn_z, sn_dxy, cat_to_index

        # read unique id of each group.
        group_uid = [w['_group_uid'] for w in xid_groups]

        # calculate inputs.
        rank_inputs = defaultdict(list)
        rank_input_group_uid = defaultdict(list)
        for i_group, grp_feat_i in enumerate(group_features):

            # calc rank scores.
            for model_j in use_models:

                # pack features.
                features_j = self._model[model_j + '_features']
                X_j = np.concatenate([grp_feat_i[t] for t in features_j])

                # do not skip groups with bad features.
                if np.any(~np.isfinite(X_j)) and skip_invalid: continue

                # scale to calculate input var.
                Xavr_j = self._model[model_j + '_Xavr']
                Xstd_j = self._model[model_j + '_Xstd']
                X_j = (X_j - Xavr_j) / Xstd_j
                rank_inputs[model_j].append(X_j)

                # also save uid of this group.
                rank_input_group_uid[model_j].append(group_uid[i_group])

        # stack into array.
        rank_inputs = {k: np.vstack(v) for k, v in rank_inputs.items()}

        # return rank scores and group id
        if return_group_id: return rank_inputs, rank_input_group_uid

        # or only return input variables.
        return rank_inputs

    def ranking_func(self, fset, cl_name):

        ''' Access ranking function directly. '''

        # get classifier.
        fc_key = fset + '_' + cl_name
        assert fc_key in self._model
        cl = self._model[fc_key]

        # get callable.
        if hasattr(cl, 'decision_function'): rfc = cl.decision_function
        elif hasattr(cl, 'predict_log_proba'): rfc = cl.predict_log_proba
        else: raise RuntimeError('!')

        return rfc

    def classifier_names(self,):
        ''' Names of classifiers. '''
        return self._model['classifiers']