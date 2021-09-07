
import re
import warnings
import datetime
from collections import defaultdict

import numpy as np
from numpy.ma.core import MaskedConstant

from astropy.coordinates import SkyCoord

# useful functions
date_stamp = lambda: datetime.datetime.now()

# remove redundant spaces.
remove_space = lambda s: re.sub(' +', ' ', s)

#
def _best_coord(ra_vals, dec_vals, refs={}, exclude_names=[]):

    ''' Find the most-referred pair of sky coordinates. '''

    # sort out RA/Dec about this event.
    crd_i, crdref_i = dict(), list()
    for ra_rec_i in ra_vals:
        ra_val_i, ra_src_i = ra_rec_i['value'], ra_rec_i['source']
        for src_i in ra_src_i.split(','):
            crd_i[src_i] = [None, None]
            crd_i[src_i][0] = ra_val_i
            crdref_i.append(src_i)
    for dec_rec_i in dec_vals:
        dec_val_i, dec_src_i = dec_rec_i['value'], dec_rec_i['source']
        for src_i in dec_src_i.split(','):
            if src_i not in crd_i:
                crd_i[src_i] = [None, None]
            crd_i[src_i][1] = dec_val_i
            crdref_i.append(src_i)

    # which ref srcs to exclude?
    ref_id_bad = [k for k, v in refs.items() if v['name'] in exclude_names]
    # print('Ref srcs to remove:', ref_id_bad)

    # warn unpaired ra/dec values.
    for src_i, (ra_val_i, dec_val_i) in crd_i.items():
        if not (ra_val_i and dec_val_i):
            warnings.warn('Unpaired RA/Dec value detected.', )

    # which is the most used value?
    for src_i in sorted(set(crdref_i), key=crdref_i.count)[::-1]:
        if src_i in ref_id_bad:
            # print('Source', src_i, 'removed.', refs[src_i])
            continue # ignore bad ref srcs.
        if crd_i[src_i][0] and crd_i[src_i][1]:
            if is_valid_coord(crd_i[src_i][0], crd_i[src_i][1]):
                return crd_i[src_i]

    return '', ''

def best_coord(ra_vals, dec_vals, refs={}, exclude_names=[]):

    ''' Find the most-referred pair of sky coordinates. '''

    # get RA/Dec pairs of events.
    crd_i = dict()
    for ra_rec_i in ra_vals:
        ra_val_i, ra_src_i = ra_rec_i['value'], ra_rec_i['source']
        for src_i in ra_src_i.split(','):
            crd_i[src_i] = [None, None]
            crd_i[src_i][0] = ra_val_i
    for dec_rec_i in dec_vals:
        dec_val_i, dec_src_i = dec_rec_i['value'], dec_rec_i['source']
        for src_i in dec_src_i.split(','):
            if src_i not in crd_i:
                crd_i[src_i] = [None, None]
            crd_i[src_i][1] = dec_val_i

    # remove pairs from bad reference sources.
    ref_id_bad = [k for k, v in refs.items() if v['name'] in exclude_names]
    # print('Ref srcs to remove:', ref_id_bad)
    for k_t in ref_id_bad:
        if k_t in crd_i: crd_i.pop(k_t)
        # print('Source', k_t, 'removed.', refs[k_t])

    # remove bad coordinates.
    crd_bad = [k for k, v in crd_i.items() if not is_valid_coord(v[0], v[1])]
    # print(crd_i)
    for crd_t in crd_bad:
        # print('Bad coord removed:', crd_t, crd_i[crd_t])
        crd_i.pop(crd_t)

    if not crd_i: return '', '', []

    # now count keys of unique values
    crd_refs_i = defaultdict(list)
    for k, v in crd_i.items(): crd_refs_i[tuple(v)].append(k)
    crd_refs_i = list(crd_refs_i.items())
    # if None and crd_refs_i.__len__() > 2:
    #     print(crd_refs_i)
    #     print(refs)
    #     print(exclude_names)
    #     print(crd_i)
    #     sys.exit()

    # the most referred?
    id_best = np.argmax([len(w[1]) for w in crd_refs_i])
    ra_best, dec_best = crd_refs_i[id_best][0]
    # print(refs)
    ref_best = [refs[t] for t in crd_refs_i[id_best][1]]
    return ra_best, dec_best, ref_best

#
def sort_by_ref(val_dict, refs={}, exclude_names=[]):

    ''' Sort values by their numbers of independent references. '''

    # which ref srcs to exclude?
    ref_id_bad = [k for k, v in refs.items() if v['name'] in exclude_names]
    # print('Ref srcs to remove:', ref_id_bad)

    # list values and reference sources, exclude bad ones.
    vals = [(w['value'], w['source'].split(',')) for w in val_dict]
    for id_i in ref_id_bad:
        for k, v in vals:
            # if (id_i in v): print('Going to remove:', id_i, refs[id_i])
            v.remove(id_i) if (id_i in v) else None
    vals = [(k, v) for k, v in vals if v]

    # find the most referred one.
    return [w[0] for w in sorted(vals, key=lambda x: -len(x[1]))]

def _fancy_and_pythonic_float_converter(s):
    try: return float(s) # I just need a 'float-or-NaN' function
    except: return np.nan

def is_valid_decimal_coord(ra, dec):
    ra_f = _fancy_and_pythonic_float_converter(ra)
    dec_f = _fancy_and_pythonic_float_converter(dec)
    return np.isfinite(ra_f) and np.isfinite(dec_f) \
            and (0 <= ra_f <= 360.) and (-90. <= dec_f <= 90.)

#
def _is_valid_coord(ra, dec):
    ''' Test if sky coord is valid. '''
    ra, dec = ra.split(':'), dec.split(':')
    cond_0 = (len(ra) == 3) and (len(dec) == 3)
    def f(s): # fancy and pythonic float number converter
        try: return float(s)
        except: return np.nan
    cond = [
        cond_0 and  ra[0].isnumeric(),
        cond_0 and  ra[1].isnumeric(),
        cond_0 and  ra[2].replace('.', '').isnumeric(),
        cond_0 and dec[0].replace('+', '').replace('-', '').isnumeric(),
        cond_0 and dec[1].isnumeric(),
        cond_0 and dec[2].replace('.', '').isnumeric(),
        cond_0 and (0. <=     f( ra[0])  <  24.),
        cond_0 and (0. <=     f( ra[1])  <= 60.),
        cond_0 and (0. <=     f( ra[2])  <= 60.),
        cond_0 and (      abs(f(dec[0])) <= 90.),
        cond_0 and (0. <=     f(dec[1])  <= 60.),
        cond_0 and (0. <=     f(dec[2])  <= 60.),
    ]
    return all(cond)

def is_valid_coord(ra, dec):
    ''' Test if sky coord is valid. '''
    try: crd_t = SkyCoord(ra=ra, dec=dec, unit=('hour', 'deg'))
    except: crd_t = None
    return not (crd_t is None)

def is_valid_sexagesimal_coord(ra, dec):
    return is_valid_coord(ra, dec)

def row_to_list(w):
    rv = list(w)
    for iv, vi in enumerate(rv):
        if isinstance(vi, MaskedConstant): # set NA values as None
            rv[iv] = None
            continue
        if isinstance(vi, bytes): # convert byte str to str.
            rv[iv] = remove_space(vi.decode())
            continue
        if isinstance(vi, (np.int64, np.int32, \
                np.int16, np.uint8, np.int)): # to int
            rv[iv] = vi.item()
            continue
        if isinstance(vi, (np.float64, np.float32, \
                np.cfloat, np.float)): # to float
            rv[iv] = vi.item()
            continue
        if isinstance(vi, np.bool_): # to normal float
            rv[iv] = bool(vi)
            continue
        if isinstance(vi, np.str_): # to normal string
            rv[iv] = str(vi)
    return rv

#
def unify_labels(claimedtype, type_synonyms):
    ''' unify classification using synonyms. '''
    new_types = list()
    for tp_i in claimedtype:
        # if '_' == tp_i[0]: continue
        is_classified = False
        for tp_name, tp_tyn in type_synonyms.items():
            if '__' in tp_name:
                continue
            if (tp_i == tp_name) or (tp_i in tp_tyn):
                new_types.append(tp_name)
                is_classified = True
        # assert is_classified, 'Cannot classify {:}'.format(tp_i)
        if not is_classified:
            print('Cannot classify {:}'.format(tp_i))
            # raise RuntimeError('')
    return list(set(new_types))

def rebuild_hierarchy(claimedtype, type_hierarchy):
    ''' Construct classification tree using definition. '''
    new_types, label_added = list(claimedtype), True
    while label_added:
        label_added = False
        for tp_i, tp_sub_i in type_hierarchy.items():
            if '__' in tp_i:
                continue # Comment or special case.
            for tp_j in tp_sub_i:
                if (tp_j in new_types) and (tp_i not in new_types):
                    new_types.append(tp_i)
                    label_added = True
    is_conflict = False
    for tp_u, tp_v in type_hierarchy["__CONFLICT_CASES"]:
        if (tp_u in new_types) and (tp_v in new_types):
            cflt_t = '_CONFLICT_{:}_{:}'.format(tp_u, tp_v)
            new_types.append( \
                    cflt_t.replace(' ', '_').replace('__', '_'))
            is_conflict = True
    if is_conflict:
        new_types.append('_CONFLICT')
    return sorted(new_types)
