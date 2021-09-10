#!/usr/bin/env python

'''
    Update list of supernovae
'''

import os
import sys
import json
import logging
import warnings
import datetime
import urllib.parse
from hashlib import md5
import itertools as itt

from pymongo import MongoClient

import numpy as np
from astropy.coordinates import SkyCoord

import shortuuid

from THExQuery.utils import date_stamp, best_coord, \
        is_valid_coord, unify_labels, rebuild_hierarchy, sort_by_ref

from collections import defaultdict

# warning messages.
PREV_REC_SKIPPED = \
        'Event {:} was skipped during update. (Moved to boneyard)'
INVALID_COORD = \
        'Failed to find the best coordinates for {:}'

# hash function for version control
_strhash = lambda s: md5(s.encode('utf-8')).hexdigest()
def read_supernovae(osc_dir):
    '''
    Iterator for JSON files in OSC.

    Parameters
    ----------
    osc_dir : str
        Directory of local OSC repositories.

    Returns
    -------
    sn_i : dict
        Supernova data decoded from original JSON document.

    sn_vcc_i : str
        MD5 hash of the original data for version check.
    '''
    for subdir, dirs, files in os.walk(osc_dir):
        if '.git' in subdir: # skip '.git' dir.
            continue
        for file_i in files:
            if '.json' != file_i.lower()[-5:]:
                continue # skip non-json files.
            with open(os.path.join(subdir, file_i), 'r') as fp:
                rawstr = fp.read()
            yield json.loads(rawstr,), _strhash(rawstr)

'''
def sort_by_ref(val_dict):
    ''' Sort values by their numbers of independent references. '''
    vals = [(w['value'], w['source'].split(',')) for w in val_dict]
    return [w[0] for w in sorted(vals, key=lambda x: -len(x[1]))]
'''
# obsolete

def save_snapshot(snapshots, rec):
    ''' Keep a snapshot of original record (in 'snap' array). '''
    snapshots.update_one({'_id': rec['_id']}, {'$addToSet': {'snap': rec}})
    snapshots.update_one({'_id': rec['_id']}, {'$set': {'last_update': date_stamp()}})

def into_boneyard(events, boneyard, rec):
    ''' Keep a snapshot of original record. '''
    boneyard.insert_one({**rec, **{'last_update': date_stamp()}})
    events.delete_one({'_id': rec['_id']})

if __name__ == '__main__':

    # get directories from environ.
    env_var_names = ['THEX_DBPATH', 'THEX_WORKDIR', 'THEX_DATADIR', 'THEX_XVALID']
    for env_i in env_var_names:
        assert (env_i in os.environ), \
                'Missing environment variable: {:}'.format(env_i)
    env_vars = {e.split('_')[1].lower(): os.environ[e] for e in env_var_names}

    # CHECK env variables.
    assert os.path.isdir(env_vars['datadir'])
    assert os.path.isdir(env_vars['workdir'])
    assert env_vars['xvalid'] in ['TRUE', 'FALSE'] # always strings.
    is_xvalid_run = (env_vars['xvalid'][0] == 'T')

    # read supernovae classification.
    with open(os.path.join(env_vars['workdir'], 'event-types.json'), 'r') as fp:
        type_synonyms = json.load(fp)

    with open(os.path.join(env_vars['workdir'], 'hierarchy.json'), 'r') as fp:
        type_hierarchy = json.load(fp)

    # read legacy supernova coordinates.
    with open('legacy-sn-coord.json', 'r') as f:
        legacy_sn_coord = json.load(f)

    # read names of bad reference sources.
    with open('bad-refsrc.json', 'r') as f:
        bad_refsrc = json.load(f)

    # initialize id generator
    suid = shortuuid.ShortUUID(alphabet='abcdefghijklmnopqrstuvwxyz')
    make_suid = lambda: suid.random(16)

    # initialize db connection, get data collections.
    db_client = MongoClient(env_vars['dbpath']) # use default now.

    if not is_xvalid_run: # a 'normal' run
        # raise RuntimeError('') # DEBUG
        events = db_client.THEx.events          # main events collection
        snapshots = db_client.THEx.snapshots    # historical version
        boneyard = db_client.THEx.boneyard      # deleted records.

    else: # cross-validation run.
        events = db_client.xvalid.events
        snapshots = db_client.xvalid.snapshots
        boneyard = db_client.xvalid.boneyard

    # TEMP
    sn_type_labels = defaultdict(list)
    xvalid_events = None

    # iterate over the list, identify new events.
    sn_data_dir = os.path.join(env_vars['datadir'], 'supernovae')
    for sn_i, sn_vcc_i in read_supernovae(sn_data_dir):

        # dataset has one SN per file.
        assert len(sn_i) == 1
        for name_i, info_i in sn_i.items():
            break
        print('---', name_i)

        # CHECK: for cross-validation, only keep those with host info!
        if is_xvalid_run:

            if xvalid_events is None: # read list of xvalid events.
                with open('xvalid-events.json', 'r') as f:
                    xvalid_events = json.load(f)

            # skip events not in this list.
            if name_i not in xvalid_events: continue

            '''
            if ('host' not in info_i) or (not info_i['host']): # no host name,
                if ('host_ra' not in info_i) or (not info_i['host_ra']): # no crd,
                    print(name_i, 'skipped.')
                    continue
            '''

        # read event name and aliases.
        alias_i = [name_i]
        if ('alias' in info_i) and info_i['alias']:
            alias_i = alias_i + [w['value'] for w in info_i['alias']]
        alias_i = list(set(alias_i))

        # CHECK: do we have these aliases in db?
        dcount_i = events.count_documents({"alias": {"$in": alias_i}})
        assert (dcount_i < 2), 'Duplicate name or alias detected.'

        # operation: new, update or remove?
        prev_rec_i = None
        if dcount_i == 1: # has previous record,
            prev_rec_i = events.find_one({"alias": {"$in": alias_i}})
            '''
            if ('sn' in prev_rec_i['vcc']) \
                    and (prev_rec_i['vcc']['sn'] == sn_vcc_i):
                continue # event info is unchanged
            if ('sn' not in prev_rec_i['vcc']):
                warnings.warn('!! Has previous record from other data source.')
            '''

        # extract reference sources.
        refs_i = {ref_k.pop('alias'): ref_k for ref_k in info_i['sources']}
        for k, v in refs_i.items():
            if 'url' in v: # trim url. (unnecessary?)
                _t = v['url'].split('/')
                v['url'] = [w for w in _t if ('http' not in w) and w][0]
            if 'name' not in v:
                if 'arxivid' in v: v['name'] = 'arXiv preprint'
                elif 'bibcode' in v: v['name'] = v['bibcode']
                elif 'doi' in v: v['name'] = v['doi']
                else:
                    print(v)
                    raise RuntimeError()

        # read type classification information.
        claimedtype_i = list()
        if ('claimedtype' in info_i) and info_i['claimedtype']:
            claimedtype_i = [w['value'] for w in info_i['claimedtype']]

        # read host information.
        host_i = list()
        if ('host' in info_i) and info_i['host']:
            host_i = sort_by_ref(info_i['host'],
                    refs=refs_i, exclude_names=bad_refsrc['name'])

        # if host_i: sys.exit()

        # read redshift (use the most referred value)
        zred_i = np.nan
        if ('redshift' in info_i) and info_i['redshift']:
            zred_i = float(sort_by_ref(info_i['redshift'])[0])

        # special case: exclude some types.
        if len(claimedtype_i) == 1:

            # exclude pure LGRBs (handled by another script later.)
            if ('LGRB' in claimedtype_i):
                if prev_rec_i: # has previous record: remove.
                    warnings.warn(PREV_REC_SKIPPED.format(name_i), )
                    into_boneyard(events, boneyard, prev_rec_i)
                continue

        # special case: exclude supernova remnants.
        if (len(claimedtype_i) == 0) and ('SNR' in '|'.join(alias_i)):
            if prev_rec_i: # has previous record: remove.
                warnings.warn(PREV_REC_SKIPPED.format(name_i), )
                into_boneyard(events, boneyard, prev_rec_i)
            continue

        # has no valid sky coord: skip.
        if ('ra' not in info_i) or ('dec' not in info_i):
            if prev_rec_i: # has previous record: remove.
                warnings.warn(PREV_REC_SKIPPED.format(name_i),)
                # into_boneyard(events, boneyard, prev_rec_i)
                # 200427: do not remove previous record if current is bad.
            continue

        # no redshift, no host name, no type -> skip.
        if not (claimedtype_i or host_i or np.isfinite(zred_i)):
            if prev_rec_i: # Case 1: has previous record: remove.
                warnings.warn(PREV_REC_SKIPPED.format(name_i), )
                into_boneyard(events, boneyard, prev_rec_i)
            continue

        # candidate w/o redshift and host.
        if not (host_i or np.isfinite(zred_i)):

            # no type label (* redundant), or the only type is 'Candidate'
            if ''.join(claimedtype_i) in ['Candidate', '']:
                if prev_rec_i: # Case 1: has previous record: remove.
                    warnings.warn(PREV_REC_SKIPPED.format(name_i), )
                    into_boneyard(events, boneyard, prev_rec_i)
                continue

        # find best coordinates in sexagesimal format (hhmmss +/- ddmmss)
        ra_i, dec_i, crdref_i = best_coord(
                info_i['ra'], info_i['dec'], refs=refs_i,)
        if not (ra_i and dec_i):
            warnings.warn(INVALID_COORD.format(name_i), )
            continue
        crd_i = SkyCoord(ra=ra_i, dec=dec_i, unit=('hour', 'deg'))

        # for legacy supernovae, check their positions with NED results.
        if (name_i in legacy_sn_coord) and legacy_sn_coord[name_i]:
            lcrd_i = SkyCoord(ra=legacy_sn_coord[name_i][2],
                              dec=legacy_sn_coord[name_i][3],
                              unit=('deg', 'deg'))

            # check separation.
            if lcrd_i.separation(crd_i).arcsec > 15:
                crd_i = lcrd_i
                ra_i, dec_i = lcrd_i.to_string('hmsdms', sep=':',).split(' ')
                crdref_i = [{'name': 'NED SN Coord'}]
                print('Using NED SN coord:', name_i, ra_i, dec_i)

        # also convert to decimal degrees
        ra_deg_i, dec_deg_i = crd_i.ra.deg, crd_i.dec.deg

        # get best host coordinates,
        host_ra_i, host_dec_i, host_crd_ref_i = '', '', []
        host_ra_deg_i, host_dec_deg_i = np.nan, np.nan
        if ('hostra' in info_i) and ('hostdec' in info_i):
            host_ra_i, host_dec_i, host_crd_ref_i = \
                    best_coord(info_i['hostra'], info_i['hostdec'],
                    refs=refs_i, exclude_names=bad_refsrc['crd'])
            if host_ra_i and host_dec_i: # has valid host coord
                host_crd_i = SkyCoord(ra=host_ra_i,
                                      dec=host_dec_i,
                                      unit=('hour', 'deg'))
                host_ra_deg_i, host_dec_deg_i = \
                        host_crd_i.ra.deg, host_crd_i.dec.deg
            else: # has host coord but cannot be parsed
                warnings.warn(INVALID_COORD.format(name_i), )

        # find host-event distance.
        host_dist_i = np.nan
        if ('hostoffsetang' in info_i) and info_i['hostoffsetang']:
            host_dist_i = np.nanmedian([float(rec_t['value']) \
                    for rec_t in info_i['hostoffsetang'] \
                    if ('derived' not in rec_t) or (not rec_t['derived'])])

        # FIX
        for t in claimedtype_i: sn_type_labels[t].append(name_i)

        # refine classification
        claimedtype_i = rebuild_hierarchy(unify_labels( \
                claimedtype_i, type_synonyms), type_hierarchy)

        # construct new record.
        event_i = dict(name=name_i, alias=alias_i,
                       ra=ra_i, dec=dec_i,
                       ra_deg=ra_deg_i, dec_deg=dec_deg_i,
                       crd_ref=crdref_i,
                       claimedtype=claimedtype_i, redshift=zred_i,
                       host=host_i,
                       host_ra=host_ra_i, host_dec=host_dec_i,
                       host_ra_deg=host_ra_deg_i,
                       host_dec_deg=host_dec_deg_i,
                       host_crd_ref=host_crd_ref_i,
                       host_dist=host_dist_i)

        if is_xvalid_run: # xvalid mode: remove host info.
            event_i['host'] = []
            event_i['host_ra'], event_i['host_dec'] = '', ''
            event_i['host_ra_deg'], event_i['host_dec_deg'] = np.nan, np.nan
            event_i['host_crd_ref'] = []
            event_i['host_dist'] = np.nan

        # Case I: no previous record, generate id and insert.
        if not prev_rec_i:
            new_rec_i = {**{'_id': make_suid()}, **event_i, \
                    **{'last_update': date_stamp(), 'vcc': {'sn': sn_vcc_i}}}
            events.insert_one(new_rec_i)
            continue

        # Has previous record:

        # compare record with existing results:
        '''
        is_identical = True
        for ki, vi in event_i.items():
            prev_ki = prev_rec_i[ki]
            if isinstance(vi, str):
                is_identical &= vi == prev_ki
            elif isinstance(vi, np.float):
                is_identical &= np.allclose(vi, prev_ki, equal_nan=True)
            elif isinstance(vi, list):
                is_identical &= sorted(vi) == sorted(prev_ki)
            elif isinstance(vi, dict):
                is_identical &= vi == prev_ki
            else:
                raise RuntimeError('Unrecognized data type.')
            if not is_identical:
                break
        '''
        prev_rec_t = {k: prev_rec_i[k] for k, v in event_i.items()}
        is_identical = json.dumps(event_i, sort_keys=True) \
                    == json.dumps(prev_rec_t, sort_keys=True)

        # Case IIa: meta info unchanged: update vcc and date only.
        if is_identical:
            update_t = {'last_update': date_stamp(), 'vcc': {'sn': sn_vcc_i}}
            events.update_one({'_id': prev_rec_i['_id']},
                              {'$set': update_t},
                              upsert=True)
            print('Existing meta data unchanged:', name_i)
            continue

        # Case IIb: info changed, update current version, save historical.
        # save_snapshot(snapshots, prev_rec_i)
        stamp_t = {'last_update': date_stamp(), 'vcc': {'sn': sn_vcc_i}}
        events.update_one({'_id': prev_rec_i['_id']},
                          {'$set': {**event_i, **stamp_t}},
                          upsert=True)
        print('Existing meta data updated:', name_i)

    # DEBUG
    if not is_xvalid_run:
        with open('sn-type-labels.json', 'w') as f:
            json.dump(sn_type_labels, f, indent=4)