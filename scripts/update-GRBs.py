#!/usr/bin/env python

'''
    Update GRBs
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

# useful functions
date_stamp = lambda: datetime.datetime.now()

# hash function for version control
_strhash = lambda s: md5(s.encode('utf-8')).hexdigest()

def read_GRBs(grb_dir):
    ''' Iterator for local GRB events. '''
    for subdir, dirs, files in os.walk(grb_dir):
        for file_i in files:
            if '.json' not in file_i.lower():
                continue # skip non-json files.
            with open(os.path.join(subdir, file_i), 'r') as fp:
                rawstr = fp.read()
            yield json.loads(rawstr,), _strhash(rawstr)

#
def save_snapshot(snapshots, rec):
    ''' Keep a snapshot of original record (in 'snap' array). '''
    snapshots.update_one({'_id': rec['_id']}, {'$addToSet': {'snap': rec}})
    snapshots.update_one({'_id': rec['_id']}, {'$set': {'last_update': date_stamp()}})

def into_boneyard(events, boneyard, rec):
    ''' Keep a snapshot of original record. '''
    boneyard.insert_one({**rec, **{'last_update': date_stamp()}})
    events.delete_one({'_id': rec['_id']})

#
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

    # initialize id generator
    suid = shortuuid.ShortUUID(alphabet='abcdefghijklmnopqrstuvwxyz')
    make_suid = lambda: suid.random(16)

    # initialize db connection.
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

    grb_data_dir = os.path.join(env_vars['datadir'], 'grb-local-new')
    for grb_i, grb_vcc_i in read_GRBs(grb_data_dir):

        # dataset has one event per file.
        assert (len(grb_i) == 1)
        for name_i, info_i in grb_i.items():
            break

        # no previous record: insert as new record, IF error circle < 60"
        if info_i['radec_err'] > 5.:
            # print('Event skipped for low position accuracy:', info_i['name'])
            continue

        # CHECK: for cross-validation, only keep those with host info!
        if is_xvalid_run:
            if ('host' not in info_i) or (not info_i['host']): # no host name,
                if ('hostra' not in info_i) or (not info_i['hostra']): # no crd,
                    print(name_i, 'skipped.')
                    continue

        # read event name and aliases.
        alias_i = [name_i]
        if ('alias' in info_i) and info_i['alias']:
            alias_i = alias_i + info_i['alias']
        alias_i = list(set(alias_i))

        # check database: do we have these aliases in db?
        dcount_i = events.count_documents({"alias": {"$in": alias_i}})
        assert (dcount_i < 2), 'Multiple name or alias matched!'

        if dcount_i == 1: # has unique previous record,
            prev_rec_i = events.find_one({"alias": {"$in": alias_i}})
            if ('grb' in prev_rec_i['vcc']) \
                    and prev_rec_i['vcc']['grb'] == grb_vcc_i:
                # has the latest record: do not update.
                continue

        # has no previous record in the master table -> new event!
        else: prev_rec_i = None

        # found previous record: update with GRB info and continue.
        if prev_rec_i:

            if len(prev_rec_i['alias']) > 1: # show other names
                print('Previous event: ', prev_rec_i['alias'])

            uid_prev_i = prev_rec_i.pop('_id')
            prev_rec_i['alias'].append(name_i)
            prev_rec_i['alias'] = list(set(prev_rec_i['alias']))

            # better r90 and pure GRB --> replace coord.
            if ('radec_err' not in prev_rec_i) or \
                    info_i['radec_err'] < prev_rec_i['radec_err']:
                if prev_rec_i['claimedtype'] and all([('GRB' in w) \
                        for w in prev_rec_i['claimedtype'] if w[0] != '_']):
                    for key_t in ['ra', 'dec', 'ra_deg', 'dec_deg',]:
                        prev_rec_i[key_t] = info_i[key_t]
                    prev_rec_i['radec_err'] = info_i['radec_err']
                    print('*** RA/Dec updated.')
                else: print('*** Better GRB radec but not used: ' + \
                        '{:}'.format(prev_rec_i['name']))

            # update type.
            prev_rec_i['claimedtype'] = list(set( \
                    prev_rec_i['claimedtype'] + info_i['claimedtype']))

            # has **new** redshift: replace original.
            if np.isfinite(info_i['redshift']) and \
                    (not np.isfinite(prev_rec_i['redshift'])):
                prev_rec_i['redshift'] = info_i['redshift']

            prev_rec_i['vcc']['grb'] = grb_vcc_i
            prev_rec_i['last_update'] = date_stamp()

            events.replace_one({'_id': uid_prev_i}, prev_rec_i)
            print('Updated event:', info_i['name'])
            continue

        info_i['_id'] = make_suid()
        info_i['last_update'] = date_stamp()
        info_i['vcc'] = dict(grb=grb_vcc_i)
        events.insert_one(info_i)
        print('New event:', info_i['name'])
