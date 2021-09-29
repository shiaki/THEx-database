#!/usr/bin/env python

'''
    Search MAST for host properties.
'''

import os
import sys
import json
import logging
import warnings
import datetime
import re
import urllib.parse
from hashlib import md5
import itertools as itt
from multiprocessing import Pool
from pprint import pprint as pp

import shortuuid
from pymongo import MongoClient

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.cosmology import WMAP9

from THExQuery.mastps1 import mastps1_query
from THExQuery.vac import best_available_coord
from THExQuery.utils import date_stamp

NO_VAC_INFO = 'Event {name} has no VAC info.'
VAC_OUTDATED = 'VAC info for event {name} is outdated.'
VAC_INCOMPLETE = 'VAC info for event {name} is incomplete.'

# per-process db connection
db_cli, mast_queries = None, None
def search_mastps1(task):

    # per-process database connection
    global db_cli, mast_queries

    # unpack task
    event_i, vac_info_i, cats, is_xvalid_run = task

    if db_cli is None: # connect to db

        if not is_xvalid_run: # normal
            db_cli = MongoClient(env_vars['dbpath'], connect=False)
            mast_queries = db_cli.THEx.mast_queries_extra

        else: # cross-validation
            db_cli = MongoClient(env_vars['dbpath'], connect=False)
            mast_queries = db_cli.xvalid.mast_queries_extra

    # dict for results.
    results = {
        'last_upate': date_stamp(),
        'vcc': event_i['vcc'],
    }

    # get best-available coordinates in VACs
    crd_i, crd_src_i, radius_i = \
            best_available_coord(event_i, vac_info_i, is_xvalid_run)

    results['coord_desc'] = crd_src_i
    crd_repr = 'coord:{:.6f},{:.6f}/radius:{:.3f}'.format(
                crd_i.ra.deg, crd_i.dec.deg, radius_i)
    results['queried_with'] = crd_repr

    '''
    crd_repr = 'coord:{:},{:}'.format(crd_i.ra.deg, crd_i.dec.deg)
    results['queried_with'] = crd_repr
    results['search_radius'] = radius_i
    results['coord_src'] = crd_src_i
    '''

    # query catalogs,
    for cat_i, cat_info_i in cats.items():
        print('Event {:} in catalog {:}'.format( \
                event_i['name'], cat_info_i['table_name']))
        results[cat_i] = mastps1_query(crd_i, radius_i, cat_info_i,)
        break # only one catalog in MAST.

    # update DB
    mast_queries.update_one({'_id': event_i['_id']},
                            {'$set': results},
                            upsert=True)

    return None

def task_to_process(events, vac_queries, mast_queries_extra,
                    mast_cats, is_xvalid_run):

    ''' Generates tasks for 'search_mastps1' '''

    # for the entire 'events' collection
    for event_i in events.find({}):

        # locate records in 'vac_queries' and 'mast_queries_extra'
        vac_rec_i = vac_queries.find_one({'_id': event_i['_id']})
        mastps1_rec = mast_queries_extra.find_one({'_id': event_i['_id']})

        '''
        if 'GRB' in ''.join(event_i['claimedtype']):
            print('Force update:', event_i['name'])
            yield event_i, vac_rec_i, mast_cats
        '''

        # if np.isfinite(event_i['host_ra_deg']) and (not event_i['host']):
        #     yield event_i, vac_rec_i, mast_cats, is_xvalid_run
        # else: continue

        # event not updated in VACs, warn and skip.
        if not vac_rec_i:
            warnings.warn(NO_VAC_INFO.format(name=event_i['name']))
            continue

        # VAC info is not up-to-date: warn and skip.
        if vac_rec_i['vcc'] != event_i['vcc']:
            warnings.warn(VAC_OUTDATED.format(name=event_i['name']))
            continue

        # has PS1 DR2 record, up-to-date, complete:
        if mastps1_rec and (mastps1_rec['vcc'] == event_i['vcc']):
            is_cat_valid = [(k in mastps1_rec) \
                            and mastps1_rec[k]['is_complete'] \
                            for k, v in mast_cats.items()]
            if all(is_cat_valid):
                continue # all complete, up-to-date, skip.

        # which catalog(s) to search in?
        if (not mastps1_rec) or (mastps1_rec['vcc'] != event_i['vcc']):
            cats_i = mast_cats # not searched before or outdated

        else: # search failed catalogs again.
            cats_i = dict()
            for cat_i, cat_info_i in mast_cats.items():
                if cat_i not in mastps1_rec:
                    cats_i[cat_i] = cat_info_i
                    continue # not searched before: search now.
                if not mastps1_rec[cat_i]['is_complete']:
                    cats_i[cat_i] = cat_info_i
                    # previous search incomplete: search now.

        # just leave it there for future catalogs based on this API
        # YJ, 080919

        # feed to the stream
        yield event_i, vac_rec_i, cats_i, is_xvalid_run

def group_process(block):
    print('Get block:', len(block))
    for task_i in block:
        if task_i:
            search_mastps1(task_i)

def group_iter(it, n):
    pieces = [iter(it)] * n
    return itt.zip_longest(*pieces, fillvalue=None)

def dummy(x):
    return None

if __name__ == '__main__':

    # get directories from environ.
    env_var_names = ['THEX_DBPATH', 'THEX_WORKDIR', 'THEX_DATADIR',
                     'THEX_XVALID', 'THEX_MOCKSN']
    for env_i in env_var_names:
        assert (env_i in os.environ), \
                'Missing environment variable: {:}'.format(env_i)
    env_vars = {e.split('_')[1].lower(): os.environ[e] for e in env_var_names}

    # CHECK env variables.
    assert os.path.isdir(env_vars['datadir'])
    assert os.path.isdir(env_vars['workdir'])
    assert env_vars['xvalid'] in ['TRUE', 'FALSE']
    assert env_vars['mocksn'] in ['TRUE', 'FALSE']
    is_xvalid_run = (env_vars['xvalid'][0] == 'T')
    is_mocksn_run = (env_vars['mocksn'][0] == 'T')

    # read definition of Datalab catalogs.
    if not is_xvalid_run: # normal run, use full col def.
        mastcatdef_file = os.path.join(env_vars['workdir'], 'MAST-PS1.json')
    else: # xvalid run, use subset of cols
        mastcatdef_file = os.path.join('./', 'MAST-PS1-short.json')

    with open(mastcatdef_file, 'r') as f:
        mastps1_catalogs = json.load(f)

    parallel_mode = 1 # True
    parallel_mode_N_proc = 16

    # use a pool of workers.
    pool = None
    if parallel_mode:
        pool = Pool(parallel_mode_N_proc)
        # pool.imap_unordered(dummy, range(512)) # force fork
        # ^ Not necessary. YJ, 080919

    # initialize db connection.
    db_client = MongoClient(env_vars['dbpath'], connect=False) # use default now.

    if not is_xvalid_run: # a 'normal' run
        events = db_client.THEx.events # main events collection
        vac_queries = db_client.THEx.vac_queries
        mast_queries_extra = db_client.THEx.mast_queries_extra

    else: # xvalid run
        events = db_client.xvalid.events # main events collection
        vac_queries = db_client.xvalid.vac_queries
        mast_queries_extra = db_client.xvalid.mast_queries_extra

    if parallel_mode:
        print('Running in parallel mode.')
        pool.imap_unordered(group_process,
                group_iter(task_to_process(events, vac_queries, \
                        mast_queries_extra, mastps1_catalogs, is_xvalid_run),
                        512), chunksize=16)
        pool.close()
        pool.join()

    else:
        for t_i in group_iter(task_to_process(
                    events, vac_queries, mast_queries_extra,
                    mastps1_catalogs, is_xvalid_run), 512):
            group_process(t_i)
            break
