#!/usr/bin/env python

'''
    Search host information on Vizier.
'''

import os
import sys
import json
import logging
import warnings
import datetime
import re
import argparse
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
import astropy.units as u

from astroquery.vizier import Vizier
from astroquery.exceptions import RemoteServiceError

from THExQuery.vac import best_available_coord
from THExQuery.utils import date_stamp
from THExQuery.vizier import vizier_query

NO_VAC_INFO = 'Event {name} has no VAC info.'
VAC_OUTDATED = 'VAC info for event {name} is outdated.'
VAC_INCOMPLETE = 'VAC info for event {name} is incomplete.'

# per-process db connection
db_cli, viz_queries = None, None
def search_vizier(task_pac):

    # global vars.
    global db_cli, viz_queries

    # unpack.
    event_i, vac_info_i, cats, is_xvalid_run = task_pac
    # print('Event:', event_i['name'])

    # connect to db
    if db_cli is None:

        if not is_xvalid_run: # normal
            db_cli = MongoClient(env_vars['dbpath'], connect=False)
            viz_queries = db_cli.THEx.vizier_queries_extra
            print('Got database conn.')

        else: # cross-validation
            db_cli = MongoClient(env_vars['dbpath'], connect=False)
            viz_queries = db_cli.xvalid.vizier_queries_extra
            print('Got database conn.')

    # dict for results.
    results = {
        'last_upate': date_stamp(),
        'vcc': event_i['vcc'],
    }

    # Removed: use routines in THExQuery.vacs instead. YJ 191027

    # get best-available coordinates in VACs
    crd_i, crd_src_i, radius_i = \
            best_available_coord(event_i, vac_info_i, is_xvalid_run)

    results['coord_desc'] = crd_src_i
    crd_repr = 'coord:{:.6f},{:.6f}/radius:{:.3f}'.format(
                crd_i.ra.deg, crd_i.dec.deg, radius_i)
    results['queried_with'] = crd_repr

    # search vizier catalogs in turn and save results.
    for cat_i, cat_info_i in cats.items():
        print('Event {} in catalog {}...'.format(event_i['name'], cat_i))
        extcols_i = cat_info_i['extra_cols'] \
                    if 'extra_cols' in cat_info_i \
                    else None
        results[cat_i] = vizier_query(crd_i, radius_i,
                                      cat_info_i['vizier_id'],
                                      extra_cols=extcols_i)

    # return results

    # update.
    viz_queries.update_one({'_id': event_i['_id']},
                           {'$set': results},
                           upsert=True)

    return None

# task scheduler
def records_to_process(events, vac_queries,
        vizier_queries, vizier_cats, force_update, is_xvalid_run):

    ''' Iterator to feed newly updated events to individual workers. '''

    # iterate over entire collection
    for event_i in events.find({}):

        # DEBUG
        if event_i['_id'] not in failed_event_uid: continue

        # locate records in 'vac_queries' and 'vizier_queries'
        vac_rec_i = vac_queries.find_one({'_id': event_i['_id']})
        viz_rec_i = vizier_queries.find_one({'_id': event_i['_id']})

        # force update GRB
        '''
        if 'GRB' in ''.join(event_i['claimedtype']):
            print('Force update:', event_i['name'])
            yield event_i, vac_rec_i, vizier_cats
        '''

        # not available in VACs: warn and skip.
        if not vac_rec_i:
            warnings.warn(NO_VAC_INFO.format(name=event_i['name']))
            continue

        # VAC info outdated: warn and skip.
        if vac_rec_i['vcc'] != event_i['vcc']:
            warnings.warn(VAC_OUTDATED.format(name=event_i['name']))
            continue

        # has vizier info, up-to-date, complete:
        if viz_rec_i and (viz_rec_i['vcc'] == event_i['vcc']):
            is_cat_valid = [(k in viz_rec_i) \
                            and viz_rec_i[k]['is_complete'] \
                            and (k not in force_update) \
                            for k, v in vizier_cats.items()]
            if all(is_cat_valid):
                continue # all complete, up-to-date, skip.

        # which catalogs to search?
        if viz_rec_i and (viz_rec_i['vcc'] == event_i['vcc']):
            cats_i = dict()
            for cat_i, cat_info_i in vizier_cats.items():
                if (cat_i not in viz_rec_i) or (cat_i in force_update):
                    cats_i[cat_i] = cat_info_i
                    continue
                if not viz_rec_i[cat_i]['is_complete']:
                    cats_i[cat_i] = cat_info_i

        else: # otherwise: search entire catalog.
            cats_i = vizier_cats

        # print(event_i['name'])

        # otherwise: feed to the stream
        yield event_i, vac_rec_i, cats_i, is_xvalid_run

def group_process(block):
    print('Get block:', len(block))
    for task_i in block:
        if task_i:
            search_vizier(task_i)

def group_iter(it, n):
    pieces = [iter(it)] * n
    return itt.zip_longest(*pieces, fillvalue=None)

def dummy(x):
    return None

if __name__ == '__main__':

    # DEBUG SOLUTION
    with open('failed-events-uid.json', 'r') as f:
        failed_event_uid = json.load(f)

    # read arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', nargs='+', help='Force update catalogs.')
    args = parser.parse_args()

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

    # always do this: import list of Vizier catalogs.
    vizcatdef_file = os.path.join(env_vars['workdir'], 'VizierCatalogs.json') \
                     if (not is_xvalid_run) else \
                     os.path.join('./', 'VizierCatalogs-short.json')
    with open(vizcatdef_file, 'r') as f:
        vizier_catalogs = json.load(f)

    # which catalogs must be updated?
    force_update = args.force if args.force else list()
    force_update = ['GALEXMIS5']
    bad_catalogs = [w for w in force_update if (w not in vizier_catalogs)]
    if any(bad_catalogs):
        raise KeyError('Catalog(s) not available: ' + ', '.join(bad_catalogs))

    parallel_mode = 1 # True
    parallel_mode_N_proc = 9

    # use a pool of workers.
    pool = None
    if parallel_mode:
        pool = Pool(parallel_mode_N_proc)
        pool.imap_unordered(dummy, range(64)) # force fork
        # ^ Not necessary. YJ, 080919

    # initialize db connection.
    db_client = MongoClient(env_vars['dbpath'], connect=False) # use default now.

    if not is_xvalid_run: # a 'normal' run
        events = db_client.THEx.events # main events collection
        vac_queries = db_client.THEx.vac_queries
        vizier_queries = db_client.THEx.vizier_queries_extra

    else:
        events = db_client.xvalid.events # main events collection
        vac_queries = db_client.xvalid.vac_queries
        vizier_queries = db_client.xvalid.vizier_queries_extra

    if parallel_mode:
        print('Running in parallel mode.')
        pool.imap_unordered(group_process,
                group_iter(records_to_process(events, vac_queries, vizier_queries,\
                        vizier_catalogs, force_update, is_xvalid_run),
                        64), chunksize=16)
        pool.close()
        pool.join()

    else:
        for t_i in group_iter(records_to_process(events, vac_queries, \
                vizier_queries, vizier_catalogs, force_update, is_xvalid_run), 64):
            group_process(t_i)
            break

    with open('done.tmp', 'w') as f:
        f.write('done')