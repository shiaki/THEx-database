#!/usr/bin/env python

'''
    Search NOAO sdss for host properties.
'''

import os
import sys
import json
import logging
import warnings
import datetime
import time
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

from THExQuery.sdss import sdss_query
from THExQuery.vac import best_available_coord
from THExQuery.utils import date_stamp

NO_VAC_INFO = 'Event {name} has no VAC info.'
VAC_OUTDATED = 'VAC info for event {name} is outdated.'
VAC_INCOMPLETE = 'VAC info for event {name} is incomplete.'

# per-process db connection
db_cli, sdss_queries = None, None
def search_sdss(task):

    # per-process database connection
    global db_cli, sdss_queries

    # unpack task
    event_i, vac_info_i, cats, is_xvalid_run = task

    if db_cli is None: # get DB conn

        if not is_xvalid_run: # Normal run
            db_cli = MongoClient(env_vars['dbpath'], connect=False)
            sdss_queries = db_cli.THEx.sdss_queries_extra

        else: # cross-validation
            db_cli = MongoClient(env_vars['dbpath'], connect=False)
            sdss_queries = db_cli.xvalid.sdss_queries_extra

    # dict for results.
    results = {
        'last_upate': date_stamp(),
        'vcc': event_i['vcc'],
    }

    # get best-available coordinates in VACs
    crd_i, crd_src_i, radius_i = best_available_coord( \
            event_i, vac_info_i, is_xvalid_run)

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
        print('Event {:}'.format(event_i['name'],))
        results[cat_i] = sdss_query(crd_i, radius_i, cat_i, is_xvalid_run)
        time.sleep(0.125)

    # update DB
    try:
        sdss_queries.update_one({'_id': event_i['_id']},
                            {'$set': results},
                            upsert=True)
    except Exception as err:
        print(err)

    return None

def task_to_process(events, vac_queries, sdss_queries, sdss_cats,
                    is_xvalid_run, force_update=[]):

    ''' Generates tasks for 'search_sdss' '''

    # d1 = datetime.datetime(2019, 11, 2, 12, 0, 0, 0,)

    # for the entire 'events' collection
    for event_i in events.find({}, no_cursor_timeout=True):

        # locate records in 'vac_queries' and 'sdss_queries'
        vac_rec_i = vac_queries.find_one({'_id': event_i['_id']})
        sdss_rec_i = sdss_queries.find_one({'_id': event_i['_id']})

        # short.
        '''
        if 'GRB' in ''.join(event_i['claimedtype']):
            print('Force update:', event_i['name'])
            yield event_i, vac_rec_i, sdss_cats, token
        '''

        # event not updated in VACs, warn and skip.
        if not vac_rec_i:
            warnings.warn(NO_VAC_INFO.format(name=event_i['name']))
            continue

        # VAC info is not up-to-date: warn and skip.
        if vac_rec_i['vcc'] != event_i['vcc']:
            warnings.warn(VAC_OUTDATED.format(name=event_i['name']))
            continue

        # has sdss info, up-to-date, complete:
        if sdss_rec_i and (sdss_rec_i['vcc'] == event_i['vcc']):
            is_cat_valid = [    (k in sdss_rec_i)             \
                            and (k not in force_update)       \
                            and sdss_rec_i[k]['is_complete']  \
                            for k, v in sdss_cats.items()]
            if all(is_cat_valid):
                continue # all complete, up-to-date, skip.

        # which catalog(s) to search in?
        if (not sdss_rec_i) or (sdss_rec_i['vcc'] != event_i['vcc']):
            cats_i = sdss_cats # not searched before or outdated,
            # all sdss catalogs

        else: # search failed catalogs again.
            cats_i = dict()
            for cat_i, cat_info_i in sdss_cats.items():
                if (cat_i not in sdss_rec_i) or (cat_i in force_update) \
                        or (not sdss_rec_i[cat_i]['is_complete']):
                    cats_i[cat_i] = cat_info_i

        # feed to the stream
        yield event_i, vac_rec_i, cats_i, is_xvalid_run

def group_process(block):
    print('Get block:', len(block))
    for task_i in block:
        if task_i:
            search_sdss(task_i)

def group_iter(it, n):
    pieces = [iter(it)] * n
    return itt.zip_longest(*pieces, fillvalue=None)

def dummy(x):
    return None

if __name__ == '__main__':

    # get directories from environ.
    env_var_names = ['THEX_DBPATH', 'THEX_WORKDIR', 'THEX_DATADIR',
            'THEX_DATALAB_USER', 'THEX_DATALAB_PASS', 'THEX_XVALID']
    for env_i in env_var_names:
        assert (env_i in os.environ), \
                'Missing environment variable: {:}'.format(env_i)
    env_vars = {'_'.join(e.split('_')[1:]).lower(): os.environ[e] \
            for e in env_var_names}

    # CHECK env variables.
    assert os.path.isdir(env_vars['datadir'])
    assert os.path.isdir(env_vars['workdir'])
    assert env_vars['xvalid'] in ['TRUE', 'FALSE'] # always strings.
    is_xvalid_run = (env_vars['xvalid'][0] == 'T')

    # read definition of catalogs.
    sdss_catalogs = {'SDSS16pz': None}

    parallel_mode = 1 # True
    parallel_mode_N_proc = 8

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
        sdss_queries = db_client.THEx.sdss_queries_extra

    else: # cross-validation run!
        events = db_client.xvalid.events # main events collection
        vac_queries = db_client.xvalid.vac_queries
        sdss_queries = db_client.xvalid.sdss_queries_extra

    if parallel_mode:
        print('Running in parallel mode.')
        pool.imap_unordered(group_process,
                group_iter(task_to_process(events, vac_queries, \
                        sdss_queries, sdss_catalogs, is_xvalid_run),
                        512), chunksize=16)
        pool.close()
        pool.join()

    else:
        for t_i in group_iter(task_to_process(events, vac_queries, \
                sdss_queries, sdss_catalogs, is_xvalid_run), 512):
            group_process(t_i)
            # break