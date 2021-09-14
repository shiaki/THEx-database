#!/usr/bin/env python

'''
    Script to search local catalogs.
'''

import os
import json
import sys
import pickle
from collections import namedtuple
from pprint import pprint as pp

from tqdm import tqdm

from pymongo import MongoClient

import tables as tb
from astropy.io import fits

from THExQuery.vac import best_available_coord
from THExQuery.localcat import *
from THExQuery.utils import *

NO_VAC_INFO = 'Event {name} has no VAC info.'
VAC_OUTDATED = 'VAC info for event {name} is outdated.'
VAC_INCOMPLETE = 'VAC info for event {name} is incomplete.'

# single task.
Task = namedtuple('Task', ['doc_id', 'catalogs',
                           'crd', 'radius', 'crd_src', 'vcc'])

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

    # read local catalog definition file.
    localcat_file = os.path.join(env_vars['workdir'], 'local-catalogs.json')
    with open(localcat_file, 'r') as f:
        local_catalogs = json.load(f)

    # initialize db connection.
    db_client = MongoClient(env_vars['dbpath']) # use default now.

    if not is_xvalid_run: # a 'normal' run
        # raise RuntimeError('')
        events = db_client.THEx.events
        vac_queries = db_client.THEx.vac_queries
        local_queries = db_client.THEx.local_queries

    else: # cross-validation run.
        events = db_client.xvalid.events
        vac_queries = db_client.xvalid.vac_queries
        local_queries = db_client.xvalid.local_queries

    # mark events to update.
    is_update_complete = True
    events_to_update, event_queue_size = list(), 1000000
    for event_i in events.find({}):

        # locate VAC records and previous local catalog results.
        vac_rec_i = vac_queries.find_one({'_id': event_i['_id']})
        loc_rec_i = local_queries.find_one({'_id': event_i['_id']})

        # VAC info not available: warn and skip.
        if not vac_rec_i:
            warnings.warn(NO_VAC_INFO.format(name=event_i['name']))
            continue

        # VAC info outdated (event table updated): warn and skip.
        if vac_rec_i['vcc'] != event_i['vcc']:
            warnings.warn(VAC_OUTDATED.format(name=event_i['name']))
            continue

        # has complete, up-to-date local catalog info: skip.
        if loc_rec_i and (loc_rec_i['vcc'] == event_i['vcc']):
            is_valid_t = [(k in loc_rec_i) for k, v in local_catalogs.items()]
            if all(is_valid_t):
                continue

        # determine which catalogs to search.
        if loc_rec_i and (loc_rec_i['vcc'] == event_i['vcc']):
            cats_i = [w for w in local_catalogs.keys() if w not in loc_rec_i]

        else: # search catalogs that are not
            cats_i = list(local_catalogs.keys())
            # no previous record: search all local catalogs.

        # get best available coordinates from original records and VACs
        crd_i, crd_src_i, radius_i = best_available_coord(event_i, vac_rec_i)

        # put into the queue
        events_to_update.append(Task(doc_id=event_i['_id'],
                                     catalogs=cats_i,
                                     crd=crd_i,
                                     radius=radius_i,
                                     crd_src=crd_src_i,
                                     vcc=event_i['vcc']))

        print('Event {:} (id: {:}) added.'.format(
                event_i['name'], event_i['_id']))

        # oversized task queue: warn and set flag.
        if len(events_to_update) > event_queue_size:
            is_update_complete = False
            warnings.warn('Not all events are updated. (Full task queue.)')
            break

    # unlike other data sources, here we iterate over catalogs.
    for catname_i, cat_info_i in local_catalogs.items():

        print('Searching catalog: {:}'.format(catname_i))

        # load catalog: FITS or HDF?
        catfn_suffix_i = cat_info_i['catalog_file'].split('.')[-1].lower()
        catfn_i = os.path.join(env_vars['workdir'], cat_info_i['catalog_file'])
        if catfn_suffix_i == 'h5':
            catfp_i = tb.open_file(catfn_i, mode='r')
            catgrp_i, catnode_i = cat_info_i['catalog_table']
            cat_i = catfp_i.get_node('/' + catgrp_i, catnode_i)
        elif catfn_suffix_i in ['fits', 'fit']: # FITS catalog,
            catfp_i = fits.open(catfn_i)
            cat_i = catfp_i[1].data
        else:
            err_msg = 'File format not supported: {:}'
            raise RuntimeError(err_msg.format(catfn_suffix_i))

        # determine index type.
        spidx_method_i = cat_info_i['spatial_index']['method'].lower()
        if spidx_method_i == 'tree': # tree form: load tree file.
            spidx_tree_file_i = os.path.join(env_vars['workdir'], \
                    cat_info_i['spatial_index']['tree_file'])
            with open(spidx_tree_file_i, 'rb') as fp:
                tree_i = pickle.load(fp)
        elif spidx_method_i == 'healpix':
            hp_cols_i = cat_info_i['spatial_index']['healpix_cols']
            hp_grp_i, hp_node_i = cat_info_i['spatial_index']['healpix_table']
            hpid_tab_i = catfp_i.get_node('/' + hp_grp_i, hp_node_i)
        else:
            err_msg = 'Spatial indexing method not supported: {:}'
            raise RuntimeError(err_msg.format(spidx_method_i))

        # determine column subset.
        if ('col_subset' in cat_info_i) and cat_info_i['col_subset']:
            col_subset_i = cat_info_i['col_subset']
        else:
            col_subset_i = None

        # for every single task: query the catalog and fill the results.
        for task_i in tqdm(events_to_update):

            # searching this catalog is not required -> skip
            if catname_i not in task_i.catalogs:
                continue

            # get previous records.
            loc_rec_i = local_queries.find_one({'_id': task_i.doc_id})

            # has no previous query record: prepare new record.
            if not loc_rec_i:
                crd_repr = 'coord:{:.6f},{:.6f}/radius:{:.3f}'.format( \
                        task_i.crd.ra.deg, task_i.crd.dec.deg, task_i.radius)
                loc_rec_i = {
                    'coord_desc': task_i.crd_src,
                    'queried_with': crd_repr,
                    'vcc': task_i.vcc,
                    # 'search_radius': task_i.radius,
                }
            else:
                loc_rec_i['vcc'] = task_i.vcc

            # do query: tree or healpix.
            if spidx_method_i == 'tree':
                stat_i = query_kdtree( \
                        cat_i, task_i.crd, task_i.radius, \
                        tree_i, col_subset=col_subset_i)

            elif spidx_method_i == 'healpix':
                stat_i = query_healpix( \
                        cat_i, task_i.crd, task_i.radius, \
                        hpid_tab_i, hp_cols_i, col_subset=col_subset_i)

            else:
                raise RuntimeError('Check spatial index method.')

            # update db.
            loc_rec_i[catname_i] = stat_i
            loc_rec_i['last_upate'] = date_stamp()

            local_queries.update_one({'_id': task_i.doc_id},
                                     {'$set': loc_rec_i},
                                     upsert=True)

        # close files
        catfp_i.close()

    # warn if task is unfinished.
    if not is_update_complete:
        warnings.warn('Local catalog update is incomplete.')
