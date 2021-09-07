#!/usr/bin/env python

'''
    Search value-added catalogs.
'''

import re
import warnings
from pprint import pprint

import requests
from urllib.parse import quote

import numpy as np
from numpy.ma.core import MaskedConstant
from numpy.ma.core import MaskedArray
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.utils.exceptions import AstropyWarning

# from astropy.cosmology import WMAP9
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(70, 0.3)

from astroquery.ned import Ned
from astroquery.simbad import Simbad
from astroquery.exceptions import RemoteServiceError
from astroquery.exceptions import TableParseError

from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

# add type fields.
simbad_extra_fields = [
    'otype', 'otypes', 'morphtype', 'z_value',
    'dim_angle', 'dim_incl', 'dim_majaxis', 'dim_minaxis'
]
for col_t in simbad_extra_fields:
    Simbad.add_votable_fields(col_t)

ned_galaxy_types = [
    "G",
    "GGroup",
    "GPair",
    "GTrpl",
    "QSO",
    "UvS",
    "Vis",
    "IrS"
]

# remove redundant spaces.
remove_space = lambda s: re.sub(' +', ' ', s)

# convert table column names to db keys
as_key = lambda w: w.lower().replace(' ', '_')

# simbad mirror to use.
simbad_url = "http://simbad.u-strasbg.fr/simbad/sim-tap/" + \
        "sync?request=doQuery&lang=adql&format=text&query="

def _f2s(v):
    if np.round(v) == v: return str(int(v))
    else: return str(v)

def row_to_list(w):
    rv = list(w)
    for iv, vi in enumerate(rv):
        if isinstance(vi, MaskedConstant): # set NA values as None
            rv[iv] = None
            continue
        if isinstance(vi, bytes): # convert byte str to str.
            rv[iv] = remove_space(vi.decode())
            continue
        if isinstance(vi, (np.int32, np.int16, np.int)): # to int
            rv[iv] = vi.item()
            continue
        if isinstance(vi, (np.float64, np.float32, \
                np.cfloat, np.float)): # to float
            rv[iv] = vi.item()
            continue
        if isinstance(vi, np.str_): # to normal float
            rv[iv] = str(vi)
        if isinstance(vi, MaskedArray): # patch for astroquery.simbad
            warnings.warn("Got MaskedArray when parsing astroquery output.")
            if vi.__len__() == 13:
                warnings.warn("Interpreted as sky coord.")
                rv[iv] = ' '.join([_f2s(t) for t in vi[:3]])
            else: raise RuntimeError('') # TODO
            continue
    return rv

def search_ned_by_name(host_i):

    # initial state
    stat = dict(queried_with='name:{:}'.format(host_i), is_complete=True)

    # turn on warning filter,
    # warnings.simplefilter("error")

    # 200802: have to turn this off. Some errors are raised as warnings by NED,
    # so I have to catch them, but astropy.io.votable throw warnings like crazy.

    # resolve name.
    try:
        restab_i = Ned.query_object(host_i, verbose=True)
        stat['is_resolved'] = True

    except RemoteServiceError as err: # failed
        stat['is_resolved'] = False
        stat['is_complete'] = 'not currently recognized' in str(err)
        # Name not recognized: consider as successful.

    except TableParseError as err:
        stat['is_resolved'] = False
        stat['is_complete'] = Ned.response.status_code == 200
        # mark as incomplete when it's a HTTP error.

    except Exception as err: # failed for other reasons.
        stat['is_resolved'] = False
        stat['is_complete'] = False

    # turn off
    # warnings.simplefilter("default")

    # not resolved for any reason: stop and return.
    if not stat['is_resolved']:
        return stat

    # marked as resolved but no result: (unhandled HTTP err)
    if not restab_i.__len__():
        stat['is_resolved'] = False
        stat['is_complete'] = True
        return stat

    # if resolved: get basic info.
    basic_tab_i = row_to_list(list(restab_i[0])) # get first row,
    colnames = [as_key(w) for w in restab_i.colnames]
    if ('(deg)' in colnames[2]) or ('(deg)' in colnames[3]):
        colnames[2] = colnames[2].replace('(deg)', '')
        colnames[3] = colnames[3].replace('(deg)', '')
    stat['basic'] = [{k: v for k, v in zip(colnames, basic_tab_i)}]
    stat['resolved_as'] = basic_tab_i[1]

    return stat

def retrieve_ned_photometry(host_i):

    ''' Get archival photometry from NED '''

    # initial state
    stat = dict(is_complete=True)

    try:
        phot_tab_i = Ned.get_table(host_i)
    except Exception as err: # failed to get photometry for any reason
        phot_tab_i = None
        stat['is_complete'] = False

    # parse photometry table.
    phot_list_i = list()
    if phot_tab_i: # has valid photometry
        colnames = [as_key(w) for w in phot_tab_i.colnames] # column names.
        for phot_rec_i in phot_tab_i:
            phot_rec_i = row_to_list(list(phot_rec_i))
            phot_list_i.append( # convert to a nested document.
                    {k: v for k, v in zip(colnames, phot_rec_i)})

    # abridge SDSS qualifiers
    for phot_rec_i in phot_list_i:
        if ('qualifiers' in phot_rec_i) and \
                ('SDSS flags' in phot_rec_i['qualifiers']):
            q_t = ';'.join([w.split(' - ')[0] for w in \
                    phot_rec_i['qualifiers'].split(';')])
            phot_rec_i['qualifiers'] = q_t

    # put archival photometry
    stat['phot'] = phot_list_i

    return stat

def search_ned_by_coord(crd_i, radius_i):

    # initial status.
    '''
    crd_repr_i = 'coord:{:.6f},{:.6f}'.format(crd_i.ra.deg, crd_i.dec.deg)
    stat = dict(is_complete=True,
                queried_with=crd_repr_i,
                search_radius=radius_i)
    '''

    crd_repr_i = 'coord:{:.6f},{:.6f}/radius:{:.3f}'.format(
                  crd_i.ra.deg, crd_i.dec.deg, radius_i)
    stat = dict(is_complete=True,
                queried_with=crd_repr_i,)
                # search_radius=radius_i)

    # define search radius for host/transient coordinates.
    rad = radius_i * u.arcsec

    # turn on warning filter,
    # warnings.simplefilter("error")

    # search nearby sources.
    try:
        restab_i = Ned.query_region(crd_i, rad)
        stat['is_resolved'] = True

    except RemoteServiceError as err:
        stat['is_resolved'] = False
        stat['is_complete'] = 'No object found' in str(err)
        # 'No object found' is considered as a successful query

    except TableParseError as err:
        stat['is_resolved'] = False
        stat['is_complete'] = Ned.response.status_code == 200
        # mark as incomplete when it's a HTTP error.

    except Exception as err:
        stat['is_resolved'] = False
        stat['is_complete'] = False
        # other errors: mark as incomplete.

    # turn off
    # warnings.simplefilter("default")

    # not resolved for any reason: stop and return.
    if not stat['is_resolved']:
        return stat

    # found sources: convert to list, sort by distance, reject other types.
    colnames = [as_key(w) for w in restab_i.colnames]
    if ('(deg)' in colnames[2]) or ('(deg)' in colnames[3]):
        colnames[2] = colnames[2].replace('(deg)', '')
        colnames[3] = colnames[3].replace('(deg)', '')
    restab_i = sorted([row_to_list(w) for w in restab_i], key=lambda x: x[9])
    restab_i = [w for w in restab_i if (w[4] in ned_galaxy_types)]

    # no object left after rejection
    if not restab_i:
        stat['is_resolved'] = False

    # parse result table.
    else:
        stat['basic'] = [ \
                {k: v for k, v in zip(colnames, src_i)} \
                for src_i in restab_i \
        ]

    return stat

def search_simbad_by_name(host_i):

    # initial state
    stat = dict(queried_with='name:{:}'.format(host_i), is_complete=True)

    # to catch warning info.
    warnings.filterwarnings('error')

    # resolve name
    try:
        restab_i = Simbad.query_object(host_i)
        stat['is_resolved'] = not (restab_i is None)
        stat['is_complete'] = not (restab_i is None)

    except Exception as err:
        err_msg = str(err)
        if ('No known catalog' in err_msg) \
                or ('an incorrect format for' in err_msg) \
                or ('ambiguous identifier' in err_msg) \
                or ('Identifier not found' in err_msg):
            # consider as successful query with empty return.
            stat['is_resolved'] = False
            stat['is_complete'] = True
        else: # other warnings: see result.
            stat['is_resolved'] = False
            stat['is_complete'] = False

    # unset warning filter.
    warnings.resetwarnings()

    # not resolved for any reason: return.
    if not stat['is_resolved']:
        return stat

    # if resolved: get basic info.
    colnames = [as_key(w) for w in restab_i.colnames]
    basic_tab_i = row_to_list(list(restab_i[0]))
    stat['basic'] = [{k: v for k, v in zip(colnames, basic_tab_i)}]
    for rec_i in stat['basic']:
        if rec_i['ra'] and rec_i['dec']:
            rec_i['ra'] = ':'.join(rec_i['ra'].split(' '))
            rec_i['dec'] = ':'.join(rec_i['dec'].split(' '))
    stat['resolved_as'] = basic_tab_i[0]

    return stat

def search_simbad_by_coord(crd_i, radius_i):

    # initial status.
    crd_repr_i = 'coord:{:.6f},{:.6f}/radius:{:.3f}'.format(
                  crd_i.ra.deg, crd_i.dec.deg, radius_i)
    stat = dict(is_complete=True,
                queried_with=crd_repr_i,)
                # search_radius=radius_i)

    # define search radius for host/transient coordinates.
    rad_i = radius_i

    # to catch warning info.
    warnings.filterwarnings('error')

    # search things inside this radius.
    try:
        restab_i = Simbad.query_criteria(
                'region(circle, ICRS, %fd %+fd, %.2fs)'%( \
                        crd_i.ra.deg, crd_i.dec.deg, rad_i),
                otypes='galaxy')
        stat['is_resolved'] = True

    except Exception as err:
        stat['is_complete'] = 'No object found' in str(err)
        stat['is_resolved'] = False

    # unset warning filter.
    warnings.resetwarnings()

    # not resolved for any reason: stop and return.
    if (not stat['is_resolved']):
        return stat

    # pack into a table.
    colnames = [as_key(w) for w in restab_i.colnames]
    restab_i = [row_to_list(w) for w in restab_i]
    stat['basic'] = [ \
            {k: v for k, v in zip(colnames, src_i)} for src_i in restab_i \
    ]

    # modify format of RA/Dec strings.
    for rec_i in stat['basic']:
        if rec_i['ra'] and rec_i['dec']:
            rec_i['ra'] = ':'.join(rec_i['ra'].split(' '))
            rec_i['dec'] = ':'.join(rec_i['dec'].split(' '))

    return stat

def retrieve_simbad_photometry(host_i):

    ''' Get archival photometry from SIMBAD '''

    # initial state
    stat = dict(is_complete=True, phot=dict())

    sql_line =  "SELECT s.*, f.*, p.rvz_redshift " + \
                "FROM basic AS p " + \
                "JOIN ident AS f ON f.oidref = p.oid " + \
                "JOIN allfluxes AS s ON s.oidref = p.oid " + \
                "WHERE id = '%s'"%(host_i)
    sql_line = quote(sql_line)

    try:
        resp = requests.get(simbad_url + sql_line)
        assert ('ERROR' not in resp.text)
    except Exception as err:
        stat['is_complete'] = False

    # failed for any reason: return
    if not stat['is_complete']:
        return stat

    resp_rec = list(filter(lambda x: x, resp.text.split('\n')))
    if len(resp_rec) < 3:
        return stat # consider as successful, no return.

    # pack into dict.
    colnames = [w.strip() for w in resp_rec[0].split('|')]
    vals = [w.strip() for w in resp_rec[2].split('|')]
    stat['phot'] = {k: v for k, v in zip(colnames, vals)}

    # convert data type.
    stat['phot']['oidref'] = int(stat['phot']['oidref'])
    stat['phot']['id'] = remove_space(stat['phot']['id'].strip('"'))
    for col_i in colnames:
        if col_i not in ['oidref', 'id']:
            stat['phot'][col_i] = float(stat['phot'][col_i]) \
                    if stat['phot'][col_i] else np.nan

    return stat

def _best_available_coord(event_i, vac_info_i, xvalid_mode=False):

    # confirmed by name in NED: search host coord.
    crd_i, crd_src_i, radius_i = None, None, 5
    if ('NED' in vac_info_i) and vac_info_i['NED']['is_resolved'] \
            and ('name:' in vac_info_i['NED']['queried_with']):
        if 'ra(deg)' in vac_info_i['NED']['basic'][0]:
            crd_i = SkyCoord(ra=vac_info_i['NED']['basic'][0]['ra(deg)'],
                             dec=vac_info_i['NED']['basic'][0]['dec(deg)'],
                             unit=('deg', 'deg'))
        else: # TODO: fix this later.
            crd_i = SkyCoord(ra=vac_info_i['NED']['basic'][0]['ra'],
                             dec=vac_info_i['NED']['basic'][0]['dec'],
                             unit=('deg', 'deg'))
        crd_src_i = 'name:NED:' + vac_info_i['NED']['resolved_as']

    # name not resolved in NED but resolved in SIMBAD: use this coord.
    if (not crd_i) and ('SIMBAD' in vac_info_i) \
            and vac_info_i['SIMBAD']['is_resolved'] \
            and ('name:' in vac_info_i['SIMBAD']['queried_with']):
        if vac_info_i['SIMBAD']['basic'][0]['ra'] \
                and vac_info_i['SIMBAD']['basic'][0]['dec']:
            crd_i = SkyCoord(ra=vac_info_i['SIMBAD']['basic'][0]['ra'],
                            dec=vac_info_i['SIMBAD']['basic'][0]['dec'],
                            unit=('hour', 'deg'))
            crd_src_i = 'name:SIMBAD:' + vac_info_i['SIMBAD']['resolved_as']
        else:
            warnings.warn('SIMBAD result available but coord is empty.')
            pprint(event_i)

    # name not resolved in either NED or SIMBAD: use host nominal coord.
    if (not crd_i) and ('host_ra_deg' in event_i) \
            and np.isfinite(event_i['host_ra_deg']):
        crd_i = SkyCoord(ra=event_i['host_ra'], dec=event_i['host_dec'],
                         unit=('hour', 'deg'),)
        radius_i, crd_src_i = 15, 'hostcrd'

    if xvalid_mode: # for cross-validation run, only use event coord.
        crd_i, crd_src_i, radius_i = None, None, 5

    # host crd is not available: use event coord.
    if not crd_i: # finally,

        crd_i = SkyCoord(ra=event_i['ra'], dec=event_i['dec'],
                         unit=('hour', 'deg'),)
        crd_src_i = 'event:'

        if ('host_dist' in event_i) and np.isfinite(event_i['host_dist']):
            radius_i = max(15., event_i['host_dist'] * 2)
            crd_src_i += 'host_dist'

        # has redshift, use project distance of 30 kpc.
        elif ('redshift' in event_i) and np.isfinite(event_i['redshift']):
            err_i = (30. * u.kpc) * cosmo.arcsec_per_kpc_proper( \
                    np.abs(event_i['redshift']))
            radius_i = err_i.to_value('arcsec')
            crd_src_i += 'redshift'

        # (GRB) has pos error, use twice this radius.
        elif ('GRB' in event_i['claimedtype']) and ('radec_err' in event_i) \
                and np.isfinite(event_i['radec_err']):
            radius_i = max(15., event_i['radec_err'] * 2.)
            crd_src_i += 'err_circle'

        # default:
        else:
            radius_i = 30.
            crd_src_i += 'default'

    # cap search radius to 2"
    radius_i = min(radius_i, 120.)

    return crd_i, crd_src_i, radius_i

# source filters.
_simbad_excluded_types = ['PoG', 'PaG', 'ClG', 'GrG', 'CGG', 'SCG',]
_ned_src_filter    = lambda r: r['type'] in ['G']
_simbad_src_filter = lambda r: all([(w not in _simbad_excluded_types) \
                                    for w in r['otypes'].split('|')]) \
                               and ('Anon' not in r['main_id']) \
                               and (r['coo_qual'] not in ['E'])
def _best_available_coord(event_i, vac_info_i, xvalid_mode=False):

    ''' Revised 082420: Exclude pairs and clusters. '''

    # Expand name-resolved radius from 5 to 15, 20210424

    # confirmed by name in NED: search host coord.
    crd_i, crd_src_i, crd_comm_i, radius_i = None, None, '', 15
    if ('NED' in vac_info_i) and vac_info_i['NED']['is_resolved'] \
            and ('name:' in vac_info_i['NED']['queried_with']):
        src_t = vac_info_i['NED']['basic'][0]
        if _ned_src_filter(src_t):
            # choose the right ra/dec keys
            if 'ra' in src_t: ra_k, dec_k = 'ra', 'dec'
            else: ra_k, dec_k = 'ra(deg)', 'dec(deg)'
            # extract ra/dec
            crd_i = SkyCoord(ra=src_t[ra_k], dec=src_t[dec_k],
                             unit=('deg', 'deg'))
            crd_src_i = 'NAME:NED:' + vac_info_i['NED']['resolved_as']
        else: crd_comm_i += 'NED_{:s}_EXCLUDED'.format(src_t['type'].upper())

    # name not resolved in NED but resolved in SIMBAD: use this coord.
    if (not crd_i) and ('SIMBAD' in vac_info_i) \
            and vac_info_i['SIMBAD']['is_resolved'] \
            and ('name:' in vac_info_i['SIMBAD']['queried_with']):
        src_t = vac_info_i['SIMBAD']['basic'][0]
        if src_t['ra'] and src_t['dec']:
            if _simbad_src_filter(src_t):
                crd_i = SkyCoord(ra=src_t['ra'], dec=src_t['dec'],
                                 unit=('hour', 'deg'))
                crd_src_i = 'NAME:SIMBAD:' \
                          + vac_info_i['SIMBAD']['resolved_as']
            else: crd_comm_i += \
                    ' SIMBAD_{:s}_EXCLUDED'.format(src_t['otypes'].upper())
        else:
            crd_comm_i += ' SIMBAD_RADEC_INVALID'
            warnings.warn('SIMBAD result available but coord is empty.')
            pprint(event_i)

    # name not resolved in either NED or SIMBAD: use host nominal coord.
    if (not crd_i) and ('host_ra_deg' in event_i) \
            and np.isfinite(event_i['host_ra_deg']):
        crd_i = SkyCoord(ra=event_i['host_ra'], dec=event_i['host_dec'],
                         unit=('hour', 'deg'),)
        radius_i, crd_src_i = 30, 'hostcrd'

    if xvalid_mode: # for cross-validation run, only use event coord.
        crd_i, crd_src_i, radius_i = None, None, 15
        crd_comm_i = 'XVALID'
        # overwrite

    # host crd is not available: use event coord.
    if not crd_i: # finally,

        crd_i = SkyCoord(ra=event_i['ra'], dec=event_i['dec'],
                         unit=('hour', 'deg'),)
        crd_src_i = 'event:'

        if None and ('host_dist' in event_i) and np.isfinite(event_i['host_dist']):
            radius_i = max(15., event_i['host_dist'] * 2)
            crd_src_i += 'host_dist'
            # XXX This branch is disabled.

        # has redshift, use project distance of 30 kpc.
        elif ('redshift' in event_i) and np.isfinite(event_i['redshift']):
            radius_i = (45. * u.kpc) * cosmo.arcsec_per_kpc_proper( \
                    np.abs(event_i['redshift']))
            radius_i = np.clip(radius_i.to_value('arcsec'), 15., 120.)
            crd_src_i += 'redshift'

        # (GRB) has pos error, use twice this radius.
        elif ('GRB' in event_i['claimedtype']) and ('radec_err' in event_i) \
                and np.isfinite(event_i['radec_err']):
            radius_i = np.clip(event_i['radec_err'] * 3., 5., 15.)
            crd_src_i += 'err_circle'

        # default:
        else:
            radius_i = 30.
            crd_src_i += 'default'

    # cap search radius to 2"
    # radius_i = np.clip(radius_i, 15., 120.)

    # update comments.
    crd_comm_i = crd_comm_i.strip()
    if crd_comm_i: crd_src_i = crd_src_i + ' / ' + crd_comm_i

    return crd_i, crd_src_i, radius_i

def best_available_coord(event_i, vac_info_i, xvalid_mode=False):

    ''' Revised 061421: . '''

    crd_i, crd_src_i, crd_comm_i = None, None, []
    radius_i, rad_src_i = 15, ''

    if not xvalid_mode: # Normal run, can use host names or coord.

        # confirmed by name in NED or SIMBAD:
        if vac_info_i['resolved_coord_best']:

            # search the best name-resolved coord
            crd_best_t = vac_info_i['resolved_coord_best']
            crd_i = SkyCoord(ra=crd_best_t['ra_deg'],
                             dec=crd_best_t['dec_deg'],
                             unit=('deg', 'deg'))
            crd_src_i = 'name_resolved'
            crd_comm_i = [crd_best_t['vac']] + crd_best_t['src'].split('/')
            radius_i, rad_src_i = 15., 'name_resolved_default'

        # name not resolved in either NED or SIMBAD:
        elif ('host_ra_deg' in event_i) and np.isfinite(event_i['host_ra_deg']):

            # use name-resolved host coord.
            crd_i = SkyCoord(ra=event_i['host_ra_deg'],
                        dec=event_i['host_dec_deg'],
                        unit=('deg', 'deg'))
            crd_src_i, crd_comm_i = 'reported_host_coord', []
            radius_i, rad_src_i = 30., 'reported_host_coord_default'
            # print('This branch is triggered.')

        else: pass

    if crd_i is None: # xvalid mode, or normal run w/o host info.

        # the default case, use event coord.
        crd_i = SkyCoord(ra=event_i['ra_deg'],
                       dec=event_i['dec_deg'],
                       unit=('deg', 'deg'),)
        crd_src_i = 'event_coord'
        crd_comm_i = ['XVALID'] if xvalid_mode else []

        # determine search radius.
        if ('redshift' in event_i) and np.isfinite(event_i['redshift']):
            radius_i = (45. * u.kpc) * cosmo.arcsec_per_kpc_proper( \
                    np.sqrt(event_i['redshift'] ** 2 + 1.e-3 ** 2))
            radius_i = np.clip(radius_i.to_value('arcsec'), 15., 120.)
            rad_src_i = 'redshift'

        elif ('GRB' in event_i['claimedtype']) and ('radec_err' in event_i) \
                and np.isfinite(event_i['radec_err']):
            radius_i = np.clip(event_i['radec_err'] * 3., 5., 15.)
            rad_src_i = 'grb_err_circle'

        else: # default case.
            radius_i = 30.
            rad_src_i = 'event_coord_default'

        # special case: Mock SN run.
        if xvalid_mode and ('_MockSN' in event_i['name']):
            radius_i = 45.
            rad_src_i = 'mock_supernova_default'
            crd_comm_i.append('MOCKSN')

    # update comments.
    crd_desc_i = {'coord_src': crd_src_i,
                  'coord_comm': crd_comm_i,
                  'coord': (crd_i.ra.deg, crd_i.dec.deg),
                  'radius': radius_i,
                  'radius_src': rad_src_i,}

    return crd_i, crd_desc_i, radius_i