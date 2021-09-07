def cross_identify(event, sources, catalogs,
                   source_filter=None, stellar_filter=None, astrom_tol=None,
                   seek_secondary=True, rank_func=None,
                   do_plot=False, fig_basename=None, fig_style=None,
                   fig_bg_image=False, fig_stamps=None, fig_legend_style=None,
                   fig_fname_ext='pdf',):

    '''
    Cross-identify sources across multiple catalogs.

    Parameters
    ----------
    event : dict
        Mongodb document (in dict form) from the event collection.

    sources : dict
        Combined list of sources from different data sources.

    catalogs : dict
        Catalog description file.

    do_plot : bool
        Switch to make figures for cross-matching results.

    fig_basename : str or None
        File for output figures.

    fig_style : dict
        Style of output figures.

    Returns
    -------
    '''

    # check arguments.
    assert ((not do_plot) or fig_basename), 'Must provide basename for figures.'
    assert ((not do_plot) or fig_style), 'No style definition for figures.'

    # set default params.
    if not stellar_filter: stellar_filter = dict()
    if not source_filter: source_filter = dict()
    if not astrom_tol: astrom_tol = dict()

    # initial status (some keys added later!)
    stat = {
        'is_identified': False,
        'identified_by': None,
        'valid_crossid': False,
        'N_groups': 0,
        'groups': [], # cross-identified sources here.
        'vcc': event['vcc'],
        'last_update': date_stamp(),
    }

    #----------------------------------------------------------
    # set reference frame

    # find center of query, calculate cartesian coord.
    crd_0, crd_src, rad_c = best_available_coord(event, sources)
    cvec_0 = np.array(_skycoord_to_cvec(crd_0))
    # 'sources' includes VAC results.

    # project onto 2d plane,
    cos_theta_0, sin_theta_0 = \
            np.cos(crd_0.dec.radian), np.sin(crd_0.dec.radian)
    cos_phi_0, sin_phi_0 = np.cos(crd_0.ra.radian), np.sin(crd_0.ra.radian)

    # (this is used to project things on a local 2-d plane)
    vn_ra =  np.array([-sin_phi_0, cos_phi_0, 0.])
    vn_dec = np.array([-sin_theta_0 * cos_phi_0, \
                       -sin_theta_0 * sin_phi_0, \
                        cos_theta_0])
    # can be done using a 3x3 rotation matrix, but let's do inner product.

    # 2-d position of event coord. (can be zero.)
    ra_ec, dec_ec = event['ra_deg'], event['dec_deg']
    crd_ec = SkyCoord(ra=ra_ec, dec=dec_ec, unit=('deg', 'deg'))
    delta_cvec_ec = np.subtract(_skycoord_to_cvec(crd_ec), cvec_0)
    dpos_ec_2d = _delta_cvec_to_2d(delta_cvec_ec, vn_ra, vn_dec)

    # redshift of the event.
    z_ec = event['redshift'] \
           if ('redshift' in event) and np.isfinite(event['redshift']) \
           else np.nan

    #----------------------------------------------------------
    # update info on positions and referencce frames.

    # status of identification.
    stat['is_identified'] = ('name' in crd_src) or ('hostcrd' in crd_src)
    stat['identified_by'] = crd_src.split(':')[0] \
                            if stat['is_identified'] \
                            else None

    # update event absolute and relative coord,
    stat['event_ra'], stat['event_dec'] = ra_ec, dec_ec
    stat['event_dxy'] = float(dpos_ec_2d[0]), float(dpos_ec_2d[0])
    stat['event_z'] = z_ec

    # event meta info.
    stat['event_name'] = event['name']
    stat['event_alias'] = event['alias']
    stat['revised_type'] = event['claimedtype']

    # existing host info.
    stat['reported_host'] = dict(name=event['host'] if 'host' in event else '',
                                 ra=event['host_ra_deg'],
                                 dec=event['host_dec_deg'],
                                 dist=event['host_dist'])

    # update field info.
    stat['field_info'] = dict(ra=crd_0.ra.deg, dec=crd_0.dec.deg,
                              coord_src=crd_src, radius=rad_c,
                              vn_ra=vn_ra.tolist(), vn_dec=vn_dec.tolist(),
                              cvec=cvec_0.tolist(),)

    #----------------------------------------------------------
    # put sources info list, do selection.

    # put catalog sources into a single list.
    src_list, excl_src_list = list(), list()
    for cat_i, cat_info_i in catalogs.items():

        if cat_i not in sources:
            continue # skip if catalog does not exist.

        # source table is 'basic' in VAC results.
        src_key_i = 'srcs' if ('srcs' in sources[cat_i]) else 'basic'
        if src_key_i not in sources[cat_i]:
            continue # Not resolved in VACs: skip.

        # use '_cat' key to indicate data source.
        for src_j in sources[cat_i][src_key_i]:
            src_list.append({**src_j, **{'_cat': cat_i}})

    # calculate positions in cartesian coord,
    cvec_delta, id_badsrc = list(), list()
    for i_src, src_i in enumerate(src_list):

        # find key names for coordinates.
        for ra_col_i, dec_col_i in catalogs[src_i['_cat']]['coord_keys']:
            if (ra_col_i in src_i) and (dec_col_i in src_i):
                break # got a valid key pair.
        assert (ra_col_i in src_i) and (dec_col_i in src_i), \
                "Missing RA/Dec keys. Check 'coord_keys'."

        # bad coordinates: skip. (it happens.)
        ra_i, dec_i = src_i[ra_col_i], src_i[dec_col_i]
        if not (_isvalid_crd(ra_i) and _isvalid_crd(dec_i)):
            id_badsrc.append((i_src, 'BAD COORD C1'))
            continue

        # invalid source: skip.
        if (src_i['_cat'] in source_filter) \
                and (not source_filter[src_i['_cat']](src_i)):
            id_badsrc.append((i_src, 'FILTER FUNC'))
            continue

        # remove sources beyond the search radius.
        if not (is_valid_decimal_coord(ra_i, dec_i) \
                or is_valid_sexagesimal_coord(ra_i, dec_i)):
            warnings.warn('Invalid coord detected!')
            id_badsrc.append((i_src, 'BAD COORD C2'))
            continue
        crd_i = SkyCoord(ra=ra_i, dec=dec_i, \
                unit=tuple(catalogs[src_i['_cat']]['coord_unit']))
        if crd_i.separation(crd_0).arcsec > rad_c:
            id_badsrc.append((i_src, 'BEYOND RADIUS'))
            continue

        # (accepted the source)

        # convert to cartesian.
        cvec_delta.append(np.array(_skycoord_to_cvec(crd_i)) - cvec_0)

        # also put this skycoord into the list.
        src_list[i_src]['_crd'] = crd_i

    # drop rejected sources in the list.
    for i_src, reason_i in id_badsrc[::-1]:
        excl_src_i = src_list.pop(i_src)
        if reason_i != 'BEYOND RADIUS':
            excl_src_list.append((excl_src_i, reason_i))

    #----------------------------------------------------------
    # calculate source positions, find connectivity

    # 2-d position of catalog sources.
    dpos_2d = [_delta_cvec_to_2d(w, vn_ra, vn_dec) for w in cvec_delta]
    dpos_2d.append((0., 0.)) # a special point, the origin. (crd_0)

    # calculate pairwise distance of catalog sources.
    N_srcs = len(src_list)
    d_srcs = np.zeros((N_srcs + 1, N_srcs + 1), dtype='i4')
    for i_src, j_src in itt.combinations(range(N_srcs + 1), 2):
        # here we also include (0., 0.) as a "virtual source".

        # ignore sources from the same catalog.
        if (i_src < N_srcs) and (j_src < N_srcs):
            if src_list[i_src]['_cat'] == src_list[j_src]['_cat']:
                continue

        # calculate distance
        d_ij = norm(np.subtract(dpos_2d[i_src], dpos_2d[j_src]))

        # calculate tolerance, zero for crd_0
        tol_i, tol_j = 0., 0.
        cat_i, cat_j = '__None', '__None'
        if i_src < N_srcs: # is a normal source, not the origin.
            cat_i, src_i = src_list[i_src]['_cat'], src_list[i_src]
            tol_i = astrom_tol[cat_i]
        if j_src < N_srcs: # normal source.
            cat_j, src_j = src_list[j_src]['_cat'], src_list[j_src]
            tol_j = astrom_tol[cat_j]
        tol_ij = np.sqrt(tol_i ** 2 + tol_j ** 2)

        # distance below threshold, different catalog -> connected.
        if (cat_i != cat_j) and (d_ij < tol_ij * 1.):
            d_srcs[i_src, j_src] = d_srcs[j_src, i_src] = 1

    # identify connected components.
    d_srcs = csr_matrix(d_srcs)
    N_cps, cps_label = connected_components(d_srcs)
    cps_0 = cps_label[-1] # the last point, crd_0, our queried coord.

    # group catalog sources into groups by component id.,
    xid_cps = [dict(_confusion=False, _confusion_cats=list()) \
               for k in np.unique(cps_label)]
    for i_src, (src_i, cps_i) in enumerate(zip(src_list, cps_label)):
        #                        here the last one (queried coord) is ignored!

        cat_i = src_i.pop('_cat')
        if cat_i not in xid_cps[cps_i]: # not in component: create new list.
            xid_cps[cps_i][cat_i] = dict(_confusion=False, srcs=list())
        xid_cps[cps_i][cat_i]['srcs'].append( \
                {**src_i, **{'_dxy': dpos_2d[i_src]}})
        # also put d_xy position into the dict.

        # check for confusion.
        xid_cps[cps_i][cat_i]['_confusion'] = \
                xid_cps[cps_i][cat_i]['srcs'].__len__() > 1
        if xid_cps[cps_i][cat_i]['_confusion'] \
                and (cat_i not in xid_cps[cps_i]['_confusion_cats']):
            xid_cps[cps_i]['_confusion'] = True
            xid_cps[cps_i]['_confusion_cats'].append(cat_i)

    # find mean dx, dy, ra, dec of each cross-matched component.
    _fc = lambda x: float(x)
    for i_cp, cp_i in enumerate(xid_cps):

        # for this component: get ra/dec, dx/dy of sources in each catalog
        radec_i = [ [(src_k['_crd'].ra.deg, src_k['_crd'].dec.deg) \
                     for src_k in catsrc_j['srcs']] \
                   for cat_j, catsrc_j in cp_i.items() if cat_j in catalogs]
        dxy_i = [[src_k['_dxy'] for src_k in catsrc_j['srcs']] \
                 for cat_j, catsrc_j in cp_i.items() if cat_j in catalogs]
        cat_nsrc_i = [(cat_j, len(catsrc_j['srcs'])) \
                      for cat_j, catsrc_j in cp_i.items() if cat_j in catalogs]

        # flatten the list.
        radec_i = list(itt.chain(*radec_i))
        dxy_i = list(itt.chain(*dxy_i))

        # if there are sources: calculate mean
        if radec_i and dxy_i:

            # position statistics.
            cp_i['_avr_radec'] = np.mean(radec_i, axis=0).tolist()
            cp_i['_avr_dxy'] = np.mean(dxy_i, axis=0).tolist()
            cp_i['_std_dxy'] = np.std(dxy_i, axis=0).tolist()
            cp_i['_cov_dxy'] = np.cov(dxy_i, rowvar=False, bias=False).tolist()

            # calculate shape stats for multiply-matched components.
            if len(dxy_i) > 1:

                cxx_t, cyy_t = cp_i['_cov_dxy'][0][0], cp_i['_cov_dxy'][1][1]
                cxy_t, cyx_t = cp_i['_cov_dxy'][0][1], cp_i['_cov_dxy'][1][0]
                pr_t = np.abs(cxy_t) / np.sqrt(cxx_t * cyy_t)
                cp_i['_shape_r'] = _fc(pr_t)
                # C_10 and C_01 should be the same.

                # shape of cov matrix.
                u_t = (cxx_t + cyy_t) / 2
                v_t = np.sqrt(4. * cxy_t * cyx_t + (cxx_t - cyy_t) ** 2) / 2
                w_t = np.clip((u_t - v_t) / (u_t + v_t), 0., 1.)
                q_t, e_t = np.sqrt(w_t), np.sqrt(1. - w_t)
                p_t = (1. - q_t) / (1. + q_t)
                cp_i['_shape_p'], cp_i['_shape_q'] = _fc(p_t), _fc(q_t)
                cp_i['_shape_e'] = _fc(e_t)

            else: # only one element, fill default values.
                cp_i['_shape_r'] = 1.
                cp_i['_shape_p'], cp_i['_shape_q'] = 1., 0.
                cp_i['_shape_e'] = 1.

            # mean distance to centroid
            dist_t = norm(np.subtract(dxy_i, cp_i['_avr_dxy']), axis=1)
            cp_i['_avr_dist'] = np.mean(dist_t).tolist()

        # calculate number of possible connections.
        if cat_nsrc_i:

            # find combinations.
            cat_nsrc_sum_i = np.sum([w[1] for w in cat_nsrc_i])
            cp_i['_N_max_conn'] = _fc( \
                    cat_nsrc_sum_i * (cat_nsrc_sum_i - 1) / 2)

            # subtract same-catalog cases.
            for cat_k, nsrc_j in cat_nsrc_i:
                if nsrc_j < 2: continue
                cp_i['_N_max_conn'] -= _fc(nsrc_j * (nsrc_j - 1) / 2)

            # count actual number of combinations.
            sidx_t = np.arange(cps_label.size - 1)[cps_label[:-1] == i_cp]
            d_srcs_sub_i = d_srcs[np.ix_(sidx_t, sidx_t)]
            cp_i['_N_conn'] = _fc(np.sum(d_srcs_sub_i) / 2)

            # "connectivity"
            cp_i['_F_conn'] = _fc(cp_i['_N_conn'] / (cp_i['_N_max_conn'] + 1))

    #----------------------------------------------------------
    # label stars, sort confusing sources.

    # label potential stellar objects.
    for cp_i in xid_cps:
        for cat_j, filter_j in stellar_filter.items():
            if cat_j not in cp_i:
                continue # No S/G separation -> skip
            for src_k in cp_i[cat_j]['srcs']: # label stellar sources.
                if filter_j(src_k): src_k['__is_stellar'] = True

    # sort confusion sources by magnitude, distance and stellarity.
    _val_to_rank = lambda a: np.argsort(np.argsort(a))
    _magc = lambda x: 99. if (x is None) or (not np.isfinite(x)) else x
    for cp_i in xid_cps:
        for cat_j in cp_i['_confusion_cats']: # only for cats with confusion

            # num of confusing cat srcs in this grp
            ccat_srcs_j = cp_i[cat_j]['srcs']
            N_ccat_src_j = len(ccat_srcs_j)

            # rank srcs by distance
            src_dist_j = [norm(w['_dxy']) for w in ccat_srcs_j]
            src_dist_rank_j = _val_to_rank(src_dist_j) # always available!

            # rank by stellarity
            is_stellar_j = [int('__is_stellar' in w) for w in ccat_srcs_j]
            is_stellar_rank_j = _val_to_rank(is_stellar_j) \
                                if np.unique(is_stellar_j).size > 1 \
                                else np.zeros(N_ccat_src_j)

            # source magnitude rank
            mag_col_t = catalogs[cat_j]['mag_col'] \
                        if ('mag_col' in catalogs[cat_j]) \
                        else None
            mag_val_i = [_magc(w[mag_col_t]) for w in ccat_srcs_j] \
                        if mag_col_t \
                        else np.zeros(N_ccat_src_j)
            mag_val_rank_i = _val_to_rank(mag_val_i) if mag_col_t \
                             else np.zeros(N_ccat_src_j)

            # the overall rank
            rank_sum_t = is_stellar_rank_j * N_ccat_src_j \
                       + mag_val_rank_i \
                       + src_dist_rank_j / N_ccat_src_j
            id_rank_j = np.argsort(rank_sum_t)

            # re-order sources.
            cp_i[cat_j]['srcs'] = [ccat_srcs_j[i] for i in id_rank_j]

    # label stellar sources (and remove inner labels)
    for cp_i in xid_cps:

        # init: list of star src. stat flag.
        cp_i['_stellar_srcs'] = list()
        cp_i['_is_stellar'] = False

        for cat_j in stellar_filter.keys(): # only cats with stellar filter.

            if cat_j not in cp_i:
                continue # not in this group -> skip.

            # iterate over potential stellar srcs.
            for k_src, src_k in enumerate(cp_i[cat_j]['srcs']):

                if '__is_stellar' not in src_k:
                    continue # not a stellar source -> skip

                # label star in list. drop label.
                cp_i['_stellar_srcs'].append((cat_j, k_src))
                src_k.pop('__is_stellar')

                if k_src == 0: # repr src is a star -> entire grp is a star
                    cp_i['_is_stellar'] = True

    #----------------------------------------------------------
    # summarize rejected sources.

    excl_src_stat = dict()
    for cat_i, catsrc_i in sources.items():
        if cat_i not in catalogs: continue
        srcs_key_t = 'basic' if ('basic' in catsrc_i) else 'srcs' # VAC/survey
        if srcs_key_t not in catsrc_i: continue
        excl_src_stat[cat_i] = dict(N_src=len(catsrc_i[srcs_key_t]),
                                    N_excl=0,
                                    excl_srcs=list())

    for src_i, reason_i in excl_src_list:
        cat_k = src_i.pop('_cat')
        excl_src_stat[cat_k]['N_excl'] += 1
        excl_src_stat[cat_k]['excl_srcs'].append(
                dict(src=src_i, reason=reason_i))

    # calculate total number
    N_src_t, N_excl_t = 0, 0
    for cat_i, rej_stat_i in excl_src_stat.items():
        N_src_t += rej_stat_i['N_src']
        N_excl_t += rej_stat_i['N_excl']

    excl_src_stat = {k: v for k, v in excl_src_stat.items() if v['N_excl']}
    stat['excluded_sources'] = {'_N_src': N_src_t, '_N_excl': N_excl_t,
                                **excl_src_stat}

    #----------------------------------------------------------
    # rank cross-matched groups to find host info

    # and sort by distance to the center of query.
    cps_dist = [(norm(cp_i['_avr_dxy']) if '_avr_dxy' in cp_i else 0.) \
                for cp_i in xid_cps]
    cps_order = np.argsort(cps_dist)

    # list for component indices included.
    cpid_included = list() # avoid duplicates

    # identified (by host name or host crd) and matches the center of query,
    hostcrd_match_star = False # set True when only a primary is found.
    if stat['is_identified'] and (cps_order[0] == cps_0):
        #       the second condition is almost always guranteed,

        # RULE 0: identified by host name or coord, the center of query
        # matches some catalogs -> confirmed (or primary, if stellar.)

        # valid cross-matching in any catalogs, (** should be real!)
        if any([(w in xid_cps[cps_0]) for w in catalogs.keys()]):
            stat['valid_crossid'] = True # mark as valid cross-match.

            # matches a galaxy-like object: mark confirmed (default case)
            stat_t = {'_stat': 'confirmed'}
            if xid_cps[cps_0]['_is_stellar']: # matched a stellar object
                stat_t['_stat'] = 'primary' # mark as primary candidate.
                stat['N_groups'] += 1
                hostcrd_match_star = True

            # put into list, mark as confirmed (or primary).
            stat['groups'].append({**xid_cps[cps_0], **stat_t})
            cpid_included.append(cps_0)

            # find best-available host ra, dec and identifier,
            stat.update(best_hostmeta(xid_cps[cps_0], catalogs))
            # either confirmed or primary

    # Nothing found in previous step, or object identified as prim candidate:
    if (not stat['groups']) or hostcrd_match_star or seek_secondary:

        # 201108: Modified, if `seek_secondary` is True, always seek secondary.

        # RULE 1:   Not searching host coord, host coord matched nothing,
        #           or host coord matches stellar objects -->
        # -->   Find and rank candidates.

        # now sort other cross-matched groups.
        '''
        cps_rank = np.zeros_like(cps_order,)
        cps_rank[cps_order] = np.arange(cps_order.size).astype(int)
        '''

        # use external regressor to rank components.
        if rank_func:
            rank_func.annotate(xid_cps, z_ec, dpos_ec_2d)
            cps_rank_score = np.array([
                    w['_rank_score'] if '_rank_score' in w else -1.e99
                    for w in xid_cps \
            ])
            # DEBUG
            # print(np.sum(cps_rank_score < -1.e-98), cps_rank_score.size)
            # if abs(np.sum(cps_rank_score < -1.e-98) - cps_rank_score.size) > 1: stat['_stop'] = None

        else: # no regressor provided: use distance directly.
            print('*** Need trained regressor to rank candidates!')
            cps_rank_score = -cps_dist

        # re-order sources.
        cps_order = np.argsort(-cps_rank_score) # higher -> more likely.

        # object matching the center of query is the best object:
        if (cps_order[0] == cps_0) \
                and any([(w in xid_cps[cps_0]) for w in catalogs.keys()]):
            stat['valid_crossid'] = True # mark as valid cross-match.

        # now put them into the list of candidate hosts.
        for cpid_i in cps_order:

            # this component is already marked as primary candidate -> skip.
            if hostcrd_match_star and (cpid_i == cps_0):
                continue

            # this component is already included as primary host
            if seek_secondary and (cpid_i in cpid_included):
                continue
            # 201108+ To avoid duplicates when seek_secondary and is_confirmed

            # special case: marked as stellar by all catalogs: skip.
            '''
            if np.sum([(k in xid_cps[cpid_i]) for k in catalogs.keys()]) \
                    == len(xid_cps[cpid_i]['_is_stellar']):
                continue
            '''
            # This cannot happen!

            # primary candidate not identified
            xidst_i = 'primary' if (not stat['groups']) else 'secondary'

            # group contains nothing, mark as event.
            if not any([w in xid_cps[cpid_i] for w in catalogs.keys()]):
                xidst_i = 'event'

            # put into the list of candidates.
            stat['groups'].append({**xid_cps[cpid_i], **{'_stat': xidst_i}})
            stat['N_groups'] += 1

            # update coord if this is a primary source.
            if xidst_i == 'primary':
                stat.update(best_hostmeta(xid_cps[cpid_i], catalogs))

    # convert '_crd' object into plain ra/dec.
    for cp_i in stat['groups']:
        for cat_j, catsrc_j in cp_i.items(): # for every catalog object
            # mammoths
            if cat_j in catalogs: # skip non-catalog keys
                for k_src, src_k in enumerate(catsrc_j['srcs']):
                    # dinosaurs
                    if '_crd' in src_k:
                        src_k['_crd'] = plain_radec(src_k['_crd'])
                        # trilobites

    # calculate offset using the best host.
    if 'host_dxy' not in stat:
        stat.update(dict(host_ra=np.nan, host_dec=np.nan,
                         host_id=None, host_coord_src=None,
                         host_dxy=[np.nan, np.nan]))

    stat['host_dist'] = norm(np.subtract(stat['host_dxy'], stat['event_dxy']))

    # and finally, assign a random id.
    stat['rand_id'] = np.random.rand()

    #----------------------------------------------------------
    # make figure if necessary

    # make a figure.
    if do_plot:
        fig_fname = '.'.join([fig_basename, \
                event['name'], fig_fname_ext]).replace(' ', '_')
        _crossid_plot(xid_cps, crd_0, event['name'], fig_fname, fig_style,
                      fig_bg_image, fig_stamps, fig_legend_style)

    return stat