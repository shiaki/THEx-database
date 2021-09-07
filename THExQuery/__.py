    if var_thres: # use per-field threshold

        # original catalog of each source, threshold axis,
        src_cats = [w['_cat'] for w in src_list] + ['_crd_0']
        thres_axis_p = np.logspace(-0.5, 0.5, 15) \
                       if thres_axis is None else thres_axis

        # results
        conn_score_th = np.zeros_like(thres_axis_p)
        conn_results_th = list() # d_srcs, N_cps, cps_label, cps_0,

        # try values and calculate "goodness-of-matching" score
        for k_th, th_k in enumerate(thres_axis_p):

            # group corss-matched components.
            d_srcs_k = csr_matrix((w_srcs < th_k).astype('i4'))
            N_cps_k, cps_label_k = connected_components(d_srcs_k)
            cps_0_k = cps_label_k[-1] # last point (crd_0), queried coord.

            # put into array
            conn_results_th.append((d_srcs_k, N_cps_k, cps_label_k, cps_0_k))

            # count fraction of properly matched events.
            cats_incl_k = [list() for _ in range(N_cps_k)]
            for w, v in zip(cps_label_k, src_cats):
                cats_incl_k[w].append(v)

            # find properly matched sources in each catalog
            for u, cats_incl_u in enumerate(cats_incl_k):
                Np_u = [w for w in cats_incl_u \
                        if (cats_incl_u.count(w) == 1)].__len__()
                conn_score_th[k_th] += Np_u * (Np_u - 1) / 2

        # find the best matching,
        i_th_best = np.argmax(conn_score_th)
        d_srcs, N_cps, cps_label, cps_0 = conn_results_th[i_th_best]
        N_cps_p = [w[1] for w in conn_results_th]
        thres = thres_axis_p[i_th_best]

    else: # use fixed threshold (thres = 1),
        d_srcs = csr_matrix((w_srcs < 1.).astype('i4'))
        N_cps, cps_label = connected_components(d_srcs)
        cps_0 = cps_label[-1] # the last point, crd_0, our queried coord.
        thres = None