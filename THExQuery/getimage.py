
import logging
from io import BytesIO
from tempfile import TemporaryDirectory as TempDir

import requests

import matplotlib.image as mpim
import matplotlib.pyplot as plt

from panstamps.downloader import downloader as ps_downloader
from panstamps.image import image as ps_image

ps1_image_cache = '~/.astropy/cache/ps1/'

ps_logger = logging.getLogger('ps-logger')
ps_logger.setLevel(40)

def retrieve_image(ra, dec, ax_rg, return_raw=False):

    baseurl = 'http://legacysurvey.org//viewer/cutout.jpg'

    # list of layers to use, also the order of preference.
    layers = ['ls-dr9', 'ls-dr8', 'ls-dr67', 'decals-dr7', 'mzls+bass-dr6',
              'des-dr1', 'sdss2', 'sdssco', ] # 'unwise-neo4']

    # return this url: image not available.
    # blank = 'http://legacysurvey.org/viewer/static/blank.jpg'

    # calculate pixel scale. (output image has fixed size!)
    pix_scal = 2. * ax_rg / 256.

    # try every layer: if available, then
    for layer_i in layers:

        # read image of this layer
        req_pl = dict(ra=ra, dec=dec, pixscale=pix_scal, layer=layer_i)
        resp = requests.get(baseurl, params=req_pl)

        # image not available: try next.
        if resp.url.endswith('blank.jpg'):
            continue

        # valid image: break
        break

    # if JPEG is requested instead of RGB arrays,
    if return_raw: return resp.content, layer_i

    # did we get a valid image? not -> terminate.
    if resp.url.endswith('blank.jpg'):
        return None, ''

    # read image and return layer name.
    try:
        img = mpim.imread(BytesIO(resp.content), 'jpeg')
    except:
        img, layer_i = None, None

    return img, layer_i

def retrieve_ps1_color(ra, dec, ax_rg, return_raw=False):

    with TempDir() as temp_dir:

        # search for image.
        fits_paths, jpeg_paths, color_path = ps_downloader(
            log=ps_logger,
            settings=False,
            downloadDirectory=temp_dir,
            fits=False,
            jpeg=True,
            arcsecSize=ax_rg * 2,
            filterSet='gri',
            color=True,
            singleFilters=False,
            ra='{:.8f}'.format(ra),
            dec='{:.8f}'.format(dec),
            imageType="stack",  # warp | stack
            mjdStart=False,
            mjdEnd=False,
            window=False
        ).get()

        # No image returned, skip.
        if (not color_path) or (color_path[0] is None): return None, ''

        if return_raw: # return raw jpeg instead of rgb image.
            with open(color_path[0], 'rb') as f: jpeg_data = f.read()
            return jpeg_data, 'ps1-stack-color'

        # read image and return layer name.
        img = mpim.imread(color_path[0], 'jpeg')
        return img, 'ps1-stack-color'

def retrieve_ps1_single(ra, dec, ax_rg, band='g', return_raw=False):

    assert band in ['g', 'r', 'i', 'z', 'y']

    with TempDir() as temp_dir:

        # search for image.
        fits_paths, jpeg_paths, color_path = ps_downloader(
            log=ps_logger,
            settings=False,
            downloadDirectory=temp_dir,
            fits=False,
            jpeg=True,
            arcsecSize=ax_rg * 2,
            filterSet=band,
            color=False,
            singleFilters=True,
            ra='{:.8f}'.format(ra),
            dec='{:.8f}'.format(dec),
            imageType="stack",  # warp | stack
            mjdStart=False,
            mjdEnd=False,
            window=False
        ).get()

        # No image returned, skip.
        if (not jpeg_paths) or (jpeg_paths[0] is None):
            return None, ''

        if return_raw: # return raw jpeg instead of rgb image.
            with open(jpeg_paths[0], 'rb') as f: jpeg_data = f.read()
            return jpeg_data, 'ps1-stack-' + band

        # read image and return layer name.
        img = mpim.imread(jpeg_paths[0], 'jpeg')
        return img, 'ps1-stack-' + band
