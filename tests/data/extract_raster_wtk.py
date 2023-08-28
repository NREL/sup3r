# -*- coding: utf-8 -*-
"""
Script to extract data subset in raster shape from flattened WTK h5 files.
"""

from rex import init_logger
from rex.resource_extraction.resource_extraction import WindX
from rex.outputs import Outputs
import matplotlib.pyplot as plt


if __name__ == '__main__':
    init_logger('rex', log_level='DEBUG')

    res_fp = '/datasets/WIND/conus/v1.0.0/wtk_conus_2013.h5'
    fout = './test_wtk_co_2013.h5'

    dsets = ['windspeed_80m', 'windspeed_100m', 'winddirection_80m',
             'winddirection_100m', 'temperature_100m', 'pressure_100m']

    target = (39.0, -105.15)
    shape = (20, 20)

    with WindX(res_fp) as res:
        meta = res.meta
        raster_index_2d = res.get_raster_index(target, shape, max_delta=20)

        for d in ('elevation', 'latitude', 'longitude'):
            data = meta[d].values[raster_index_2d]
            a = plt.imshow(data)
            plt.colorbar(a, label=d)
            plt.savefig(d + '.png')
            plt.close()

        raster_index = sorted(raster_index_2d.ravel())

        attrs = {k: res.resource.attrs[k] for k in dsets}
        chunks = {k: None for k in dsets}
        dtypes = {k: res.resource.dtypes[k] for k in dsets}
        meta = meta.iloc[raster_index].reset_index(drop=True)
        time_index = res.time_index
        shapes = {k: (len(time_index), len(meta)) for k in dsets}
        print(shapes)

        Outputs.init_h5(fout, dsets, shapes, attrs, chunks, dtypes, meta,
                        time_index=time_index)

        with Outputs(fout, mode='a') as f:
            for d in dsets:
                f[d] = res[d, :, raster_index]

        with Outputs(fout, mode='r') as f:
            meta = f.meta
            for d in dsets:
                data = f[d].mean(axis=0)
                data = data.reshape(shape)
                a = plt.imshow(data)
                plt.colorbar(a, label=d)
                plt.savefig(d + '.png')
                plt.close()
