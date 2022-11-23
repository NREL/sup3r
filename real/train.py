import os
import time
import logging
import numpy as np
import pandas as pd
from inspect import signature
from rex import init_logger
from sup3r import CONFIG_DIR
from sup3r.models import Sup3rCondMom
from sup3r.preprocessing.data_handling import DataHandlerH5 as DataHandler
import argparse

parser = argparse.ArgumentParser(description="Train Cond Mom")
parser.add_argument(
        "-im",
        "--indModel",
        type=int,
        metavar="",
        required=True,
        help="Model ID",
        default=None,
    )
parser.add_argument(
        "-imom",
        "--indMoment",
        type=int,
        metavar="",
        required=True,
        help="Moment index (1 or 2)",
        default=None,
    )
parser.add_argument(
        "-sp",
        "--s_padding",
        type=int,
        metavar="",
        required=False,
        help="Spatial Padding",
        default=None,
    )
parser.add_argument(
        "-tp",
        "--t_padding",
        type=int,
        metavar="",
        required=False,
        help="Temporal Padding",
        default=None,
    )
parser.add_argument(
        "-sf",
        "--subfilter",
        action="store_true",
        help="Learn subfilter",
    )
group1 = parser.add_mutually_exclusive_group()
group1.add_argument(
    "-s", "--spatial", action="store_true", help="Only spatial"
)
group1.add_argument(
    "-st", "--spatioTemporal", action="store_true", help="Spatio temporal"
)

parser.add_argument(
        "-m1f",
        "--mom1_file",
        type=str,
        metavar="",
        required=False,
        help="model file for first moment",
        default=None,
    )


args = parser.parse_args()

if args.spatioTemporal:
    spatio_temp = True
    coord_str = 'st'
    t_en = 12
    samp_shape = (20, 20, 36)
else: 
    spatio_temp = False
    coord_str = 's'
    t_en = 1
    samp_shape = (20, 20)
if not args.subfilter:
    sf_str = ''
else: 
    sf_str = '_sf'
if not args.indMoment==1:
    mom_str = '_mom2'
else:
    mom_str = ''

if spatio_temp:
    if args.indMoment==1:
        if args.subfilter:
            from sup3r.preprocessing.batch_handling import BatchHandlerMom1SF as BatchHandler
        else:
            from sup3r.preprocessing.batch_handling import BatchHandler as BatchHandler
    elif args.indMoment==2:
        if args.mom1_file is None:
            print("ERROR: Need first moment file to train second moment")
            stop
        if args.subfilter:
            from sup3r.preprocessing.batch_handling import BatchHandlerMom2SF as BatchHandler
        else:
            from sup3r.preprocessing.batch_handling import BatchHandlerMom2 as BatchHandler
    else:
        print("ERROR: Moment must be 1 or 2")
        stop 
else:
    if args.indMoment==1:
        if args.subfilter:
            from sup3r.preprocessing.batch_handling import SpatialBatchHandlerMom1SF as BatchHandler
        else:
            from sup3r.preprocessing.batch_handling import SpatialBatchHandler as BatchHandler
    elif args.indMoment==2:
        if args.mom1_file is None:
            print("ERROR: Need first moment file to train second moment")
            stop
        if args.subfilter:
            from sup3r.preprocessing.batch_handling import SpatialBatchHandlerMom2SF as BatchHandler
        else:
            from sup3r.preprocessing.batch_handling import SpatialBatchHandlerMom2 as BatchHandler
    else:
        print("ERROR: Moment must be 1 or 2")
        stop 

if args.s_padding is None:
    spad = 0
else:
    spad = args.s_padding
if args.t_padding is None:
    tpad = 0
else:
    tpad = args.t_padding


localFolder_file = '%s_mom%d%s_%d_spad%d_tpad%d' % (coord_str, args.indMoment, sf_str, args.indModel, spad, tpad)
model_otpt = '/projects/nationalwind/conditionalMoment/output/' + localFolder_file 
os.makedirs('/projects/nationalwind/conditionalMoment/output', exist_ok=True)
os.makedirs(model_otpt, exist_ok=True)


logger = logging.getLogger(__name__)
if spatio_temp:
    fp_gen = '/projects/nationalwind/conditionalMoment/real_configs/gen_2x_12x_14f%s_%d.json' % (mom_str, args.indModel)
else:
    fp_gen = '/projects/nationalwind/conditionalMoment/real_configs/gen_2x_1x_14f%s_%d.json' % (mom_str, args.indModel)

saved_model = None  # train model from scratch
out_dir= '/projects/nationalwind/conditionalMoment/' + model_otpt + '/cond_mom_e{epoch}/'

if __name__ == '__main__':
    log_file = model_otpt + '/train.log'
    init_logger(__name__, log_level='DEBUG', log_file=log_file)
    init_logger('sup3r', log_level='DEBUG', log_file=log_file)

    os.makedirs('/scratch/mhassana/wtk_cache', exist_ok=True)
    os.makedirs('/scratch/mhassana/wtk_cache/' + localFolder_file, exist_ok=True)
    os.makedirs('/scratch/mhassana/wtk_cache/' + localFolder_file + '/15_min', exist_ok=True)

    cache_dir = '/scratch/mhassana/wtk_cache/' + localFolder_file + '/15_min'
    year = 2007
    grid_size = (100, 100)
    target = (32, -122) 
    raster_id = f'{"x".join([str(g) for g in grid_size])}_'
    raster_id += f'{"_".join([str(t) for t in target])}'
    os.makedirs(model_otpt + '/wtk_raster_ind_cache', exist_ok=True)
    raster_file = model_otpt + '/wtk_raster_ind_cache/raster_conus_'
    raster_file += f'{raster_id}.txt'
   
 
    features = []
    wtk_files = [] 
    cache_pattern = f'{cache_dir}/cached_features_{year}'+'_{feature}.pkl'
     
    for height in [10, 40, 80, 100, 120, 160, 200]:
        
        features.append(f'U_{height}m')
        features.append(f'V_{height}m')
        wtk_files.append(f'/datasets/WIND/conus/v1.0.0/{year}/wtk_conus_{year}_{height}m.h5')
   
    stds_file = f'{cache_dir}/stds_full_wind_model_uv_{raster_id}_{year}.pkl'
    means_file = f'{cache_dir}/means_full_wind_model_uv_{raster_id}_{year}.pkl'
  
  
    kwargs = dict(file_paths=wtk_files,
                  features=features,
                  target=target,
                  shape=grid_size,
                  temporal_slice=slice(None, None, 3),
                  time_chunk_size=50,
                  raster_file=raster_file,
                  cache_pattern=cache_pattern,
                  val_split=0.1,
                  sample_shape=samp_shape,
                  stdevs_file=stds_file,
                  means_file=means_file,
                  batch_size=64,
                  s_enhance=2,
                  t_enhance=t_en,
                  n_batches=100,
                  load_workers=None,
                  norm_workers=None,
                  extract_workers=None,
                  compute_workers=None,
                  overwrite_cache=True,
                  overwrite_stats=False,
                  s_padding=args.s_padding,
                  t_padding=args.t_padding,
                  model_mom1=args.mom1_file)

    sig = signature(DataHandler)
    dh_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    data_handler = DataHandler(**dh_kwargs) 
    sig = signature(BatchHandler)
    bh_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    batch_handler = BatchHandler([data_handler], **bh_kwargs)

    if saved_model is None:
        model = Sup3rCondMom(fp_gen, learning_rate=5e-5)
    else:
        model = Sup3rCondMom.load(saved_model)
        model.update_optimizer(learning_rate=5e-5)
        logger.info(f'optimizer configuration: {model.get_optimizer_config}') 

    try:
        # GAN training
        model.train(batch_handler, n_epoch=10000,
                    checkpoint_int=10, out_dir=out_dir)
    except Exception as e:
        logger.exception('Training failed!')
        raise e
