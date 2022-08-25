# -*- coding: utf-8 -*-
"""sup3r batch utilities based on reV's batch module"""
from sup3r.pipeline.pipeline import Sup3rPipeline
from sup3r.pipeline.pipeline_cli import pipeline_monitor_background
from reV.batch.batch import BatchJob as RevBatchJob


class BatchJob(RevBatchJob):
    """Framework for building a batched job suite."""

    # Class attributes to set the software's pipeline class and run+monitor in
    # background method. These can be updated in dependent software packages
    # for other workflows that want to utilize reV's pipeline and batch
    # features.
    PIPELINE_CLASS = Sup3rPipeline
    PIPELINE_BACKGROUND_METHOD = pipeline_monitor_background
