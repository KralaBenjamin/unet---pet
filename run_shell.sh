#!/bin/bash


apptainer shell --mount type=bind,src=/data/datapool3/datasets/krala/unet-pet,dst=/data/ \
 --mount type=bind,src=.,dst=/code/ \
--nv \
  conda_env.sif