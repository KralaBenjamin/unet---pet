apptainer exec --mount type=bind,src=/data/datapool3/datasets/krala/unet-pet,dst=/data/ \
 --mount type=bind,src=.,dst=/code/ \
  conda_env.sif python ./training_unet_only_pet_segmentation.py