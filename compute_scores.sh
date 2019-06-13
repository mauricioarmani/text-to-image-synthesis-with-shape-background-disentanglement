# env = hdgan2
name="GDESGAN_64_birds"
epoch="500"
device=0

export CUDA_VISIBLE_DEVICES=${device}
h5_file="$name""_G_epoch_""$epoch.h5"

# IS
python Evaluation/inception_score/inception_score.py \
            --checkpoint_dir Evaluation/inception_score/inception_finetuned_models/birds_valid299/model.ckpt \
            --image_folder Results \
            --h5_file ${h5_file} \
            --batch_size 10 \
            --num_classes 50

# # FID
# python Evaluation/fid/fid_example.py \
#         --image_folder Results \
#         --h5_file ${h5_file}