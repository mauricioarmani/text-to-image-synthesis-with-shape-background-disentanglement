# env = hdgan2
name="model4_birds"
epoch="500"
device=1

export CUDA_VISIBLE_DEVICES=${device}
h5_file="$name""_G_epoch_""$epoch.h5"

# IS
python evaluation/inception_score/inception_score.py \
            --checkpoint_dir evaluation/inception_score/inception_finetuned_models/birds_valid299/model.ckpt \
            --image_folder results \
            --h5_file ${h5_file} \
            --batch_size 10 \
            --num_classes 50

# FID
python evaluation/fid/fid_example.py \
        --image_folder results \
        --h5_file ${h5_file}


# model4 500

# IS500  aligned        3.83
# IS500  shape_aligned  3.55
# IS500  bkg_aligned    3.61
# IS500  misaligned     3.45

# FID500 aligned        18.92
# FID500 shape_aligned  
# FID500 bkg_aligned    
# FID500 misaligned     