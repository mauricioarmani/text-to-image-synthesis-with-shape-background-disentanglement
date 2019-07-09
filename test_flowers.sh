# # ALL ALIGNED
name='SEG10_BG50L2_birds'
epoch=500

python gan/test_worker.py \
     --dataset flowers \
     --model_name ${name} \
     --load_from_epoch ${epoch} \
     --test_sample_num 10 \
     --save_visual_results \
     --batch_size 64 \
     --align all \
