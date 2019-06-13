name='model1'
CUDA_VISIBLE_DEVICES="1" python Gan/test_worker.py \
                                    --dataset birds \
                                    --model_name ${name} \
                                    --load_from_epoch 500 \
                                    --test_sample_num 10 \
                                    --save_visual_results \
                                    --batch_size 64 \
                                    # --random_seg_noise
