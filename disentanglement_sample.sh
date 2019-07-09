name='SEG10_BG10L1_birds'
epoch='500'
device=2

CUDA_VISIBLE_DEVICES=${device} python evaluation/disentanglement/disentanglement_sampler.py --model_name ${name} \
                                                            --load_from_epoch ${epoch} \
