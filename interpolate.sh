name="model4"
epoch="500"
device=0

# text, shape, background

CUDA_VISIBLE_DEVICES=${device} python sampling/interpolation.py --model ${name} \
                                                                --epoch ${epoch} \
                                                                --nb_interp 10 \
                                                                --interpolate text \
                                                                --fx1_id 3 \
                                                                --fx2_id 0 \
                                                                --mv1_id 1 \
                                                                --mv2_id 2