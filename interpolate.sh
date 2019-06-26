name="model4"
epoch="550"
device=0

# text, shape, background

# CUDA_VISIBLE_DEVICES=${device} python sampling/1Dinterpolation.py --model ${name} \
#                                                                   --epoch ${epoch} \
#                                                                   --nb_interp 5 \
#                                                                   --interpolate shape \
#                                                                   # --fx1_id 71 \
#                                                                   # --fx2_id 34 \
#                                                                   # --mv_ida 14 \
#                                                                   # --mv_idb 5

CUDA_VISIBLE_DEVICES=${device} python sampling/2Dinterpolation.py --model ${name} \
                                                                  --epoch ${epoch} \
                                                                  --nb_interp 5 \
                                                                  --fix text \
                                                                  # --fx_id 6 \
                                                                  # --mv1_ida 88 \
                                                                  # --mv1_idb 27 \
                                                                  # --mv2_ida 13 \
                                                                  # --mv2_idb 7

# CUDA_VISIBLE_DEVICES=${device} python sampling/3Dinterpolation.py --model ${name} \
#                                                                   --epoch ${epoch} \
#                                                                   --nb_interp 5 \
#                                                                   # --mv1_ida 70 \
#                                                                   # --mv1_idb 68 \
#                                                                   # --mv2_ida 66 \
#                                                                   # --mv2_idb 71 \
#                                                                   # --mv3_idb 71 \
#                                                                   # --mv3_idb 71                                                               