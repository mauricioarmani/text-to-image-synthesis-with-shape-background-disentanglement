name="model4"
epoch="550"
device=0

CUDA_VISIBLE_DEVICES=${device} python sampling/sampler.py --model ${name} --epoch ${epoch} --n_samples 1 --batch_size 10