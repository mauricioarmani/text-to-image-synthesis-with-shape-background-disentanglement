name='neural_dist'
device=0
CUDA_VISIBLE_DEVICES=${device} python train_nd_worker.py --dataset birds --batch_size 32 --model_name ${name} --lr 0.001