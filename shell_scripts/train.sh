python main.py -mode train\
    -cfg  lr=1e-4\
    gradient_accumulation_steps=4 batch_size=8\
    epoch_num=50\
    cuda_device=$1\