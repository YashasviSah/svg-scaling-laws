
out_dir = 'out-svg-sp-medium'

eval_interval = 500
eval_iters = 100
log_interval = 50

dataset = 'svg'
vocab_size = 4096

gradient_accumulation_steps = 4
batch_size = 8
block_size = 512

n_layer = 6
n_head = 8
n_embd = 256
dropout = 0.1

learning_rate = 0.003
max_iters = 6653
lr_decay_iters = 6653
min_lr = 0.00030000000000000003
beta2 = 0.95
warmup_iters = 100

always_save_checkpoint = True
wandb_log = False

track_metrics = True
metrics_log_file = 'metrics.csv'
