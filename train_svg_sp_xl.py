
out_dir = 'out-svg-sp-xl'

eval_interval = 500
eval_iters = 100
log_interval = 50

dataset = 'svg'
vocab_size = 4096

gradient_accumulation_steps = 32
batch_size = 1
block_size = 512

n_layer = 12
n_head = 12
n_embd = 768
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
