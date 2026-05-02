
out_dir = 'out-svg-mup-small-final'

eval_interval = 500
eval_iters = 100
log_interval = 50
always_save_checkpoint = False

dataset = 'svg'
vocab_size = 4096
init_from = 'scratch'

gradient_accumulation_steps = 4
batch_size = 8
block_size = 512

n_layer = 6
n_head = 6
n_embd = 192
dropout = 0.1
bias = False

learning_rate = 0.01
max_iters = 6653
lr_decay_iters = 6653
min_lr = 0.001
beta2 = 0.95
warmup_iters = 100

wandb_log = False
compile = False

track_metrics = True
metrics_log_file = 'metrics.csv'
