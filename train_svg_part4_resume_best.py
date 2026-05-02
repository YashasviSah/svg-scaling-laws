
out_dir = 'out-svg-mup-small-final'
init_from = 'resume'

dataset = 'svg'
vocab_size = 4096

block_size = 512

batch_size = 8
gradient_accumulation_steps = 4

learning_rate = 0.01
min_lr = 0.001
max_iters = 26612
lr_decay_iters = 26612
warmup_iters = 100
beta2 = 0.95
dropout = 0.1

eval_interval = 500
eval_iters = 100
log_interval = 50

always_save_checkpoint = True
wandb_log = False

track_metrics = True
metrics_log_file = 'metrics.csv'
