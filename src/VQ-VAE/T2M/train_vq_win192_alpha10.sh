python train_vq.py \
--batch-size 128 \
--lr 2e-4 \
--total-iter 300000 \
--lr-scheduler 200000 \
--nb-code 512 \
--down-t 2 \
--depth 3 \
--dilation-growth-rate 3 \
--out-dir output \
--dataname ubiphysio \
--vq-act relu \
--quantizer ema_reset \
--loss-bio 1.0  \
--recons-loss l1_smooth \
--exp-name VQVAE_win192_alpha10 \
--window-size 192 \
--gpu '1'
# The window size is 64 for 20Hz data previously, now changed to 192 for 60Hz data.