import torch
import random
from networks.modules import *
from networks.transformer import *
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
# import tensorflow as tf
from collections import OrderedDict
from utils.utils import *
from os.path import join as pjoin
from data.dataset import collate_fn
import codecs as cs
from torch.utils.tensorboard import SummaryWriter

class Logger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, global_step=step)
        self.writer.flush()

class Trainer(object):


    @staticmethod
    def zero_grad(opt_list):
        for opt in opt_list:
            opt.zero_grad()

    @staticmethod
    def clip_norm(network_list):
        for network in network_list:
            clip_grad_norm_(network.parameters(), 0.5)

    @staticmethod
    def step(opt_list):
        for opt in opt_list:
            opt.step()

    def ones_like(self, tensor, val=1.):
        return torch.FloatTensor(tensor.size()).fill_(val).to(self.opt.gpu_id)

    # @staticmethod
    def zeros_like(self, tensor, val=0.):
        return torch.FloatTensor(tensor.size()).fill_(val).to(self.opt.gpu_id)

    def forward(self, batch_data):
        pass

    def backward(self):
        pass

    def update(self):
        pass

class TransformerT2MTrainer(Trainer):
    def __init__(self, args, t2m_transformer):
        self.opt = args
        self.t2m_transformer = t2m_transformer
        # self.quantizer = quantizer
        # self.vq_decoder = vq_decoder
        self.device = args.device

        # self.trg_pad_index = args.trg_pad_index
        # self.trg_start_index = args.trg_start_index
        # self.trg_end_index = args.trg_end_index
        # self.trg_num_vocab = args.trg_num_vocab

        if args.is_train:
            self.logger = Logger(args.log_dir)

    def forward(self, batch_data):
        word_emb, word_tokens, caption, cap_lens, m_tokens, _ = batch_data
        word_emb = word_emb.detach().to(self.device).float()
        # pos_ohot = pos_ohot.detach().to(self.device).float()
        m_tokens = m_tokens.detach().to(self.device).long()
        word_tokens = word_tokens.detach().to(self.device).long()

        self.cap_lens = cap_lens
        self.caption = caption

        trg_input, self.gold = m_tokens[:, :-1], m_tokens[:, 1:]

        if self.opt.t2m_v2:
            self.trg_pred = self.t2m_transformer(word_tokens, trg_input)
        else:
            self.trg_pred = self.t2m_transformer(word_emb, trg_input, cap_lens)

        # one_hot_indices = F.one_hot(encoding_indices, num_classes=self.args.trg_num_vocab)

    def backward(self):
        # print(self.trg_pred.shape, self.gold.shape)
        trg_pred = self.trg_pred.view(-1, self.trg_pred.shape[-1]).clone()
        # print(trg_pred[0])
        gold = self.gold.contiguous().view(-1).clone()
        self.loss, self.pred_seq, self.n_correct, self.n_word = cal_performance(trg_pred, gold, self.opt.mot_pad_idx,
                                                            smoothing=self.opt.label_smoothing)
        # print(gold, self.pred_seq)
        # self.loss = loss / n_word
        loss_logs = OrderedDict({})
        loss_logs['loss'] = self.loss.item() / self.n_word
        loss_logs['accuracy'] = self.n_correct / self.n_word

        return loss_logs

    def update(self):
        self.zero_grad([self.opt_t2m_transformer])
        # time2_0 = time.time()
        # print("\t\t Zero Grad:%5f" % (time2_0 - time1))
        loss_logs = self.backward()
        self.loss.backward()

        # time2_3 = time.time()
        # print("\t\t Clip Norm :%5f" % (time2_3 - time2_2))
        self.step([self.opt_t2m_transformer])

        return loss_logs

    def save(self, file_name, ep, total_it):

        state = {
            't2m_transformer': self.t2m_transformer.state_dict(),

            'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),

            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.t2m_transformer.load_state_dict(checkpoint['t2m_transformer'])

        self.opt_t2m_transformer.load_state_dict(checkpoint['opt_t2m_transformer'])
        # if self.opt.use_gan:
        #     self.discriminator.load_state_dict(checkpoint['discriminator'])
        #     self.opt_discriminator.load_state_dict(checkpoint['opt_discriminator'])
        return checkpoint['ep'], checkpoint['total_it']


    def train(self, train_dataloader, val_dataloader, plot_eval):
        self.t2m_transformer.to(self.device)
        # self.vq_decoder.to(self.device)
        # self.quantizer.to(self.device)

        self.opt_t2m_transformer = optim.Adam(self.t2m_transformer.parameters(), lr=self.opt.lr)


        epoch = 0
        it = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_dataloader)
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_dataloader), len(val_dataloader)))
        val_loss = 0
        val_accuracy = 0
        min_val_loss = np.inf
        min_val_epoch = epoch
        logs = OrderedDict()
        while epoch < self.opt.max_epoch:
            for i, batch_data in enumerate(train_dataloader):
                self.t2m_transformer.train()

                self.forward(batch_data)

                log_dict = self.update()
                # continue
                # time3 = time.time()
                # print('Update Time: %.5f s' % (time3 - time2))
                # time0 = time3
                for k, v in log_dict.items():
                    if k not in logs:
                        logs[k] = v
                    else:
                        logs[k] += v

                it += 1
                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict({'val_loss': val_loss, 'val_accuracy':val_accuracy})
                    self.logger.scalar_summary('val_loss', val_loss, it)
                    self.logger.scalar_summary('val_accuracy', val_accuracy, it)

                    for tag, value in logs.items():
                        self.logger.scalar_summary(tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = OrderedDict()
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch, i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            epoch += 1
            if epoch % self.opt.save_every_e == 0:
                self.save(pjoin(self.opt.model_dir, 'E%04d.tar' % (epoch)), epoch, total_it=it)

            print('Validation time:')

            val_loss = 0
            val_accuracy = 0
            with torch.no_grad():
                for i, batch_data in enumerate(val_dataloader):
                    self.forward(batch_data)
                    self.backward()
                    val_loss += self.loss.item() / self.n_word
                    val_accuracy += self.n_correct / self.n_word
                    # val_loss_rec += self.l1_criterion(self.recon_motions, self.motions).item()
                    # val_loss_emb += self.embedding_loss.item()

            val_loss = val_loss / len(val_dataloader)
            val_accuracy = val_accuracy / len(val_dataloader)
            # val_loss = val_loss / (len(val_dataloader) + 1)
            # val_loss_rec = val_loss_rec / (len(val_dataloader) + 1)
            # val_loss_emb = val_loss_emb / (len(val_dataloader) + 1)
            print(self.gold[0])
            print(self.pred_seq.view(self.gold.shape)[0])

            print('Validation Loss: %.5f Validation Accuracy: %.4f' % (val_loss, val_accuracy))

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                min_val_epoch = epoch
                self.save(pjoin(self.opt.model_dir, 'finest.tar'), epoch, it)
                print('Best Validation Model So Far!~')

            # if epoch % self.opt.eval_every_e == 0:
                # self.quantizer.eval()
                # self.vq_decoder.eval()
                # with torch.no_grad():
                #     pred_seq = self.pred_seq.view(self.gold.shape)[0:1]
                #     # print(pred_seq.shape)
                #     non_pad_mask = self.gold[0:1].ne(self.opt.trg_pad_idx)
                #     pred_seq = pred_seq.masked_select(non_pad_mask).unsqueeze(0)
                #     # print(non_pad_mask.shape)
                #     # print(pred_seq.shape)
                #     # print(pred_seq)
                #     # print(self.gold[0:1])
                #     vq_latent = self.quantizer.get_codebook_entry(pred_seq)
                #     # print(vq_latent.shape)
                #
                #     rec_motion = self.vq_decoder(vq_latent)
                #
                # save_dir = pjoin(self.opt.eval_dir, 'E%04d' % (epoch))
                # os.makedirs(save_dir, exist_ok=True)
                # plot_eval(rec_motion.detach().cpu().numpy(), self.caption[0:1], save_dir)
                # save_dir = pjoin(self.opt.eval_dir, 'E%04d' % epoch)
                # os.makedirs()

            if epoch - min_val_epoch >= 5:
                print('Early Stopping!~')
                # break

class TransformerM2TTrainer(Trainer):
    def __init__(self, args, m2t_transformer):
        self.opt = args
        self.m2t_transformer = m2t_transformer
        # self.quantizer = quantizer
        # self.vq_decoder = vq_decoder
        self.device = args.device

        # self.trg_pad_index = args.trg_pad_index
        # self.trg_start_index = args.trg_start_index
        # self.trg_end_index = args.trg_end_index
        # self.trg_num_vocab = args.trg_num_vocab

        if args.is_train:
            self.logger = Logger(args.log_dir)

    def forward(self, batch_data):
        word_emb, word_tokens, caption, cap_lens, m_tokens, _ = batch_data
        word_emb = word_emb.detach().to(self.device).float()
        # pos_ohot = pos_ohot.detach().to(self.device).float()
        m_tokens = m_tokens.detach().to(self.device).long()
        word_tokens = word_tokens.detach().to(self.device).long()

        cap_lens = cap_lens - 1
        self.cap_lens = cap_lens
        self.caption = caption

        self.gold = word_tokens[:, 1:]

        """Input pretrained word vector"""
        if self.opt.m2t_v3:
            trg_input = word_emb[:, :-1]
            self.trg_pred = self.m2t_transformer(m_tokens, trg_input, cap_lens)
        else:
            trg_input = word_tokens[:, :-1]
            self.trg_pred = self.m2t_transformer(m_tokens, trg_input)

        # one_hot_indices = F.one_hot(encoding_indices, num_classes=self.args.trg_num_vocab)

    def backward(self):
        # print(self.trg_pred.shape, self.gold.shape)
        trg_pred = self.trg_pred.view(-1, self.trg_pred.shape[-1]).clone()
        # print(trg_pred[0])
        gold = self.gold.contiguous().view(-1).clone()
        self.loss, self.pred_seq, self.n_correct, self.n_word = cal_performance(trg_pred, gold, self.opt.txt_pad_idx,
                                                            smoothing=self.opt.label_smoothing)
        # print(gold, self.pred_seq)
        # self.loss = loss / n_word
        loss_logs = OrderedDict({})
        loss_logs['loss'] = self.loss.item() / self.n_word
        loss_logs['accuracy'] = self.n_correct / self.n_word

        return loss_logs

    def update(self):
        self.zero_grad([self.opt_m2t_transformer])
        # time2_0 = time.time()
        # print("\t\t Zero Grad:%5f" % (time2_0 - time1))
        loss_logs = self.backward()
        self.loss.backward()

        # time2_3 = time.time()
        # print("\t\t Clip Norm :%5f" % (time2_3 - time2_2))
        self.step([self.opt_m2t_transformer])

        return loss_logs

    def save(self, file_name, ep, total_it):

        state = {
            'm2t_transformer': self.m2t_transformer.state_dict(),

            'opt_m2t_transformer': self.opt_m2t_transformer.state_dict(),

            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.m2t_transformer.load_state_dict(checkpoint['m2t_transformer'], strict=False)

        self.opt_m2t_transformer.load_state_dict(checkpoint['opt_m2t_transformer'], strict=False)
        # if self.opt.use_gan:
        #     self.discriminator.load_state_dict(checkpoint['discriminator'])
        #     self.opt_discriminator.load_state_dict(checkpoint['opt_discriminator'])
        print('--------Model Weights Loaded--------')
        return checkpoint['ep'], checkpoint['total_it']


    def train(self, train_dataloader, val_dataloader, w_vectorizer):
        self.m2t_transformer.to(self.device)
        # self.vq_decoder.to(self.device)
        # self.quantizer.to(self.device)

        self.opt_m2t_transformer = optim.Adam(self.m2t_transformer.parameters(), lr=self.opt.lr)

        epoch = 0
        it = 0
        if self.opt.is_continue:
            # model_dir = pjoin(self.opt.model_dir, 'latest.tar') # Train and test with the same dataset
            # epoch, it = self.resume(model_dir)
            model_dir = pjoin('./checkpoints/t2m/M2T_EL4_DL4_NH8_PS/model', 'finest.tar') # Finetune from T2M dataset
            epoch, it = self.resume(model_dir)
            epoch, it = 0, 0
            
        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_dataloader)
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_dataloader), len(val_dataloader)))
        val_loss = 0
        val_accuracy = 0
        min_val_loss = np.inf
        min_val_epoch = epoch
        logs = OrderedDict()
        while epoch < self.opt.max_epoch:
            for i, batch_data in enumerate(train_dataloader):
                self.m2t_transformer.train()

                self.forward(batch_data)

                log_dict = self.update()
                # continue
                # time3 = time.time()
                # print('Update Time: %.5f s' % (time3 - time2))
                # time0 = time3
                for k, v in log_dict.items():
                    if k not in logs:
                        logs[k] = v
                    else:
                        logs[k] += v

                it += 1
                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict({'val_loss': val_loss, 'val_accuracy':val_accuracy})
                    self.logger.scalar_summary('val_loss', val_loss, it)
                    self.logger.scalar_summary('val_accuracy', val_accuracy, it)

                    for tag, value in logs.items():
                        self.logger.scalar_summary(tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = OrderedDict()
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch, i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            epoch += 1
            if epoch % self.opt.save_every_e == 0:
                self.save(pjoin(self.opt.model_dir, 'E%04d.tar' % (epoch)), epoch, total_it=it)

            print('Validation time:')

            val_loss = 0
            val_accuracy = 0
            with torch.no_grad():
                for i, batch_data in enumerate(val_dataloader):
                    self.forward(batch_data)
                    self.backward()
                    val_loss += self.loss.item() / self.n_word
                    val_accuracy += self.n_correct / self.n_word
                    # val_loss_rec += self.l1_criterion(self.recon_motions, self.motions).item()
                    # val_loss_emb += self.embedding_loss.item()

            val_loss = val_loss / len(val_dataloader)
            val_accuracy = val_accuracy / len(val_dataloader)
            # val_loss = val_loss / (len(val_dataloader) + 1)
            # val_loss_rec = val_loss_rec / (len(val_dataloader) + 1)
            # val_loss_emb = val_loss_emb / (len(val_dataloader) + 1)

            print('Validation Loss: %.5f Validation Accuracy: %.4f' % (val_loss, val_accuracy))

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                min_val_epoch = epoch
                self.save(pjoin(self.opt.model_dir, 'finest.tar'), epoch, it)
                print('Best Validation Model So Far!~')

            print(self.gold[0])
            print(self.pred_seq.view(self.gold.shape)[0])
            gt_seq = ' '.join(w_vectorizer.itos(i) for i in self.gold[0, :self.cap_lens[0]].cpu().numpy())
            pred_seq = ' '.join(
                w_vectorizer.itos(i) for i in self.pred_seq.view(self.gold.shape)[0, :self.cap_lens[0]].cpu().numpy())
            print(gt_seq)
            print(pred_seq)
            # if epoch % self.opt.eval_every_e == 0:
                # self.quantizer.eval()
                # self.vq_decoder.eval()
                # with torch.no_grad():
                #     pred_seq = self.pred_seq.view(self.gold.shape)[0:1]
                #     # print(pred_seq.shape)
                #     non_pad_mask = self.gold[0:1].ne(self.opt.trg_pad_idx)
                #     pred_seq = pred_seq.masked_select(non_pad_mask).unsqueeze(0)
                #     # print(non_pad_mask.shape)
                #     # print(pred_seq.shape)
                #     # print(pred_seq)
                #     # print(self.gold[0:1])
                #     vq_latent = self.quantizer.get_codebook_entry(pred_seq)
                #     # print(vq_latent.shape)
                #
                #     rec_motion = self.vq_decoder(vq_latent)
                #
                # save_dir = pjoin(self.opt.eval_dir, 'E%04d' % (epoch))
                # os.makedirs(save_dir, exist_ok=True)
                # plot_eval(rec_motion.detach().cpu().numpy(), self.caption[0:1], save_dir)
                # save_dir = pjoin(self.opt.eval_dir, 'E%04d' % epoch)
                # os.makedirs()

            if epoch - min_val_epoch >= 5:
                print('Early Stopping!~')
                # break

class VQTokenizerTrainerV3(Trainer):
    def __init__(self, args, vq_encoder, quantizer, vq_decoder, discriminator=None):
        self.opt = args
        self.vq_encoder = vq_encoder
        self.vq_decoder = vq_decoder
        self.quantizer = quantizer
        # self.mov_encoder = mov_encoder
        self.discriminator = discriminator
        self.device = args.device

        if args.is_train:
            self.logger = Logger(args.log_dir)
            self.l1_criterion = torch.nn.L1Loss()
            self.gan_criterion = torch.nn.BCEWithLogitsLoss()
            self.disc_loss = self.hinge_d_loss

    def hinge_d_loss(self, logits_real, logits_fake):
        loss_real = torch.mean(F.relu(1. - logits_real))
        loss_fake = torch.mean(F.relu(1. + logits_fake))
        d_loss = 0.5 * (loss_real + loss_fake)
        return d_loss

    # def ones_like(self, tensor, val=1.):
    #     return torch.FloatTensor(tensor.size()).fill_(val).to(self.opt.gpu_id)
    #
    # # @staticmethod
    # def zeros_like(self, tensor, val=0.):
    #     return torch.FloatTensor(tensor.size()).fill_(val).to(self.opt.gpu_id)

    def forward(self, batch_data):
        motions = batch_data
        self.motions = motions.detach().to(self.device).float()
        # print(self.motions.shape)
        self.pre_latents = self.vq_encoder(self.motions[..., :-4])
        # print(self.pre_latents.shape)
        self.embedding_loss, self.vq_latents, _, self.perplexity = self.quantizer(self.pre_latents)
        # print(self.vq_latents.shape)
        self.recon_motions = self.vq_decoder(self.vq_latents)

    def calculate_adaptive_weight(self, rec_loss, gan_loss, last_layer):
        rec_grads = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
        gan_grads = torch.autograd.grad(gan_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(rec_grads) / (torch.norm(gan_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.opt.lambda_adv
        return d_weight


    def backward_G(self):
        self.loss_rec_mot = self.l1_criterion(self.recon_motions, self.motions)
        self.loss_G = self.loss_rec_mot + self.embedding_loss

        if self.opt.start_use_gan:
            _, logits_fake = self.discriminator(self.recon_motions)
            self.loss_G_adv = -torch.mean(logits_fake)
            # last_layer = self.vq_decoder.main[9].weight
            #
            # try:
            #     self.d_weight = self.calculate_adaptive_weight(self.loss_rec_mot, self.loss_G_adv, last_layer=last_layer)
            # except RuntimeError:
            #     assert not self.opt.is_train
            #     self.d_weight = torch.tensor(0.0)
            # self.loss_G += self.d_weight * self.loss_G_adv
            self.loss_G += self.opt.lambda_adv * self.loss_G_adv


    def backward_D(self):
        self.real_feats, real_labels = self.discriminator(self.motions.detach())
        fake_feats, fake_labels = self.discriminator(self.recon_motions.detach())

        self.loss_D = self.disc_loss(real_labels, fake_labels) * self.opt.lambda_adv
        # self.loss_D = (self.loss_D_T + self.loss_D_F) * self.opt.lambda_adv


    def update(self):
        loss_logs = OrderedDict({})

        if self.opt.start_use_gan:
            self.zero_grad([self.opt_discriminator])
            self.backward_D()
            self.loss_D.backward(retain_graph=True)
            self.step([self.opt_discriminator])

        self.zero_grad([self.opt_vq_encoder, self.opt_quantizer, self.opt_vq_decoder])
        self.backward_G()
        self.loss_G.backward()
        self.step([self.opt_vq_encoder, self.opt_quantizer, self.opt_vq_decoder])

        loss_logs['loss_G'] = self.loss_G.item()
        loss_logs['loss_G_rec_mot'] = self.loss_rec_mot.item()
        loss_logs['loss_G_emb'] = self.embedding_loss.item()
        loss_logs['perplexity'] = self.perplexity.item()

        if self.opt.start_use_gan:
            # loss_logs['d_weight'] = self.d_weight.item()
            loss_logs['loss_G_adv'] = self.loss_G_adv.item()
            loss_logs['loss_D'] = self.loss_D.item()

        return loss_logs

    def save(self, file_name, ep, total_it):
        state = {
            'vq_encoder': self.vq_encoder.state_dict(),
            'quantizer': self.quantizer.state_dict(),
            'vq_decoder': self.vq_decoder.state_dict(),
            'discriminator': self.discriminator.state_dict(),

            'opt_vq_encoder': self.opt_vq_encoder.state_dict(),
            'opt_quantizer': self.opt_quantizer.state_dict(),
            'opt_vq_decoder': self.opt_vq_decoder.state_dict(),
            'opt_discriminator': self.opt_discriminator.state_dict(),

            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.vq_encoder.load_state_dict(checkpoint['vq_encoder'])
        self.quantizer.load_state_dict(checkpoint['quantizer'])
        self.vq_decoder.load_state_dict(checkpoint['vq_decoder'])

        self.opt_vq_encoder.load_state_dict(checkpoint['opt_vq_encoder'])
        self.opt_quantizer.load_state_dict(checkpoint['opt_quantizer'])
        self.opt_vq_decoder.load_state_dict(checkpoint['opt_vq_decoder'])

        # if self.opt.use_gan:
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.opt_discriminator.load_state_dict(checkpoint['opt_discriminator'])
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_dataloader, val_dataloader, plot_eval):
        self.vq_encoder.to(self.device)
        self.quantizer.to(self.device)
        self.vq_decoder.to(self.device)
        self.discriminator.to(self.device)

        self.opt_vq_encoder = optim.Adam(self.vq_encoder.parameters(), lr=self.opt.lr)
        self.opt_quantizer = optim.Adam(self.quantizer.parameters(), lr=self.opt.lr)
        self.opt_vq_decoder = optim.Adam(self.vq_decoder.parameters(), lr=self.opt.lr)
        self.opt_discriminator = optim.Adam(self.discriminator.parameters(), lr=self.opt.lr)

        epoch = 0
        it = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)
            print("Load model epoch:%d iterations:%d"%(epoch, it))

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_dataloader)
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_dataloader), len(val_dataloader)))
        val_loss = 0
        min_val_loss = np.inf
        min_val_epoch = epoch
        logs = OrderedDict()
        while epoch < self.opt.max_epoch:
            self.opt.start_use_gan = (epoch >= self.opt.start_dis_epoch)
            for i, batch_data in enumerate(train_dataloader):
                self.vq_encoder.train()
                self.quantizer.train()
                self.vq_decoder.train()
                # if self.opt.use_percep:
                #     self.mov_encoder.train()
                if self.opt.start_use_gan:
                    # print('Introducing Adversarial Loss!~')
                    self.discriminator.train()

                self.forward(batch_data)

                log_dict = self.update()
                # time3 = time.time()
                # print('Update Time: %.5f s' % (time3 - time2))
                # time0 = time3
                for k, v in log_dict.items():
                    if k not in logs:
                        logs[k] = v
                    else:
                        logs[k] += v

                it += 1
                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict({'val_loss': val_loss})
                    self.logger.scalar_summary('val_loss', val_loss, it)

                    for tag, value in logs.items():
                        self.logger.scalar_summary(tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = OrderedDict()
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch, i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            epoch += 1
            if epoch % self.opt.save_every_e == 0:
                self.save(pjoin(self.opt.model_dir, 'E%04d.tar' % (epoch)), epoch, total_it=it)

            print('Validation time:')

            val_loss_rec = 0
            val_loss_emb = 0
            with torch.no_grad():
                for i, batch_data in enumerate(val_dataloader):
                    self.forward(batch_data)
                    val_loss_rec += self.l1_criterion(self.recon_motions, self.motions).item()
                    # val_loss_emb += self.embedding_loss.item()

            val_loss = val_loss_rec / (len(val_dataloader) + 1)
            # val_loss = val_loss / (len(val_dataloader) + 1)
            # val_loss_rec = val_loss_rec / (len(val_dataloader) + 1)
            # val_loss_emb = val_loss_emb / (len(val_dataloader) + 1)

            print('Validation Loss: %.5f' % (val_loss))

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                min_val_epoch = epoch
                self.save(pjoin(self.opt.model_dir, 'finest.tar'), epoch, it)
                print('Best Validation Model So Far!~')

            if epoch % self.opt.eval_every_e == 0:
                data = torch.cat([self.recon_motions[:4], self.motions[:4]], dim=0).detach().cpu().numpy()
                save_dir = pjoin(self.opt.eval_dir, 'E%04d' % (epoch))
                os.makedirs(save_dir, exist_ok=True)
                plot_eval(data, save_dir)

            if epoch - min_val_epoch >= 5:
                print('Early Stopping!~')
                # break