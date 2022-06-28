import json
import torch
import utils.utils as utils
import time
import gc

class Trainer():
    """Trainer"""

    def __init__(self, opt, train_loader, val_loader, model, executor, device):
        self.opt = opt
        self.device = device
        self.reinforce = opt.reinforce
        self.reward_decay = opt.reward_decay
        self.entropy_factor = opt.entropy_factor
        self.num_iters = opt.num_iters
        self.run_dir = opt.run_dir
        self.display_every = opt.display_every
        self.checkpoint_every = opt.checkpoint_every
        self.visualize_training = opt.visualize_training
        if opt.dataset == 'clevr':
            self.vocab = utils.load_vocab(opt.clevr_vocab_path)
        elif opt.dataset == 'clevr-humans':
            self.vocab = utils.load_vocab(opt.human_vocab_path)
        else:
            raise ValueError('Invalid dataset')

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.executor = executor
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.seq2seq.parameters()),
                                          lr=opt.learning_rate)

        self.stats = {
            'train_losses': [],
            'train_batch_accs': [],
            'train_accs_ts': [],
            'val_losses': [],
            'val_accs': [],
            'val_accs_ts': [],
            'best_val_acc': -1,
            'model_t': 0,
            'train_count_reason': [],
            'train_count_error': [],
            'train_count_re':[],
            'train_dir':[],
            'train_dirp':[],
            'train_rp':[],
            'val_count_reason': [],
            'val_count_error': [],
            'val_count_re':[],
            'val_dir' : [],
            'val_dirp':[],
            'val_rp':[],
        }
        if opt.visualize_training:
            from reason.utils.logger import Logger
            self.logger = Logger('%s/logs' % opt.run_dir)

    def train(self):
        training_mode = 'reinforce' if self.reinforce else 'seq2seq'
        print('| start training %s, running in directory %s' % (training_mode, self.run_dir))
        t = 0
        epoch = 0
        baseline = 0
        batches_list_x = []
        batches_list_y = []
        batches_list_ans = []
        batches_list_idx = []
        batches_list_ciidx = []
        batches_list_ct = []
        for x, y, ans, idx, complete_image_idx, constraint_type in self.train_loader:
            x = x.to(device = self.device)
            y = y.to(device = self.device)
            ans = ans.to(device = self.device)
            idx = idx.to(device = self.device)
            complete_image_idx = complete_image_idx.to(device = self.device)
            constraint_type =  constraint_type.to(device = self.device)
            batches_list_x.append(x)
            batches_list_y.append(y)
            batches_list_ans.append(ans)
            batches_list_idx.append(idx)
            batches_list_ciidx.append(complete_image_idx)
            batches_list_ct.append(constraint_type)
        """
        x_all = torch.stack(batches_list_x)
        y_all = torch.stack(batches_list_y)
        ans_all = torch.stack(batches_list_ans)
        idx_all = torch.stack(batches_list_idx)
        ciidx_all = torch.stack(batches_list_ciidx)
        ct_all = torch.stack(batches_list_ct)

        print(x_all.get_device())
        input('--------------------')
        """
        val_batches_list_x = []
        val_batches_list_y = []
        val_batches_list_ans = []
        val_batches_list_idx = []
        val_batches_list_ciidx = []
        val_batches_list_ct = []
        for x, y, ans, idx, complete_image_idx, constraint_type in self.val_loader:
            x = x.to(device = self.device)
            y = y.to(device = self.device)
            ans = ans.to(device = self.device)
            idx = idx.to(device = self.device)
            complete_image_idx = complete_image_idx.to(device = self.device)
            constraint_type =  constraint_type.to(device = self.device)
            val_batches_list_x.append(x)
            val_batches_list_y.append(y)
            val_batches_list_ans.append(ans)
            val_batches_list_idx.append(idx)
            val_batches_list_ciidx.append(complete_image_idx)
            val_batches_list_ct.append(constraint_type)
         
        while t < self.num_iters:
            ts = time.time()
            epoch += 1
            #print("Loading train..:")
            #for x, y, ans, idx, complete_image_idx, constraint_type in self.train_loader:
            for x, y, ans, idx, complete_image_idx, constraint_type in zip(batches_list_x, batches_list_y, batches_list_ans, batches_list_idx, batches_list_ciidx, batches_list_ct):
                #torch.cuda.empty_cache()
                """
                x = x.to(device = self.device)
                y = y.to(device = self.device)
                ans.to(device = self.device)
                idx.to(device = self.device)
                complete_image_idx = complete_image_idx.to(device = self.device)
                constraint_type =  constraint_type.to(device = self.device)
                """

                t += 1
                loss, reward = None, None
                self.model.set_input(x, y)
                self.optimizer.zero_grad()
                if self.reinforce:
                    
                    pred = self.model.reinforce_forward()
                    #print('TIME BEFORE GET_BATCH_REWARD:  ', time.time())                    
                    reward, count_reason, count_error, count_re, count_dir, count_dirp, count_rp = self.get_batch_reward(pred, ans, idx, complete_image_idx, constraint_type, 'train')
                    #print('TIME AFTER GET_BATCH_REWARD:  ', time.time())                    
                    #print("Loss of bactch in:", (time.time()-ts))
                    ##reward = self.get_batch_reward(pred, ans, idx, 'train')
                    print("Reward ="+str(reward)+" at epoch:", epoch)
                    baseline = reward * (1 - self.reward_decay) + baseline * self.reward_decay
                    advantage = reward - baseline
                    self.model.set_reward(advantage)
                    #print('TIME BEFORE BACKWARD:  ', time.time())                    
                    loss = self.model.reinforce_backward(self.entropy_factor)
                    #print('TIME AFTER BACKWARD:  ', time.time())                    
                else:
                    loss = self.model.supervised_forward()
                    print("Loss ="+str(loss)+" at epoch:", epoch)
                    self.model.supervised_backward()
                self.optimizer.step()

                if t % self.display_every == 0:
                    if self.reinforce:
                        self.stats['train_batch_accs'].append(reward)
                        self.stats['train_count_reason'].append(count_reason)
                        self.stats['train_count_error'].append(count_error)
                        self.stats['train_count_re'].append(count_re)
                        self.stats['train_dir'].append(count_dir)
                        self.stats['train_dirp'].append(count_dirp)
                        self.stats['train_rp'].append(count_rp)
                        self.stats['train_losses'].append(loss)
                        self.log_stats('training batch reward', reward, t)
                        print('| iteration %d / %d, epoch %d, reward %f' % (t, self.num_iters, epoch, reward))
                    else:
                        self.stats['train_losses'].append(loss)
                        self.log_stats('training batch loss', loss, t)
                        print('| iteration %d / %d, epoch %d, loss %f' % (t, self.num_iters, epoch, loss))
                    self.stats['train_accs_ts'].append(t)

                if t % self.checkpoint_every == 0 or t >= self.num_iters:
                    print('| checking validation accuracy')
                    #print('TIME BEFORE VALIDATION CHECK:  ', time.time())                    
                    val_acc, count_reason, count_error, count_re, count_dir, count_dirp, count_rp = self.check_val_accuracy(val_batches_list_x, val_batches_list_y, val_batches_list_ans, val_batches_list_idx, val_batches_list_ciidx, val_batches_list_ct)
                    #print('TIME AFTER VALIDATION CHECK:  ', time.time())                    
                    print('| validation accuracy %f' % val_acc)
                    if val_acc >= self.stats['best_val_acc']:
                        print('| best model')
                        self.stats['best_val_acc'] = val_acc
                        self.stats['model_t'] = t
                        self.model.save_checkpoint('%s/checkpoint_best.pt' % self.run_dir)
                        #self.model.save_checkpoint('%s/checkpoint_iter%08d.pt' % (self.run_dir, t))
                    if not self.reinforce:
                        val_loss= self.check_val_loss()
                        print('| validation loss %f' % val_loss)
                        self.stats['val_losses'].append(val_loss)
                        self.log_stats('val loss', val_loss, t)
                    #print('TIME BEFORE STATS:  ', time.time())                        
                    self.stats['val_accs'].append(val_acc)
                    self.stats['val_count_reason'].append(count_reason)
                    self.stats['val_count_error'].append(count_error)
                    self.stats['val_count_re'].append(count_re)
                    self.stats['val_dir'].append(count_dir)
                    self.stats['val_dirp'].append(count_dirp)
                    self.stats['val_rp'].append(count_rp)
                    self.log_stats('val accuracy', val_acc, t)
                    self.stats['val_accs_ts'].append(t)
                    self.model.save_checkpoint('%s/checkpoint.pt' % self.run_dir)
                    with open('%s/stats.json' % self.run_dir, 'w') as fout:
                        json.dump(self.stats, fout)
                    self.log_params(t)
                    #print('TIME AFTER STATS:  ', time.time())                    
               
                if t >= self.num_iters:
                    break
            #print('Time taken for an epoch:::', (time.time()-ts))
          
    def check_val_loss(self):
        loss = 0
        t = 0
        #question, program, answer, image_idx, complete_image, constraint_type
        for x, y, _, _ ,_,_ in self.val_loader:
            self.model.set_input(x, y)
            loss += self.model.supervised_forward()
            t += 1
        return loss / t if t is not 0 else 0

    def check_val_accuracy(self, val_batches_list_x, val_batches_list_y, val_batches_list_ans, val_batches_list_idx, val_batches_list_ciidx, val_batches_list_ct):
        reward = 0
        count_reason = 0
        count_error = 0
        count_re = 0
        count_dir = 0
        count_dirp = 0
        count_rp = 0
        t = 0
        
        #for x, y, ans, idx, complete_image_idx, constraint_type in self.val_loader:
        for x, y, ans, idx, complete_image_idx, constraint_type in zip(val_batches_list_x, val_batches_list_y, val_batches_list_ans, val_batches_list_idx, val_batches_list_ciidx, val_batches_list_ct):
            """
            x = x.to(device = self.device)
            y.to(device = self.device)
            ans = ans.to(device = self.device)
            idx  = idx.to(device = self.device)
            complete_image_idx = complete_image_idx.to(device = self.device)
            constraint_type =  constraint_type.to(device = self.device)
            """
            self.model.set_input(x, y)
            pred = self.model.parse()
            #print("Pred_val:", pred)
            reward1, r, e, re, d, dp, rp = self.get_batch_reward(pred, ans, idx, complete_image_idx, constraint_type, 'val')
            reward += reward1
            count_reason += r
            count_error += e
            count_re += re
            count_dir += d
            count_dirp += dp 
            count_rp += rp 
            ##reward += self.get_batch_reward(pred, ans, idx, 'val')
            t += 1
        reward = reward / t if t is not 0 else 0
        return reward, count_reason, count_error, count_re, count_dir, count_dirp, count_rp 

    def get_batch_reward(self, programs, answers, image_idxs, complete_image_idxs, constraint_type, split):
    ##def get_batch_reward(self, programs, answers, image_idxs, split):
        pg_np = programs.cpu().detach().numpy()
        ans_np = answers.cpu().detach().numpy()
        idx_np = image_idxs.cpu().detach().numpy()
        ct_np = constraint_type.cpu().detach().numpy()
        cs_np = complete_image_idxs.cpu().detach().numpy()
        """
        pg_np = programs._to_numpy()
        ans_np = answers._to_numpy()
        idx_np = image_idxs._to_numpy()
        ct_np = constraint_type._to_numpy()
        cs_np = complete_image_idxs._to_numpy()
        """
        reward = 0
        #print("idx_np:",idx_np)
        count_reason = 0
        count_error = 0
        count_re = 0
        count_dir = 0
        count_dirp = 0
        count_rp = 0
        #for i in range(pg_np.shape[0]):
        for i in range(len(pg_np)):
            pred, r, e, re, d, g = self.executor.run(pg_np[i], idx_np[i], cs_np[i], ct_np[i], split)
            count_reason += r
            count_error += e
            count_re += re 
            count_dir += d
            ##pred = self.executor.run(pg_np[i], idx_np[i], split)
            ans = self.vocab['answer_idx_to_token'][ans_np[i]]
            if pred == ans:
                reward += 1.0
                if d == 1:
                  count_dirp += 1
                elif r == 1:
                  count_rp += 1

        reward /= pg_np.shape[0]
        #reward /= len(pg_np)
        #print(split,'::Count-reason', count_reason)
        #print(split,'::Count-error', count_error)
        #print(split,'::Count-re', count_re)
        #print('---------------------------------------')
        return reward, count_reason, count_error, count_re, count_dir, count_dirp, count_rp

    def log_stats(self, tag, value, t):
        if self.visualize_training and self.logger is not None:
            self.logger.scalar_summary(tag, value, t)

    def log_params(self, t):
        if self.visualize_training and self.logger is not None:
            for tag, value in self.model.seq2seq.named_parameters():
                tag = tag.replace('.', '/')
                self.logger.histo_summary(tag, self._to_numpy(value), t)
                if value.grad is not None:
                    self.logger.histo_summary('%s/grad' % tag, self._to_numpy(value.grad), t)

    def _to_numpy(self, x):
        return x.data.cpu().numpy()
