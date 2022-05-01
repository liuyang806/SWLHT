from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from exp.exp_basic import Exp_Basic
from models.model import SWLHT

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

class Exp_SWLHT(Exp_Basic):
    def __init__(self, args):
        super(Exp_SWLHT, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'swlht':SWLHT,
        }
        if self.args.model=='swlht' or self.args.model=='swlhtstack':
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len,
                self.args.num_segmemts,
                self.args.smem_len,
                self.args.lmem_len,
                self.args.mem_layers,
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.device
            )
        
        return model.double()

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ETTh1':Dataset_ETT_hour,
            'ETTh2':Dataset_ETT_hour,
            'ETTm1':Dataset_ETT_minute,
            'ETTm2':Dataset_ETT_minute,
            'Airquality1':Dataset_Custom,
            'Airquality4': Dataset_Custom,
            'Metrology1': Dataset_Custom,
            'Metrology4': Dataset_Custom,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size
        
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=args.freq
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()

        pre_pred_len = int(self.args.pred_len / self.args.num_segmemts)
        split_pred_len = lambda x: x.split(pre_pred_len, dim=1)

        total_loss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
            outputs = []

            batch_x = batch_x.double().to(self.device)
            batch_y = batch_y.double()
            groud_truth = batch_y
            
            batch_x_mark = batch_x_mark.double().to(self.device)
            batch_y_mark = batch_y_mark.double().to(self.device)

            batch_y, batch_y_mark = map(split_pred_len, (batch_y, batch_y_mark))

            # init mem
            tmp_x = batch_x[:, :self.args.seq_len, :]
            tmp_x_mark = batch_x_mark[:, :self.args.seq_len, :]
            tmp_inp = torch.zeros_like(batch_x[:, self.args.label_len: self.args.label_len+pre_pred_len:, :]).double().to(self.device)
            tmp_inp = torch.cat([batch_x[:, :self.args.label_len, :], tmp_inp], dim=1)
            tmp_inp_mark = batch_x_mark[:, :(self.args.label_len + pre_pred_len), :]
            mem = None
            _, mem = self.model(tmp_x, tmp_x_mark, tmp_inp, tmp_inp_mark, memories=mem)
            mems = [mem] * self.args.num_segmemts

            for ind, (batch_y, batch_y_mark) in enumerate(zip(batch_y, batch_y_mark)):
                if ind is 0:
                    mem_in = mems[0]
                else:
                    mem_in = mems[ind-1]

                # encoder input
                batch_x = batch_x[:, -self.args.seq_len:, :]
                batch_x_mark = batch_x_mark[:, -self.args.seq_len:, :]
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -pre_pred_len:, :]).double().to(self.device)
                dec_inp = torch.cat([batch_x[:, -self.args.label_len:, :], dec_inp], dim=1)

                dec_inp_mark = torch.cat([batch_x_mark, batch_y_mark], dim=1)[:, -(self.args.label_len + pre_pred_len):, :]
                # encoder - decoder
                if self.args.output_attention:
                    output, new_mem, _ = self.model(batch_x, batch_x_mark, dec_inp, dec_inp_mark, memories=mem_in)
                    outputs.append(output)
                    # new_mems.append(new_mem)
                else:
                    output, new_mem = self.model(batch_x, batch_x_mark, dec_inp, dec_inp_mark, memories=mem_in)
                    outputs.append(output)
                    # new_mems.append(new_mem)
                # print(outputs.shape) torch.Size([32, 24, 7])
                batch_x = torch.cat([batch_x, output], dim=1)
                batch_x_mark = torch.cat([batch_x_mark, batch_y_mark], dim=1)
                mems[ind] = new_mem

            outputs = torch.cat(outputs, dim=1)
            f_dim = -1 if self.args.features=='MS' else 0
            groud_truth = groud_truth[:,-self.args.pred_len:,f_dim:].to(self.device)

            pred = outputs.detach().cpu()
            true = groud_truth.detach().cpu()

            loss = criterion(pred, true) 

            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
        
    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        path = './checkpoints/'+setting
        if not os.path.exists(path):
            os.makedirs(path)
        
        train_steps = len(train_loader)
        print('train_loader len: ', len(train_loader))

        time_now = time.time()
        print('train starting time: ', time.asctime(time.localtime(time.time())))
        
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        pre_pred_len = int(self.args.pred_len / self.args.num_segmemts)
        split_pred_len = lambda x: x.split(pre_pred_len, dim=1)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
                # print(batch_x.shape)      torch.Size([32, 96, 7])
                # print(batch_x_mark.shape) torch.Size([32, 96, 4])
                # print(batch_y.shape)      torch.Size([32, 48, 7])
                # print(batch_y_mark.shape) torch.Size([32, 48, 4])

                iter_count += 1
                outputs = []

                batch_x = batch_x.double().to(self.device)
                batch_y = batch_y.double()
                groud_truth = batch_y
                
                batch_x_mark = batch_x_mark.double().to(self.device)
                batch_y_mark = batch_y_mark.double().to(self.device)

                batch_y, batch_y_mark = map(split_pred_len, (batch_y, batch_y_mark))

                # init mem
                tmp_x = batch_x[:, :self.args.seq_len, :]
                tmp_x_mark = batch_x_mark[:, :self.args.seq_len, :]
                tmp_inp = torch.zeros_like(
                    batch_x[:, self.args.label_len: self.args.label_len + pre_pred_len:, :]).double().to(self.device)
                tmp_inp = torch.cat([batch_x[:, :self.args.label_len, :], tmp_inp], dim=1)
                tmp_inp_mark = batch_x_mark[:, :(self.args.label_len + pre_pred_len), :]
                mem = None
                _, mem = self.model(tmp_x, tmp_x_mark, tmp_inp, tmp_inp_mark, memories=mem)
                mems = [mem] * self.args.num_segmemts

                for ind, (batch_y, batch_y_mark) in enumerate(zip(batch_y, batch_y_mark)):
                    if ind is 0:
                        mem_in = mems[0]
                    else:
                        mem_in = mems[ind - 1]

                    # encoder input
                    batch_x = batch_x[:, -self.args.seq_len:, :]
                    batch_x_mark = batch_x_mark[:, -self.args.seq_len:, :]
                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -pre_pred_len:, :]).double().to(self.device)
                    dec_inp = torch.cat([batch_x[:, -self.args.label_len:, :], dec_inp], dim=1)

                    dec_inp_mark = torch.cat([batch_x_mark, batch_y_mark], dim=1)[:,
                                   -(self.args.label_len + pre_pred_len):, :]
                    # encoder - decoder
                    if self.args.output_attention:
                        output, new_mem, _ = self.model(batch_x, batch_x_mark, dec_inp, dec_inp_mark, memories=mem_in)
                        outputs.append(output)
                        # new_mems.append(new_mem)
                    else:
                        output, new_mem = self.model(batch_x, batch_x_mark, dec_inp, dec_inp_mark, memories=mem_in)
                        outputs.append(output)
                        # new_mems.append(new_mem)
                    # print(outputs.shape) torch.Size([32, 24, 7])
                    batch_x = torch.cat([batch_x, output], dim=1)
                    batch_x_mark = torch.cat([batch_x_mark, batch_y_mark], dim=1)
                    mems[ind] = new_mem

                outputs = torch.cat(outputs, dim=1)

                model_optim.zero_grad()
                f_dim = -1 if self.args.features=='MS' else 0
                groud_truth = groud_truth[:,-self.args.pred_len:,f_dim:].to(self.device)
                loss = criterion(outputs, groud_truth)
                train_loss.append(loss.item())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                loss.backward()
                model_optim.step()

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
            
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        print('train finishing time: ', time.asctime(time.localtime(time.time())))
        
        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()

        pre_pred_len = int(self.args.pred_len / self.args.num_segmemts)
        split_pred_len = lambda x: x.split(pre_pred_len, dim=1)
        
        preds = []
        trues = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
            outputs = []

            batch_x = batch_x.double().to(self.device)
            batch_y = batch_y.double()
            groud_truth = batch_y
            batch_x_mark = batch_x_mark.double().to(self.device)
            batch_y_mark = batch_y_mark.double().to(self.device)

            batch_y, batch_y_mark = map(split_pred_len, (batch_y, batch_y_mark))

            # init mem
            tmp_x = batch_x[:, :self.args.seq_len, :]
            tmp_x_mark = batch_x_mark[:, :self.args.seq_len, :]
            tmp_inp = torch.zeros_like(
                batch_x[:, self.args.label_len: self.args.label_len + pre_pred_len:, :]).double().to(self.device)
            tmp_inp = torch.cat([batch_x[:, :self.args.label_len, :], tmp_inp], dim=1)
            tmp_inp_mark = batch_x_mark[:, :(self.args.label_len + pre_pred_len), :]
            mem = None
            _, mem = self.model(tmp_x, tmp_x_mark, tmp_inp, tmp_inp_mark, memories=mem)
            mems = [mem] * self.args.num_segmemts

            for ind, (batch_y, batch_y_mark) in enumerate(zip(batch_y, batch_y_mark)):
                if ind is 0:
                    mem_in = mems[0]
                else:
                    mem_in = mems[ind - 1]

                # encoder input
                batch_x = batch_x[:, -self.args.seq_len:, :]
                batch_x_mark = batch_x_mark[:, -self.args.seq_len:, :]
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -pre_pred_len:, :]).double().to(self.device)
                dec_inp = torch.cat([batch_x[:, -self.args.label_len:, :], dec_inp], dim=1)

                dec_inp_mark = torch.cat([batch_x_mark, batch_y_mark], dim=1)[:, -(self.args.label_len + pre_pred_len):,
                               :]
                # encoder - decoder
                if self.args.output_attention:
                    output, new_mem, _ = self.model(batch_x, batch_x_mark, dec_inp, dec_inp_mark, memories=mem_in)
                    outputs.append(output)
                    # new_mems.append(new_mem)
                else:
                    output, new_mem = self.model(batch_x, batch_x_mark, dec_inp, dec_inp_mark, memories=mem_in)
                    outputs.append(output)
                    # new_mems.append(new_mem)
                # print(outputs.shape) torch.Size([32, 24, 7])
                batch_x = torch.cat([batch_x, output], dim=1)
                batch_x_mark = torch.cat([batch_x_mark, batch_y_mark], dim=1)
                mems[ind] = new_mem

            outputs = torch.cat(outputs, dim=1)
            f_dim = -1 if self.args.features=='MS' else 0
            groud_truth = groud_truth[:,-self.args.pred_len:,f_dim:].to(self.device)
            
            pred = outputs.detach().cpu().numpy()#.squeeze()
            true = groud_truth.detach().cpu().numpy()#.squeeze()
            
            preds.append(pred)
            trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path+'metrics', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+'pred', preds)
        np.save(folder_path+'true', trues)

        for i in range(3):
            # B * L * var_num
            pre_dig = preds[i]
            Y_dig = trues[i]

            plt.plot(range(preds.shape[-2]), pre_dig[:, 0], color="red")
            plt.plot(range(trues.shape[-2]), Y_dig[:, 0], color="blue")
            path_dir = folder_path + str(i) + "_zeroB.png"
            plt.savefig(path_dir)  # 默认像素dpi是80
            plt.clf()

        return
