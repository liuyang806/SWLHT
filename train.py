import argparse
import os
import torch
import time

from exp.exp_swlht import Exp_SWLHT


def count_parameter_amount(model=None):
    for name, parameter in model.named_parameters():
        print(name, ':', parameter.size(), "---", parameter.numel())

    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print({'Total': total_num, 'Trainable': trainable_num})

parser = argparse.ArgumentParser(description='[] Long Sequences Forecasting')

parser.add_argument('--model', type=str, default='swlht',help='model of experiment, options: [swlht, swlhtstack, swlhtlight(TBD)]')
# , required=True
parser.add_argument('--data', type=str, default='ETTh1', help='data')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS(TBD)]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding')

parser.add_argument('--seq_len', type=int, default=48, help='input sequence length of SWLHT encoder')
parser.add_argument('--label_len', type=int, default=32, help='start token length of SWLHT decoder')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--num_segmemts', type=int, default=4, help='divide prediction sequence into segmemts parts')
# decoder input: concat[start token series(label_len), zero padding series(pred_len)]
parser.add_argument('--smem_len', type=int, default=48, help='short memory length')
parser.add_argument('--lmem_len', type=int, default=12, help='long memory length')
parser.add_argument('--mem_layers', type=list, default=[2], help='which layer to use memory')

parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=1024, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder', default=False)
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--attn', type=str, default='full', help='attention used in encoder, options:[prob, full]')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu',help='activation')
parser.add_argument('--output_attention', action='store_true', default=False, help='whether to output attention in ecoder')

parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=3, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=6, help='train epochs')#3
parser.add_argument('--batch_size', type=int, default=36, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=1, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test',help='exp description')
parser.add_argument('--loss', type=str, default='mse',help='loss function')
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() else False

data_parser = {
    'ETTh1':{'data':'ETTh1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTh2':{'data':'ETTh2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm1':{'data':'ETTm1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm2':{'data':'ETTm2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
}

for dataId in range(1):
    dataName = ['ETTh1']
    args.data = dataName[dataId]
    if args.data in data_parser.keys():
        data_info = data_parser[args.data]
        args.data_path = data_info['data']
        args.target = data_info['T']
        args.enc_in, args.dec_in, args.c_out = data_info[args.features]
    print('Args in experiment:')
    print(args)

    Exp = Exp_SWLHT

    torch.cuda.empty_cache()

    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_eb{}_dt{}_{}_{}'.format(args.model,
                                                                                                   args.data,
                                                                                                   args.features,
                                                                                                   args.seq_len,
                                                                                                   args.label_len,
                                                                                                   args.pred_len,
                                                                                                   args.d_model,
                                                                                                   args.n_heads,
                                                                                                   args.e_layers,
                                                                                                   args.d_layers,
                                                                                                   args.d_ff, args.attn,
                                                                                                   args.embed,
                                                                                                   args.distil,
                                                                                                   args.des, ii)

        exp = Exp(args)  # set experiments
        print('training : {}'.format(setting))
        time_start = time.time()
        exp.train(setting)
        time_end = time.time()
        print('time cost: ', time_end - time_start, 's')

        print('testing : {}'.format(setting))
        exp.test(setting)

        torch.cuda.empty_cache()