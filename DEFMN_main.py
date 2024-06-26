import sys
import os
import shutil
from datetime import datetime
import time
import csv
import torch.nn as nn
from torchsummary import summary
import argparse
from configparser import ConfigParser
import logging
from metrics import evaluate
from utils import *
from DEFMN import DEFMN
from sklearn.preprocessing import StandardScaler
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

def getModel(mode):
    model = DEFMN(num_nodes=args.num_nodes, input_dim=args.input_dim, output_dim=args.output_dim, seq_len=args.seq_len,
                   out_len=args.out_len, b_layers = args.b_layers, top_k=args.top_k, d_model=args.d_model,
                   d_ff = args.d_ff, num_kernels = args.num_kernels, mem_num=args.mem_num, mem_dim=args.mem_dim,
                   cheb_k=args.max_diffusion_step, cl_decay_steps=args.cl_decay_steps).to(device)
    if mode == 'train':
        summary(model, [(args.seq_len, args.num_nodes, args.input_dim), (args.out_len, args.num_nodes, args.output_dim)], batch_dim=0, device=device)
        print_params(model)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
    return model

def evaluateModel(model, data_iter):
    if args.loss == 'MAE': criterion = nn.L1Loss()
    if args.loss == 'MSE': criterion = nn.MSELoss()
    if args.loss == 'BCE': criterion = nn.BCEWithLogitsLoss()
    separate_loss = nn.TripletMarginLoss(margin=1.0)
    compact_loss = nn.MSELoss()
        
    model.eval()
    loss_sum, n, YS_pred = 0.0, 0, []
    loss_sum1, loss_sum2, loss_sum3 = 0.0, 0.0, 0.0
    with torch.no_grad():
        for x, y, y_cov in data_iter:
            y_pred, h_att, query, pos, neg = model(x, y_cov)
            loss1 = criterion(y_pred, y)
            # loss_mape = torch.mean(torch.abs((y_pred - y) / y))
            loss2 = separate_loss(query, pos.detach(), neg.detach())
            loss3 = compact_loss(query, pos.detach())
            loss = args.lamb1 * loss1 + args.lamb2 * loss2 + args.lamb3 * loss3
            loss_sum += loss.item() * y.shape[0]
            loss_sum1 += loss1.item() * y.shape[0]
            loss_sum2 += loss2.item() * y.shape[0]
            loss_sum3 += loss3.item() * y.shape[0]
            n += y.shape[0]
            YS_pred.append(y_pred.cpu().numpy())     
    loss, loss1, loss2, loss3 = loss_sum / n, loss_sum1 / n, loss_sum2 / n, loss_sum3 / n
    YS_pred = np.vstack(YS_pred)
    return loss, loss1, loss2, loss3, YS_pred

def trainModel(name, mode, XS, YS, YCov):
    model = getModel(mode)
    
    XS_torch, YS_torch, YCov_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device), torch.Tensor(YCov).to(device)
    trainval_data = torch.utils.data.TensorDataset(XS_torch, YS_torch, YCov_torch)
    trainval_size = len(trainval_data)
    train_size = int(trainval_size * (1 - args.val_ratio))
    train_data = torch.utils.data.Subset(trainval_data, list(range(0, train_size)))
    val_data = torch.utils.data.Subset(trainval_data, list(range(train_size, trainval_size)))
    train_iter = torch.utils.data.DataLoader(train_data, args.batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_data, args.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if args.lr_decay == 'MultiStepLR': lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.steps, gamma=args.lr_decay_ratio)
    if args.lr_decay == 'ExponentialLR': lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8, last_epoch=-1)

    if args.loss == 'MAE': criterion = nn.L1Loss()
    if args.loss == 'MSE': criterion = nn.MSELoss()
    if args.loss == 'BCE': criterion = nn.BCEWithLogitsLoss()
    separate_loss = nn.TripletMarginLoss(margin=1.0)
    compact_loss = nn.MSELoss()
        
    min_val_loss = np.inf
    wait = 0
    batches_seen = 0
    for epoch in range(args.epochs):
        starttime = datetime.now()     
        loss_sum, n = 0.0, 0
        loss_sum1, loss_sum2, loss_sum3 = 0.0, 0.0, 0.0
        model.train()
        for x, y, ycov in train_iter:
            optimizer.zero_grad()
            y_pred, h_att, query, pos, neg = model(x, ycov, y, batches_seen)
            loss1 = criterion(y_pred, y)
            loss2 = separate_loss(query, pos.detach(), neg.detach())
            loss3 = compact_loss(query, pos.detach())
            loss = args.lamb1 * loss1 + args.lamb2 * loss2 + args.lamb3 * loss3
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.shape[0]
            loss_sum1 += loss1.item() * y.shape[0]
            loss_sum2 += loss2.item() * y.shape[0]
            loss_sum3 += loss3.item() * y.shape[0]
            n += y.shape[0]
            batches_seen += 1
        lr_scheduler.step()
        train_loss, train_loss1, train_loss2, train_loss3 = loss_sum / n, loss_sum1 / n, loss_sum2 / n, loss_sum3 / n
        val_loss, val_loss1, val_loss2, val_loss3, _ = evaluateModel(model, val_iter)
        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            torch.save(model.state_dict(), modelpt_path)
        else:
            wait += 1
            if wait == args.patience:
                logger.info('Early stopping at epoch: %d' % epoch)
                break
        endtime = datetime.now()
        epoch_time = (endtime - starttime).seconds
        logger.info("epoch", epoch, "time used:", epoch_time, "seconds", 
                    "train loss:", '%.6f %.6f %.6f %.6f' % (train_loss, train_loss1, train_loss2, train_loss3),
                    "validation loss:", '%.6f %.6f %.6f %.6f' % (val_loss, val_loss1, val_loss2, val_loss3))
        with open(epochlog_path, 'a') as f:
            f.write("%s, %d, %s, %d, %s, %s, %.6f, %s, %.6f\n" % ("epoch", epoch, "time used", epoch_time, "seconds", "train loss", train_loss, "valid loss:", val_loss))
    
def testModel(name, mode, XS, YS, YCov, scalers):
    model = getModel(mode).cuda()
    model.load_state_dict(torch.load(modelpt_path))
    
    XS_torch, YS_torch, YCov_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device), torch.Tensor(YCov).to(device)
    test_data = torch.utils.data.TensorDataset(XS_torch, YS_torch, YCov_torch)
    test_iter = torch.utils.data.DataLoader(test_data, args.batch_size, shuffle=False)
    loss, loss1, loss2, loss3, YS_pred = evaluateModel(model, test_iter)

    
    logger.info('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    # YS, YS_pred = np.squeeze(YS), np.squeeze(YS_pred)
    YS, YS_pred = YS.reshape(-1, 1177, YS.shape[-1]), YS_pred.reshape(-1, 1177, YS_pred.shape[-1])
    YS[:, :, 0], YS_pred[:, :, 0] = scalers[0].inverse_transform(YS[:, :, 0]), scalers[0].inverse_transform(YS_pred[:, :, 0])
    YS[:, :, 1], YS_pred[:, :, 1] = scalers[1].inverse_transform(YS[:, :, 1]), scalers[1].inverse_transform(YS_pred[:, :, 1])
    YS[:, :, 2], YS_pred[:, :, 2] = scalers[2].inverse_transform(YS[:, :, 2]), scalers[2].inverse_transform(YS_pred[:, :, 2])
    YS, YS_pred = YS.reshape(-1, args.out_len, YS.shape[-2], YS.shape[-1]), YS_pred.reshape(-1, args.out_len, YS.shape[-2], YS_pred.shape[-1])
    # net load
    net = YS[:, :, :, 0] - YS[:, :, :, 1] - YS[:, :, :, 2]
    net_pred = YS_pred[:, :, :, 0] - YS_pred[:, :, :, 1] - YS_pred[:, :, :, 2]
    # net, net_pred = np.squeeze(net), np.squeeze(net_pred)

    net_out_load = net[0, :, :]
    net_out_load = np.transpose(net_out_load)  #
    with open(path + f'/{name}_net_out_load.csv', 'w+', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for row in net_out_load:
            writer.writerow(row)

    net_pred_out_load = net_pred[0, :, :]
    net_pred_out_load = np.transpose(net_pred_out_load)  #
    with open(path + f'/{name}_net_pred_out_load.csv', 'w+', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for row in net_pred_out_load:
            writer.writerow(row)

    #output
    YS_out_load = YS[0, :, :, 0]
    YS_out_load = np.transpose(YS_out_load)  #
    with open(path + f'/{name}_YS_out_load.csv', 'w+', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for row in YS_out_load:
            writer.writerow(row)

    YS_out_pv = YS[0, :, :, 1]
    YS_out_pv = np.transpose(YS_out_pv)  #
    with open(path + f'/{name}_YS_out_pv.csv', 'w+', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for row in YS_out_pv:
            writer.writerow(row)

    YS_out_wind = YS[0, :, :, 2]
    YS_out_wind = np.transpose(YS_out_wind)  #
    with open(path + f'/{name}_YS_out_wind.csv', 'w+', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for row in YS_out_wind:
            writer.writerow(row)
    #
    YS_pre_out_load = YS_pred[0, :, :, 0]
    YS_pre_out_load = np.transpose(YS_pre_out_load)
    with open(path + f'/{name}_YS_pre_out_load.csv', 'w+', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for row in YS_pre_out_load:
            writer.writerow(row)

    YS_pre_out_pv = YS_pred[0, :, :, 1]
    YS_pre_out_pv = np.transpose(YS_pre_out_pv)
    with open(path + f'/{name}_YS_pre_out_pv.csv', 'w+', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for row in YS_pre_out_pv:
            writer.writerow(row)

    YS_pre_out_wind = YS_pred[0, :, :, 2]
    YS_pre_out_wind = np.transpose(YS_pre_out_wind)
    with open(path + f'/{name}_YS_pre_out_wind.csv', 'w+', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for row in YS_pre_out_wind:
            writer.writerow(row)

    np.save(path + f'/{name}_prediction', YS_pred)
    np.save(path + f'/{name}_groundtruth', YS)
    MSE, RMSE, MAE, MAPE, skewness, kurtosis = evaluate(YS, YS_pred)
    MSE_n, RMSE_n, MAE_n, MAPE_n, skewness_n, kurtosis_n = evaluate(net, net_pred)
    
    logger.info("%s, %s, test loss, loss1, loss2, loss3, %.6f, %.6f, %.6f, %.6f" % (name, mode, loss, loss1, loss2, loss3))
    logger.info("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, skewness, kurtosis, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f" % (name, mode, MSE, RMSE, MAE, MAPE, skewness, kurtosis))
    logger.info("all pred steps, %s, %s, MSE_n, RMSE_n, MAE_n, MAPE_n, skewness_n, kurtosis_n, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f" % (name, mode, MSE_n, RMSE_n, MAE_n, MAPE_n, skewness_n, kurtosis_n))
    with open(score_path, 'a') as f:
        f.write("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, skewness, kurtosis, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f\n" % (name, mode, MSE, RMSE, MAE, MAPE, skewness, kurtosis))
        for i in range(args.out_len):
            MSE, RMSE, MAE, MAPE, skewness, kurtosis = evaluate(YS[:, i, :, :], YS_pred[:, i, :, :])
            MSE_n, RMSE_n, MAE_n, MAPE_n, skewness_n, kurtosis_n = evaluate(net[:, i, :], net_pred[:, i, :])
            logger.info("%d step, %s, %s, MSE, RMSE, MAE, MAPE, skewness, kurtosis, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f" % (i+1, name, mode, MSE, RMSE, MAE, MAPE, skewness, kurtosis))
            f.write("%d step, %s, %s, MSE, RMSE, MAE, MAPE, skewness, kurtosis, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f\n" % (i+1, name, mode, MSE, RMSE, MAE, MAPE, skewness, kurtosis))
            logger.info("%d step, %s, %s, MSE_n, RMSE_n, MAE_n, MAPE_n, skewness_n, kurtosis_n, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f" % (i + 1, name, mode, MSE_n, RMSE_n, MAE_n, MAPE_n, skewness_n, kurtosis_n))
            f.write("%d step, %s, %s, MSE_n, RMSE_n, MAE_n, MAPE_n, skewness_n, kurtosis_n, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f\n" % (i + 1, name, mode, MSE_n, RMSE_n, MAE_n, MAPE_n, skewness_n, kurtosis_n))


#########################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['power_DG', 'power_DG_20'], default='power_DG', help='which dataset to run')  # power_DG is PDS Scenario 1  power_DG_20 is PDS Scenario 2
parser.add_argument('--month', type=str, default='power_DG_200712', help='which experiment setting (month) to run as testing data')
parser.add_argument('--val_ratio', type=float, default=0.25, help='the ratio of validation data among the trainval ratio')
parser.add_argument('--top_k', type=int, default=3, help='for TimesBlock')
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
parser.add_argument('--d_ff', type=int, default=64, help='dimension of fcn')
parser.add_argument('--seq_len', type=int, default=24, help='input sequence length')
parser.add_argument('--out_len', type=int, default=3, help='output sequence length')
parser.add_argument('--input_dim', type=int, default=3, help='number of input channel')
parser.add_argument('--output_dim', type=int, default=3, help='number of output channel')
parser.add_argument('--b_layers', type=int, default=1, help='num of block layers')
parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
parser.add_argument('--max_diffusion_step', type=int, default=3, help='max diffusion step or Cheb K')
parser.add_argument('--mem_num', type=int, default=10, help='number of meta-nodes/prototypes')
parser.add_argument('--mem_dim', type=int, default=64, help='dimension of meta-nodes/prototypes')
parser.add_argument("--loss", type=str, default='MAE', help="MAE, MSE, BCE")
parser.add_argument('--lamb1', type=float, default=1.4, help='lamb1 value for MAE loss')
parser.add_argument('--lamb2', type=float, default=0.5, help='lamb2 value for separate loss')
parser.add_argument('--lamb3', type=float, default=0.5, help='lamb3 value for compact loss')
parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="base learning rate")
parser.add_argument("--steps", type=eval, default=[90], help="steps")
parser.add_argument("--lr_decay", type=str, choices=['MultiStepLR', 'ExponentialLR'], default='MultiStepLR', help="lr_scheduler")
parser.add_argument("--patience", type=int, default=10, help="patience used for early stop")
parser.add_argument("--lr_decay_ratio", type=float, default=0.1, help="lr_decay_ratio")
parser.add_argument("--cl_decay_steps", type=int, default=1500, help="cl_decay_steps")
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
parser.add_argument('--seed', type=int, default=42, help='random seed')
args = parser.parse_args()

config = ConfigParser()
config.read('params_DG.txt', encoding='UTF-8')
train_month = eval(config[args.month]['train_month'])
test_month = eval(config[args.month]['test_month'])
power_path = config[args.month]['power_path']
subroad_path = config[args.dataset]['subroad_path']
road_path = config['common']['road_path']
adj_path = config['common']['adj01_path']
num_variable = len(np.loadtxt(subroad_path).astype(int))
N_link = config.getint('common', 'N_link')
args.num_nodes = num_variable

model_name = 'DEFMN'
timestring = time.strftime('%Y%m%d%H%M%S', time.localtime())
path = f'../save/{args.dataset}_{model_name}_{timestring}'
logging_path = f'{path}/{model_name}_{timestring}_logging.txt'
score_path = f'{path}/{model_name}_{timestring}_scores.txt'
epochlog_path = f'{path}/{model_name}_{timestring}_epochlog.txt'
modelpt_path = f'{path}/{model_name}_{timestring}.pt'
if not os.path.exists(path): os.makedirs(path)
shutil.copy2(sys.argv[0], path)
shutil.copy2(f'{model_name}.py', path)
shutil.copy2('utils.py', path)
shutil.copy2('metrics.py', path)
shutil.copy2('params_DG.txt', path)

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
class MyFormatter(logging.Formatter):
    def format(self, record):
        spliter = ' '
        record.msg = str(record.msg) + spliter + spliter.join(map(str, record.args))
        record.args = tuple() # set empty to args
        return super().format(record)
formatter = MyFormatter()
handler = logging.FileHandler(logging_path, mode='a')
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(console)

logger.info('model', model_name)
logger.info('dataset', args.dataset)
logger.info('val_ratio', args.val_ratio)
logger.info('num_nodes', args.num_nodes)
logger.info('seq_len', args.seq_len)
logger.info('out_len', args.out_len)
logger.info('input_dim', args.input_dim)
logger.info('output_dim', args.output_dim)
logger.info('max_diffusion_step', args.max_diffusion_step)
logger.info('mem_num', args.mem_num)
logger.info('mem_dim', args.mem_dim)
logger.info('loss', args.loss)
logger.info('MAE loss lamb1', args.lamb1)
logger.info('separate loss lamb2', args.lamb2)
logger.info('compact loss lamb3', args.lamb3)
logger.info('epochs', args.epochs)
logger.info('batch_size', args.batch_size)
logger.info('lr', args.lr)
logger.info('steps', args.steps)
logger.info('patience', args.patience)

cpu_num = 1
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)
device = torch.device("cuda:{}".format(args.gpu)) if torch.cuda.is_available() else torch.device("cpu")
# Please comment the following three lines for running experiments multiple times.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)
#####################################################################################################

def create_scalers(dim):
    scalers = []
    for _ in range(dim):
        scaler = StandardScaler()
        scalers.append(scaler)
    return scalers

def main():
    train_data = [get_data(config[month]['power_path'], N_link, subroad_path, ['power', 'pv', 'wind']) for month in train_month]
    test_data = [get_data(config[month]['power_path'], N_link, subroad_path, ['power', 'pv', 'wind']) for month in test_month]
    train_time = [get_time(config[month]['power_path'], N_link, subroad_path) for month in train_month]
    test_time = [get_time(config[month]['power_path'], N_link, subroad_path) for month in test_month]
    # adj_bank = get_adj(adj_path, subroad_path)

    load_data = [data[:, :, 0] for data in train_data + test_data]  # better to only use train_data
    pv_data = [data[:, :, 1] for data in train_data + test_data]
    wind_data = [data[:, :, 2] for data in train_data + test_data]

    load_data = np.vstack(load_data)
    pv_data = np.vstack(pv_data)
    wind_data = np.vstack(wind_data)

    scalers = create_scalers(args.input_dim)

    scalers[0].fit(load_data)
    scalers[1].fit(pv_data)
    scalers[2].fit(wind_data)

    for data in train_data + test_data:
        data[:, :, 0] = scalers[0].transform(data[:, :, 0])
        data[:, :, 1] = scalers[1].transform(data[:, :, 1])
        data[:, :, 2] = scalers[2].transform(data[:, :, 2])

    logger.info(args.dataset, args.month, 'training started', time.ctime())
    trainXS, trainYS = getXSYS(train_data, args.seq_len, args.out_len)
    trainXCov, trainYCov = getXSYS(train_time, args.seq_len, args.out_len)
    logger.info('TRAIN XS.shape YS.shape, XCov.shape, YCov.shape', trainXS.shape, trainYS.shape, trainXCov.shape, trainYCov.shape)
    trainModel(model_name, 'train', trainXS, trainYS, trainYCov)
    logger.info(args.dataset, args.month, 'training ended', time.ctime())
    logger.info('=' * 90)

    logger.info(args.dataset, args.month, 'testing started', time.ctime())
    testXS, testYS = getXSYS(test_data, args.seq_len, args.out_len)
    testXCov, testYCov = getXSYS(test_time, args.seq_len, args.out_len)
    logger.info('TEST XS.shape, YS.shape, XCov.shape, YCov.shape', testXS.shape, testYS.shape, testXCov.shape, testYCov.shape)
    testModel(model_name, 'test', testXS, testYS, testYCov, scalers)
    logger.info(args.dataset, args.month, 'testing ended', time.ctime())
    logger.info('=' * 90)


if __name__ == '__main__':
    main()