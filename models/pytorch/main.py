"""training models"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
import pandas as pd
import time

from glob import glob
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from lr_finder import LRFinder
import argparse
import torch
from torch.utils import data
from torch.optim import lr_scheduler
from tqdm import tqdm
import numpy as np
from adabound import AdaBound
from utils import init_weights
from data_loader import NiftyLoader, NiftyLoaderTransform, get_transforms
from inceptionv3 import InceptionAttention3D
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter


def get_paramters():
    """get parameters"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-lr", required=True, default=0.0001, type=float)
    parser.add_argument("-drop", required=True, default=0.5, type=float)
    parser.add_argument("-batch", required=True, default=8, type=int)
    parser.add_argument("-seed", default=1, type=int)

    parser.add_argument("-epoch", required=True, default=50, type=int)
    parser.add_argument("-gpu", default="0")
    parser.add_argument("-resume", action="store_true")
    parser.add_argument("-findlr", action="store_true")
    parser.add_argument("-endlr", type=float, help='only useful for find learning rate')
    parser.add_argument("-bids", required=True, default="results/bids_head1.csv", type=str)
    parser.add_argument("-modelpath", required=True, default="", type=str, help='model store root path')
    parser.add_argument("-machine", default="", type=str)
    parser.add_argument("-exclude", required=True, default="check_datasets/bad_ids_thrown_ids_for_antspy2.txt", type=str)
    parser.add_argument("-modality", required=True, type=str, default='0', help='data modality, default: FLAIR')
    parser.add_argument("-symptom", required=True, type=str, default='Leukomalacia', help='choose one from Cyst,Leukomalacia,Hemorrhage,Myelin or Develop,Expansion,Lesion')
    return parser.parse_args()


def test(model, test_loader, device, loss, return_outputs=True):
    model.eval()
    all_outputs = []
    all_labels = []
    all_preds = []
    with torch.no_grad():
        correct = 0
        val_loss = 0.0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            val_loss += loss(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels[:, 1].long()).sum().item()

            all_outputs.append(outputs[:, 1])
            all_preds.append(predicted)
            all_labels.append(labels[:, 1])
    if return_outputs:
        predicted = torch.cat(all_preds, dim=0).cpu()
        outputs = torch.cat(all_outputs, dim=0).cpu()
        labels  = torch.cat(all_labels,  dim=0).cpu()
        #print(confusion_matrix(labels, predicted, labels=[0, 1]))
        #print(average_precision_score(labels, outputs))
        #print(roc_auc_score(labels, outputs))
        #print(f1_score(labels, predicted, pos_label=1))
        return correct/total, val_loss/total, predicted, outputs, labels
    return correct/total, val_loss/total


def train_valid(model, train_loader, valid_loader, optimizer, epoch, device, loss, args, schedule, prev_valid_loss, prev_valid_acc, checkpoint):
    training_loss = 0.0
    training_acc = 0.0
    total = 0.0
    model.train()

    for index, (train_image_data, train_label_data) in enumerate(train_loader):
        train_image_data, train_label_data = train_image_data.to(device), train_label_data.to(device)

    #for index in range(50):
    #    train_data = next(train_loader)
    #    train_image_data, train_label_data = torch.from_numpy(train_data['data']).to(device), torch.from_numpy(train_data['labels']).to(device)

        output = model(train_image_data)
        labels = train_label_data
        criterion = loss(output, labels)

        optimizer.zero_grad()
        criterion.backward()
        optimizer.step()
        training_loss += criterion.item()

        _, pred = torch.max(output, 1)
        training_acc += (pred == train_label_data[:, 1].long()).sum().item()
        total += train_label_data.size(0)
        train_acc, train_loss = training_acc/total, training_loss/total
        if (index+1)%10 == 0:
            print('[%d, %5d] train loss: %.4f, acc: %.4f, lr:%.9f' % (epoch+1, index+1, training_loss/total, training_acc/total, optimizer.param_groups[0]['lr']))

    valid_acc, valid_loss = test(model, valid_loader, device, loss, False)
    print('[%d, %5d] valid loss: %.4f, acc: %.4f, lr:%.9f' % (epoch+1, index+1, valid_loss, valid_acc, optimizer.param_groups[0]['lr']))

    if valid_loss < prev_valid_loss:
        prev_valid_loss = valid_loss
        prev_valid_acc = valid_acc

        # too much disk usage, only save the current best models
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'schedule': schedule.state_dict(),
                    'args': args,
                    'epoch': epoch+1},
                    '%s/params%s_dataiter_%s_currentlr_%s_validloss_%s_validacc_%s_train_loss_%s_train_acc_%s.ckpt' % (checkpoint, epoch+1, index+1, optimizer.param_groups[0]['lr'], round(valid_loss, 5), round(valid_acc, 5), round(train_loss, 5), round(train_acc, 5)))

    schedule.step(valid_loss)
    return prev_valid_loss, prev_valid_acc


def train_test_valid(bids, exclude, machine, modality, batch, symptom, seed=1, pure_control=False):
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    all_bids = pd.read_csv(bids, index_col=0)
    with open(exclude) as fin:
        exclude_ids = list(map(int, fin.readlines()))
    all_bids = all_bids.loc[~all_bids.index.isin(exclude_ids), :]
    all_symptoms = ['Cyst', 'Leukomalacia', 'Hemorrhage', 'Myelin_or_Develop', 'Expansion', 'Lesion']

    if pure_control:
        pure_control = (all_bids.Cyst == 0) & (all_bids.Leukomalacia == 0) & (all_bids.Hemorrhage==0) & \
                       (all_bids.Expansion == 0) & (all_bids.Lesion == 0) & (all_bids.Myelin_or_Develop==0)
        pure_control = all_bids.loc[pure_control, :]
        bids_symptom = all_bids.loc[all_bids.loc[:, symptom] == 1, :]
        all_bids = pd.concat([bids_symptom, pure_control], axis=0)

    if machine != '':
        all_bids = all_bids.loc[all_bids.merge_machine_info.isin(machine.split(',')), :]

    print('**'*20)
    print(all_bids.shape)

    # the basic split
    train, test = train_test_split(all_bids.index, test_size=0.3, shuffle=True, random_state=seed, stratify=all_bids.loc[:, symptom])
    print('**'*20)
    print(np.intersect1d(train, test))
    test_bids = all_bids.loc[test, :]
    train_bids = all_bids.loc[train, :]

    train, valid = train_test_split(train_bids.index, test_size=0.3, shuffle=True, random_state=seed, stratify=train_bids.loc[:, symptom])
    valid_bids = train_bids.loc[valid, :]
    train_bids = train_bids.loc[train, :]

    print('**'*20)
    print(np.intersect1d(train, test))
    print(np.intersect1d(train, valid))
    print(np.intersect1d(test, valid))
    print('**'*20)

    train_bids.to_csv('results/seed%s_train.csv' % (seed))
    valid_bids.to_csv('results/seed%s_valid.csv' % (seed))
    test_bids.to_csv('results/seed%s_test.csv' % (seed))

    print(train_bids.loc[:, symptom].sum(), len(train))
    print(valid_bids.loc[:, symptom].sum(), len(valid))
    print(test_bids.loc[:, symptom].sum(), len(test))

    train_nifty = NiftyLoader(train_bids, modality, symptom, 635)    ## hard code: template id for this project
    train_loader = data.DataLoader(train_nifty, shuffle=True, batch_size=batch, num_workers=10)
    #train_nifty = NiftyLoaderTransform(train_bids, modality, symptom, batch, 10)    ## hard code: template id for this project
    #train_loader = MultiThreadedAugmenter(train_nifty, get_transforms(), 10, 3, seeds=None, pin_memory=True)
    
    valid_nifty = NiftyLoader(valid_bids, modality, symptom, 635)
    valid_loader = data.DataLoader(valid_nifty, shuffle=True, batch_size=batch, num_workers=10)

    test_nifty = NiftyLoader(test_bids, modality, symptom, 635)
    test_loader = data.DataLoader(test_nifty, shuffle=False, batch_size=batch, num_workers=10)
    return train_loader, valid_loader, test_loader


def custom_load_checkpoint(path):
    checkpoints= glob(os.path.abspath(path)+'/*')
    epochs, valid_loss, valid_acc = [], [], []
    modal = []
    paths = []
    for epoch_param in checkpoints:
        epoch = int(os.path.basename(epoch_param).split('_')[0].replace('params', ''))        
        vloss = float(os.path.basename(epoch_param).split('_')[6].replace('params', ''))  
        vacc = float(os.path.basename(epoch_param).split('_')[8].replace('params', ''))
        epochs.append(epoch)
        valid_loss.append(vloss)  
        valid_acc.append(vacc)
        modal.append(os.path.basename(os.path.dirname(epoch_param)).split('_')[1].replace('modal', ''))
        paths.append(epoch_param)
    metrics_all = pd.DataFrame(dict(epoch=epochs, validloss=valid_loss, validacc=valid_acc, modal=modal, path=paths))
    metrics_all.sort_values('validloss', inplace=True)
    return metrics_all

def main():
    """ main interface"""

    start = time.time()
    print(start)
    args = get_paramters()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data files
    modality = list(map(int, args.modality.split(',')))
    train_loader, valid_loader, test_loader = train_test_valid(args.bids, args.exclude, args.machine,
                                                               modality, args.batch, args.symptom, args.seed, True)

    model = InceptionAttention3D(len(modality), args.drop).cuda()
    #model = torch.nn.DataParallel(model) # use multiple gpus

    loss = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4) # for PVL 1e-4 l2 is enough with lr 0.00017

    if args.findlr:
        lr_finder = LRFinder(model, optimizer, loss, device="cuda")
        lr_finder.range_test(train_loader, end_lr=args.endlr, num_iter=200, step_mode='exp')
        lr_finder.plot(log_lr=False)
        plt.savefig('%s/%s_lr_test%s_%s.pdf' % (args.modelpath, args.symptom, args.lr, args.endlr))
        sys.exit(0)

    checkpoint = '%s/%s_modal%s_class_models_initiallr%.6f_batch%s_drop%s' % (args.modelpath, args.symptom, args.modality, float(args.lr), args.batch, args.drop)
    if os.path.exists(checkpoint) and args.resume:
        metrics_all = custom_load_checkpoint(checkpoint)
        print(metrics_all.iloc[0, :].loc['path'])
        params = torch.load(metrics_all.iloc[0, :].loc['path'])
        print(params.keys())
        model.load_state_dict(params['model'])
        print('resume....')
        test_acc, test_loss, test_predict, test_outputs, test_labels = test(model, test_loader, device, loss, True)
        print(test_acc, test_loss)
        print(np.vstack([test_predict, test_outputs, test_labels]).T.shape)
        output_test = pd.DataFrame(np.vstack([test_predict, test_outputs, test_labels]).T, columns=['predict', 'outputs', 'labels'])
        output_test.to_csv(f'{checkpoint}_test_resume.csv')
        sys.exit(0)
    else:
        init_weights(model, init_type='kaiming')

    os.system('mkdir -p %s' % checkpoint)
     
    #scheduler = lr_scheduler.StepLR(optimizer,
    #                                step_size=10, gamma=0.3333)
    #scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
    #                                           10, eta_min=1e-6) # PVL
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.3333, patience=5)

    valid_loss = np.inf
    valid_acc = 0.0
    for epoch in range(args.epoch):
        valid_loss, valid_acc = train_valid(model, train_loader, valid_loader,
                                 optimizer, epoch, device, loss, args,
                                 scheduler, valid_loss, valid_acc, checkpoint)
    end = time.time()
    print(end)
    print(end - start)

    # load best checkpoints
    metrics_all = custom_load_checkpoint(checkpoint)
    model.load_state_dict(torch.load(metrics_all.iloc[0, :].loc['path'])['model'])
    test_acc, test_loss, test_predict, test_outputs, test_labels = test(model, test_loader, device, loss, True)
    print(test_acc, test_loss)
    print(np.vstack([test_predict, test_outputs, test_labels]).T.shape)
    output_test = pd.DataFrame(np.vstack([test_predict, test_outputs, test_labels]).T, columns=['predict', 'outputs', 'labels'])
    output_test.to_csv(f'{checkpoint}_test.csv')
    metrics_all.to_csv(f'{checkpoint}_valid.csv')


if __name__ == '__main__':
    main()
