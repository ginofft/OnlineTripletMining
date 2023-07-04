import argparse
from pathlib import Path
import torch
import torch.optim as optim

from src.dataset import OnlineTripletImageDataset, ImageDataset
from src.loss import OnlineTripletLoss
from src.utils import save_checkpoint, load_checkpoint, plot_retrieval_images, str2bool, construct_model
from src.train import train, validate
from src.query import query, calculate_netvlads

parser = argparse.ArgumentParser(description = 'torch-image_classifier-online_triplet_mining')
#Hyper Parameters
##Sampler 
parser.add_argument('--P', type=int, default=4, 
                    help='no. classes for Online Triplet Mining')
parser.add_argument('--K', type=int, default=8, 
                    help='no. images per class for Online Triplet Mining')
##Optimizer
parser.add_argument('--optim', type=str, default = 'Adam', 
                    help='optimizer to use', choices=['SGD', 'Adam'])
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
##Loss
parser.add_argument('--margin', type=float, default=0.1**0.5, 
                    help='Margin for triplet loss')
parser.add_argument('--tripletLoss', type=str, default='batchall', 
                    help='Type of triplet loss to use. There are three available: naive (random triplet), online triplet mining - hard variation, online triplet mining - semi-hard variation',
                    choices=['naive', 'batchall', 'batchhard'])
##Model
parser.add_argument('--encoder', type=str, default='VGG16',
                    choices=['VGG13','VGG16','VGG19',
                            'Resnet18','Resnet34','Resnet50','Resnet101','Resnet152'])

#Training Arguments 
parser.add_argument('--nEpochs', type =  int, default = 500, help='no. epochs')
parser.add_argument('--mode', type=str, default='train', 
                    help='Traning mode or Testing(inference) mode', 
                    choices=['train', 'test'],
                    required=True)
parser.add_argument('--trainPath', type=str, default=None, 
                    help='Path of training set')
parser.add_argument('--valPath', type=str, default=None, 
                    help='Path of validation set')
parser.add_argument('--savePath', type=str, default='', 
                    help='Path to save checkpoint to')
parser.add_argument('--loadPath', type=str, default='', 
                    help='Path to load checkpoint from - used for resume or testing')
parser.add_argument('--resetLoss', type = str2bool, nargs = '?', default= False,
                    help='Default is False - resume training with stored val and train loss.\n You should set this to True when switching loss function')
parser.add_argument('--saveEvery', type=int, default=25, 
                    help='no. epoch before a save is created')

#Inference Arguments
parser.add_argument('--dbPath', type=str, default='', 
                    help='Path to database folder (NOT training set folder)')
parser.add_argument('--queryPath', type=str, default='',
                    help='Path to query folder')
parser.add_argument('--outPath', type=str, default='', 
                    help="Path where to store: database's netvlads, query's netvlads and retrieval results")

if __name__ == "__main__":
  opt = parser.parse_args()
  cuda = torch.cuda.is_available()
  if cuda:
    print("GPU found, runing on GPU")
    device = torch.device("cuda")
  else:
    print("No GPU found, running on CPU")
    device = torch.device("cpu")
  
  #Setup model
  if opt.loadPath:
    model = construct_model(opt.encoder, opt.n_vocabs, use_pretrained_weight= False)
  else:
    model = construct_model(opt.encoder, opt.n_vocabs, use_pretrained_weight= True)
  model = model.to(device)

  if opt.mode.lower() == 'train':
    startEpoch = 0
    val_loss = float('inf')
    train_loss = float('inf')
    accuracy = 0

    if opt.trainPath:
      trainSet = OnlineTripletImageDataset(Path(opt.trainPath))
    else:
      raise Exception("Please provide a trainset using --trainPath")
    if opt.valPath:
      valSet = OnlineTripletImageDataset(Path(opt.valPath))
    else:
      raise Exception("Please provide a validation set using --valPath")

    if opt.tripletLoss.lower() == 'batchhard':
      criterion = OnlineTripletLoss(margin = opt.margin, hard=True).to(device)
    elif opt.tripletLoss.lower() == 'batchall':
      criterion = OnlineTripletLoss(margin = opt.margin, hard=False).to(device)
    elif opt.tripletLoss.lower() == 'naive':
      raise Exception('naive triplet is not implemented yet\n(cause im lazy, deal with it)')
    
    if opt.optim.lower() =='adam':
      optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                             lr = opt.lr)
    else:
      pass #TODO SGD optimizer

    if opt.loadPath: #loading stuff
      startEpoch, train_loss, val_loss, accuracy = load_checkpoint(Path(opt.loadPath),
                                                        device,
                                                        model, 
                                                        optimizer)
    if opt.resetLoss == True: #condition for when you switch loss function
      val_loss = float('inf')
      train_loss = float('inf')
      accuracy = 0
    
    for epoch in range(startEpoch+1, opt.nEpochs+1):
      # train & validate
      epoch_train_loss = train(device, model, epoch,
                            trainSet, opt.P, opt.K,
                            criterion, optimizer)
      epoch_val_loss, epoch_accuracy = validate(device, model, 
                                valSet, opt.P, opt.K,
                                criterion)

      state_dict = {
            'epoch': epoch,
            'train_loss': epoch_train_loss,
            'val_loss': epoch_val_loss,
            'accuracy': epoch_accuracy,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
      #saving stuff
      if (epoch_accuracy > accuracy): #lowest loss on val set
        accuracy = epoch_accuracy
        save_checkpoint(state_dict, Path(opt.savePath), 'best.pth.tar')
      if (epoch % opt.saveEvery) == 0: #save every epoch
        save_checkpoint(state_dict, Path(opt.savePath), 'epoch{}.pth.tar'.format(epoch))      
  else:  
    raise NotImplementedError