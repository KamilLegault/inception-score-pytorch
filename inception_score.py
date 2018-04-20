import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy

class IgnoreLabelDataset(torch.utils.data.Dataset):
    def __init__(self, orig):
        self.orig = orig

    def __getitem__(self, index):
        return self.orig[index][0]

    def __len__(self):
        return len(self.orig)

def w_distance(data, noise_tensor, generator, discriminator, batch_size=32, cuda=True):

    sample_num=len(noise_tensor)
    train_sampler = range(sample_num)

    gen_dataloader =  torch.utils.data.DataLoader(data, batch_size=batch_size, sampler=train_sampler, num_workers=2)
    noise_dataset  = torch.utils.data.TensorDataset(noise_tensor,torch.zeros(len(noise_tensor)))
    noise_dataloader = torch.utils.data.DataLoader(noise_dataset, batch_size=batch_size, shuffle=False)

    generator.eval()
    discriminator.eval()
    
    distances=[]
    for x, z in zip(gen_dataloader, noise_dataloader):
        x ,_ = x
        z ,_ = z
        x,z = Variable(x.cuda()), Variable(z.cuda())
        D_x = discriminator(x) 
        D_z = discriminator(generator(z))
        distance=-(D_x - D_z)
        distances.append(distance.data[0])
    generator.train()
    discriminator.train()

    return np.mean(distances)
        
        
def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    imgs = IgnoreLabelDataset(imgs)
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:(i+1)*batch_size] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)