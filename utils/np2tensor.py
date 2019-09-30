import torch

def np2tensor(imo_l):

    imo_l = imo_l.transpose(0, -1, -3, -2).astype('float32')
    imo_l = (imo_l - 128.)/255.
    imo_l = torch.from_numpy(imo_l)
    # imo_l = Variable(imo_l)
    imo_l = imo_l.cuda()
    return imo_l

def npBN(np_img):
    return (np_img - 128.)/255.