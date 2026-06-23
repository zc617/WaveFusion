import argparse
import torch
from models import fusion_model
from input_data import ImageDataset
from uitils import *
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os
import logging
import os.path as osp
from loss_ssim import *
from models import Semantic

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(6)

plt.rcParams['font.family'] = ['serif']
plt.rcParams['font.sans-serif'] = ['Times New Roman']

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", default="/mnt/disk/ch/data/MSRS-main/train/", type=str)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--image_size", type=int, default=(128, 128)) # 64 * 64 测试
parser.add_argument("--epoch", type=int, default=150)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/")
parser.add_argument('--loss_weight', default='[1, 10, 0.1, 1]', type=str,metavar='N', help='loss weight')
   #  2 10 3
# Loss_ssim = kornia.losses.SSIM(11, reduction='mean')


if __name__ == "__main__":
    opt = parser.parse_args()
    if not os.path.exists(opt.checkpoint_dir):
        os.makedirs(opt.checkpoint_dir)
    x = opt.epoch - 50
    writer = SummaryWriter('./runs/logdir')
    net = fusion_model.FusionNet().to(device)
    start_epoch = 0
  
    ############################################
    modelpth = './checkpoints/'
    n_classes = 9
    segmodel = Semantic.BiSeNet(n_classes=n_classes)
    save_pth = osp.join(modelpth, 'model_final.pth')

    segmodel.load_state_dict(torch.load(save_pth))
    segmodel.cuda()
    segmodel.eval()
    for p in segmodel.parameters():
        p.requires_grad = False
    print('Load Segmentation Model {} Sucessfully~'.format(save_pth))
    score_thres = 0.7
    ignore_idx = 255
    n_min = 8 * opt.image_size[0] * opt.image_size[1] // 8
    criteria_p = Semantic_Loss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    criteria_16 = Semantic_Loss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    #############################################

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,net.parameters()),lr=opt.lr)
    train_datasets = ImageDataset(opt.data_root, opt.image_size)
    lens = len(train_datasets)
    print('data lens', lens)
    log_file = './log_dir'
    dataloader = torch.utils.data.DataLoader(train_datasets,batch_size=opt.batch_size, shuffle=True)
    runloss = 0.
    total_params = sum(p.numel() for p in net.parameters())
    print('total parameters:', total_params)
    global_step = 0
    w1_vis = 1
    i = 0
    n_min =  opt.image_size[0] * opt.image_size[1]
    Semantic = Semantic_Loss(thresh=0.7, n_min=n_min, ignore_lb=255.0)
    se_loss = 0.0
    grad_loss = 0.0
    inti_loss = 0.0
    ssi_loss = 0.0
    sem_loss = 0.0
    t1, t2, t3, t4 = eval(opt.loss_weight)
    for epoch in range(opt.epoch):
        current_lr = build_schedule(opt.lr, epoch, x)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        net.train()
        num=0
        for index, data in enumerate(dataloader):
            if len(data[0].shape) > 4:
                    data[0] = data[0].squeeze(1)
                    data[1] = data[1].squeeze(1)
                    data[2] = data[2].squeeze(1)
                    data[3] = data[3].squeeze(1)
            nc, c, h, w = data[0].size()
            nc2, c2, h2, w2 = data[1].size()
            infrared = data[0].to(device)
            visible = data[1].to(device)
            label = Variable(data[2]).to(device)
            color = data[3]
            color, Cb, Cr = RGB2YCrCb(color)
            color = color.to(device)
            Cb = Cb.to(device)
            Cr = Cr.to(device)
            fused_img, disp_ir_feature, disp_vis_feature = net(infrared, visible)

            fused_img = clamp(fused_img)
            int_loss = Int_Loss(fused_img, visible, infrared, w1_vis).to(device)
            gradient_loss = gradinet_Loss(fused_img, visible, infrared).to(device)
         
            fused_rgb = YCrCb2RGB(fused_img, Cb, Cr)
            fused_r = torch.squeeze(fused_rgb, 1)
            lab = torch.squeeze(label, 1)
            if epoch > x:     
                out, mid = segmodel(fused_rgb)
                lossp = criteria_p(out, lab)
                loss2 = criteria_16(mid, lab)
                se_loss = (lossp + 0.1 * loss2) * 0.5
            ssim_loss = SSIM_Loss(fused_img, visible, infrared).to(device)
           
            loss = t1 * int_loss + t2 * gradient_loss  + t3 * se_loss   + t4 * (1-ssim_loss)
            runloss += loss.item()
            grad_loss += gradient_loss.item()
            inti_loss  += int_loss.item()
            ssi_loss += (1-ssim_loss).item()
            sem_loss += se_loss
    
            if index % 200 == 0:  #
                writer.add_scalar('training loss', runloss / 200, epoch * len(dataloader) + index)
                writer.add_scalar('int loss', inti_loss / 200, epoch * len(dataloader) + index)
                writer.add_scalar('gradient loss', grad_loss / 200, epoch * len(dataloader) + index)
                writer.add_scalar('ssim loss', ssi_loss / 200, epoch * len(dataloader) + index) 
                runloss = 0.
                inti_loss = grad_loss = ssi_loss = 0.
            if epoch > x and index % 10 == 0:
                writer.add_scalar('sem loss', sem_loss / 10, epoch * len(dataloader) + index)
                sem_loss = 0.0
            current_lr = optimizer.param_groups[0]['lr']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 1 == 0:
            print('write_data, epoch=', epoch)

            print(
                'epoch [{}/{}], images [{}/{}], Int loss is {:.5}, gradient loss is {:.5}, Semantic loss is {:.5}, SSIM loss is {:.5}, total loss is  {:.5}, Learning Rate: {:.6}'.
                format(epoch + 1, opt.epoch, (index + 1) * data[0].shape[0], lens, int_loss.item(),
                       gradient_loss.item(), se_loss, ssim_loss.item(), loss.item(), current_lr))
            writer.add_images('IR_images', infrared, dataformats='NCHW')
            writer.add_images('VIS_images', visible, dataformats='NCHW')
            writer.add_images('Fusion_images', fused_img, dataformats='NCHW')
    writer.close()
    torch.save(net.state_dict(), './checkpoints/wavefusion.pth'.format(opt.lr, log_file[2:]))

    print('training is complete!')

