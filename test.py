import argparse
import torchvision.transforms as transforms
from models import fusion_model
from PIL import Image
import time
from uitils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.cuda.set_device(0)


# /data1/cc/data/M3FD_test/
# /data1/cc/data/TNO_test/
# /data1/cc/data/RoadScene_test/
# /data1/cc/data/LLVIP_test/
#/data1/cc/data/MSRS_test/

parser = argparse.ArgumentParser()
parser.add_argument("--infrared_dataroot", default="./test/ir/", type=str)
parser.add_argument("--visible_dataroot", default="./test/vi/", type=str)
parser.add_argument('--scale', type=int, default=1, help='scale factor: 1, 2, 3, 4, 8')  
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--output_root", default="./Ours_Wave/", type=str)
parser.add_argument("--image_size", type=int, default=[128, 128])
parser.add_argument("--epoch", type=int, default=1)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/")
args = parser.parse_args()
patch_size = 4

if __name__ == "__main__":
    opt = parser.parse_args()
    if not os.path.exists(opt.checkpoint_dir):
        os.makedirs(opt.checkpoint_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(opt.output_root):
        os.makedirs(opt.output_root)

    net = fusion_model.FusionNet().to(device)
    net.load_state_dict(torch.load("./checkpoints/wavefusion.pth"))
    net.eval()
    transform = transforms.Compose([transforms.ToTensor()])
    dirname_ir = os.listdir(opt.infrared_dataroot)
    dirname_vi = os.listdir(opt.visible_dataroot)
    tmp_len = len(dirname_ir)
    with torch.no_grad():
        t = []
        for i in range(tmp_len):
            index = i
            if i != 0:
                start = time.time()
          
            infrared  = Image.open(os.path.join(opt.infrared_dataroot, dirname_ir[i])).convert('L')
            infrared = transform(infrared).unsqueeze(0).to(device)
            visible = Image.open(os.path.join(opt.visible_dataroot, dirname_vi[i]))
            visible = transform(visible)
            visible = visible.unsqueeze(0)
            vis_y_image, vis_cb_image, vis_cr_image = RGB2YCrCb(visible)
            vis_y_image = vis_y_image.to(device)
            vis_cb_image = vis_cb_image.to(device)
            vis_cr_image = vis_cr_image.to(device)   # show color
            visible = visible.squeeze(0)
            _, h_old, w_old = visible.size()
            h_pad = (h_old // patch_size + 1) * patch_size - h_old
            w_pad = (w_old // patch_size + 1) * patch_size - w_old
            vis_y_image = torch.cat([vis_y_image, torch.flip(vis_y_image, [2])], 2)[:, :, :h_old + h_pad, :]
            vis_y_image = torch.cat([vis_y_image, torch.flip(vis_y_image, [3])], 3)[:, :, :, :w_old + w_pad]
            infrared = torch.cat([infrared, torch.flip(infrared, [2])], 2)[:, :, :h_old + h_pad, :]
            infrared = torch.cat([infrared, torch.flip(infrared, [3])], 3)[:, :, :, :w_old + w_pad] 
            fused_img, _, _,  = net(infrared,vis_y_image)
            fused_img = fused_img[..., :h_old * args.scale, :w_old * args.scale]
            if i != 0:
                end = time.time()
                print('consume time:', end - start)
                t.append(end - start)
   
            fused_img = YCrCb2RGB(fused_img, vis_cb_image, vis_cr_image)  
            fused_img = fused_img.squeeze(0)
            fused_img = transforms.ToPILImage()(fused_img)
            fused_img.save(os.path.join(opt.output_root, str(dirname_ir[i])))  
        print("mean:%s, std: %s" % (np.mean(t), np.std(t)))

