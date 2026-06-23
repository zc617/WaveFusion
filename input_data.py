from torch.utils.data import Dataset
from torchvision import transforms
#from flowlib import read, read_weights_file
from PIL import Image
import re
from uitils import *
from augment_image import *
import torchvision
import torchvision.transforms.functional as TF


class ImageDataset(Dataset):
    def __init__(self, dataroot, image_size):
        """

        """
        self.vis_folder = os.path.join(dataroot, 'vi')
        self.ir_folder = os.path.join(dataroot, 'ir')
        self.label_folder = os.path.join(dataroot, 'Segmentation_labels')
        self.vis_list = sorted(os.listdir(self.vis_folder))
        self.ir_list = sorted(os.listdir(self.ir_folder))
        self.label_list = sorted(os.listdir(self.label_folder))
        self.image_size = image_size

        k = 0
        tmp_len = len(self.vis_list)
        print(self.vis_folder, tmp_len)
        self.crop = transforms.CenterCrop(self.image_size)


    def __len__(self):
        return self.lens

    def __getitem__(self, index):
        """
        idx must be between 0 to len-1
        assuming flow[0] contains flow in x direction and flow[1] contains flow in y
        """
        image_name = self.ir_list[index]
        vis_path = os.path.join(self.vis_folder, image_name)
        ir_path = os.path.join(self.ir_folder, image_name)
        label_path = os.path.join(self.label_folder, image_name)
     
        vis = self.imread(path=vis_path)
        ir = self.imread(path=ir_path)
        label = self.imread(path=label_path, label=True)
        color = self.imread(path=vis_path, color=True)
        ir = self.crop(ir)
        vis = self.crop(vis)
        label = self.crop(label)
        color = self.crop(color)
        return ir, vis, label, color
    
    def __len__(self):
        return len(self.vis_list)
    
    @staticmethod
    def imread(path, label=False, color=False):
        if label:
            img = np.array(Image.open(path))
            img = np.asarray(Image.fromarray(img), dtype=np.int64)
            im_ts = torch.tensor(img)
        elif color:
            img = Image.open(path)
            im_ts = TF.to_tensor(img).unsqueeze(0) 
        else :
            img = Image.open(path).convert('L')
            im_ts = TF.to_tensor(img).unsqueeze(0) 
        return im_ts


if __name__ == "__main__":
    date_root="./MSRS-main/train/"
    image = ImageDataset(date_root,[256,256])
    print('data lens', len(image))
    dataloader = torch.utils.data.DataLoader(image, batch_size=1)
    for index, item in enumerate(dataloader):
        print('vi:', item[0].shape)
        print('ir:', item[1].shape)
        print('label:', item[2].shape)
 
    print('OK')





