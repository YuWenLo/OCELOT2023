import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from user.fc.CatConv2d.catconv2d import CatConv2d

from skimage import feature
import cv2
import os
from util.constants import SAMPLE_SHAPE
from torchvision import transforms
import torchvision.transforms.functional as tf

class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, dropout=0.1):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel,
                                          stride=stride, padding=kernel//2, bias = False))
        self.add_module('norm', nn.BatchNorm2d(out_channels))
        self.add_module('relu', nn.ReLU(inplace=True))

        #print(kernel, 'x', kernel, 'x', in_channels, 'x', out_channels)

    def forward(self, x):
        return super().forward(x)
        
class BRLayer(nn.Sequential):
    def __init__(self, in_channels):
        super().__init__()
        
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(True))
    def forward(self, x):
        return super().forward(x)

class HarDBlock_v2(nn.Module):
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
          return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
          dv = 2 ** i
          if layer % dv == 0:
            k = layer - dv
            link.append(k)
            if i > 0:
                out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
          ch,_,_ = self.get_link(i, base_ch, growth_rate, grmul)
          in_channels += ch
        return out_channels, in_channels, link


    def get_out_ch(self):
        return self.out_channels

    def __init__(self, in_channels, growth_rate, grmul, n_layers, keepBase=False, residual_out=False, dwconv=False, list_out=False):
        super().__init__()
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.grmul = grmul
        self.n_layers = n_layers
        self.keepBase = keepBase
        self.links = []
        self.list_out = list_out
        layers_ = []
        self.out_channels = 0

        for i in range(n_layers):
          outch, inch, link = self.get_link(i+1, in_channels, growth_rate, grmul)
          self.links.append(link)
          use_relu = residual_out
          layers_.append(CatConv2d(inch, outch, (3,3), relu=True))

          if (i % 2 == 0) or (i == n_layers - 1):
            self.out_channels += outch
        # print("Blk out =",self.out_channels)
        self.layers = nn.ModuleList(layers_)

    def transform(self, blk):
        for i in range(len(self.layers)):
            self.layers[i].weight[:,:,:,:] = blk.layers[i][0].weight[:,:,:,:]
            self.layers[i].bias[:] = blk.layers[i][0].bias[:]

    def forward(self, x):
        layers_ = [x]
        #self.res = []
        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])

            out = self.layers[layer](tin)
            #self.res.append(out)
            layers_.append(out)
        t = len(layers_)
        out_ = []
        for i in range(t):
          if (i == 0 and self.keepBase) or \
             (i == t-1) or (i%2 == 1):
              out_.append(layers_[i])
        if self.list_out:
            return out_
        else:
            return torch.cat(out_, 1)

class HarDBlock(nn.Module):
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
          return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
          dv = 2 ** i
          if layer % dv == 0:
            k = layer - dv
            link.append(k)
            if i > 0:
                out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
          ch,_,_ = self.get_link(i, base_ch, growth_rate, grmul)
          in_channels += ch
        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels
 
    def __init__(self, in_channels, growth_rate, grmul, n_layers, keepBase=False, residual_out=False):
        super().__init__()
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.grmul = grmul
        self.n_layers = n_layers
        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0 # if upsample else in_channels
        for i in range(n_layers):
          outch, inch, link = self.get_link(i+1, in_channels, growth_rate, grmul)
          self.links.append(link)
          use_relu = residual_out
          layers_.append(ConvLayer(inch, outch))
          if (i % 2 == 0) or (i == n_layers - 1):
            self.out_channels += outch
        #print("Blk out =",self.out_channels)
        self.layers = nn.ModuleList(layers_)


    def forward(self, x):
        layers_ = [x]
        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:
                x = torch.cat(tin, 1)
            else:
                x = tin[0]
            out = self.layers[layer](x)
            layers_.append(out)
        t = len(layers_)
        out_ = []
        for i in range(t):
          if (i == 0 and self.keepBase) or \
             (i == t-1) or (i%2 == 1):
              out_.append(layers_[i])
        out = torch.cat(out_, 1)
        return out

class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #print("upsample",in_channels, out_channels)

    def forward(self, x, skip, concat=True):
        is_v2 = type(skip) is list
        if is_v2:
            skip_x = skip[0]
        else:
            skip_x = skip
        out = F.interpolate(
                x,
                size=(skip_x.size(2), skip_x.size(3)),
                mode="bilinear",
                align_corners=True,
                            )
        if concat:       
          if is_v2:
            out = [out] + skip
          else:                     
            out = torch.cat([out, skip], 1)
          
        return out

class hardnet(nn.Module):
    def __init__(self, n_classes=19):
        super(hardnet, self).__init__()

        first_ch  = [16,24,32,48]
        ch_list = [  64, 96, 160, 224, 320]
        grmul = 1.7
        gr       = [  10,16,18,24,32]
        n_layers = [   4, 4, 8, 8, 8]

        blks = len(n_layers) 
        self.shortcut_layers = []

        self.base = nn.ModuleList([])
        self.base.append (
             ConvLayer(in_channels=3, out_channels=first_ch[0], kernel=3,
                       stride=2) )
        self.base.append ( ConvLayer(first_ch[0], first_ch[1],  kernel=3) )
        self.base.append ( ConvLayer(first_ch[1], first_ch[2],  kernel=3, stride=2) )
        self.base.append ( ConvLayer(first_ch[2], first_ch[3],  kernel=3) )

        skip_connection_channel_counts = []
        ch = first_ch[3]
        for i in range(blks):
            blk = HarDBlock(ch, gr[i], grmul, n_layers[i])
            ch = blk.get_out_ch()
            skip_connection_channel_counts.append(ch)
            self.base.append ( blk )
            if i < blks-1:
              self.shortcut_layers.append(len(self.base)-1)

            self.base.append ( ConvLayer(ch, ch_list[i], kernel=1) )
            ch = ch_list[i]
            
            if i < blks-1:            
              self.base.append ( nn.AvgPool2d(kernel_size=2, stride=2) )


        cur_channels_count = ch
        prev_block_channels = ch
        n_blocks = blks-1
        self.n_blocks =  n_blocks

        #######################
        #   Upsampling path   #
        #######################

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        self.conv1x1_up    = nn.ModuleList([])
        
        for i in range(n_blocks-1,-1,-1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]
            self.conv1x1_up.append(ConvLayer(cur_channels_count, cur_channels_count//2, kernel=1))
            cur_channels_count = cur_channels_count//2

            blk = HarDBlock(cur_channels_count, gr[i], grmul, n_layers[i])
            
            self.denseBlocksUp.append(blk)
            prev_block_channels = blk.get_out_ch()
            cur_channels_count = prev_block_channels


        self.finalConv = nn.Conv2d(in_channels=cur_channels_count,
               out_channels=n_classes, kernel_size=1, stride=1,
               padding=0, bias=True)
    
    def v2_transform(self):        
        for i in range( len(self.base)):
            if isinstance(self.base[i], HarDBlock):
                blk = self.base[i]
                self.base[i] = HarDBlock_v2(blk.in_channels, blk.growth_rate, blk.grmul, blk.n_layers, list_out=True)
                self.base[i].transform(blk)
            elif isinstance(self.base[i], nn.Sequential):
                blk = self.base[i]
                sz = blk[0].weight.shape
                if sz[2] == 1:
                    self.base[i] = CatConv2d(sz[1],sz[0],(1,1), relu=True)
                    self.base[i].weight[:,:,:,:] = blk[0].weight[:,:,:,:]
                    self.base[i].bias[:] = blk[0].bias[:]

        for i in range(self.n_blocks):
            blk = self.denseBlocksUp[i]
            self.denseBlocksUp[i] = HarDBlock_v2(blk.in_channels, blk.growth_rate, blk.grmul, blk.n_layers, list_out=False)
            self.denseBlocksUp[i].transform(blk)
  
        for i in range(len(self.conv1x1_up)):
            blk = self.conv1x1_up[i]
            sz = blk[0].weight.shape
            if sz[2] == 1:
                self.conv1x1_up[i] = CatConv2d(sz[1],sz[0],(1,1), relu=True)
                self.conv1x1_up[i].weight[:,:,:,:] = blk[0].weight[:,:,:,:]
                self.conv1x1_up[i].bias[:] = blk[0].bias[:]                 

    def forward(self, x):
        
        skip_connections = []
        size_in = x.size()
        
        
        for i in range(len(self.base)):
            x = self.base[i](x)
            if i in self.shortcut_layers:
                skip_connections.append(x)
        out = x
        
        for i in range(self.n_blocks):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip, True)
            out = self.conv1x1_up[i](out)
            out = self.denseBlocksUp[i](out)
        
        out = self.finalConv(out)
        
        out = F.interpolate(
                            out,
                            size=(size_in[2], size_in[3]),
                            mode="bilinear",
                            align_corners=True)
        return out


class PytorchaFcCellModel():
    """
    Parameters
    ----------
    metadata: Dict
        Dataset metadata in case you wish to compute statistics

    """
    def __init__(self, metadata):
        self.device = torch.device('cuda:0')
        self.metadata = metadata
        self.resize_to = 1024 # The model is trained with 1024 resolution
        # RGB images and 2 class prediction
        self.cell_n_classes =  1 # Two cell classes and background
        self.tissue_n_classes =  3

        # cell
        self.cell_hardnet = hardnet(n_classes=self.cell_n_classes)
        # self.load_checkpoint(mode='cell', weight_path = "checkpoints/all_cell_1c_r5_2nd_epoch199_5165.pth")
        # self.cell_hardnet = self.cell_hardnet.to(self.device)
        # self.cell_hardnet.eval()
        self.cell_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.resize_to, self.resize_to)),
            transforms.ToTensor(), 
            transforms.Normalize([0.7610, 0.5776, 0.6962], [0.1515, 0.1870, 0.1426])])
        
        # tissue
        self.tissue_hardnet = hardnet(n_classes=self.tissue_n_classes)
        # self.load_checkpoint(mode='tissue', weight_path = "checkpoints/tissue_epoch649_9651.pth")
        # self.tissue_hardnet = self.tissue_hardnet.to(self.device)
        # self.tissue_hardnet.eval()
        self.tissue_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.resize_to, self.resize_to)),
            transforms.ToTensor(), 
            transforms.Normalize([0.7653, 0.5833, 0.6994], [0.1548, 0.1969, 0.1566])])

    def load_checkpoint(self, mode='cell', weight_path = "checkpoints/all_cell_1c_r5_2nd_epoch199_5165.pth"):
        """Loading the trained weights to be used for validation"""
        _curr_path = os.path.split(__file__)[0]
        _path_to_checkpoint = os.path.join(_curr_path, weight_path)
        # state_dict = torch.load(_path_to_checkpoint, map_location=torch.device('cpu'))
        if mode == 'cell':
            # self.cell_hardnet.load_state_dict(state_dict, strict=True)
            self.cell_hardnet.load_state_dict(torch.load(_path_to_checkpoint))
            print("Cell Weights were successfully loaded!")
        else:
            # self.tissue_hardnet.load_state_dict(state_dict, strict=True)
            self.tissue_hardnet.load_state_dict(torch.load(_path_to_checkpoint))
            print("Tissue Weights were successfully loaded!")

    def prepare_input(self, cell_patch, tissue_patch):
        """This function prepares the cell patch array to be forwarded by
        the model

        Parameters
        ----------
        cell_patch: np.ndarray[uint8]
            Cell patch with shape [1024, 1024, 3] with values from 0 - 255
        tissue_patch: np.ndarray[uint8] 
            Tissue patch with shape [1024, 1024, 3] with values from 0 - 255

        Returns
        -------
            torch.tensor of shape [1, 3, 1024, 1024] where the first axis is the batch
            dimension
        """
        # cell_patch = torch.from_numpy(cell_patch).permute((2, 0, 1)).unsqueeze(0)
        # cell_patch = cell_patch.to(self.device).type(torch.cuda.FloatTensor)
        # cell_patch = cell_patch / 255 # normalize [0-1]
        # if self.resize_to is not None:
        #     cell_patch= F.interpolate(
        #             cell_patch, size=self.resize_to, mode="bilinear", align_corners=True
        #     ).detach()


        cell_patch = self.cell_transform(cell_patch)
        tissue_patch = self.tissue_transform(tissue_patch)

        return cell_patch.unsqueeze(0).cuda(), tissue_patch.unsqueeze(0).cuda()

        # return cell_patch
        
    def find_cells_1ch(self, heatmap):
        """This function detects the cells in the output heatmap

        Parameters
        ----------
        heatmap: torch.tensor
            output heatmap of the model,  shape: [1, 3, 1024, 1024]

        Returns
        -------
            List[tuple]: for each predicted cell we provide the tuple (x, y, cls, score)
        """
        arr = heatmap[0,:,:,:].cpu().detach().numpy()
        # arr = np.transpose(arr, (1, 2, 0)) # CHW -> HWC

        # arr, pred_wo_bg = np.split(arr, (1,), axis=0) # Background and non-background channels
        #bg = np.squeeze(bg, axis=0)
        #obj = 1.0 - bg
        arr= np.squeeze(arr, axis=0)

        arr = cv2.GaussianBlur(arr, (0, 0), sigmaX=3)
        peaks = feature.peak_local_max(
            arr, min_distance=10, exclude_border=0, threshold_abs=0.55
        ) # List[y, x]

        #maxval = np.max(pred_wo_bg, axis=0)

        # Filter out peaks if background score dominates
        #peaks = np.array([peak for peak in peaks if bg[peak[0], peak[1]] < maxval[peak[0], peak[1]]])
        if len(peaks) == 0:
            return []

        # Get score and class of the peaks
        scores = arr[peaks[:, 0], peaks[:, 1]]

        predicted_cells = [(x, y, 1, float(s)) for [y, x], s in zip(peaks, scores)]

        return predicted_cells

    def post_process(self, logits, mode='cell'):
        """This function applies some post processing to the
        output logits
        
        Parameters
        ----------
        logits: torch.tensor
            Outputs of U-Net

        Returns
        -------
            torch.tensor after post processing the logits
        """
        if self.resize_to is not None:
            logits = F.interpolate(logits, size=SAMPLE_SHAPE[:2],
                mode='bilinear', align_corners=False
            )
            
        if mode == 'cell':
            # torch.softmax(logits, dim=1)
            return torch.sigmoid(logits)
        else:
            maxcls_0 = np.argmax(logits[0].cpu().detach().numpy(), axis=0)
            return maxcls_0

    def __call__(self, cell_patch, tissue_patch, pair_id):
        """This function detects the cells in the cell patch using Pytorch U-Net.

        Parameters
        ----------
        cell_patch: np.ndarray[uint8]
            Cell patch with shape [1024, 1024, 3] with values from 0 - 255
        tissue_patch: np.ndarray[uint8] 
            Tissue patch with shape [1024, 1024, 3] with values from 0 - 255
        pair_id: str
            Identification number of the patch pair

        Returns
        -------
            List[tuple]: for each predicted cell we provide the tuple (x, y, cls, score)
        """
        
        # data process
        cell_patch, tissue_patch = self.prepare_input(cell_patch, tissue_patch)
        
        # cell
        _curr_path = os.path.split(__file__)[0]
        weights_dir = os.path.join(_curr_path, 'checkpoints/all5fold_cell_1c_r5_1ch/')
        weights = sorted(os.listdir(weights_dir))
        logits = None
        for k, weight in enumerate(weights):
            # print('weight = ', weight)
            self.load_checkpoint(mode='cell', weight_path = os.path.join('checkpoints/all5fold_cell_1c_r5_1ch/', weight))
            self.cell_hardnet = self.cell_hardnet.to(self.device)
            self.cell_hardnet.eval()

            fold_result = self.cell_hardnet(cell_patch)
            fold_result = (fold_result + tf.vflip(tf.hflip(self.cell_hardnet(tf.vflip(tf.hflip(cell_patch))))))/2

            if k == 0:
                logits = fold_result
            else:
                logits += fold_result

        logits = logits / len(weights)
        heatmap = self.post_process(logits, mode='cell')
        cell_predictions = self.find_cells_1ch(heatmap)
        del cell_patch, logits, heatmap
        
        # print("cell weight num = ", len(weights))
        
        # tissue
        _curr_path = os.path.split(__file__)[0]
        weights_dir = os.path.join(_curr_path, 'checkpoints/all5fold_tissue_strloss_1000e/')
        weights = sorted(os.listdir(weights_dir))
        tissue_logits = None
        for k, weight in enumerate(weights):
            # print('weight = ', weight)
            self.load_checkpoint(mode='tissue', weight_path = os.path.join('checkpoints/all5fold_tissue_strloss_1000e/', weight))
            self.tissue_hardnet = self.tissue_hardnet.to(self.device)
            self.tissue_hardnet.eval()

            fold_result = self.tissue_hardnet(tissue_patch)
            fold_result = (fold_result + tf.vflip(tf.hflip(self.tissue_hardnet(tf.vflip(tf.hflip(tissue_patch))))))/2

            if k == 0:
                tissue_logits = fold_result
            else:
                tissue_logits += fold_result

        tissue_logits = tissue_logits / len(weights)
        tissue_heatmap_maxcls = self.post_process(tissue_logits, mode='tissue')
        del tissue_patch, tissue_logits
        # print("tissue weight num = ", len(weights))
        
        # metadata
        cell_half_size = 128
        info = self.metadata[pair_id]
        # cell_info = info['cell']
        # tissue_info = info['tissue']
        patch_x_offset = info['patch_x_offset']
        patch_y_offset = info['patch_y_offset']
        height, width = 1024, 1024

        cell_y_c = int(height * patch_y_offset)
        cell_x_c = int(width * patch_x_offset)
        cell_area = tissue_heatmap_maxcls[cell_y_c-cell_half_size:cell_y_c+cell_half_size, cell_x_c-cell_half_size:cell_x_c+cell_half_size]
        # cell_area = ndimage.zoom(cell_area, zoom=(4, 4), order=1)
        # cell_area = np.resize(cell_area, (cell_area.shape[0]*4, cell_area.shape[1]*4))
        
        # print("origin ------------------------------------------------")
        # for point in cell_predictions:
        #     print(point)
        
        # tissue filter
        for i, point in enumerate(cell_predictions):
            # print("point = ", point)
            x, y, point_class, score = point
            x, y = int(x/4), int(y/4)

            if cell_area[y, x] != point_class:
                if cell_area[y, x] == 1 or cell_area[y, x] == 2:
                    point_tmp = list(point)
                    point_tmp[2] = cell_area[y, x]
                    cell_predictions[i] = tuple(point_tmp)

            # print(point)
        # print("after ------------------------------------------------")
        # for point in cell_predictions:
        #     print(point)
        return cell_predictions


