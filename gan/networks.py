import torch
import torch.nn as nn
from torch.nn import functional as F


class CA(nn.Module):
    def __init__(self, noise_dim, emb_dim):
        super(CA, self).__init__()

        self.noise_dim = noise_dim
        self.emb_dim = emb_dim
        self.linear  = nn.Linear(noise_dim, emb_dim*2)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def sample_encoded_context(self, mean, logsigma, kl_loss=False, epsilon=None):
    
        if not isinstance(epsilon, torch.Tensor):
            epsilon = torch.cuda.FloatTensor(mean.size()).normal_()

        stddev  = logsigma.exp()

        return epsilon.mul(stddev).add_(mean)

    def forward(self, inputs, kl_loss=True, epsilon=None):
        '''
        inputs: (B, dim)
        return: mean (B, dim), logsigma (B, dim)
        '''
        out = self.relu(self.linear(inputs))
        mean = out[:, :self.emb_dim]
        log_sigma = out[:, self.emb_dim:]

        c = self.sample_encoded_context(mean, log_sigma, epsilon=epsilon)
        return c, mean, log_sigma


class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        
        self.res_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
    
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(dim)
        )

    def forward(self, input):
        return self.res_block(input) + input


class VecToFeatMap(nn.Module):
    # used to project a sentence code into a set of feature maps
    def __init__(self, in_dim, row, col, channel, activ=None):
        super(VecToFeatMap, self).__init__()
        
        out_dim = row*col*channel
        
        self.row = row
        self.col = col
        self.channel = channel

        self.out = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim)                
        )

    def forward(self, inputs):
        output = self.out(inputs)
        output = output.view(-1, self.channel, self.row, self.col)
        return output


class ImgEncoder(nn.Module):
    def __init__(self, num_chan, out_dim):

        super(ImgEncoder, self).__init__()

        self.node = nn.Sequential(
            nn.Conv2d(num_chan, out_dim//16, kernel_size=3, padding=1, bias=False, stride=2),
            nn.LeakyReLU(0.2, True), # 32

            nn.Conv2d(out_dim//16, out_dim//8, kernel_size=3, padding=1, bias=False, stride=2),
            nn.BatchNorm2d(out_dim//8),
            nn.LeakyReLU(0.2, True), # 16

            nn.Conv2d(out_dim//8, out_dim//4, kernel_size=3, padding=1, bias=False, stride=2),
            nn.BatchNorm2d(out_dim//4),
            nn.LeakyReLU(0.2, True), # 8

            nn.Conv2d(out_dim//4, out_dim//2, kernel_size=3, padding=1, bias=False, stride=2),
            nn.BatchNorm2d(out_dim//2),
            nn.LeakyReLU(0.2, True), # 4

            nn.Conv2d(out_dim//2, out_dim, kernel_size=4, padding=0, bias=False, stride=1),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(0.2, True) # 1
        )

    def forward(self, segs):
        return self.node(segs).squeeze(-1).squeeze(-1)


class Generator(nn.Module):
    def __init__(self, tcode_dim, scode_dim, emb_dim, hid_dim):
        """
        Parameters:
        ----------
        tcode_dim: int
            the dimension of sentence embedding
        scode_dim: int
            the dimension of segmentation embedding
        emb_dim : int
            the dimension of compressed sentence embedding.
        hid_dim: int
            used to control the number of feature maps.
        scode_dim : int
            the dimension of the segmentation embedding.
        """

        super(Generator, self).__init__()

        self.TCA = CA(tcode_dim, emb_dim)
        self.SCA = CA(scode_dim, emb_dim)
        self.BCA = CA(scode_dim, emb_dim)

        self.vec_to_tensor = VecToFeatMap(emb_dim*3, 4, 4, hid_dim*8) # TXT BG SEG

        cur_dim = hid_dim*8

        self.scale_4 = ResnetBlock(cur_dim)

        self.scale_8 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(cur_dim, cur_dim//2, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(cur_dim//2),
            nn.ReLU(True), 
            
            ResnetBlock(cur_dim//2),
        )

        self.scale_16 = nn.Sequential(
            ResnetBlock(cur_dim//2),
        )

        self.scale_32 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(cur_dim//2, cur_dim//4, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(cur_dim//4),
            nn.ReLU(True),
            
            ResnetBlock(cur_dim//4),
        )

        self.scale_64 = nn.Sequential(
            ResnetBlock(cur_dim//4),
        )

        self.tensor_to_img_64 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(cur_dim//4, 3, kernel_size=3, padding=0, bias=False),
            nn.Tanh()
        )

        self.apply(weights_init)

        self.txt_encoder_f = nn.GRUCell(300, 512)
        self.txt_encoder_b = nn.GRUCell(300, 512)

    def forward(self, txt_data=None, txt_len=None, seg_cond=None, bkg_cond=None, z_list=None,
        shape_noise=False, background_noise=False):

        out = []

        if not z_list:
            txt_data = txt_data.permute(1,0,2)

            hi_f = torch.zeros(txt_data.size(1), 512, device=txt_data.device)
            hi_b = torch.zeros(txt_data.size(1), 512, device=txt_data.device)
            h_f = []
            h_b = []
            mask = []
            for i in range(txt_data.size(0)):
                mask_i = (txt_data.size(0) - 1 - i < txt_len).float().unsqueeze(1).cuda()
                mask.append(mask_i)
                hi_f = self.txt_encoder_f(txt_data[i], hi_f)
                h_f.append(hi_f)
                hi_b = mask_i * self.txt_encoder_b(txt_data[-i - 1], hi_b) + (1 - mask_i) * hi_b
                h_b.append(hi_b)
            mask = torch.stack(mask[::-1])
            h_f = torch.stack(h_f) * mask
            h_b = torch.stack(h_b[::-1])
            h = (h_f + h_b) / 2
            txt_cond = h.sum(0) / mask.sum(0)

            z_t, tmean, tlogsigma   = self.TCA(txt_cond)
            z_s, smean, slogsigma = self.SCA(seg_cond)
            z_b, bmean, blogsigma = self.BCA(bkg_cond)

            if shape_noise:
                z_s = torch.cuda.FloatTensor(z_s.size()).normal_()
            if background_noise:
                z_b = torch.cuda.FloatTensor(z_b.size()).normal_()

            z_list = [z_t, z_s, z_b]
            
            out.append((tmean, tlogsigma))
            out.append((smean, slogsigma))
            out.append((bmean, blogsigma))

        z = torch.cat(z_list, dim=1) # BG SEG

        x = self.vec_to_tensor(z)

        x_4  = self.scale_4(x)
        x_8  = F.interpolate(x_4, scale_factor=2, mode='nearest')
        x_8  = self.scale_8(x_8)
        x_16 = F.interpolate(x_8, scale_factor=2, mode='nearest')
        x_16 = self.scale_16(x_16)
        x_32 = F.interpolate(x_16, scale_factor=2, mode='nearest')
        x_32 = self.scale_32(x_32)
        x_64 = F.interpolate(x_32, scale_factor=2, mode='nearest')
        x_64 = self.scale_64(x_64)

        img_64 = self.tensor_to_img_64(x_64)

        out.append(img_64)
        out.append(z_list)

        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.eps = 1e-7

        self.encoder_1 = nn.Sequential(
            nn.Conv2d(3, 128, 4, 2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.encoder_2 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.encoder_3 = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.GAP_1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.GAP_2 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.GAP_3 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # text feature
        self.txt_encoder_f = nn.GRUCell(300, 512)
        self.txt_encoder_b = nn.GRUCell(300, 512)

        self.gen_filter = nn.ModuleList([
            nn.Linear(512, 256 + 1),
            nn.Linear(512, 512 + 1),
            nn.Linear(512, 512 + 1)
        ])
        self.gen_weight = nn.Sequential(
            nn.Linear(512, 3),
            nn.Softmax(-1)
        )

        self.classifier = nn.Conv2d(512, 1, 4)

        self.apply(init_weights)

    def forward(self, img, txt_data, txt_len, negative=False):
        txt_data = txt_data.permute(1,0,2)

        img_feat_1 = self.encoder_1(img)
        img_feat_2 = self.encoder_2(img_feat_1)
        img_feat_3 = self.encoder_3(img_feat_2)
        img_feats = [self.GAP_1(img_feat_1), self.GAP_2(img_feat_2), self.GAP_3(img_feat_3)]
        D = self.classifier(img_feat_3).squeeze()

        # text attention
        u, m, mask = self._encode_txt(txt_data, txt_len)
        att_txt = (u * m.unsqueeze(0)).sum(-1)
        att_txt_exp = att_txt.exp() * mask.squeeze(-1)
        att_txt = (att_txt_exp / att_txt_exp.sum(0, keepdim=True))

        weight = self.gen_weight(u).permute(2, 1, 0)

        sim = 0
        for i in range(3):
            img_feat = img_feats[i]
            W_cond = self.gen_filter[i](u).permute(1, 0, 2)
            W_cond, b_cond = W_cond[:, :, :-1], W_cond[:, :, -1].unsqueeze(-1)
            img_feat = img_feat.mean(-1).mean(-1).unsqueeze(-1)

            sim += torch.sigmoid(torch.bmm(W_cond, img_feat) + b_cond).squeeze(-1) * weight[i]

        sim = torch.clamp(sim + self.eps, max=1).t().pow(att_txt).prod(0)

        return D, sim

    def _encode_txt(self, txt, txt_len):
        hi_f = torch.zeros(txt.size(1), 512, device=txt.device)
        hi_b = torch.zeros(txt.size(1), 512, device=txt.device)
        h_f = []
        h_b = []
        mask = []
        for i in range(txt.size(0)):
            mask_i = (txt.size(0) - 1 - i < txt_len).float().unsqueeze(1).cuda()
            mask.append(mask_i)
            hi_f = self.txt_encoder_f(txt[i], hi_f)
            h_f.append(hi_f)
            hi_b = mask_i * self.txt_encoder_b(txt[-i - 1], hi_b) + (1 - mask_i) * hi_b
            h_b.append(hi_b)
        mask = torch.stack(mask[::-1])
        h_f = torch.stack(h_f) * mask
        h_b = torch.stack(h_b[::-1])
        u = (h_f + h_b) / 2
        m = u.sum(0) / mask.sum(0)
        return u, m, mask


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        if m.weight.requires_grad:
            m.weight.data.normal_(std=0.02)
        if m.bias is not None and m.bias.requires_grad:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm2d) and m.affine:
        if m.weight.requires_grad:
            m.weight.data.normal_(1, 0.02)
        if m.bias.requires_grad:
            m.bias.data.fill_(0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1: 
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)