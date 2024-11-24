import torch
import torch.nn as nn
import numpy as np
from GT import GlobalTokenTransformer
from ConvNeXt.ConvNeXt2 import ConvNeXt2
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

def window_partition(x, window_size):

    B,C, H, W  = x.shape
    x = x.view(B,C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size*window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / ((H // window_size) * (W // window_size)))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)
    return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        
class PatchExpand_X2_cel(nn.Module):
    def __init__(self, dim, norm_layer=nn.BatchNorm2d, patch_size=[2,4], factor=2):
        super().__init__()
        self.dim = dim
        self.reductions = nn.ModuleList()
        self.patch_size = patch_size 
        self.norm = norm_layer(dim) 
        # W,H,C
        for i, ps in enumerate(patch_size):
            if i == len(patch_size) - 1:
                out_dim = ( dim // 2 ** i) // factor 
            else:
                out_dim = (dim // 2 ** (i + 1)) // factor 
            stride = 2
            padding = (ps - stride) // 2 # 0,0;1,1
            self.reductions.append(nn.ConvTranspose2d(dim, out_dim, kernel_size=ps, stride=stride, padding=padding))

    def forward(self, x):
        # B,C,W,H
        x = self.norm(x) 
        xs = []
        for i in range(len(self.reductions)):
            tmp_x = self.reductions[i](x)
            xs.append(tmp_x)
        x = torch.cat(xs, dim=1)
        # B,C/2,2W,2H
        return x

class FinalPatchExpand_X4_cel(nn.Module):
    def __init__(self,dim, norm_layer=nn.BatchNorm2d, patch_size=[4,8,16,32]):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(dim) 
        self.reductions = nn.ModuleList()
        self.patch_size = patch_size
     
        # W,H,C
        for i, ps in enumerate(patch_size):
            if i == len(patch_size) - 1:
                out_dim = ( dim // 2 ** i) 
            else:
                out_dim = (dim // 2 ** (i + 1)) 
            stride = 4
            padding = (ps - stride) // 2 
            self.reductions.append(nn.ConvTranspose2d(dim, out_dim, kernel_size=ps, 
                                                stride=stride, padding=padding)) 

            
    def forward(self, x):
        """
        x: B,C,H,W
        """
        x = self.norm(x) 
        xs = []
        for i in range(len(self.reductions)):
            tmp_x = self.reductions[i](x)
            xs.append(tmp_x)
        x = torch.cat(xs, dim=1)
        # x: B,C,4H,4W
        return x

def Get_encoder( model_name="ConvNeXt",num_tokens=10):
    if model_name == "ConvNeXt":
        encoder_model = ConvNeXt2(in_chans=3, 
                                depths=[3, 3, 3, 3], 
                                dims=[96, 192, 384, 768], 
                                drop_path_rate=0.2,
                                layer_scale_init_value=1e-6, 
                                out_indices=[0, 1, 2, 3])
        
    elif model_name == "GlobalTokenTransformer":
        encoder_model = GlobalTokenTransformer(
                                            img_size=224,          # 입력 이미지 크기
                                            patch_size=4,          # 패치 크기
                                            in_chans=3,            # 입력 채널 수 (RGB 이미지의 경우 3)
                                            num_tokens=num_tokens,
                                            num_classes=2,      # 출력 클래스 수 (예: ImageNet의 경우 1000)
                                            embed_dim=96,          # 임베딩 차원
                                            depths=[2, 2, 2, 2],   # 각 스테이지의 레이어 수
                                            num_heads=[3, 6, 12, 24],  # 각 레이어의 헤드 수
                                            window_size=7,         # 윈도우 크기
                                            mlp_ratio=4.0,         # MLP의 hidden dimension 비율
                                            qkv_bias=True,         # QKV에 바이어스 추가
                                            drop_rate=0.3,         # 드롭아웃 비율
                                            attn_drop_rate=0.2     # 어텐션 드롭아웃 비율
                                        )

    else:
        raise AssertionError("Not implemented model")
    return encoder_model
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class Attention_Fusion_Module(nn.Module):
    def __init__(self, layer_i, in_dim, out_dim):
        super().__init__()

        self.layer_i = layer_i
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fuse = TokenFusionModule(dim=in_dim)

        if layer_i == 3:
            self.imagedecoder = nn.Sequential(
                                nn.Conv2d(in_dim*2, in_dim, 3, padding=1, bias=False),
                                LayerNorm(in_dim),
                                nn.GELU(),
                                FinalPatchExpand_X4_cel(dim=in_dim))# final cel x4   # out_dim = in_dim
        elif layer_i == 0:
            self.imagedecoder = nn.Sequential(
                                nn.Conv2d(in_dim, in_dim, 3, padding=1, bias=False),
                                LayerNorm(in_dim),
                                nn.GELU(),
                                PatchExpand_X2_cel(dim=in_dim, factor=2),)  # out_dim = in_dim // 2
        else:
            self.imagedecoder = nn.Sequential(
                                nn.Conv2d(in_dim*2, in_dim, 3, padding=1, bias=False),
                                LayerNorm(in_dim),
                                nn.GELU(),
                                PatchExpand_X2_cel(dim=in_dim, factor=2),) # out_dim = in_dim // 2


    def forward(self, l_x, g_x, f_out=None, global_token=None):
        out = self.fuse(l_x, g_x, global_token) 
        if f_out is not None:
            out = torch.cat([out, f_out],dim=1)
        out = self.imagedecoder(out)
        return out


class TokenFusionModule(nn.Module):
    def __init__(self, dim, window_size=7):
        super(TokenFusionModule, self).__init__()
        self.dim = dim
        self.window_size = window_size

        # Attention mechanism
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.lv = nn.Linear(dim, dim)

        self.assemble = nn.Sequential(
            nn.Conv2d(dim * 2, int(dim), 5, padding=2),
            LayerNorm(int(dim)),
            nn.Conv2d(int(dim), int(dim*4), 1),
            nn.GELU(),
            nn.Conv2d(int(dim*4), int(dim), 1),
        )

    def forward(self, local, glob,G_T):
        B, C, H, W = local.shape
        B,Nw,Nw,N,C = G_T.permute(0,3,4,2,1).contiguous().shape
        # Local and global features를 윈도우 단위로 분할
        local_windows = window_partition(local, self.window_size)  # nW*B, window_size*window_size, C
        G_T = G_T.view(-1,N,C)
        # Attention computation within each window
        keys = self.k(G_T)
        queries = self.q(local_windows)
        local_values = self.lv(G_T)

        # Window-based attention weights
        energy = torch.matmul(queries,keys.transpose(-2, -1))  # nW*B, window_size*window_size, window_size*window_size
        attention = torch.softmax(energy, dim=-1)
        # Apply attention within each window
        attended_local_windows = torch.matmul(attention, local_values) # nW*B, window_size*window_size, C
        # Reshape back to original image shape
        attended_local_feature = window_reverse(attended_local_windows, self.window_size, H, W)  # B, C , H, W
        final = torch.cat([attended_local_feature,glob],dim=1)
        # Combine local and global features
        combined = self.assemble(final)
        return combined    

class TokenSelectionModel(nn.Module):
    def __init__(self,in_channels):
        super(TokenSelectionModel, self).__init__()
        # 채널 축소: 768 -> 128 (예시)
        self.channel_reduction = nn.Conv1d(in_channels=in_channels, out_channels=128, kernel_size=1)
        
        # 토큰 중요도 학습: 각 토큰의 중요도를 평가
        self.token_attention = nn.Linear(128, 1)
        
        # 최종 분류기: 중요한 토큰을 선택한 후 이진 분류
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        # x의 shape: [batch_size, 768, 10, 1, 1] -> [batch_size, 768, 10]
        x = x.squeeze(-1).squeeze(-1)  # 차원 축소 [batch_size, 768, 10]
        
        # 채널 축소: [batch_size, 768, 10] -> [batch_size, 128, 10]
        x = self.channel_reduction(x)
        
        # 토큰 중요도 계산: [batch_size, 128, 10] -> [batch_size, 10]
        token_weights = self.token_attention(x.transpose(1, 2)).squeeze(-1)  # [batch_size, 10]
        token_weights = F.softmax(token_weights, dim=-1)  # 소프트맥스로 가중치 계산
        
        x = torch.matmul(x, token_weights.unsqueeze(-1)).squeeze(-1)
        
        # 최종 이진 분류: [batch_size, 128] -> [batch_size, 1]
        output = self.classifier(x)
        return output
    
class MFNet(nn.Module):
    def __init__(self, Global_branch="GlobalTokenTransformer", Local_branch="ConvNeXt", num_classes=1,ds=False,num_tokens=10):
        super().__init__()

        encoder_depth = len([ 2, 2, 2, 2 ])
        embed_dim = 96

        self.encoder_depth = encoder_depth
        self.decoder_depth = encoder_depth
        self.embed_dim = embed_dim
        
        # Global and Local encoder branch
        self.L_encoder = Get_encoder( Local_branch)
        self.G_encoder = Get_encoder( Global_branch,num_tokens)

        # attention fusion decoder
        self.Att_fusion = nn.ModuleList()
        for i in range(encoder_depth):
            input_dim = embed_dim*2**(encoder_depth - i - 1)
            att_fusion = Attention_Fusion_Module(
                                                layer_i=i,
                                                in_dim=input_dim, 
                                                out_dim=input_dim//2 if i < encoder_depth - 1 else input_dim
                                                )

            self.Att_fusion.append(att_fusion)
        
        # Segmentation Head
        self.segment = nn.Conv2d(in_channels=embed_dim, out_channels=num_classes, kernel_size=1, bias=False)
        self.token_classifier = TokenSelectionModel(embed_dim*8)
        ######## deep_supervision #########
        self.ds = ds
        if self.ds:
            self.deep_supervision = nn.ModuleList([
                nn.Sequential(nn.Conv2d(embed_dim*4, num_classes, 3, padding=1), nn.Upsample(scale_factor=16)),
                nn.Sequential(nn.Conv2d(embed_dim, num_classes, 3, padding=1), nn.Upsample(scale_factor=4))
            ])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)       

    def forward(self,x_l, x_g):
        if x_l.size()[1] == 1:
            x_l = x_l.repeat(1,3,1,1)
        if x_g.size()[1] == 1:
            x_g = x_g.repeat(1,3,1,1)
        # Obtain the intermediate layer features obtained by the encoder model (from bottom to top)
        G_features,G_tokens = self.G_encoder(x_g)
        L_features = self.L_encoder(x_l)

        assert len(G_features) == len(L_features), "the length of encoder does not match!"

        # deep supervision
        if self.ds:
            self.ds_out = []

        # The decoder fuses the features and restores the image resolution
        for idx in range(self.decoder_depth):
            if idx == 0:
                out = self.Att_fusion[idx](L_features[idx], G_features[idx], None, G_tokens[idx])
            else:
                out = self.Att_fusion[idx](L_features[idx], G_features[idx], out, G_tokens[idx]) 
            if self.ds:
                if idx % 2 == 0:
                    self.ds_out.append(self.deep_supervision[idx//2](out))

        # Segmentation Head
        out = self.segment(out)
        
        token_class = self.token_classifier(G_tokens[0])

        if self.ds:
            self.ds_out.append(out)
            
            return self.ds_out[-1],self.ds_out[-2],self.ds_out[-3],token_class
        else:
            return out


