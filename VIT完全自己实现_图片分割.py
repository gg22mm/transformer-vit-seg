# 四种天气图片数据分类 - 卷积 - 创建Dataloader和可视化图片 自己准备的数据 - Conv2d
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision #读取图片方法
import os
import shutil
from torchvision import transforms
import math
# 
import dill
from sklearn.metrics import accuracy_score

# pip install einops -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install dill -i https://pypi.tuna.tsinghua.edu.cn/simple
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# #
from torch.utils import data 
import glob #正则获取所有图片所有路径
from PIL import Image
from torch.optim import lr_scheduler
####################### 数据集一 ##################################


# 1、训练数据图片

# 获取所有的png图片（主图与蒙贝图一起，因为都放一个目录中了）
all_pics = sorted( glob.glob(r'dataset/HKdataset/training/*.png') ) #这里可以不用加sorted，因为只读一次，不像其它的是读两次才要排序规则是一样的
all_pics[:6]

# (x for x in if)  或 ( x.处理 for x in )
images = [p for p in all_pics if 'matte' not in p] #只获取主图，这里用了 matte not in 不包含matte意思就是不获取蒙版的所有图
len(images) #1700 ，这么多图
images[:6]

# 蒙版图-只获取包含matte的，而这些都是蒙版图
annotations = [p for p in all_pics if 'matte' in p]
len(annotations) #1700 ，这么多图，对得上主图了
images[:6],annotations[:6] #看了一下对得上

# 随机打乱
np.random.seed(2021) #固定随机
index = np.random.permutation(len(images))

# 分批获取图片
images = np.array(images)[index]    #主图
anno = np.array(annotations)[index] #蒙板图
images,anno

# 2、测试数据
all_test_pics = sorted( glob.glob(r'dataset/HKdataset/testing/*.png') )
all_test_pics[:6]
test_images = [p for p in all_test_pics if 'matte' not in p]
test_anno = [p for p in all_test_pics if 'matte' in p]
test_images[:6],test_anno[:6] #看了一下对得上

# 数据增强方法转换，下面调用此方法，先定义数据增强方法
transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
])
class Mydataset(data.Dataset):
    def __init__(self, img_paths, anno_paths):#主图，蒙版图
        self.imgs = img_paths    #主图
        self.annos = anno_paths  #蒙版图
        
    def __getitem__(self, index): #实现切片方法-分割获取
        img = self.imgs[index]
        anno = self.annos[index]

        #修复主图
        pil_img = Image.open(img)       #读取打开主图
        pil_img = pil_img.convert('RGB') 
        img_tensor = transform(pil_img) #打开后-格式转换  #正常三维图直接转transform         
      
        # 蒙版图
        pil_anno = Image.open(anno)       #读取蒙版图
        anno_tensor = transform(pil_anno) #格式转换
        #
        anno_tensor = torch.squeeze(anno_tensor).type(torch.long) #为1的维度去掉如：1，200，200 去为1的维度变成：200*200，为什么要删除维度为1的呢，因为要保持跟传进来的 y 也就是蒙版图格式一样是二维的，模型预测最后也是返加二维的
        anno_tensor[anno_tensor > 0] = 1  #对蒙版实现0和1，人为二分类
        
        return img_tensor, anno_tensor
    
    def __len__(self):
        return len(self.imgs)


# 生成dataset
trainDataset = Mydataset(images, anno)
validRow = Mydataset(test_images, test_anno)
# 生成dataRwo
batch = 8
trainRow = data.DataLoader(
                           trainDataset,
                           batch_size=batch,
                           shuffle=True,
)
validRow = data.DataLoader(
                          validRow,
                          batch_size=batch,
)

# 拿一批数据来测试一下
imgs_batch, annos_batch = next(iter(trainRow))
# print(imgs_batch.shape) #torch.Size([8, 3, 224, 224])
# print(annos_batch.shape)  #torch.Size([8, 224, 224])


img = imgs_batch[0].permute(1,2,0).numpy() #这个是三维图片，所以才要转通道
anno = annos_batch[0].numpy()              #这个上面已经去了1维，所以是正的是二维灰度的图片，plt不需要转

# plt.subplot(1,2,1)
# plt.imshow(img)
# plt.show()

# plt.subplot(1,2,2)
# plt.imshow(anno)
# plt.show()


####################### 模型定义二 ##################################


# 注意力 是 Multi_Head_Attention 子逻辑方法
class Attention(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        emb = torch.matmul(torch.matmul(q, k.transpose(-1, -2))/math.sqrt(x.shape[-1]), v)
        return emb

# 编码2 - 注意力逻辑
class Multi_Head_Attention(nn.Module):
    def __init__(self, dim, n_head) -> None:
        super().__init__()
        assert dim % n_head == 0 
        self.n_head = n_head
        self.attention = Attention(dim=dim//self.n_head)

    def split(self, x):
        n_batch, num, length = x.shape
        x = x.view(n_batch, self.n_head, num, length//self.n_head).contiguous()
        return x 

    def concat(self, x):
        n_batch, n_head, num, length = x.shape
        x = x.view(n_batch, num, n_head*length).contiguous()
        return x 

    def forward(self, x):
        x = self.split(x)
        # print('---split---:', x.shape)
        x = self.attention(x)
        x = self.concat(x)
        # print('---concat--:', x.shape)
        return x 

# 编码2 - LayerNorm逻辑
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model)) #torch.Size([64]) 值全是1的
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        out = (x - mean) / (std + self.eps)
        out = self.gamma * out + self.beta
        return out

# 编码2 - 前馈网络逻辑
class FeedForward(nn.Module):
    def __init__(self, dim, dropout) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        ) 
    def forward(self, x):
        x = self.net(x)
        return x 
       
# 编码1 - 层数堆叠 - 编码控制器 - 这是方法会被循环多次
class Encoder_layer(nn.Module):
    def __init__(self, dim, n_head, dropout) -> None:
        super().__init__()
        self.norm = LayerNorm(dim)                          #编码2 - LayerNorm逻辑
        self.dropout = nn.Dropout(dropout)
        self.attention = Multi_Head_Attention(dim, n_head)  #编码2 - 注意力逻辑
        self.ffn = FeedForward(dim=dim, dropout=dropout)    #编码2 - 前馈网络逻辑

    def forward(self, x):
        # 残差连接
        x_ = x
        x = self.attention(x)
        x = self.norm(x + x_)
        x = self.dropout(x)
        x_ = x
        x = self.ffn(x)
        x = self.norm(x + x_)
        x = self.dropout(x)
        return x 


        
# 编码入口
class Encoder(nn.Module):
    def __init__(self, dim, n_head, dropout, n_layers) -> None:
        super().__init__()
        self.n_layers = nn.ModuleList([Encoder_layer(dim, n_head, dropout) for _ in range(n_layers)])

    def forward(self, x):
        for layers in self.n_layers:
            x = layers(x)
        return x 

def unpadding(y, target_size):
    H, W = target_size
    H_pad, W_pad = y.size(2), y.size(3)
    # crop predictions on extra pixels coming from padding
    extra_h = H_pad - H
    extra_w = W_pad - W
    if extra_h > 0:
        y = y[:, :, :-extra_h]
    if extra_w > 0:
        y = y[:, :, :, :-extra_w]
    return y

# 投影 - 解码
class Mlp_Head(nn.Module):
    def __init__(self, hidden_dim, num_class, dropout) -> None:
        super().__init__()
        # self.net = nn.Sequential(
        #     Rearrange('b n l -> b (n l)'),
        #     nn.Linear(hidden_dim, num_class), 
        #     nn.Dropout(dropout)
        # )

        self.n_cls=num_class
        self.proj_dec = nn.Linear(192, hidden_dim) #进来后，统一改成 hidden_dim
        self.cls_emb = nn.Parameter(torch.randn(1, num_class, hidden_dim))                           #self.cls_emb = nn.Parameter(torch.randn(1, n_cls, dim))
        self.proj_patch = nn.Parameter(hidden_dim ** -0.5 * torch.randn(hidden_dim, hidden_dim))     #self.proj_patch = nn.Parameter(dim ** -0.5 * torch.randn(dim, dim))
        self.proj_classes = nn.Parameter(hidden_dim ** -0.5 * torch.randn(hidden_dim, hidden_dim))   #self.proj_classes = nn.Parameter(dim ** -0.5 * torch.randn(dim, dim))        
        self.mask_norm = nn.LayerNorm(num_class)                                                     #self.mask_norm = nn.LayerNorm(n_cls)
        

    def forward(self, x):        
        # x = self.net(x)
        # return x


        # print('解码x1：',x.shape) #torch.Size([8, 197, 192])

        #################
        # # # 说明例子, 假设
        # # # 正确的应该是这样: torch.Size([8, 64, 256, 256])
        # # # 而vit确是这样:   torch.Size([8, 257, 64]) 所以有了下面的骚操作，少了一维，就要添加进来一维
        ################


        # 之前vit中多添加的,现在进行删除
        # remove CLS/DIST tokens for decoding 删除用于解码的CLS/DIST令牌       
        x = x[:, 1:]#从第一个开始取，删除了第0个
        # print('X7-之前vit中多添加的,现在进行删除：',x.shape) #torch.Size([8, 196, 192])

        x = self.proj_dec(x)
        # print('x1:',x.shape) #torch.Size([8, 196, 192])


        ################ 开始不一样的地方 扩展维度 ###################

        # 1、
        
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        # print('cls_emb:',cls_emb.shape) #torch.Size([8, 150, 192]) ， 现 torch.Size([8, 2, 192])
        
        x = torch.cat((x, cls_emb), 1)
        # print('x2:',x.shape) #torch.Size([8, 346, 192]) ， 现 torch.Size([8, 198, 192])
        

        n_cls=self.n_cls #分类
        patches, cls_seg_feat = x[:, : -n_cls], x[:, -n_cls :]
        patches = patches @ self.proj_patch
        cls_seg_feat = cls_seg_feat @ self.proj_classes

        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)

        masks = patches @ cls_seg_feat.transpose(1, 2)
        masks = self.mask_norm(masks)
        masks = rearrange(masks, "b (h w) n -> b n h w", h=n_cls)
        # print('masks:',masks.shape) #torch.Size([8, 150, 14, 14])   ， 现  torch.Size([8, 2, 2, 98])          

        #2、有上面的masks才能用下面的

        # 原来通过这个加一维的 - 224 这个配置要与图片尺寸保持一样
        image_height=224
        image_width=224
        masks = F.interpolate(masks, size=(image_height, image_width), mode="bilinear")       
        masks = unpadding(masks, (image_height, image_width))
        # print('masks22:',masks.shape) #torch.Size([8, 150, 14, 14])   ， 现  torch.Size([8, 2, 224, 224])
        # exit() 

        return  masks


# 编码
class VIT(nn.Module):
    
    def __init__(self):        

        super().__init__()

        # 图片 transforms.Resize 尺寸对应
        image_height=224
        image_width=224 

        # Transformer 切成的图片尺寸
        patch_height=16
        patch_width=16

        # 图片维度
        in_channels=3

        # 
        dim=192 #改成与v2一样
        n_head=8
        p=0.1
        n_layers=5
        
        
        # 整除操作 - 
        assert image_height % patch_height == 0 and image_width % patch_width == 0 # 
        
        # 计算图片 patch
        patch_dim = patch_height*patch_width*in_channels #Transformer图片尺寸768= 16 * 16 * 3        
        num_patches = (image_height*image_width)//(patch_height*patch_width) #64=(128*128)/(16*16)

        # 切割图片，并扁平化处理 - 直接分割即把图像直接分成多块
        self.patches_embedding = nn.Sequential(
            # 直接分割即把图像直接分成多块。在代码实现上需要使用einops这个库，完成的操作是将（B，C，H，W）的shape调整为（B，(H/P *W/P)，P*P*C）
            # Rearrange 说明：https://blog.51cto.com/u_15279692/5756365
            Rearrange(pattern='b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width), 
            nn.Linear(patch_dim, dim) #768,dim=64
        )

        # torch.nn.Parameter()将一个不可训练的tensor转换成可以训练的类型parameter，并将这个parameter绑定到这个module里面。即在定义网络时这个tensor就是一个可以训练的参数了。使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。
        # https://blog.csdn.net/weixin_44878336/article/details/124733598
         
        # 位置编码  torch.randn(size=(1, int(num_patches)+1, dim)) = torch.Size([1, 65, 64])
        # nn.Parameter() 后的格式 torch.Size([1, 65, 64])
        self.pos_embedding = nn.Parameter(torch.randn(size=(1, int(num_patches)+1, dim)))#dim=64

        # 类别信息
        # torch.randn(size=(1, 1, dim)) = torch.Size([1, 1, 64])
        self.cls_token = nn.Parameter(torch.randn(size=(1, 1, dim)))#dim=64
        self.dropout = nn.Dropout(p) #自定义才会有这个0.1,官方版的自带默认也是这个0.1
        
        # # 编码器 - 用官方的效果极差
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim,nhead=n_head)#定义encoder_layer下面要用,注意： nhead 要被 embed_dim 整除
        # self.encoder = nn.TransformerEncoder(self.encoder_layer,num_layers=n_layers)   

        # 编码器 - 自己写的效果极好
        self.encoder = Encoder(dim, n_head, p, n_layers)#dim=64
        
        # 投影
        self.mlp = Mlp_Head(hidden_dim=dim, num_class=2, dropout=p) #dim=64


    def forward(self, x):
        
        # im = x  # # 扩展一维出来 - 方式二、v2中的代码

        # print('X1-直接输入:', x.shape) #torch.Size([8, 3, 224, 224])
        
        x = self.patches_embedding(x)
        # print('X2-emb:', x.shape) #torch.Size([8, 196, 192])
        
                
        batch, n, _ = x.shape #196,192
        
        #-----------
        # 扩展一维出来 - 方式一 、拼接class_token信息
        cls_token = repeat(self.cls_token, '1 1 d -> b 1 d', b=batch)
        # print('添加一维:', cls_token.shape) #torch.Size([8, 1, 192])

        x = torch.cat((cls_token, x), dim=1)  
        # print('X3-合并新添加的一维:', x.shape) #torch.Size([8, 197, 192])

        # # 扩展一维出来 - 方式二、v2中的代码
        # B, _, H, W = im.shape
        # cls_tokens = self.cls_token.expand(B, -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)
        # print('X3-合并新添加的一维:', x.shape) #torch.Size([8, 197, 192])
        #-----------        
        

        # batch里每个图片均加入相同的位置编码信息 - 只是添加位置信息
        # print(self.pos_embedding.shape) #torch.Size([1, 197, 192])           
        x += self.pos_embedding[:, :(n + 1) , :]
        # print('X4-batch里每个图片均加入相同的位置编码信息:',x.shape) #torch.Size([8, 197, 192])
        
        # 
        x = self.dropout(x)
        
        # Transformer 编码器 - 自己写的效果极好 - 用官方的效果极差
        x = self.encoder(x)
        # print('X5-放到vit编码中后：',x.shape) #torch.Size([8, 197, 192])
        # exit()
        
        # 投影 - 解码
        x = self.mlp(x)  
        # print('投影 - 解码后：',x.shape)
        # exit()       

        return x




# 预训练
device = 'cpu'
epoches=60
model = VIT()
model.to(device)


# 定义交叉熵损失函数和优化器
Loss = []
loss_fn = nn.CrossEntropyLoss()
opt = optim.AdamW(model.parameters(), lr=0.005)

# 模型训练
for epoch in range(epoches):
    train_loss = []
    predict = []
    image_label = []

    for x, y in trainRow:
        x=x.to(device)
        y=y.to(device)
        
        # print('输入x：',x.shape) #torch.Size([8, 3, 224, 224])
        # print('输入y：',y.shape) #torch.Size([8, 224, 224])

        y_pred = model(x)            #预测结果

        # print('y_pred：',y_pred.shape) #torch.Size([8, 2, 224, 224])

        loss = loss_fn(y_pred, y)   #预测值，真实值          
        opt.zero_grad()             #默认梯度为0
        loss.backward()             #损失反向传播放        
        opt.step()                  #优化模型
        
        # 损失
        train_loss.append(loss.detach().cpu().item())

        # # 网络预测
        # predict.extend(y_pred.argmax(dim=1).detach().cpu().tolist())
        # image_label.extend(y.detach().cpu().tolist())

    # assert len(predict) == len(image_label)
    # acc = accuracy_score(image_label, predict)
    
    # 每个Epoch损失平均值
    Loss.append(np.mean(np.array(train_loss)))
    print('Epoch:', epoch, 'Model Loss:', Loss[-1], 'Accuracy: ', 0)

    # 模型保存
    torch.save(model.state_dict(), './output.pth', _use_new_zipfile_serialization=False)

import matplotlib.pyplot as plt
plt.title('The Train Loss of Vision Transformer')
plt.plot(range(len(Loss)), Loss, label='Loss')
plt.legend()
plt.show()


# # 预测 
# vit = VIT(image_height=128, image_width=128, in_channels=3, patch_height=16, patch_width=16, dim=64, n_head=8, p=0.1, n_layers=5, num_class=6)
# vit.load_state_dict(torch.load(r'./Model/vit.pth'))

# from sklearn.metrics import accuracy_score

# def test(path):
#     predict = []
#     label = []
    
#     # 导入测试数据
#     with open(r'./test_loader.pth', 'rb') as f: 
#         test_image = dill.load(f)
#         for image_item, label_item in test_image:
#             output = vit(image_item)
#             predict.append(output.argmax(dim=1).detach().cpu().item())
#             label.append(label_item.item())
        
#     print('测试数据：', len(label))
#     print('Accuracy:', accuracy_score(label, predict))



