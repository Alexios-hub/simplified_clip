import argparse
import os
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import datasets
import clip
from tqdm import tqdm
from modules.model import ShapedAttentionBlock,ResidualAttentionBlock
from torch.nn import Sequential
import wandb
import numpy as np
import random
import time



parser = argparse.ArgumentParser(description="Train a CLIP model on COCO2017 with DDP")
parser.add_argument('--data_path', type=str, default="data/COCO2017", help='Path to the COCO2017 dataset')
parser.add_argument('--batch_size', type=int, default=512, help='Input batch size for training')
parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs to train')
parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
parser.add_argument('--weight_decay',type=float,default=0.2,help="wd for adamw")
parser.add_argument('--local_rank', type=int,default=0, help='Local rank. Necessary for using the torch.distributed.launch utility.')
parser.add_argument('--simplify',action="store_true",help="use simplified_transformer blocks")
args = parser.parse_args()

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

wandb.init(
    project="Simplified-Clip-COCO2017",
    config=args
)

# # 设置DDP：初始化进程组
# torch.distributed.init_process_group(backend='nccl')

# 设置设备
torch.cuda.set_device(args.local_rank)
device = torch.device("cuda", args.local_rank)



COCO_DIR = os.path.join(os.getcwd(), args.data_path)
ds = datasets.load_dataset("ydshieh/coco_dataset_script", "2017", data_dir=COCO_DIR)

class COCODataset(Dataset):
    def __init__(self,ds,preprocess) -> None:
        super().__init__()
        self.ds = ds
        self.preprocess = preprocess
        self.texts = clip.tokenize([item['caption'] for item in self.ds])
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        image = self.preprocess(Image.open(self.ds[index]['image_path']))
        text = self.texts[index]
        return image,text
    
device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
model, preprocess = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training

def simplify_clip(model):
    simplified_blocks = []
    for i in range(len(model.visual.transformer.resblocks)):
        simplified_block = ShapedAttentionBlock(
            d_model=model.visual.transformer.resblocks[i].attn.embed_dim,
            n_head=model.visual.transformer.resblocks[i].attn.num_heads,
            attn_mask=model.visual.transformer.resblocks[i].attn_mask
        )
        simplified_blocks.append(simplified_block)
    model.visual.transformer.resblocks = Sequential(*simplified_blocks)
    
    simplified_blocks = []
    for i in range(len(model.transformer.resblocks)):
        simplified_block = ShapedAttentionBlock(
            d_model=model.transformer.resblocks[i].attn.embed_dim,
            n_head=model.transformer.resblocks[i].attn.num_heads,
            attn_mask=model.transformer.resblocks[i].attn_mask
        )
        simplified_blocks.append(simplified_block)
    model.transformer.resblocks = Sequential(*simplified_blocks)

def no_simplify_clip(model):
    blocks = []
    for i in range(len(model.visual.transformer.resblocks)):
        block = ResidualAttentionBlock(
            d_model=model.visual.transformer.resblocks[i].attn.embed_dim,
            n_head=model.visual.transformer.resblocks[i].attn.num_heads,
            attn_mask=model.visual.transformer.resblocks[i].attn_mask
        )
        blocks.append(block)
    model.visual.transformer.resblocks = Sequential(*blocks)

    blocks=[]
    for i in range(len(model.transformer.resblocks)):
        block = ResidualAttentionBlock(
            d_model=model.transformer.resblocks[i].attn.embed_dim,
            n_head=model.transformer.resblocks[i].attn.num_heads,
            attn_mask=model.transformer.resblocks[i].attn_mask
        )
        blocks.append(block)
    model.transformer.resblocks = Sequential(*blocks)



if args.simplify:
    print("simplified transformer")
    simplify_clip(model)
else:
    no_simplify_clip(model)
model = model.to(device)

print(model)

train_dataset = COCODataset(ds=ds['train'],preprocess=preprocess)
train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size)

def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 


if device == "cpu":
  model.float()
else :
  clip.model.convert_weights(model) # Actually this line is unnecessary since clip by default already on float16

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(),lr=args.learning_rate,betas=(0.9,0.98),eps=1e-6,weight_decay=args.weight_decay)

model_parameters = filter(lambda p: p.requires_grad,model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
wandb.log({"total_parameters": params})
print(f"Total parameters: {params}")

start_time = time.time()
for epoch in range(args.num_epochs):
    model.train()
    total_loss = 0.0

    with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{args.num_epochs}") as pbar:
        for images, texts in train_dataloader:
            optimizer.zero_grad()
            images = images.to(device)
            texts = texts.to(device)

            logits_per_image, logits_per_text = model(images, texts)
            ground_truth = torch.arange(images.size(0), dtype=torch.long, device=device)

            loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
            wandb.log({"loss":loss})
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            pbar.set_description(f"Epoch {epoch+1}/{args.num_epochs} Loss: {loss.item():.4f}")
            pbar.update(1)
    model_save_path = 'checkpoints/simplified/'+str(epoch)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    torch.save(model.state_dict(),model_save_path + '/model.pth')

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch Completed: {epoch+1}/{args.num_epochs}, Average Loss: {avg_loss:.4f}")
    wandb.log({"avg_loss":avg_loss})
end_time = time.time()
wandb.log({"time":end_time-start_time})






