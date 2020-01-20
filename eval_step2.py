from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.metrics import RunningScore
import numpy as np
import torch.nn as nn
from PIL import Image
palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

if __name__ == '__main__':

    opt_val = TestOptions().parse()

    # 加载验证数据集
    opt_val.dataset_mode = 'single'
    opt_val.batch_size = 1
    dataset_val = create_dataset(opt_val)
    dataset_val_size = len(dataset_val)
    print('The number of valling images = %d' % dataset_val_size)

    # 创建验证模型
    model_val = create_model(opt_val)
    model_val.eval()

    # 从这一轮保存的权重恢复
    model_val.setup(opt_val)
    metrics = RunningScore(opt_val.num_classes)
    metrics.reset()
    for i, data in enumerate(dataset_val):
        model_val.set_input(data)
        model_val.forward()
        gt = data["B_label"].numpy().squeeze()  # [H, W]
        output = model_val.pre # [N, C, H, W]
        output = nn.functional.softmax(output, dim=1) # [N, C, H, W]
        output = nn.functional.upsample(output, (1024, 2048), mode='bilinear', align_corners=True).cpu().data[
            0].numpy() # [C, H, W]
        output = output.transpose(1, 2, 0) # [H, W, C]
        output_nomask = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)  # [H, W]

        # 保存成彩色图
        output_col = colorize_mask(output_nomask)
        output_col.save('%s/%s_color.png' % (opt_val.results_dir, data['name'][0].split('/')[-1].split('.')[0]))

        # 计算iou
        if len(gt.flatten()) != len(output_nomask.flatten()):
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}'.format(len(gt.flatten()),len(output_nomask.flatten())))
            continue
        metrics.update(gt, output_nomask)
    metrics.get_scores()
