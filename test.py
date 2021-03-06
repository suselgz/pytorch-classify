import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import json
###测试数据预处理
"""
测试数据处理方法需要跟训练时一直才可以
crop操作的目的是保证输入的大小是一致的
标准化操作也是必须的，用跟训练数据相同的mean和std,但是需要注意一点训练数据是在0-1上进行标准化，所以测试数据也需要先归一化
最后一点，PyTorch中颜色通道是第一个维度，跟很多工具包都不一样，需要转换
"""
def process_image(image_path):
    # 读取测试数据
    img = Image.open(image_path)
    # Resize,thumbnail方法只能进行缩小，所以进行了判断
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))
    # Crop操作
    left_margin = (img.width - 224) / 2
    bottom_margin = (img.height - 224) / 2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin,
                    top_margin))
    # 相同的预处理方法
    img = np.array(img) / 255
    mean = np.array([0.485, 0.456, 0.406])  # provided mean
    std = np.array([0.229, 0.224, 0.225])  # provided std
    img = (img - mean) / std
    # 注意颜色通道应该放在第一个位置
    img = img.transpose((2, 0, 1))
    return img


"""展示数据"""
def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    # 颜色通道还原
    image = np.array(image).transpose((1, 2, 0))
    # 预处理还原
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    ax.set_title(title)
    return ax

### 注意tensor的数据需要转换成numpy的格式，而且还需要还原回标准化的结果
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image

def main(config):
    logger = config.get_logger('test')

    ### 读取标签对应的实际名字
    with open('./data/cat_dog_data/cat_dog.json', 'r') as f:
        cat_to_name = json.load(f)
    # 数据
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=16,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    )

    # 模型
    module = config.init_obj('arch', module_arch)
    #model = module
    model = module.model_ft
    logger.info(model)

    # 同样的
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # 准备测试
    train_on_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            # save sample images, or do something with output here
            # 显示每个batch图片结果

            output.shape
            # 得到概率最大的那个
            _, preds_tensor = torch.max(output, 1)
            preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())

            ### 展示预测结果
            fig = plt.figure(figsize=(20, 20))
            columns = 4
            rows = 4
            for idx in range(columns * rows):
                ax = fig.add_subplot(rows, columns, idx + 1, xticks=[], yticks=[])
                plt.imshow(im_convert(data[idx]))
                ax.set_title("{} ({})".format(cat_to_name[str(preds[idx])], cat_to_name[str(target[idx].item())]),
                             color=("green" if cat_to_name[str(preds[idx])] == cat_to_name[
                                 str(target[idx].item())] else "red"))
            plt.show()

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='data/cat_dog_data/config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default='saved/models/CatDog_ResNet/1113_091352/model_best.pth', type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
