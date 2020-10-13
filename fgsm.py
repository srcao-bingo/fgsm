from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from CNNarchitecture import CNN
epsilons = [0, .05, .1, .15, .2, .25, .3]

# epsilons - 要用于运行的epsilon值列表。
# 从直觉上说epsilon越大，扰动越明显，但攻击越有效，降低了模型的准确性。由于数据的范围是 [0,1]
# ，任何epsilon值都不应超过1。
pretrained_model = "cnn_params.pkl"
use_cuda=False

def fgsm_attack(image, epsilon, data_grad):
    # 获取样本梯度的符号
    sign_data_grad = data_grad.sign()
    # 生成对抗样本
    perturbed_image = image + epsilon*sign_data_grad
    # 饱和运算，比0小的都置零，比1大的都置一
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # 返回结果
    return perturbed_image


# MNIST Test dataset 和 dataloader 声明
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist/', train=False, download=False, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=1, shuffle=True)

# 定义要使用的设备
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# 初始化网络
model = CNN().to(device)

# 加载预训练模型
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

# 将模型设置为评估模式. 这是为了 Dropout layers。
model.eval()

def test( model, device, test_loader, epsilon ):

    # Accuracy counter
    correct = 0
    adv_examples = []

    # 遍历所有测试样本
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        # 显示设置对输入样本求梯度，否则，无得到样本的梯度方向
        data.requires_grad = True
        # 维度转换
        origin_img = torch.squeeze(data).detach().numpy()
        # 前向传播
        output,Noues = model(data)
        init_pred = torch.max(output, 1)[1].data.numpy() # 得到类别的下标

        # 对于本来就判错的样本直接跳过
        if init_pred.item() != target.item():
            continue
        # print(target,output)

        # 计算损失
        loss = F.nll_loss(output, target)

        # 梯度清零
        model.zero_grad()

        # 反向传播，计算误差
        loss.backward()

        # 得到样本的梯度
        data_grad = data.grad.data

        # 生成对抗样本
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # 用对抗样本测试网络
        output,Nouse = model(perturbed_data)

        # 检查是否攻击成功
        final_pred = output.max(1, keepdim=True)[1] # 得到网络预测的对抗样本的类别
        if final_pred.item() == target.item():
            correct += 1
            # 保留 epsilon = 0 的例子
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex,origin_img) )
        else:
            # 留下一些对抗样本用于可视化
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex,origin_img) )

    # 计算这个epsilon下的准确率
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # 返回准确率和生成的对抗样本
    return final_acc, adv_examples
accuracies = []
examples = []

# 对每个epsilon进行测试
for eps in epsilons:
    acc, ex = test(model, device, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)

plt.figure(figsize=(5,5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.show()

# 显示不同epsilon下的对抗样本
cnt = 0
plt.figure(figsize=(8,10))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons),len(examples[0]),cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        orig,adv,ex,orig_img = examples[i][j]
        plt.title("{} -> {}".format(orig, adv))
        plt.imshow(ex, cmap="gray")
plt.tight_layout()
plt.show()

plt.figure(5)
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        orig, adv, ex, orig_img = examples[i][j]
        # 绘制原图
        plt.subplot(131)
        plt.imshow(orig_img, cmap="gray")
        plt.xlabel("Origin")
        plt.xticks(())
        plt.yticks(())
        plt.title("target:{}".format(orig))
        # 绘制结果与原图之差
        plt.subplot(132)
        plt.imshow(orig_img-ex, cmap="gray")
        plt.xlabel("Difference")
        plt.xticks(())
        plt.yticks(())
        # 绘制最终的对抗样本
        plt.subplot(133)
        plt.imshow(ex, cmap="gray")
        plt.xlabel("Adversarial")
        plt.title("predict:{}".format( adv))
        plt.xticks(())
        plt.yticks(())
        plt.waitforbuttonpress()


