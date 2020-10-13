import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from CNNarchitecture import CNN

# 检查是否安装sklearn,便于后面使用TSNE降维，显示分类结果
try: from sklearn.manifold import TSNE; HAS_SK = True
except: HAS_SK = False; print('Please install sklearn for layer visualization')

# torch.manual_seed(1)    # reproducible

# 设置超参数
EPOCH = 1               # 用整个样本集训练的次数
BATCH_SIZE = 128
LR = 0.001              # learning rate
DOWNLOAD_MNIST = False
TEST_NUM = 2000        # 选取2000个测试样本

# 检查手写数字集是否已下载
if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True

# 读取训练数据，通过ToTensor，将数据转换成(channels,height,width),并进行压缩[0.0,1.0]
train_data = torchvision.datasets.MNIST(
    root = './mnist/',
    train = True,
    transform = torchvision.transforms.ToTensor(),
    download = DOWNLOAD_MNIST,
)

# 将训练集划分成若干批，每批128张数字，image batch shape (128, 1, 28, 28)，每一批提取时都进行打乱
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# 提取测试集
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
# shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:TEST_NUM]/255.
test_y = test_data.test_labels[:TEST_NUM]

# 生成网络，并指定优化方法为adam，采用交叉熵作为损失函数
cnn = CNN()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # 将所有参数交给优化器
loss_func = nn.CrossEntropyLoss()                       # 交叉熵输入的标签不是 one-hotted

# 定义绘图函数
def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)

plt.ion()

flag = True                                 # 是否进行网络的训练
loss_record = [];
Recall_record = [];
if flag:
# 训练和测试
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):

            output = cnn(b_x)[0]            # 得到网络的输出
            loss = loss_func(output, b_y)   # 计算交叉熵损失
            optimizer.zero_grad()           # 梯度清零
            loss.backward()                 # 反向传播计算梯度
            optimizer.step()                # 更新权重

            if step % 20 == 0:
                cnn.eval()                  # 切换到预测模式，关闭DropOut
                test_output, last_layer = cnn(test_x)
                pred_y = torch.max(test_output, 1)[1].data.numpy()
                # 一次算一个batch的平均准确度
                accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
                cnn.train()                 # 回到训练模式，开启DropOut
                loss_record.append(loss.data.numpy())
                Recall_record.append(accuracy)
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
                if HAS_SK:
                    # Visualization of trained flatten layer (T-SNE)
                    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
                    plot_only = 500
                    low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
                    labels = test_y.numpy()[:plot_only]
                    plot_with_labels(low_dim_embs, labels)
    plt.ioff()
# 保存网络参数
torch.save(cnn.state_dict(),'cnn_params.pkl')

plt.figure()
plt.plot(np.arange(len(loss_record)) + 1,loss_record,'c',label = r'$Loss$')
plt.plot(np.arange(len(Recall_record)) + 1, Recall_record,'r',label=r'$Recall$')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.title('训练过程')
plt.legend()
plt.waitforbuttonpress()


# 加载权重参数
if ~flag:
    cnn = CNN()
    cnn.load_state_dict(torch.load("cnn_params.pkl"))

def my_show_predict(index,pred_y):
    #显示真实图像与预测值
    plt.figure(5)
    plt.imshow(test_data.test_data[index].numpy(), cmap='gray')
    plt.title('predict:%i' % pred_y)
    plt.waitforbuttonpress()

# 得到最终网络的预测结果
cnn.eval()
test_output, _ = cnn(test_x[:TEST_NUM])
pred_y = torch.max(test_output, 1)[1].data.numpy()
label = test_y[:TEST_NUM].numpy()

# 统计每一类数字的个数
classNum = 10
pred_y_count = dict.fromkeys(range(classNum),0)
label_count = dict.fromkeys(range(classNum),0)
errorindex = []
for it in range(TEST_NUM):
    if pred_y[it] == label[it]:
        pred_y_count[pred_y[it]] += 1         # 该类数字预测对的个数
    else:
        errorindex.append(it)
    label_count[label[it]] += 1               # 原来各类数字的个数
print(label_count)
# 计算召回率
sumVal = 0
res = []
for it in range(classNum):
    res.append( 1.0 - abs(pred_y_count[it] - label_count[it])/label_count[it])#(label_count[it]/100)#
    sumVal += res[it]
    print("\n number:%d recall is %.3f"%(it,res[it]))     #计算每一个数字的准确率
print("\n Average recall:%.3f"%(sumVal/classNum))        #计算所有数字平均的准确率
# 以条形图形式显示召回率
res.append(sumVal/classNum)
X = range(classNum+1)
plt.figure(2)
ticks = ['0','1','2','3','4','5','6','7','8','9','avg']
plt.bar(X,res,facecolor = '#9999ff',edgecolor='white')
# 条形图参数设置
for x,y in zip(X,res):
    plt.text(x,y,'%.2f'%y,ha = 'center',va = 'bottom')
plt.xticks(X,ticks)
plt.ylim([0.0,1.2])
plt.title(r"$Recall$")
plt.waitforbuttonpress()

# 对于判错的测试图像,显示预测结果
for i in errorindex:
    my_show_predict(i, pred_y[i])
