import numpy as np
from matplotlib import pyplot as plt
from torch import nn, optim, device
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
from tqdm import tqdm #检查进度
import torch
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import time
from PIL import Image
# def calculate_mean_std(dataset):
#     """计算数据集的均值和标准差"""
#     mean = 0.
#     std = 0.
#     total_images = 0
#
#     for images, _ in dataset:
#         # images 是一个批次的图像（Tensor），我们需要处理每张图像
#         for image in images:
#             # 将图像从 Tensor 转回 PIL Image
#             image = transforms.ToPILImage()(image)
#             # 计算每个图像的均值和标准差
#             image = transforms.ToTensor()(image)
#             mean += image.mean([1, 2])
#             std += image.std([1, 2])
#             total_images += 1
#
#     mean /= total_images
#     std /= total_images
#     # print(mean)
#     # print(std)
#     return mean.tolist(), std.tolist()
#
#
# # 使用数据集进行计算
# data_transforms = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])
#
# dataset = datasets.ImageFolder('dataset', transform=data_transforms)
# mean, std = calculate_mean_std(dataset)

# print(f"Mean: {mean}")
# print(f"Std: {std}")

# 数据预处理
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(0.481, 0.221)  # 使用计算得到的均值和标准差
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(0.481, 0.221)  # 使用计算得到的均值和标准差
    ]),
}


data_dir = 'dataset'

def evaluate_model(model, dataloader, device):

    start_time = time.time()
    model.eval()  # 切换到评估模式
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    tp = np.sum((all_labels == 1) & (all_preds == 1))
    tn = np.sum((all_labels == 0) & (all_preds == 0))
    fp = np.sum((all_labels == 0) & (all_preds == 1))
    fn = np.sum((all_labels == 1) & (all_preds == 0))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # 结束训练时间
    end_time = time.time()
    train_time = end_time - start_time
    print(f"Test time: {train_time:.2f} seconds")

    return accuracy, sensitivity, specificity, recall, precision, f1
def main():

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}


    dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4)
                   for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Data augmentation of Alexnet well done")


    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    #model = models.alexnet(weights='DEFAULT')
    num_features = model.classifier[6].in_features

    model.classifier[6] = nn.Linear(num_features, len(class_names))
    model = model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss() #使用交叉熵损失函数（nn.CrossEntropyLoss）来衡量预测与实际标签之间的差异。
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  #使用 Adam 优化器（optim.Adam）来更新模型参数，学习率为 0.0001。


    def train_model(model, criterion, optimizer, num_epochs=50):
        # 设置时间
        start_time = time.time()

        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []

        for epoch in range(num_epochs):
            print(f'Epoch {epoch + 1}/{num_epochs}')
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                correct_preds = 0

                dataloader = tqdm(dataloaders[phase], desc=f'{phase} phase')

                for inputs, labels in dataloader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    correct_preds += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = correct_preds.double() / dataset_sizes[phase]

                if phase == 'train':
                    train_losses.append(epoch_loss)
                    train_accuracies.append(epoch_acc.item())
                else:
                    val_losses.append(epoch_loss)
                    val_accuracies.append(epoch_acc.item())

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        # 结束训练时间
        end_time = time.time()
        train_time = end_time - start_time

        print(f"Training time: {train_time:.2f} seconds")
        return model, train_losses, val_losses, train_accuracies, val_accuracies

    model, train_losses, val_losses, train_accuracies, val_accuracies = train_model(model, criterion, optimizer, num_epochs=50)
    #model = train_model(model, criterion, optimizer, num_epochs=10)

    # 保存模型
    torch.save(model.state_dict(), 'model_Alexnet.pth')
    print("Test model of alexnet saved successfully.")

    # 训练和验证损失图
    plt.subplot(1, 2, 1)
    epochs = range(1, len(train_losses) + 1)  # 确保 epochs 与训练和验证数据长度一致
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    # 训练和验证准确率图
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(0.481, 0.221)
    ])

    test_dataset = datasets.ImageFolder('dataset/test', transform=test_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    accuracy, sensitivity, specificity, recall, precision, f1 = evaluate_model(model, test_dataloader, device)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Sensitivity (Recall): {sensitivity:.4f}')
    print(f'Specificity: {specificity:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'F1 Score: {f1:.4f}')

    return model, device, class_names


if __name__ == "__main__":
    model, device, class_names = main()














