import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torch.nn as nn
import torchvision.datasets as datasets
from operator import truediv
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ==================== Dataset Class ====================
class InRAMImageFolder(Dataset):
    """
    Custom Dataset class that loads all images into RAM for faster access.

    Attributes:
        images (list): List of preloaded image tensors.
        labels (list): Corresponding class labels for images.
        classes (list): Sorted list of class names.
        class_to_idx (dict): Mapping from class name to numeric index.
    """
    def __init__(self, root, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        
        classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        self.classes = classes
        
        for idx, cls_name in enumerate(classes):
            self.class_to_idx[cls_name] = idx
            cls_dir = os.path.join(root, cls_name)
            for fname in os.listdir(cls_dir):
                path = os.path.join(cls_dir, fname)
                if os.path.isfile(path):
                    img = Image.open(path).convert('RGB')
                    if self.transform:
                        img = self.transform(img)
                    else:
                        img = transforms.ToTensor()(img)
                    self.images.append(img)
                    self.labels.append(idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# ==================== Logging Function ====================
def log_output(oa_ae, aa_ae, top2_acc, top5_acc, element_acc, lr, epoch, path):
    """
    Logs model evaluation results into a text file.

    Args:
        oa_ae (float): Overall top-1 accuracy.
        aa_ae (float): Average accuracy across classes.
        top2_acc (float): Top-2 accuracy.
        top5_acc (float): Top-5 accuracy.
        element_acc (list): Class-wise accuracies.
        lr (float): Learning rate.
        epoch (int): Number of epochs.
        path (str): Path to save the log file.
    """
    f = open(path, 'a')
    f.write(f"\n\n\nFor Learning_Rate : {lr} & Epochs : {epoch}   The Result is ->\n")
    sentence1 = 'OA(Top-1 Accuracy) is: ' + str(oa_ae) + '\n'
    f.write(sentence1)
    sentence2 = 'AA is: ' + str(aa_ae) +'\n'
    f.write(sentence2)
    sentence3 = 'Top-2 Accuracy is: '+ str(top2_acc) + '\n'
    f.write(sentence3)
    sentence4 = 'Top-5 Accuracy is: '+ str(top5_acc) + '\n'
    f.write(sentence4)
    element_mean = list(element_acc)
    sentence5 = "Class wise accuracy: " + str(element_mean) + '\n'
    f.write(sentence5)
    f.close()

# ==================== Identity Layer ====================
class Identity(nn.Module):
    """
    Identity Layer that outputs its input unchanged.
    Useful for replacing layers like fully connected in pretrained models.
    """
    def __init__(self):
        super(Identity,self).__init__()

    def forward(self,x):
        return x

# ==================== Model Building & Training ====================
def build_model(model, root_dir, folder, name, EPOCHS, LEARNING_RATE):
    """
    Train, validate, and evaluate a PyTorch model on a dataset loaded in RAM.

    Includes:
        - Data loading
        - Training with early stopping
        - Validation and checkpointing
        - Test evaluation
        - Accuracy and loss plotting
        - Confusion matrix calculation
        - Logging results to file

    Args:
        model (nn.Module): PyTorch model instance.
        root_dir (str): Root directory containing train/val/test folders.
        folder (str): Folder to save results and checkpoints.
        name (str): Model name.
        EPOCHS (int): Number of training epochs.
        LEARNING_RATE (float): Learning rate for optimizer.
    
    Returns:
        model: Trained model loaded with best checkpoint.
    """

    # ==================== Setup ====================
    CHECKPOINT_PATH = f'./{folder}/checkpoint_{name}.pth'

    transform =transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    print("Detecting Dataset... ")

    # Create folder if it does not exist
    os.makedirs(folder, exist_ok=True)

    TRAIN_DIRECTORY = f'{root_dir}/train'
    VALID_DIRECTORY = f'{root_dir}/val'
    TEST_DIRECTORY = f'{root_dir}/test'
    print("Dataset Detected... Loading... ")

    # ==================== Load Datasets ====================
    train_dataset = InRAMImageFolder(root=TRAIN_DIRECTORY, transform=transform)
    valid_dataset = InRAMImageFolder(root=VALID_DIRECTORY, transform=transform)
    test_dataset = InRAMImageFolder(root=TEST_DIRECTORY, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    class_list = train_dataset.classes

    epochs = EPOCHS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Starting Training on {device}...")

    # ==================== Loss & Optimizer ====================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    # ==================== Early Stopping Setup ====================
    best_loss = float('inf')
    best_accuracy = 0.0
    patience = 10
    counter = 0

    # Lists to store accuracy and loss values
    train_acc_list = []
    train_loss_list = []
    val_acc_list = []
    val_loss_list = []

    # ==================== Training Loop ====================
    for epoch in range(epochs):
        # -------------------- Train --------------------
        model.train()
        train_loss = 0.0
        train_correct = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)

        # -------------------- Validation --------------------
        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels).sum().item()

            val_loss /= len(valid_loader.dataset)
            val_acc = val_correct / len(valid_loader.dataset)
            val_acc_list.append(val_acc)
            val_loss_list.append(val_loss)

        print(f"Epoch: {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # -------------------- Early Stopping --------------------
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping!")
                break

        # -------------------- Save Best Model --------------------
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)

    # ==================== Load Best Checkpoint ====================
    model.load_state_dict(torch.load(CHECKPOINT_PATH))

    # ==================== Test Evaluation ====================
    model.eval()
    test_loss = 0.0
    test_correct = 0
    top2_correct = 0
    top5_correct = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            _, top2preds = torch.topk(outputs.data,2,1)
            _, top5preds = torch.topk(outputs.data,5,1)
            test_correct += (predicted == labels).sum().item()
            for i,label in enumerate(labels):
                if label in top5preds[i]:
                    top5_correct += 1
                if label in top2preds[i]:
                    top2_correct += 1

        test_loss /= len(test_loader.dataset)
        test_acc = test_correct / len(test_loader.dataset)
        top2_acc = top2_correct / len(test_loader.dataset)
        top5_acc = top5_correct / len(test_loader.dataset)

    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Top-2 Acc: {top2_acc:.4f} | Top-5 Acc: {top5_acc:.4f}")

    # ==================== Plot Accuracy and Loss ====================
    plt.figure(figsize=(20, 20))
    plt.plot(train_acc_list, label='Train')
    plt.plot(val_acc_list, label='Validation')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.savefig(f'./{folder}/Acc_{name}.png')
    plt.show()

    plt.figure(figsize=(20, 20))
    plt.plot(train_loss_list, label='Train')
    plt.plot(val_loss_list, label='Validation')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.savefig(f'./{folder}/loss_{name}.png')
    plt.show()

    # ==================== Confusion Matrix ====================
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    confusion_mat = confusion_matrix(y_true, y_pred)
    df = pd.DataFrame(confusion_mat)
    df.to_csv(f'./{folder}/confuse_matrix_{name}.csv')
    counter = confusion_mat.shape[0]
    list_diag = np.diag(confusion_mat)
    list_raw_sum = np.sum(confusion_mat, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    ax = sns.heatmap(confusion_mat, annot=True, cmap='Blues', fmt='g')
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values')
    ax.xaxis.set_ticklabels(class_list)
    ax.yaxis.set_ticklabels(class_list)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.savefig(f'./{folder}/confusion_matrix_{name}.png', dpi=400)
    plt.show()

    # ==================== Log Final Results ====================
    log_output(test_acc, average_acc, top2_acc, top5_acc, each_acc, LEARNING_RATE, EPOCHS, f'./{folder}/results_{name}.txt')

    return model