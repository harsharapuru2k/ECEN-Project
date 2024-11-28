import warnings
import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the CIFAR-10 dataset (testing data only)
mean, std = [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transforms)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

# Load the pre-trained model
model_path = "simple_model.pth"  # Update with your model file path
model = torchvision.models.resnet34()
model.fc = torch.nn.Linear(model.fc.in_features, 10)  # Adjust output layer for CIFAR-10
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

print("Model loaded successfully!")

# Evaluate the model
y_true = []
y_pred = []

with torch.no_grad():
    for x_test, y_test in test_loader:
        x_test, y_test = x_test.to(device), y_test.to(device)
        outputs = model(x_test)
        _, predictions = torch.max(outputs, 1)
        y_true.extend(y_test.cpu().numpy())
        y_pred.extend(predictions.cpu().numpy())

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Classification report
report = classification_report(y_true, y_pred, target_names=test_dataset.classes)
print("Classification Report:")
print(report)
