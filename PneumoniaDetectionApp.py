
import os
import json
from tkinter import filedialog, Tk, Label, Button, messagebox, ttk, Entry
from PIL import Image, ImageTk
from torchvision import models, transforms
from torch import nn
import torch

import tkinter as tk

#加载类别名称
def load_class_names(filename='class_names.json'):
    with open(filename, 'r') as f:
        return json.load(f)

def save_class_names(class_names, filename='class_names.json'):
    with open(filename, 'w') as f:
        json.dump(class_names, f)

def load_model(device, class_names):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(class_names))
    model.load_state_dict(torch.load('model-resnet18.pth'))
    model = model.to(device)
    model.eval()  # 设置模型为评估模式
    return model

class PneumoniaDetectionApp:
    def __init__(self, root, model, device, class_names):
        self.root = root
        self.root.title("Pneumonia Detection System")


        self.root.geometry("800x600") #窗口大小

        self.model = model
        self.device = device
        self.class_names = class_names

        main_frame = tk.Frame(root, padx=20, pady=20,)
        main_frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

        # 图像上传
        self.label = Label(main_frame, text="Please select a chest X-ray image", font=('Arial', 16))
        self.label.pack(pady=10)

        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=10)

        self.upload_button = Button(button_frame, text="Upload Image", command=self.upload_image, width=15, bg='#FF6347')
        self.upload_button.pack(side=tk.LEFT, padx=10)

        self.predict_button = Button(button_frame, text="Predict", command=self.predict, width=15, bg='#FF6347')
        self.predict_button.pack(side=tk.LEFT, padx=10)

        # 显示图像
        self.image_label = Label(main_frame, text="No image uploaded", font=('Arial', 14))
        self.image_label.pack(pady=10)

        # 显示结果区域
        self.result_label = Label(main_frame, text="", font=('Arial', 16))
        self.result_label.pack(pady=10)

        # 添加进度条
        self.progress = ttk.Progressbar(main_frame, orient='horizontal', length=300, mode='indeterminate')
        self.progress.pack(pady=20)
        self.progress.start()

        # 添加退出按钮
        self.exit_button = Button(button_frame, text="Exit", command=self.exit_app, width=15, bg='#FF6347')
        self.exit_button.pack(side=tk.LEFT, padx=10)

    def upload_image(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if self.file_path:
            self.display_image(self.file_path)

    def exit_app(self):
        self.root.quit()  # 退出应用程序

    def display_image(self, image_path):
        try:
            image = Image.open(image_path)
            image = image.resize((224, 224), Image.LANCZOS)
            self.image_tk = ImageTk.PhotoImage(image)

            # 更新 Label 显示图像
            self.image_label.config(image=self.image_tk, text="")
            self.image_label.image = self.image_tk

        except Exception as e:
            print(f"Error displaying image: {e}")
            messagebox.showerror("Error", "Failed to display image")

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        image = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])(image).unsqueeze(0)
        return image.to(self.device)

    def predict(self):
        if hasattr(self, 'file_path'):
            self.progress.start()
            self.root.update_idletasks()  # 更新进度条

            image_tensor = self.preprocess_image(self.file_path)
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(image_tensor)
                _, preds = torch.max(outputs, 1)
                result = self.class_names[preds[0]]
                self.result_label.config(text=f'Prediction: {result}')

            self.progress.stop()  # 停止进度条
        else:
            messagebox.showerror("Error", "Please upload an image first")

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.multiprocessing.set_start_method('spawn')

    model_path = 'model-resnet18.pth'
    class_names_path = 'class_names.json'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    if os.path.exists(model_path) and os.path.exists(class_names_path):
        print("Loading model from file...")
        class_names = load_class_names(class_names_path)
        model = load_model(device, class_names)
    else:
        print("Training model...")
        from data_augmentation import main
        model, device, class_names = main()
        save_class_names(class_names)

    root = Tk()
    root.configure(bg='lightblue')


    app = PneumoniaDetectionApp(root, model, device, class_names)
    root.mainloop()
