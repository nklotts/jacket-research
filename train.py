import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from ultralytics import YOLO
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import cv2

# Импорт нашей модели
import sys
sys.path.append('.')
from models.mlp import LEDControlMLP


def parse_labelstudio_annotations(json_path):
    """Парсит JSON с аннотациями"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    annotations_dict = {}
    for task in data:
        image_name = os.path.basename(task['data']['image'])
        polygons = []
        
        if task['annotations']:
            for annotation in task['annotations']:
                for result in annotation['result']:
                    if result['type'] == 'polygonlabels':
                        polygon_data = {
                            'points': result['value']['points'],
                            'label': result['value']['polygonlabels'][0],
                            'original_width': result['original_width'],
                            'original_height': result['original_height']
                        }
                        polygons.append(polygon_data)
        
        annotations_dict[image_name] = polygons
    
    return annotations_dict


def polygon_to_mask(points, original_width, original_height, target_width, target_height):
    """Конвертирует полигон в маску"""
    mask = np.zeros((target_height, target_width), dtype=np.uint8)
    
    scaled_points = []
    for x, y in points:
        x_scaled = int((x / 100.0) * target_width)
        y_scaled = int((y / 100.0) * target_height)
        scaled_points.append([x_scaled, y_scaled])
    
    scaled_points = np.array(scaled_points, dtype=np.int32)
    cv2.fillPoly(mask, [scaled_points], color=1)
    
    return mask


def create_mask_from_annotations(annotation_polygons, target_height, target_width):
    """Создает маску из всех полигонов"""
    combined_mask = np.zeros((target_height, target_width), dtype=np.uint8)
    
    for polygon_data in annotation_polygons:
        points = polygon_data['points']
        orig_w = polygon_data['original_width']
        orig_h = polygon_data['original_height']
        
        poly_mask = polygon_to_mask(points, orig_w, orig_h, target_width, target_height)
        combined_mask = np.maximum(combined_mask, poly_mask)
    
    mask_tensor = torch.from_numpy(combined_mask).float().unsqueeze(0)
    return mask_tensor


def smooth_mask_edges(mask, kernel_size=5):
    """Сглаживает края маски"""
    import torch.nn.functional as F
    
    sigma = kernel_size / 3.0
    x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    
    kernel = gauss.unsqueeze(0) * gauss.unsqueeze(1)
    kernel = kernel.unsqueeze(0).unsqueeze(0).to(mask.device)
    
    padding = kernel_size // 2
    smoothed = F.conv2d(mask, kernel, padding=padding)
    
    return smoothed


class AnnotatedImageDataset(Dataset):
    """Датасет с аннотациями для масок"""
    
    def __init__(self, image_dir, annotations_path, transform=None, n_sensors=10):
        self.image_dir = image_dir
        self.transform = transform
        self.n_sensors = n_sensors
        
        # Загружаем аннотации
        self.annotations = parse_labelstudio_annotations(annotations_path)
        
        # Список файлов с аннотациями
        self.image_files = [f for f in os.listdir(image_dir) 
                           if f.endswith(('.jpg', '.png', '.jpeg')) and f in self.annotations]
        
        print(f"Loaded {len(self.image_files)} images with annotations")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img_path = os.path.join(self.image_dir, filename)
        
        # Загружаем изображение
        image = Image.open(img_path).convert('RGB')
        
        # Применяем трансформации
        if self.transform:
            # Запоминаем оригинальный размер для маски
            orig_width, orig_height = image.size
            image = self.transform(image)
            target_height, target_width = image.shape[1], image.shape[2]
        else:
            image_np = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image_np).permute(2, 0, 1)
            target_height, target_width = image.shape[1], image.shape[2]
        
        # Создаем маску из аннотаций
        polygons = self.annotations[filename]
        mask = create_mask_from_annotations(polygons, target_height, target_width)
        
        # Генерируем случайные параметры датчиков
        sensors = torch.rand(self.n_sensors)
        
        return image, sensors, mask


def overlay_led_pattern_with_mask(image, led_colors, mask, alpha=0.7, blur_edges=True):
    """
    Накладывает LED паттерн ТОЛЬКО в области маски
    
    Args:
        image: [B, 3, H, W]
        led_colors: [B, n_leds, 3]
        mask: [B, 1, H, W]
        alpha: интенсивность наложения
        blur_edges: сглаживать края
    """
    batch_size = image.size(0)
    _, h, w = image.size()[1:]
    n_leds = led_colors.size(1)
    
    # Определяем размер сетки LED
    grid_h = int(np.sqrt(n_leds))
    grid_w = int(np.ceil(n_leds / grid_h))
    total_grid_size = grid_h * grid_w
    
    # Дополняем LED цвета до полной сетки
    if total_grid_size > n_leds:
        padding = torch.zeros(batch_size, total_grid_size - n_leds, 3, 
                             device=led_colors.device)
        led_colors_padded = torch.cat([led_colors, padding], dim=1)
    else:
        led_colors_padded = led_colors
    
    # Создаем LED изображение
    led_image = led_colors_padded.view(batch_size, grid_h, grid_w, 3)
    led_image = led_image.permute(0, 3, 1, 2)  # [B, 3, grid_h, grid_w]
    
    # Ресайзим LED паттерн до размера изображения
    led_resized = nn.functional.interpolate(
        led_image, 
        size=(h, w), 
        mode='bilinear', 
        align_corners=False
    )
    
    # Сглаживаем края маски для плавного перехода
    if blur_edges:
        mask = smooth_mask_edges(mask, kernel_size=7)
    
    # Расширяем маску до 3 каналов
    mask_3ch = mask.expand(-1, 3, -1, -1)
    
    # КЛЮЧЕВОЙ МОМЕНТ: накладываем LED только там где mask=1
    modified_image = image * (1 - alpha * mask_3ch) + led_resized * alpha * mask_3ch
    modified_image = torch.clamp(modified_image, 0, 1)
    
    return modified_image


def compute_detection_loss(predictions, target_conf=0.0):
    """Вычисляет loss для минимизации детекции"""
    loss = 0.0
    count = 0
    
    for pred in predictions:
        if pred.boxes is not None and len(pred.boxes) > 0:
            person_mask = pred.boxes.cls == 0
            if person_mask.sum() > 0:
                person_conf = pred.boxes.conf[person_mask]
                loss += person_conf.mean()
                count += 1
    
    if count > 0:
        loss = loss / count
    
    return loss


def compute_color_loss(original_image, modified_image, mask=None):
    """
    Вычисляет MSE между средними RGB значениями
    Опционально только в области маски
    """
    if mask is not None:
        # Считаем loss только в области маски
        mask_3ch = mask.expand(-1, 3, -1, -1)
        
        # Извлекаем пиксели в области маски
        orig_masked = original_image * mask_3ch
        mod_masked = modified_image * mask_3ch
        
        # Средние значения
        mask_sum = mask_3ch.sum(dim=[2, 3])
        orig_mean = orig_masked.sum(dim=[2, 3]) / (mask_sum + 1e-8)
        mod_mean = mod_masked.sum(dim=[2, 3]) / (mask_sum + 1e-8)
    else:
        orig_mean = original_image.mean(dim=[2, 3])
        mod_mean = modified_image.mean(dim=[2, 3])
    
    color_loss = nn.functional.mse_loss(mod_mean, orig_mean)
    return color_loss


def train_epoch(model, dataloader, yolo_model, optimizer, device, epoch):
    """Один epoch обучения"""
    model.train()
    
    # ФИКС: Переключаем BatchNorm слои в eval режим если batch_size=1
    def set_bn_eval(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()
    model.apply(set_bn_eval)
    
    total_loss = 0.0
    total_det_loss = 0.0
    total_color_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, (images, sensors, masks) in enumerate(pbar):
        images = images.to(device)
        sensors = sensors.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Генерируем LED паттерн
        led_colors = model(images, sensors)
        
        # Накладываем LED ТОЛЬКО в области маски
        modified_images = overlay_led_pattern_with_mask(
            images, led_colors, masks, alpha=0.7, blur_edges=True
        )
        
        # Конвертируем для YOLO
        yolo_input_list = []
        for i in range(modified_images.size(0)):
            img = modified_images[i]
            img = img.permute(1, 2, 0)
            img = (img * 255).clamp(0, 255).byte().cpu().numpy()
            img_bgr = img[:, :, ::-1].copy()
            yolo_input_list.append(img_bgr)
        
        # Получаем предсказания YOLO
        with torch.no_grad():
            yolo_predictions = yolo_model(yolo_input_list, verbose=False)
        
        # Вычисляем loss детекции
        detection_loss = compute_detection_loss(yolo_predictions)
        detection_loss = torch.tensor(detection_loss, device=device, requires_grad=True)
        
        # Вычисляем color loss (только в области маски)
        color_loss = compute_color_loss(images, modified_images, mask=masks)
        
        # Общий loss
        lambda_color = 5
        total_batch_loss = detection_loss + lambda_color * color_loss
        
        # Обратное распространение
        total_batch_loss.backward()
        optimizer.step()
        
        # Статистика
        total_loss += total_batch_loss.item()
        total_det_loss += detection_loss.item()
        total_color_loss += color_loss.item()
        
        pbar.set_postfix({
            'loss': f'{total_batch_loss.item():.4f}',
            'det': f'{detection_loss.item():.4f}',
            'color': f'{color_loss.item():.4f}'
        })
    
    avg_loss = total_loss / max(len(dataloader), 1)  # Защита от деления на 0
    avg_det_loss = total_det_loss / max(len(dataloader), 1)
    avg_color_loss = total_color_loss / max(len(dataloader), 1)
    
    return avg_loss, avg_det_loss, avg_color_loss


def visualize_results(original, modified, mask, epoch, save_path='results'):
    """Визуализация с маской"""
    os.makedirs(save_path, exist_ok=True)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    orig_np = original[0].permute(1, 2, 0).cpu().numpy()
    mod_np = modified[0].permute(1, 2, 0).cpu().numpy()
    mask_np = mask[0, 0].cpu().numpy()
    
    # Создаем overlay маски
    mask_colored = np.zeros_like(orig_np)
    mask_colored[mask_np > 0.5] = [1, 0, 0]
    overlay = orig_np * 0.7 + mask_colored * 0.3
    
    axes[0].imshow(orig_np)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(mask_np, cmap='gray')
    axes[1].set_title('LED Application Mask')
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title('Mask Overlay')
    axes[2].axis('off')
    
    axes[3].imshow(mod_np)
    axes[3].set_title('Modified (LED in mask area)')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/epoch_{epoch}.png', dpi=150)
    plt.close()


def main():
    # Гиперпараметры
    BATCH_SIZE = 1  # ИЗМЕНЕНО: разрешаем batch_size=1
    NUM_EPOCHS = 250
    LEARNING_RATE = 3e-4
    IMAGE_SIZE = 256
    N_SENSORS = 10
    N_LEDS = 11776
    
    # Пути
    IMAGE_DIR = 'data/images/annotated'  # Папка с изображениями
    ANNOTATIONS_PATH = './data/images/annotated/annotated.json'
    
    # Устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Трансформации
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    
    # Загрузка датасета с аннотациями
    print('Loading annotated dataset...')
    dataset = AnnotatedImageDataset(
        image_dir=IMAGE_DIR,
        annotations_path=ANNOTATIONS_PATH,
        transform=transform,
        n_sensors=N_SENSORS
    )
    
    if len(dataset) == 0:
        raise ValueError(f"No annotated images found! Check {IMAGE_DIR} and {ANNOTATIONS_PATH}")
    
    print(f"Dataset size: {len(dataset)} images")
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                           num_workers=0, drop_last=False)  # drop_last=False!
    
    # Создание модели
    print('Creating model...')
    model = LEDControlMLP(
        image_size=IMAGE_SIZE,
        n_sensors=N_SENSORS,
        n_leds=N_LEDS,
        hidden_dims=[2048, 4096, 8192, 4096],
        dropout=0.3
    ).to(device)
    
    print(f'Model parameters: {sum(p.numel() for p in model.parameters())}')
    
    # Загрузка YOLO
    print('Loading YOLO model...')
    yolo_model = YOLO('yolov8n.pt')
    yolo_model.to(device)
    
    # Оптимизатор
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Обучение
    print('Starting training...')
    best_loss = float('inf')
    
    for epoch in range(1, NUM_EPOCHS + 1):
        avg_loss, avg_det_loss, avg_color_loss = train_epoch(
            model, dataloader, yolo_model, optimizer, device, epoch
        )
        
        scheduler.step()
        
        print(f'\nEpoch {epoch}/{NUM_EPOCHS}:')
        print(f'  Avg Loss: {avg_loss:.4f}')
        print(f'  Detection Loss: {avg_det_loss:.4f}')
        print(f'  Color Loss: {avg_color_loss:.4f}')
        print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Сохранение лучшей модели
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, 'best_model.pth')
            print(f'  Saved best model!')
        
        # Визуализация каждые 5 epochs
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                sample_img, sample_sensors, sample_mask = next(iter(dataloader))
                sample_img = sample_img.to(device)
                sample_sensors = sample_sensors.to(device)
                sample_mask = sample_mask.to(device)
                
                led_colors = model(sample_img, sample_sensors)
                modified = overlay_led_pattern_with_mask(
                    sample_img, led_colors, sample_mask, alpha=0.7
                )
                
                visualize_results(sample_img, modified, sample_mask, epoch)
            model.train()
    
    print('Training completed!')


if __name__ == '__main__':
    main()