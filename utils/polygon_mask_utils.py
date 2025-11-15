import json
import torch
import numpy as np
from PIL import Image, ImageDraw
import cv2


def parse_labelstudio_annotations(json_path):
    """
    Парсит JSON файл с аннотациями из Label Studio
    
    Args:
        json_path: путь к JSON файлу
    
    Returns:
        annotations_dict: словарь {image_path: [polygons]}
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    annotations_dict = {}
    
    for task in data:
        # Путь к изображению
        image_path = task['data']['image']
        
        # Список полигонов для этого изображения
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
        
        annotations_dict[image_path] = polygons
    
    return annotations_dict


def polygon_to_mask(points, original_width, original_height, target_width, target_height):
    """
    Конвертирует полигон в бинарную маску
    
    Args:
        points: список координат [[x1,y1], [x2,y2], ...]
        original_width: ширина при аннотировании
        original_height: высота при аннотировании
        target_width: целевая ширина маски
        target_height: целевая высота маски
    
    Returns:
        mask: numpy array [H, W] со значениями 0/1
    """
    # Создаем пустую маску
    mask = np.zeros((target_height, target_width), dtype=np.uint8)
    
    # Масштабируем координаты
    scaled_points = []
    for x, y in points:
        # Координаты в Label Studio в процентах (0-100)
        x_scaled = int((x / 100.0) * target_width)
        y_scaled = int((y / 100.0) * target_height)
        scaled_points.append([x_scaled, y_scaled])
    
    scaled_points = np.array(scaled_points, dtype=np.int32)
    
    # Заполняем полигон
    cv2.fillPoly(mask, [scaled_points], color=1)
    
    return mask


def create_mask_from_annotations(annotation_polygons, target_height, target_width):
    """
    Создает комбинированную маску из всех полигонов
    
    Args:
        annotation_polygons: список полигонов из parse_labelstudio_annotations
        target_height: высота целевого изображения
        target_width: ширина целевого изображения
    
    Returns:
        mask: torch tensor [1, H, W] со значениями 0/1
    """
    combined_mask = np.zeros((target_height, target_width), dtype=np.uint8)
    
    for polygon_data in annotation_polygons:
        points = polygon_data['points']
        orig_w = polygon_data['original_width']
        orig_h = polygon_data['original_height']
        
        # Создаем маску для этого полигона
        poly_mask = polygon_to_mask(points, orig_w, orig_h, target_width, target_height)
        
        # Добавляем к общей маске
        combined_mask = np.maximum(combined_mask, poly_mask)
    
    # Конвертируем в torch tensor
    mask_tensor = torch.from_numpy(combined_mask).float().unsqueeze(0)
    
    return mask_tensor


def apply_led_colors_to_mask(image, led_colors, mask, alpha=0.8, blur_edges=True):
    """
    Применяет LED цвета только в области маски
    
    Args:
        image: исходное изображение [B, 3, H, W] или [3, H, W]
        led_colors: цвета LED [B, n_leds, 3] или просто [3] для однородного цвета
        mask: бинарная маска [1, H, W] или [B, 1, H, W]
        alpha: интенсивность наложения (0-1)
        blur_edges: сглаживать края маски
    
    Returns:
        modified_image: изображение с наложенными LED цветами
    """
    # Обработка размерностей
    if image.dim() == 3:
        image = image.unsqueeze(0)
        single_image = True
    else:
        single_image = False
    
    batch_size, _, h, w = image.shape
    
    # Если mask без batch размерности, добавляем
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)
    
    # Если маска одна, но batch несколько - дублируем
    if mask.size(0) == 1 and batch_size > 1:
        mask = mask.expand(batch_size, -1, -1, -1)
    
    # Сглаживание краев маски
    if blur_edges:
        mask = smooth_mask_edges(mask, kernel_size=5)
    
    # Случай 1: led_colors это тензор цветов для каждого LED
    if led_colors.dim() == 3:  # [B, n_leds, 3]
        # Создаем цветное изображение из LED цветов
        n_leds = led_colors.size(1)
        grid_h = int(np.sqrt(n_leds))
        grid_w = int(np.ceil(n_leds / grid_h))
        
        # Дополняем до полной сетки если нужно
        total_grid = grid_h * grid_w
        if total_grid > n_leds:
            padding = torch.zeros(batch_size, total_grid - n_leds, 3, 
                                 device=led_colors.device, dtype=led_colors.dtype)
            led_colors = torch.cat([led_colors, padding], dim=1)
        
        # Reshape в изображение
        led_image = led_colors.view(batch_size, grid_h, grid_w, 3)
        led_image = led_image.permute(0, 3, 1, 2)  # [B, 3, grid_h, grid_w]
        
        # Resize до размера целевого изображения
        import torch.nn.functional as F
        led_pattern = F.interpolate(led_image, size=(h, w), 
                                     mode='bilinear', align_corners=False)
    
    # Случай 2: led_colors это один цвет [3] или [B, 3]
    elif led_colors.dim() <= 2:
        if led_colors.dim() == 1:
            led_colors = led_colors.unsqueeze(0)  # [1, 3]
        
        # Создаем однородное цветное изображение
        led_pattern = led_colors.view(batch_size, 3, 1, 1).expand(-1, -1, h, w)
    
    else:
        raise ValueError(f"Unexpected led_colors shape: {led_colors.shape}")
    
    # Расширяем маску до 3 каналов
    mask_3ch = mask.expand(-1, 3, -1, -1)
    
    # Применяем LED цвета только в области маски
    modified_image = image * (1 - alpha * mask_3ch) + led_pattern * alpha * mask_3ch
    modified_image = torch.clamp(modified_image, 0, 1)
    
    if single_image:
        modified_image = modified_image.squeeze(0)
    
    return modified_image


def smooth_mask_edges(mask, kernel_size=5):
    """
    Сглаживает края маски для плавного перехода
    
    Args:
        mask: [B, 1, H, W]
        kernel_size: размер ядра размытия
    
    Returns:
        smoothed_mask: [B, 1, H, W]
    """
    import torch.nn.functional as F
    
    # Создаем ядро гауссова размытия
    sigma = kernel_size / 3.0
    x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    
    # 2D ядро
    kernel = gauss.unsqueeze(0) * gauss.unsqueeze(1)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]
    kernel = kernel.to(mask.device)
    
    # Применяем convolution для размытия
    padding = kernel_size // 2
    smoothed = F.conv2d(mask, kernel, padding=padding)
    
    return smoothed


def visualize_mask_overlay(image, mask, save_path=None):
    """
    Визуализирует маску наложенную на изображение
    
    Args:
        image: [3, H, W] или [H, W, 3]
        mask: [1, H, W] или [H, W]
        save_path: путь для сохранения
    """
    import matplotlib.pyplot as plt
    
    # Конвертируем в numpy
    if torch.is_tensor(image):
        if image.dim() == 3 and image.size(0) == 3:
            image = image.permute(1, 2, 0)
        image = image.cpu().numpy()
    
    if torch.is_tensor(mask):
        if mask.dim() == 3:
            mask = mask.squeeze(0)
        mask = mask.cpu().numpy()
    
    # Создаем цветную версию маски
    mask_colored = np.zeros((*mask.shape, 3))
    mask_colored[mask > 0] = [1, 0, 0]  # Красный цвет
    
    # Накладываем
    overlay = image * 0.6 + mask_colored * 0.4
    overlay = np.clip(overlay, 0, 1)
    
    # Визуализация
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Mask')
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


# Пример использования
if __name__ == '__main__':
    # 1. Парсим аннотации
    annotations = parse_labelstudio_annotations('project-1-at-2025-11-09-11-58-baa04fe3.json')
    
    print(f"Found {len(annotations)} annotated images")
    
    for image_path, polygons in annotations.items():
        print(f"\nImage: {image_path}")
        print(f"  Polygons: {len(polygons)}")
        for i, poly in enumerate(polygons):
            print(f"    Polygon {i+1}: {poly['label']}, {len(poly['points'])} points")
    
    # 2. Загружаем изображение
    # image_path = list(annotations.keys())[0]
    # image = Image.open(image_path).convert('RGB')
    # image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
    # image_tensor = image_tensor.permute(2, 0, 1)  # [3, H, W]
    
    # 3. Создаем маску
    # polygons = annotations[image_path]
    # h, w = image_tensor.shape[1], image_tensor.shape[2]
    # mask = create_mask_from_annotations(polygons, h, w)
    
    # 4. Применяем LED цвета
    # led_color = torch.tensor([1.0, 0.0, 0.0])  # Красный цвет
    # modified = apply_led_colors_to_mask(image_tensor, led_color, mask, alpha=0.7)
    
    # 5. Визуализируем
    # visualize_mask_overlay(image_tensor, mask, 'mask_overlay.png')