import torch
import torch.nn as nn
import torch.nn.functional as F


class LEDControlMLP(nn.Module):
    """
    MLP для генерации adversarial LED паттернов
    БЕЗ BatchNorm - работает с любым batch_size включая 1
    """
    
    def __init__(
        self,
        image_size=256,
        n_sensors=10,
        n_leds=11776,
        hidden_dims=[2048, 4096, 8192, 4096],
        dropout=0.3
    ):
        super(LEDControlMLP, self).__init__()
        
        self.image_size = image_size
        self.n_sensors = n_sensors
        self.n_leds = n_leds
        
        # Энкодер изображения БЕЗ BatchNorm
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Размер выхода image encoder
        image_features_size = 512 * 4 * 4  # 8192
        
        # Объединенный размер входа
        input_size = image_features_size + n_sensors
        
        # Строим MLP слои с LayerNorm
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))  # LayerNorm работает с batch_size=1
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        self.mlp = nn.Sequential(*layers)
        
        # Выходной слой для генерации LED цветов
        self.output_layer = nn.Linear(prev_size, n_leds * 3)
        
        # Инициализация весов
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Инициализация весов модели"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, images, sensors):
        """
        Forward pass
        
        Args:
            images: [batch_size, 3, H, W] - входные изображения
            sensors: [batch_size, n_sensors] - данные датчиков
        
        Returns:
            led_colors: [batch_size, n_leds, 3] - RGB цвета для каждого LED
        """
        batch_size = images.size(0)
        
        # Извлекаем признаки из изображения
        image_features = self.image_encoder(images)
        image_features = image_features.view(batch_size, -1)
        
        # Объединяем признаки изображения и датчиков
        combined = torch.cat([image_features, sensors], dim=1)
        
        # Пропускаем через MLP
        hidden = self.mlp(combined)
        
        # Генерируем LED цвета
        led_output = self.output_layer(hidden)
        
        # Reshape в [batch_size, n_leds, 3]
        led_colors = led_output.view(batch_size, self.n_leds, 3)
        
        # Применяем sigmoid для нормализации в [0, 1]
        led_colors = torch.sigmoid(led_colors)
        
        return led_colors


# Тест модели
if __name__ == '__main__':
    print("Testing LEDControlMLP with batch_size=1...")
    
    model = LEDControlMLP(
        image_size=256,
        n_sensors=10,
        n_leds=100,
        hidden_dims=[512, 1024, 512],
        dropout=0.3
    )
    
    # Тест с batch_size=1
    test_image = torch.randn(1, 3, 256, 256)
    test_sensors = torch.randn(1, 10)
    
    model.eval()
    with torch.no_grad():
        output = model(test_image, test_sensors)
        print(f"✓ Output shape: {output.shape}")
        print(f"✓ Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # Тест с batch_size=4
    print("\nTesting with batch_size=4...")
    test_image_batch = torch.randn(4, 3, 256, 256)
    test_sensors_batch = torch.randn(4, 10)
    
    with torch.no_grad():
        output_batch = model(test_image_batch, test_sensors_batch)
        print(f"✓ Output shape: {output_batch.shape}")
    
    print("\n✓ All tests passed!")