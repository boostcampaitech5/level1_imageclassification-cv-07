import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm


class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)

class ResNet18(nn.Module):
    def __init__(self, num_classes=18):
        """ResNet18 모델에 fc layer만 변경한 클래스

        Args:
            num_classes (int, optional): 분류할 클래스의 개수. Defaults to 18.
        """
        super(ResNet18, self).__init__()

        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.fc = nn.Linear(in_features=512, out_features=num_classes)
    
    def forward(self, x):
        x = self.resnet18(x)

        return x
    

class ResNet50(nn.Module):
    def __init__(self, num_classes=18):
        """ResNet 50 모델에 fc layer만 변경한 클래스

        Args:
            num_classes (int, optional): 분류할 클래스의 개수. Defaults to 18.
        """
        super(ResNet50, self).__init__()

        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.fc = nn.Linear(in_features=2048, out_features=num_classes)

    def forward(self, x):
        x = self.resnet50(x)

        return x

class ResNet152(nn.Module):
    def __init__(self, num_classes):
        super(ResNet152, self).__init__()

        self.resnet152 = models.resnet152(pretrained=True)
        self.resnet152.fc = nn.Linear(in_features=2048, out_features=num_classes)

    def forward(self, x):
        x = self.resnet152(x)

        return x


class EfficientNetV2_S(nn.Module):
    def __init__(self, num_classes=18):
        super(EfficientNetV2_S, self).__init__()

        self.backbone = timm.models.efficientnetv2_s()
        self.backbone.classifier = nn.Linear(in_features=1280, out_features=num_classes)
    
    def forward(self, x):
        x = self.backbone(x)

        return x


class Beit(nn.Module):
    def __init__(self, num_classes=18):
        super(Beit, self).__init__()

        self.backbone = timm.models.beit_large_patch16_224(pretrained=True)
        self.backbone.head = nn.Linear(in_features=1024, out_features=num_classes, bias=True)
    
    def forward(self, x):
        x = self.backbone(x)

        return x
    

class MyEnsemble(nn.Module):
    def __init__(self, num_classes=18):
        """학습한 모델들을 앙상블해서 결과를 출력하는 클래스

        Args:
            num_classes (int, optional): 분류할 클래스의 개수. Defaults to 18.

        how-to-use:
            앙상블할 모델을 선언하고, parameters를 load한 뒤 eval 모드로 바꾸어준다.
        """
        super(MyEnsemble, self).__init__()


        self.modelA = ResNet18()
        self.modelA.load_state_dict(torch.load("/opt/ml/level1_imageclassification-cv-07/cv-07_image-classification/model/fold_model2/resnet18_sfold_ce/best.pth"))
        self.modelA.eval()

        self.modelB = ResNet18()
        self.modelB.load_state_dict(torch.load("/opt/ml/level1_imageclassification-cv-07/cv-07_image-classification/model/fold_model2/resnet18_sfold_ce2/best.pth"))
        self.modelB.eval()

        self.modelC = ResNet18()
        self.modelC.load_state_dict(torch.load("/opt/ml/level1_imageclassification-cv-07/cv-07_image-classification/model/fold_model2/resnet18_sfold_ce3/best.pth"))
        self.modelC.eval()

        self.modelD = ResNet18()
        self.modelD.load_state_dict(torch.load("/opt/ml/level1_imageclassification-cv-07/cv-07_image-classification/model/fold_model2/resnet18_sfold_ce4/best.pth"))
        self.modelD.eval()

        self.modelE = ResNet18()
        self.modelE.load_state_dict(torch.load("/opt/ml/level1_imageclassification-cv-07/cv-07_image-classification/model/fold_model2/resnet18_sfold_ce5/best.pth"))
        self.modelE.eval()
        
    def forward(self, x):
        out1 = self.modelA(x)
        out2 = self.modelB(x)
        out3 = self.modelC(x)
        out4 = self.modelD(x)
        out5 = self.modelE(x)
        
        out = out1 + out2 + out3 + out4 + out5
        out = torch.softmax(out, dim=-1)

        return out

# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return x

# Test code
if __name__ == '__main__':
    # model = ResNet18(num_classes=18)
    # # overview
    # for name, module in model.named_modules():
    #     print(name, module)

    # input = torch.randn(4, 3, 224, 224)
    # print(model(input).shape)

    ensemble = MyEnsemble()

