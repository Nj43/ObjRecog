"""
This file contains links main.py, data.py and model.py
"""


from data import ResNet152_data, VisionTransformer_data, basic_CNN_data, DeiT_data, ViT_EVA_Giant_data
from model import Net,VisionTransformer,ResNet152, DeiT, ViT_MAE, ViT_EVA_Giant


class ModelFactory:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = self.init_model()
        self.train_transform, self.val_transform, self.test_transform = self.init_transform()

    def init_model(self):
        if self.model_name == "basic_cnn":
            return Net()
        elif self.model_name == "resnet152" :
            return ResNet152()
        elif self.model_name == "vit" :
            return VisionTransformer()
        elif self.model_name == "deit":
            return DeiT()
        elif self.model_name == "mae" :
            return ViT_MAE()
        elif self.model_name == "eva" :
            return ViT_EVA_Giant()
        else:
            raise NotImplementedError("Model not implemented")

    def init_transform(self):
        if self.model_name == "basic_cnn":
            return basic_CNN_data
        elif self.model_name == "resnet152" :
            return ResNet152_data()
        elif self.model_name == "vit" :
            return VisionTransformer_data()
        elif self.model_name == "deit":
            return DeiT_data()
        elif self.model_name == "mae" :
            return ResNet152_data()
        elif self.model_name == "eva" :
            return ViT_EVA_Giant_data()
        else:
            raise NotImplementedError("Transform not implemented")

    def get_model(self):
        return self.model

    def get_transform(self):
        return self.transform

    def get_all(self):
        return self.model, self.train_transform, self.val_transform, self.test_transform