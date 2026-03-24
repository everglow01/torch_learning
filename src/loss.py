from torch import nn
class LossFunction(nn.Module):
    def __init__(self):
        super().__init__()
        self.location_loss = nn.MSELoss()
        self.classification_loss = nn.CrossEntropyLoss()  
        
    def forward(self, predictions, targets, batch_size) -> torch.Tensor:
        # predictions shape: (batch_size, 8) -> (batch_size, 4) for coordinates and (batch_size, 4) for class probabilities
        pred_locations = predictions[:, :4]  # 前四个元素是坐标预测   左闭 右开
        pred_classes= predictions[:, 4:]   # 后四个元素是类别概率 
        
        target_locations = targets[:, :4]  # 前四个元素是坐标标签
        target_classes = targets[:, 4:].long()  # 后四个元素是类别标签，转换为long类型以适应CrossEntropyLoss
        
        location_loss_value = self.location_loss(pred_locations, target_locations)
        classification_loss_value = self.classification_loss(pred_classes, target_classes)
        
        total_loss = location_loss_value + classification_loss_value
        return total_loss