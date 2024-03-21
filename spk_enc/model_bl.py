import torch
import torch.nn as nn


# D-向量说话人嵌入
# 继承自nn.Module基类
class D_VECTOR(nn.Module):
    """d vector speaker embedding."""
    # LSTM层复用3次，隐藏单元数量256个
    def __init__(self, num_layers=3, dim_input=40, dim_cell=256, dim_emb=64):
        super(D_VECTOR, self).__init__()
        self.lstm = nn.LSTM(input_size=dim_input, hidden_size=dim_cell, 
                            num_layers=num_layers, batch_first=True) 
        # embedding对输入的数据进行线性变换，输入大小为256，输出大小为64
        self.embedding = nn.Linear(dim_cell, dim_emb)
        
    
    # 网络模型的计算过程    
    def forward(self, x):
        self.lstm.flatten_parameters()
        # 获得LSTM层的输出
        lstm_out, _ = self.lstm(x)
        # LSTM的输入embedding中，得到Linear层输出
        embeds = self.embedding(lstm_out[:,-1,:])
        # 对最后一个维度计算2范数、保持维度不变
        norm = embeds.norm(p=2, dim=-1, keepdim=True) 
        # embeds除以norm得到标准化的embeds
        embeds_normalized = embeds.div(norm)
        return embeds_normalized
    
