import torch
from torch.utils.data import Dataset, DataLoader
import re
from collections import Counter
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 读取文本数据
with open('text_data.txt', 'r') as file:
    text = file.read()

# 预处理：分词、去标点符号
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text.split()

words = preprocess_text(text)

# 构建词汇表
word_counts = Counter(words)
vocab = ['<UNK>'] + [word for word, count in word_counts.items() if count > 1]
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}

# 将文本转换为索引
indexed_text = [word2idx[word] if word in word2idx else word2idx['<UNK>'] for word in words]

# 创建数据集
class CBOWDataset(Dataset):
    def __init__(self, indexed_text, context_size=2):
        self.indexed_text = indexed_text
        self.context_size = context_size
        self.data = []

        for i in range(context_size, len(indexed_text) - context_size):
            context = indexed_text[i - context_size:i] + indexed_text[i + 1:i + 1 + context_size]
            target = indexed_text[i]
            self.data.append((context, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        return torch.tensor(context), torch.tensor(target)

# 创建数据集和数据加载器
dataset = CBOWDataset(indexed_text, context_size=2)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 定义CBOW模型
class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context):
        embeds = self.embeddings(context)
        sum_embeds = embeds.sum(dim=1)
        out = self.linear(sum_embeds)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

# 初始化模型、损失函数和优化器
vocab_size = len(vocab)
embedding_dim = 100
model = CBOWModel(vocab_size, embedding_dim)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
losses = []

for epoch in range(num_epochs):
    total_loss = 0
    for contexts, targets in dataloader:
        model.zero_grad()
        log_probs = model(contexts)
        loss = criterion(log_probs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    losses.append(avg_loss)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')

# 绘制损失曲线
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# 评估模型
def find_similar_words(word, top_k=5):
    word_idx = word2idx[word]
    embeddings = model.embeddings.weight.data
    word_vector = embeddings[word_idx]
    
    distances = [(i, torch.dist(word_vector, embeddings[i]).item()) for i in range(vocab_size)]
    distances.sort(key=lambda x: x[1])
    
    for i, dist in distances[:top_k]:
        print(f'{idx2word[i]}: {dist:.4f}')

# 查找与"good"最相似的词
find_similar_words('good')