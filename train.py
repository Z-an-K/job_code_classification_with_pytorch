import torch
from torch import nn
from torch.utils.data import DataLoader
from model import load_pretrained_model,BERTClassifier
from processor_data import JobDataset
from d2l import torch as d2l



def train_batch_ch13(net, X, y, loss, trainer, devices):
    if isinstance(X, list):
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum

def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus()):
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')


model_name = "bert-base-chinese"
MODEL_PATH = "/home/zhanghan/PycharmProjects/learn_pytorch/bert-classification/bert-base-chinese"
vocab_name = "vocab.txt"
bert,tokenizer = load_pretrained_model(model_name,vocab_name,MODEL_PATH)
net = BERTClassifier(bert)
batch_size,sentence_lens = 32,32
seq_lens = 24
train_set = JobDataset(seq_lens,tokenizer,True)
test_set = JobDataset(seq_lens,tokenizer,False)
train_iter = DataLoader(train_set,batch_size,shuffle=True,num_workers=4)
test_iter = DataLoader(test_set,batch_size,shuffle=False,num_workers=4)
devices = ["cuda" if torch.cuda.is_available() else "cpu"]
lr, num_epochs = 1e-4, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction='none')
train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
    devices)
