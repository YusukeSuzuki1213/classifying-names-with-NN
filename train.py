import torch
from data_prepare import *
from model_rnn import *
from predict import *
import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


n_hidden = 128
n_epochs = 100000
print_every = 5000
plot_every = 1000
learning_rate = 0.005

# 1番確率の高いcategoryを見つける
def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

# トレーニングデータを取得
def randomTrainingExample():
    category = randomChoice(all_categories) # type: String, Example: 'French', 'Japanese'
    line = randomChoice(category_lines[category]) # type: String, Example: 'Abandonato', 'Abatangelo'
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long) # type: torch.Tensor, Example: tensor([3])
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

neural_net = RNN(n_letters, n_hidden, n_categories)
optimizer  = torch.optim.SGD(neural_net.parameters(), lr=learning_rate)
criterion  = nn.NLLLoss()

def train(category_tensor, line_tensor):
    hidden = neural_net.initHidden()

    neural_net.zero_grad() # Sets gradients of all model parameters to zero.

    for i in range(line_tensor.size()[0]):
        output, hidden = neural_net(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    # パラメータの更新？
    for p in neural_net.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()

# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for epoch in range(1, n_epochs + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # Print epoch number, loss, name and guess
    if epoch % print_every == 0:
        guess, guess_i = categoryFromOutput(output) # Type: tuple[str, int], Example: ('Japanese', 8)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (epoch, epoch / n_epochs * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if epoch % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

torch.save(neural_net, 'classifying-names-with-rnn.pt')



# モデルの評価
# lossのグラフ
plt.figure()
plt.plot(all_losses)
plt.savefig('result/loss_figure.png')

# Keep track of correct guesses in a confusion matrix
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000

# Go through a bunch of examples and record which are correctly guessed
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1

# Normalize by dividing every row by its sum
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.savefig('result/confusion_matrix.png')

