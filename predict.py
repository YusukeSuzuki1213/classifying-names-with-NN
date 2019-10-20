from data_prepare import *
import sys

def evaluate(line_tensor):
    neural_net = torch.load('classifying-names-with-rnn.pt')
    hidden = neural_net.initHidden()

    # 順伝播させる
    for i in range(line_tensor.size()[0]):
        output, hidden = neural_net(line_tensor[i], hidden)

    return output

def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))

        # n_predictionsの値分、上からカテゴリを取得
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])
        
        return predictions

if __name__ == '__main__':
    predict(sys.argv[1])
