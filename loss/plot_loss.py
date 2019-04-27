import matplotlib.pyplot as plt
import numpy as np

def showPlot(plot_epoches, plot_losses, plot_val_losses, dataset, learning_rate, batch_size):
    plt.plot(plot_epoches, plot_losses, color='red', label='train loss')
    plt.plot(plot_epoches, plot_val_losses, color='blue', label='validate loss')
    plt.ylim(0, )
    title = str(dataset).split('_')[1]+'_lr='+str(learning_rate)+'_batchsize='+str(batch_size)+'_epoch='+str(plot_epoches[-1])
    plt.title(title)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss (per sentence)')
    plt.savefig(title + '.jpg')
    plt.show()
    return

# Load
file_path = 'loss.npy'
dic = np.load(file_path).item()
plot_epoches = dic['plot_epoches']
plot_losses = dic['plot_losses']
plot_val_losses = dic['plot_val_losses']  
dataset = dic['dataset']
learning_rate = dic['learning_rate']
batch_size = dic['batch_size']  
showPlot(plot_epoches, plot_losses, plot_val_losses, dataset, learning_rate, batch_size)