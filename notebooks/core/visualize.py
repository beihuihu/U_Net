#   Author: Ankit Kariryaa, University of Bremen

import matplotlib.pyplot as plt  # plotting tools
from matplotlib.patches import Polygon
import numpy as np

def display_images(img, titles=None, cmap=None, norm=None, interpolation=None):
    """Display the given set of images, optionally with titles.
    images: array of image tensors in Batch * Height * Width * Channel format.
    titles: optional. A list of titles to display with each image.
    cmap: Optional. Color map to use. For example, "Blues".
    norm: Optional. A Normalize instance to map values to colors.
    interpolation: Optional. Image interpolation to use for display.
    """
    cols = img.shape[-1]
    rows = img.shape[0]
#     titles = titles if titles is not None else [""] * (rows*cols)

    plt.figure(figsize=(14, 14 * rows // cols))
    for i in range(rows):
        for j in range(cols):
            plt.subplot(rows, cols, (i*cols) + j + 1 )
            plt.axis('off')
            plt.imshow(img[i,...,j], cmap=cmap, norm=norm, interpolation=interpolation)
#             plt.title(titles[(i*cols) + j ])
#     plt.suptitle(titles)
#     plt.show()

def plot(hist,optimizerName,lossName, patchSize, epochNum, batchSize,chs):
    plt.figure()
    train_loss = hist['loss']
    val_loss = hist['val_loss']
    x_ticks= np.arange(1, epochNum+1,10) 
    y_ticks = np.arange(0.2,1,0.05)
    y_ticks_loss = np.arange(0,0.8,0.05)
    epochs = np.arange(1, len(train_loss)+1,1)
    plt.plot(epochs,train_loss, 'b', label='Training Loss')
    plt.plot(epochs,val_loss, 'r', label='Validation Loss')
    plt.grid(color='gray', linestyle='--')
    plt.legend()
    plt.xticks(x_ticks)
    plt.yticks(y_ticks_loss)
    plt.title('OP={} LN={} PS={} Epochs={} Batch={}'.format(optimizerName,lossName,patchSize,epochNum, batchSize))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    fig_name = 'loss_{}_{}_{}_{}_{}_{}.png'.format(optimizerName,lossName,patchSize,epochNum,batchSize,chs)
    plt.savefig(fig_name)
    plt.close()

#     dice_loss = hist['dice_loss'] 
#     val_dice_loss = hist['val_dice_loss']
#     plt.figure()
#     plt.plot(epochs, dice_loss, 'b', label='Training Dice loss')
#     plt.plot(epochs, val_dice_loss, 'r', label='Validation Dice loss')
#     plt.grid(color='gray', linestyle='--')
#     plt.legend()  
#     plt.xticks(x_ticks)
#     plt.yticks(y_ticks_loss)
#     plt.title('OP={} LN={} PS={} Epochs={} Batch={}'.format(optimizerName,lossName,patchSize,epochNum, batchSize))
#     plt.xlabel('Epochs')
#     plt.ylabel('Dice_loss')
#     fig_name = 'dice_loss_{}_{}_{}_{}_{}_{}.png'.format(optimizerName,lossName,patchSize,epochNum,batchSize,chs)
#     plt.savefig(fig_name)
#     plt.close()
    
#     acc = hist['accuracy'] 
#     val_acc = hist['val_accuracy']
#     plt.plot(epochs, acc, 'b', label='Training accuracy')
#     plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
#     plt.grid(color='gray', linestyle='--')
#     plt.legend()  
#     plt.xticks(x_ticks)
#     plt.yticks(y_ticks)
#     plt.title('OP={} LN={} PS={} Epochs={} Batch={}'.format(optimizerName,lossName,patchSize,epochNum, batchSize))
#     plt.xlabel('Epochs')
#     plt.ylabel('accuracy')
#     fig_name = 'accuracy_{}_{}_{}_{}_{}_{}.png'.format(optimizerName,lossName,patchSize,epochNum,batchSize,chs)
#     plt.savefig(fig_name)
#     plt.close()

    recall = hist['recall'] 
    val_recall = hist['val_recall']
    plt.plot(epochs, recall, 'b', label='Training recall')
    plt.plot(epochs, val_recall, 'r', label='Validation recall')
    plt.grid(color='gray', linestyle='--')
    plt.legend()  
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    plt.title('OP={} LN={} PS={} Epochs={} Batch={}'.format(optimizerName,lossName,patchSize,epochNum, batchSize,chs))
    plt.xlabel('Epochs')
    plt.ylabel('recall')
    fig_name = 'recall_{}_{}_{}_{}_{}_{}.png'.format(optimizerName,lossName,patchSize,epochNum,batchSize,chs)
    plt.savefig(fig_name)
    plt.close()

    IoU = hist['IoU'] 
    val_IoU = hist['val_IoU']
    plt.plot(epochs, IoU, 'b', label='Training IoU')
    plt.plot(epochs, val_IoU, 'r', label='Validation IoU')
    plt.grid(color='gray', linestyle='--')
    plt.legend()    
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    plt.title('OP={} LN={} PS={} Epochs={} Batch={}'.format(optimizerName,lossName,patchSize,epochNum, batchSize))
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    fig_name = 'IoU_{}_{}_{}_{}_{}_{}.png'.format(optimizerName,lossName,patchSize,epochNum,batchSize,chs)
    plt.savefig(fig_name)
    plt.close()
    
    precision = hist['precision'] 
    val_precision = hist['val_precision']
    plt.plot(epochs, precision, 'b', label='Training precision')
    plt.plot(epochs, val_precision, 'r', label='Validation precision')
    plt.grid(color='gray', linestyle='--')
    plt.legend()   
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    plt.title('OP={} LN={} PS={} Epochs={} Batch={}'.format(optimizerName,lossName,patchSize,epochNum, batchSize))
    plt.xlabel('Epochs')
    plt.ylabel('precision')
    fig_name = 'precision_{}_{}_{}_{}_{}_{}.png'.format(optimizerName,lossName,patchSize,epochNum,batchSize,chs)
    plt.savefig(fig_name)
    plt.close()