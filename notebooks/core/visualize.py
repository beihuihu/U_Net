#   Author: Ankit Kariryaa, University of Bremen

import matplotlib.pyplot as plt  # plotting tools
from matplotlib.patches import Polygon
import numpy as np
import earthpy.plot as ep

def display_images(img,ann,pre=[],pad=0,output_size=512, fn=None,titles=None,cmap=None, norm=None, interpolation=None):
    """Display the given set of images, optionally with titles.
    images: array of image tensors in Batch * Height * Width * Channel format.
    titles: optional. A list of titles to display with each image.
    cmap: Optional. Color map to use. For example, "Blues".
    norm: Optional. A Normalize instance to map values to colors.
    interpolation: Optional. Image interpolation to use for display.
    """
    if len(pre)!=0:
        cols = 5
    else:
        cols = 4
    rows = img.shape[0]

    plt.figure(figsize=(20, 20 * rows // cols),dpi=150)
    plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.9,wspace=0.05,hspace=0.05)
    for i in range(rows):
        for j in range(cols):
            ax=plt.subplot(rows, cols, (i*cols) + j + 1 )
            ax.set_axis_off()
            if (i==0) and (titles!=None):
                ax.set_title(titles[j],fontsize=12)
            if j==0:
                plt.imshow(img[i,...,j], cmap=cmap, norm=norm, interpolation=interpolation)
                plt.hlines(y=pad,xmin=pad,xmax=pad+output_size,color='red',linewidth=1)
                plt.hlines(y=pad+output_size,xmin=pad,xmax=pad+output_size,color='red',linewidth=1)
                plt.vlines(x=pad,ymin=pad,ymax=pad+output_size,color='red',linewidth=1)
                plt.vlines(x=pad+output_size,ymin=pad,ymax=pad+output_size,color='red',linewidth=1)
            elif j==1:
                rgb=np.transpose(img[i,...,j:j+3], axes=(2,0,1))
                ep.plot_rgb(rgb,ax=ax, stretch=True,str_clip=0.5)
                plt.hlines(y=pad,xmin=pad,xmax=pad+output_size,color='red',linewidth=1)
                plt.hlines(y=pad+output_size,xmin=pad,xmax=pad+output_size,color='red',linewidth=1)
                plt.vlines(x=pad,ymin=pad,ymax=pad+output_size,color='red',linewidth=1)
                plt.vlines(x=pad+output_size,ymin=pad,ymax=pad+output_size,color='red',linewidth=1)
            elif j==2:
                plt.imshow(img[i,...,j+2], norm=norm,interpolation=interpolation)
                plt.hlines(y=pad,xmin=pad,xmax=pad+output_size,color='red',linewidth=1)
                plt.hlines(y=pad+output_size,xmin=pad,xmax=pad+output_size,color='red',linewidth=1)
                plt.vlines(x=pad,ymin=pad,ymax=pad+output_size,color='red',linewidth=1)
                plt.vlines(x=pad+output_size,ymin=pad,ymax=pad+output_size,color='red',linewidth=1)
            elif j==3:
                plt.imshow(ann[i,:output_size,:output_size], cmap=cmap, norm=norm, interpolation=interpolation)
                plt.xlim(-pad,pad+output_size)
                plt.ylim(pad+output_size,-pad)
            else:
                plt.imshow(pre[i,:output_size,:output_size], cmap=cmap, norm=norm, interpolation=interpolation)
#                 plt.xlim(-pad,pad+output_size)
#                 plt.ylim(pad+output_size,-pad)
    if fn!=None:
        plt.savefig(fn,bbox_inches='tight',pad_inches = 0.1)
        plt.close()

def display_images_2(img,pad=0,output_size=512, fn=None,titles=None, cmap=None, norm=None, interpolation=None):
    """Display the given set of images, optionally with titles.
    images: array of image tensors in Batch * Height * Width * Channel format.
    titles: optional. A list of titles to display with each image.
    cmap: Optional. Color map to use. For example, "Blues".
    norm: Optional. A Normalize instance to map values to colors.
    interpolation: Optional. Image interpolation to use for display.
    """
    cols = img.shape[-1]-2
    rows = img.shape[0]
#     titles = titles if titles is not None else [""] * (rows*cols)

    plt.figure(figsize=(20, 20 * rows // cols),dpi=150)
    plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.9,wspace=0.05,hspace=0.05)
#     plt.suptitle(title,x=0.5,y=0.89,fontsize=18)
    plt.tight_layout()
    for i in range(rows):
        for j in range(cols):
            ax=plt.subplot(rows, cols, (i*cols) + j + 1 )
            ax.set_axis_off()
            if i==0 and (titles!=None):
                ax.set_title(titles[j],fontsize=12)
            if j==0:
                plt.imshow(img[i,...,j], cmap=cmap, norm=norm, interpolation=interpolation)
            elif j==1:
                rgb=np.transpose(img[i,...,j:j+3], axes=(2,0,1))
                ep.plot_rgb(rgb,ax=ax, stretch=True,str_clip=0.5)
            else:
                plt.imshow(img[i,...,j+2], cmap=cmap, norm=norm, interpolation=interpolation)
            plt.hlines(y=pad,xmin=pad,xmax=pad+output_size,color='red',linewidth=1)
            plt.hlines(y=pad+output_size,xmin=pad,xmax=pad+output_size,color='red',linewidth=1)
            plt.vlines(x=pad,ymin=pad,ymax=pad+output_size,color='red',linewidth=1)
            plt.vlines(x=pad+output_size,ymin=pad,ymax=pad+output_size,color='red',linewidth=1)
    if fn!=None:
        plt.savefig(fn,bbox_inches='tight',pad_inches = 0.1)
        plt.close()
    
def plot(hist,timestr,optimizerName,lossName, patchSize, epochNum, batchSize,chs):
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
    plt.title('OP={} LN={} PS={} Batch={}'.format(optimizerName,lossName,patchSize,batchSize))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    fig_name = '{}_loss_{}_{}_{}_{}_{}.png'.format(timestr,optimizerName,lossName,patchSize,batchSize,chs)
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

    IoU = hist['IoU'] 
    val_IoU = hist['val_IoU']
    plt.plot(epochs, IoU, 'b', label='Training IoU')
    plt.plot(epochs, val_IoU, 'r', label='Validation IoU')
    plt.grid(color='gray', linestyle='--')
    plt.legend()    
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    plt.title('OP={} LN={} PS={} Batch={}'.format(optimizerName,lossName,patchSize,batchSize))
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    fig_name = '{}_IoU_{}_{}_{}_{}_{}.png'.format(timestr,optimizerName,lossName,patchSize,batchSize,chs)
    plt.savefig(fig_name)
    plt.close()

    F1s = hist['F1_score'] 
    val_F1s = hist['val_F1_score']
    plt.plot(epochs, F1s, 'b', label='Training F1 score')
    plt.plot(epochs, val_F1s, 'r', label='Validation F1 score')
    plt.grid(color='gray', linestyle='--')
    plt.legend()    
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    plt.title('OP={} LN={} PS={} Batch={}'.format(optimizerName,lossName,patchSize,batchSize))
    plt.xlabel('Epochs')
    plt.ylabel('F1 score')
    fig_name = '{}_F1_score_{}_{}_{}_{}_{}.png'.format(timestr,optimizerName,lossName,patchSize,batchSize,chs)
    plt.savefig(fig_name)
    plt.close()


    recall = hist['recall'] 
    val_recall = hist['val_recall']
    plt.plot(epochs, recall, 'b', label='Training recall')
    plt.plot(epochs, val_recall, 'r', label='Validation recall')
    plt.grid(color='gray', linestyle='--')
    plt.legend()  
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    plt.title('OP={} LN={} Epochs={} Batch={}'.format(optimizerName,lossName,patchSize,batchSize))
    plt.xlabel('Epochs')
    plt.ylabel('recall')
    fig_name = '{}_recall_{}_{}_{}_{}_{}.png'.format(timestr,optimizerName,lossName,patchSize,batchSize,chs)
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
    plt.title('OP={} LN={} PS={} Batch={}'.format(optimizerName,lossName,patchSize,batchSize))
    plt.xlabel('Epochs')
    plt.ylabel('precision')
    fig_name = '{}_precision_{}_{}_{}_{}_{}.png'.format(timestr,optimizerName,lossName,patchSize,batchSize,chs)
    plt.savefig(fig_name)
    plt.close()