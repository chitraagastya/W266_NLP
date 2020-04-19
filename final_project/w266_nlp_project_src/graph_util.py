import matplotlib.pyplot as plt

def plot_graph(hist, epoch, graphname, type='answer'):
    # visualizing losses and accuracy
    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    if type == 'answer':
        train_acc = hist.history['sparse_categorical_accuracy']
        val_acc = hist.history['val_sparse_categorical_accuracy']
    else:
        train_acc = hist.history['binary_accuracy']
        val_acc = hist.history['val_binary_accuracy']
    xc = range(epoch)

    plt.figure(1, figsize=(7, 5))
    plt.plot(xc, train_loss)
    plt.plot(xc, val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train', 'val'])
    # print plt.style.available # use bmh, classic,ggplot for big pictures
    plt.style.use(['classic'])
    plt.savefig('{}_loss.png'.format(graphname))

    plt.figure(2, figsize=(7, 5))
    plt.plot(xc, train_acc)
    plt.plot(xc, val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train', 'val'], loc=4)
    # print plt.style.available # use bmh, classic,ggplot for big pictures
    plt.style.use(['classic'])

    plt.savefig('{}_accuracy.png'.format(graphname))
    plt.clf()
    plt.close()

def plot_graph2(hist, epoch, graphname, type='answer'):
    # visualizing losses and accuracy
    train_loss = hist['loss']
    val_loss = hist['val_loss']
    if type == 'answer':
        train_acc = hist['sparse_categorical_accuracy']
        val_acc = hist['val_sparse_categorical_accuracy']
    else:
        train_acc = hist['binary_accuracy']
        val_acc = hist['val_binary_accuracy']
    xc = range(epoch)

    plt.figure(1, figsize=(7, 5))
    plt.plot(xc, train_loss)
    plt.plot(xc, val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train', 'val'])
    # print plt.style.available # use bmh, classic,ggplot for big pictures
    plt.style.use(['classic'])
    plt.savefig('{}_loss.png'.format(graphname))

    plt.figure(2, figsize=(7, 5))
    plt.plot(xc, train_acc)
    plt.plot(xc, val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train', 'val'], loc=4)
    # print plt.style.available # use bmh, classic,ggplot for big pictures
    plt.style.use(['classic'])

    plt.savefig('{}_accuracy.png'.format(graphname))
    plt.clf()
    plt.close()