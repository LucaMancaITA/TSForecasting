
def adjust_learning_rate(optimizer, epoch, learning_rate):
    """Adjust training learning rate (exponential decay).

    Args:
        optimizer (object): training optimizer.
        epoch (int): training epoch.
        learning_rate (float): learning rate
    """
    lr_adjust = {epoch: learning_rate * (0.5 ** ((epoch-1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))
