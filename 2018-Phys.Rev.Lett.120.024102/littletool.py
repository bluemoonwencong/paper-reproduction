import os
import numpy as np
import matplotlib.pyplot as plt
import logging

'''
原文，0.25*70000=17000
0.1*100000=10000
考虑reservior维度和最先的一个半成功demo，可行。
如果0.1*50000=5000，可能就不好了。

import numpy as np
for i in ('A0','Win0','A1','Win1','A2','Win1','A2','Win2','A3','Win3','A4','Win4'):
    data = np.loadtxt(i + '.dat')
    np.save(i + '.npy', data)

'''

logger = logging.getLogger()
logger.setLevel('DEBUG')
BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
DATE_FORMAT = '%Y/%m/%d %H:%M:%S'
formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
handler1 = logging.StreamHandler()
handler1.setFormatter(formatter)
handler1.setLevel('INFO')#控制台就info级别吧
handler2 = logging.FileHandler('logging.log')#日志文件debug级别
handler2.setFormatter(formatter)
logger.addHandler(handler1)
logger.addHandler(handler2)
logger.debug('debug message')
logger.info('info message')
logger.warning('warning message')
logger.error('error message')
logger.critical('critical message')



def dataload(win_key, a_key):

    utrain = np.load('u-shape=(95000,200)-train-dt0.1-dx0.25.npy')[0:95000]
    udev1   = np.load('u-shape=(3000,200)-dev1-dt0.1-dx0.25.npy')[0:3000]
    udev2  = np.load('u-shape=(2000,200)-dev2-dt0.1-dx0.25.npy')[0:2000]
    Win    = np.load(win_key + '.npy')
    A      = np.load(a_key + '.npy')

    T1  = utrain.shape[0]
    T2  = udev1.shape[0]
    T3  = udev2.shape[0]
    Din = utrain.shape[1]
    Dr  = A.shape[0]
    hp_width = 1

    fixedparameter = (T1, T2, T3, Din, Dr, hp_width, Win, A, a_key, win_key)

    logger.info('数据特征:')
    logger.warning(win_key + a_key)
    logger.info(f'mean(Win)={Win.mean()}')
    logger.info(f'mean(abs(Win))={np.abs(Win).mean()}')
    logger.info(f'(Win(#)==0 % : {(Win == 0).sum()/(Dr*Din)}')
    logger.info(f'det(A)={np.linalg.det(A)}')
    logger.info(f'mean(A)={A.mean()}')
    logger.info(f'(A(#)==0 % : {(A == 0).sum()/(Dr*Dr)}')
    logger.info(f'T1, T2, T3, Din, Dr : {[T1, T2, T3, Din, Dr]}')

    return utrain, udev1, udev2, fixedparameter


def caculate_rtrain(utrain, findparameter, fixedparameter):
    """
    return xtrain, rtrain_nearfuture

    """
    
    T1, T2, T3, Din, Dr, hp_width, Win, A, a_key, win_key = fixedparameter
    hp_a, hp_win, hp_a_index, hp_win_index = findparameter
    rtrain = np.zeros((T1, Dr))
    rtrain[0] = np.zeros(Dr)

    for i in range(1, T1):
        rtrain[i] = np.tanh( hp_width * (hp_a*A.dot(rtrain[i-1]) + hp_win*Win.dot(utrain[i-1]) ) )
    rtrain_nearfuture = np.tanh( hp_width * (hp_a*A.dot(rtrain[T1-1]) + hp_win*Win.dot(utrain[T1-1]) ) )

    xtrain = np.hstack( (rtrain, np.power(rtrain, 2)) )

    assert utrain.shape == (T1, Din), 'utrain.shape error'
    assert rtrain.shape == (T1, Dr), 'rtrain.shape error'
    assert xtrain.shape == (T1, 2*Dr), 'xtrain.shape error'

    see = T1 - 1
    path = f"./best-parameter/hp_a[{hp_a_index}]" # 
    imgname = f'T1={T1},hp_a[{hp_a_index}],hp_win[{hp_win_index}]=({hp_a:0.9f},{hp_win:0.9f})' + a_key + win_key + '.jpg'
    if not os.path.isdir(path):
        os.mkdir(path)
    plt.figure(figsize=(20,3))

    plt.subplot(1,4,1)
    plt.ylim((-1,1))
    plt.plot(rtrain[see],'r.')
    plt.title(f'rtrain[{see}]')

    plt.subplot(1,4,2)
    plt.plot( hp_width * hp_a * A.dot(rtrain[see]), 'g.' )
    plt.title(f'hp_width * hp_a * A * rtrain[{see}]')

    plt.subplot(1,4,3)
    plt.ylim((-1,1))
    plt.plot( hp_width * hp_win * Win.dot(utrain[see]), 'b.' )
    plt.title(f'hp_width * hp_win * Win * utrain[{see}]')

    plt.subplot(1,4,4)
    plt.plot(np.linspace(-5,5,200), np.tanh(hp_width*np.linspace(-5,5,200)))
    plt.title(f'tanh(#)')

    plt.savefig(path + '/' + imgname)
    plt.close('all')

    del rtrain
    logger.info(f"hp_a[{hp_a_index}], hp_win[{hp_win_index}], rtrain have been deduced and deleted, xtrain get, over.")
    return xtrain, rtrain_nearfuture


def see_dev1predict(udev1, P, beta_index, beta, rtrain_nearfuture, findparameter, fixedparameter):
    """
    返回 rdev1_nearfuture
    """

    T1, T2, T3, Din, Dr, hp_width, Win, A, a_key, win_key = fixedparameter
    hp_a, hp_win, hp_a_index, hp_win_index = findparameter

    rdev1 = np.zeros((T2, Dr))
    xdev1 = np.zeros((T2, 2*Dr))
    rdev1[0] = rtrain_nearfuture
    xdev1[0] = np.hstack( (rdev1[0], np.power(rdev1[0], 2)) )
    for i in range(1,T2):
        rdev1[i] = np.tanh( hp_width * (hp_a*A.dot(rdev1[i-1]) + hp_win*Win.dot( xdev1[i-1].dot(P) ) ) )
        xdev1[i] = np.hstack( (rdev1[i], np.power(rdev1[i], 2)) )

    assert udev1.shape == (T2, Din), 'udev1.shape error'
    assert rdev1.shape == (T2, Dr), 'rdev1.shape error'
    assert xdev1.shape == (T2, 2*Dr), 'xdev1.shape error'

    udev1_predict = xdev1.dot(P)
    dev1_rmse = np.sqrt( np.power(udev1 - udev1_predict, 2).mean(axis=1) )

    dev1_count = 0
    while dev1_rmse[dev1_count]<1:
        dev1_count += 1

    path = f"./best-beta/hp_a[{hp_a_index}]" # 
    if not os.path.isdir(path):
        os.mkdir(path)
    imgname = f'T1={T1},hp_a[{hp_a_index}],hp_win[{hp_win_index}],beta[{beta_index}]=({hp_a:0.9f},{hp_win:0.9f},{beta:0.9f})-' + a_key + win_key + '-reinit' + str(T1) + f'-dev1count={dev1_count}' + '.jpg'
    np.save(path + '/' + imgname + '.npy', dev1_rmse)
    
    plt.figure(figsize=(12,8))
    plt.subplot(3,1,1)
    plt.title(imgname)
    plt.imshow(udev1_predict[0:1250].T)
    plt.subplot(3,1,2)
    plt.imshow(udev1[0:1250].T)
    plt.subplot(3,1,3)
    plt.plot(dev1_rmse[0:1250],'.')
    plt.plot([0, 1250], [1,1], 'r')
    plt.ylim((0, 3))
    plt.title('dev1_rmse.mean =' + str(dev1_rmse.mean()) + f'      dev1_count={dev1_count}')
    plt.savefig(path + '/' + imgname)
    plt.close('all')

    logger.info(f"hp_a[{hp_a_index}], hp_win[{hp_win_index}], beta[{beta_index}], P over, rdev1 over.")


def see_dev2predict(udev1, udev2, P, beta_index, beta, findparameter, fixedparameter):

    T1, T2, T3, Din, Dr, hp_width, Win, A, a_key, win_key = fixedparameter
    hp_a, hp_win, hp_a_index, hp_win_index = findparameter
    
    rdev1 = np.zeros((T2, Dr))
    rdev1[0] = np.zeros(Dr)
    for i in range(1, T2):
        rdev1[i] = np.tanh( hp_width * (hp_a*A.dot(rdev1[i-1]) + hp_win*Win.dot(udev1[i-1]) ) )
    rdev1_nearfuture = np.tanh( hp_width * (hp_a*A.dot(rdev1[T2-1]) + hp_win*Win.dot(udev1[T2-1]) ) )
    
    rdev2 = np.zeros((T3, Dr))
    xdev2 = np.zeros((T3, 2*Dr))
    rdev2[0] = rdev1_nearfuture
    xdev2[0] = np.hstack( (rdev2[0], np.power(rdev2[0], 2)) )
    for i in range(1,T3):
        rdev2[i] = np.tanh( hp_width * (hp_a*A.dot(rdev2[i-1]) + hp_win*Win.dot( xdev2[i-1].dot(P) ) ) )
        xdev2[i] = np.hstack( (rdev2[i], np.power(rdev2[i], 2)) )

    assert udev2.shape == (T3, Din), 'udev2.shape error'
    assert rdev2.shape == (T3, Dr), 'rdev2.shape error'
    assert xdev2.shape == (T3, 2*Dr), 'xdev2.shape error'

    udev2_predict = xdev2.dot(P)
    dev2_rmse = np.sqrt( np.power(udev2 - udev2_predict, 2).mean(axis=1) )

    dev2_count = 0
    while dev2_rmse[dev2_count]<1:
        dev2_count += 1

    path = f"./best-beta/hp_a[{hp_a_index}]" # 
    if not os.path.isdir(path):
        os.mkdir(path)
    imgname = f'T1={T1},hp_a[{hp_a_index}],hp_win[{hp_win_index}],beta[{beta_index}]=({hp_a:0.9f},{hp_win:0.9f},{beta:0.9f})-' + a_key + win_key + '-reinit' + str(T2) + f'-dev2count={dev2_count}' + '.jpg'
    np.save(path + '/' + imgname + '.npy', dev2_rmse)

    plt.figure(figsize=(12,8))
    plt.subplot(3,1,1)
    plt.title(imgname)
    plt.imshow(udev2_predict[0:1250].T)
    plt.subplot(3,1,2)
    plt.imshow(udev2[0:1250].T)
    plt.subplot(3,1,3)
    plt.plot(dev2_rmse[0:1250],'.')
    plt.plot([0, 1250], [1,1], 'r')
    plt.ylim((0, 3))
    plt.title('dev2_rmse.mean =' + str(dev2_rmse.mean()) + f'      dev2_count={dev2_count}')
    plt.savefig(path + '/' + imgname)
    plt.close('all')

    logger.info(f"hp_a[{hp_a_index}], hp_win[{hp_win_index}], beta[{beta_index}], rdev2 over.")




print('hello world! 111')

if __name__ == '__main__':
    print('hello world! 222')

'''
作为模块被调用，会打印111；直接调用，会分别打印111，222.



'''

