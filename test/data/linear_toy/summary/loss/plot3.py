import numpy as np
import matplotlib.pyplot as plt
loss1=[]
loss2=[]
for i in range(1,6):

    loss1 .append( np.array(np.load('losssim-%i.npy'%i),dtype=np.float32))
    loss2.append(np.array(np.load('loss2-%i.npy' % i), dtype=np.float32))

loss1=np.vstack(loss1)
loss2=np.vstack(loss2)
xs=np.arange(loss1.shape[1])
plt.figure()
plt.plot(np.mean(loss1,0),color='r',label='plain gradient descent')
plt.fill_between(xs,np.mean(loss1,0)-np.std(loss1,0),np.mean(loss1,0)+np.std(loss1,0),color='r',alpha=0.1)
plt.plot(np.mean(loss2,0),color='g',label='MAML')
plt.fill_between(xs,np.mean(loss2,0)-np.std(loss2,0),np.mean(loss2,0)+np.std(loss2,0),color='g',alpha=0.1)
plt.legend()
plt.title('loss after one step adaption')
plt.show()