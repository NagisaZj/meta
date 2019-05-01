import numpy as np
import torch
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from base_net import Base_net
from fast_net import Fast_net
class exp_generator():
    def __init__(self,samples):
        self.samples = samples
        self.step = 2*np.pi/samples
        self.sample_points = np.arange(self.samples)*self.step
        self.data = np.zeros((samples,1))

    def generate(self,amp,phase):
        self.data = amp / (1+np.exp(phase/np.pi-self.sample_points))
        return self.data,self.sample_points

class sin_generator():
    def __init__(self,samples):
        self.samples = samples
        self.step = 2*np.pi/samples
        self.sample_points = np.arange(self.samples)*self.step
        self.data = np.zeros((samples,1))

    def generate(self,amp,phase):
        self.data = amp * np.sin(self.sample_points+phase)
        return self.data,self.sample_points

class trainer():
    def __init__(self,num_samples,num_tasks,num_updates, step_size,meta_step_size, batch_size, meta_batch_size,num_iterations,num_clusters):
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.meta_batch_size = meta_batch_size
        self.generator = sin_generator(num_samples)
        self.generator_exp = exp_generator(num_samples)
        self.num_tasks = num_tasks
        self.num_samples = num_samples
        self.sample = np.zeros((self.num_tasks*2,num_samples,1))
        self.test_data = np.zeros((self.num_tasks*2, num_samples, 1))
        self.x = np.zeros((self.num_tasks*2,num_samples,1))
        self.num_updates = num_updates
        self.step_size = step_size
        self.meta_step_size = meta_step_size
        self.num_clusters = num_clusters
        amp = 0.5+np.random.rand(self.num_tasks*4) * 2.5
        phase = np.random.rand(self.num_tasks*4)*np.pi
        for i in range(self.num_tasks):
            self.sample[i,:,0],self.x[i,:,0] = self.generator.generate(amp[i],phase[i])
        for i in range(self.num_tasks):
            self.test_data[i,:,0],_ = self.generator.generate(amp[self.num_tasks+i],phase[self.num_tasks+i])
        for i in range(self.num_tasks):
            self.sample[self.num_tasks+i,:,0],self.x[self.num_tasks+i,:,0] = self.generator_exp.generate(
                amp[self.num_tasks*2+i],phase[self.num_tasks*2+i])
        for i in range(self.num_tasks):
            self.test_data[self.num_tasks+i,:,0],_ = self.generator_exp.generate(amp[self.num_tasks*3+i],phase[self.num_tasks*3+i])
        self.base_net = []
        for i in range(self.num_clusters):
            self.base_net.append(Base_net(nn.MSELoss()))
            self.base_net[i].cuda()
        self.base_net_plain = Base_net(nn.MSELoss())
        self.fast_net = Fast_net(nn.MSELoss(),num_updates, step_size, batch_size, meta_batch_size)
        self.fast_net.cuda()
        self.base_net_plain.cuda()
        self.opt = []
        for j in range(self.num_clusters):
            self.opt.append(torch.optim.Adam(self.base_net[j].parameters(), lr=meta_step_size))
        #self.opt_plain = torch.optim.Adam(self.base_net.parameters(), lr=meta_step_size)
        return

    def train_plain(self): #naive learning by learning tasks together
        total_loss = []
        test_loss = []
        for it in range(self.num_iterations):
            t_loss = 0
            for i in range(self.num_tasks):
                indices = np.random.choice(self.num_samples, size=self.batch_size)
                indices_meta = np.random.choice(self.num_samples, size=self.meta_batch_size)
                sample_x = self.x[i,indices,:]
                sample_y = self.sample[i,indices,:]
                sample_x_meta = self.x[i, indices_meta, :]
                sample_y_meta = self.sample[i, indices_meta, :]
                sample_x,sample_y,sample_x_meta,sample_y_meta = torch.FloatTensor(sample_x).cuda(),torch.FloatTensor(sample_y
                ).cuda(),torch.FloatTensor(sample_x_meta).cuda(),torch.FloatTensor(sample_y_meta).cuda()
                out = self.base_net_plain.forward_ori(sample_x)
                loss = self.base_net_plain.loss_fn(out,sample_y)
                self.opt_plain.zero_grad()
                loss.backward()
                self.opt.step()
                t_loss = t_loss + loss
            # Perform the meta update
            if (it+1) %100==0:
                tt_loss = self.test_plain().data.cpu().numpy()
                print('Update', it,'train_loss:',t_loss.data.cpu().numpy(),'test loss:',tt_loss)
                test_loss.append(tt_loss)
            total_loss.append(t_loss.data.cpu().numpy())

        plt.figure()
        plt.plot(np.array(total_loss))
        plt.title('train loss(naive learning)')
        plt.figure()
        plt.plot(np.array(test_loss))
        plt.title('test loss(naive learning)')
        #plt.show()

    def train(self):
        loss_rem_1 = []
        loss_rem_2 = []
        grads = []
        for j in range(self.num_clusters):
            grads.append([])
        for it in range(self.num_iterations):
            for j in range(self.num_clusters):
                grads[j] = []
            total_loss = 0
            index = np.zeros((self.num_tasks*2,),dtype=np.int32)
            for i in range(self.num_tasks*2):
                indices = np.random.choice(self.num_samples, size=self.batch_size)
                indices_meta = np.random.choice(self.num_samples, size=self.meta_batch_size)
                sample_x = self.x[i,indices,:]
                sample_y = self.sample[i,indices,:]
                sample_x_meta = self.x[i, indices_meta, :]
                sample_y_meta = self.sample[i, indices_meta, :]
                sample_x,sample_y,sample_x_meta,sample_y_meta = torch.FloatTensor(sample_x).cuda(),torch.FloatTensor(sample_y
                ).cuda(),torch.FloatTensor(sample_x_meta).cuda(),torch.FloatTensor(sample_y_meta).cuda()
                loss_rem = np.zeros((self.num_clusters,))
                grad_rem = []
                for j in range(self.num_clusters):
                    self.fast_net.copy_weights(self.base_net[j])
                    g,loss = self.fast_net.forward(sample_x,sample_y,sample_x_meta,sample_y_meta,self.num_updates)
                    loss_rem[j] = loss.cpu().data.numpy()
                    grad_rem.append(g)
                possibilities = np.exp(-1*loss_rem)/np.sum(np.exp(-1*loss_rem))
                index[i] = np.random.choice(np.arange(self.num_clusters),1,p=possibilities)
                #print(loss_rem, possibilities,index[i])
                total_loss = total_loss + loss_rem[index[i]]
                grads[index[i]].append(grad_rem[index[i]])
            # Perform the meta update
            loss_rem_1.append(total_loss)
            if (it+1) %100==0:
                loss_test = self.test()
                print('Meta update', it,'train_loss:',total_loss,'test loss:',loss_test)
                print(index)
                loss_rem_2.append(loss_test)
            for j in range(self.num_clusters):
                self.meta_update(sample_x,sample_y, grads[j],j)

        plt.figure()
        plt.plot(np.array(loss_rem_1))
        plt.title('train loss')
        #plt.show()
        plt.figure()
        plt.plot(np.array(loss_rem_2))
        plt.title('test loss')
        #plt.show()

    def meta_update(self, sample_x,sample_y, ls,j):
        out = self.base_net[j].forward_ori(sample_x,None)
        loss = self.base_net[j].loss_fn(out,sample_y)
        # Unpack the list of grad dicts
        if ls == []:
            return
        gradients = {k: sum(d[k] for d in ls) for k in ls[0].keys()}
        # Register a hook on each parameter in the net that replaces the current dummy grad
        # with our grads accumulated across the meta-batch
        hooks = []
        for (k, v) in self.base_net[j].named_parameters():
            def get_closure():
                key = k
                def replace_grad(grad):
                    return gradients[key]
                return replace_grad
            hooks.append(v.register_hook(get_closure()))
        # Compute grads for current step, replace with summed gradients as defined by hook
        self.opt[j].zero_grad()
        loss.backward()
        # Update the net parameters with the accumulated gradient according to optimizer
        self.opt[j].step()
        # Remove the hooks before next training phase
        for h in hooks:
            h.remove()

    def plt_test(self):
        plt.figure()
        for i in range(self.num_tasks*2):
            plt.subplot(self.num_tasks*2, 1, i+1)
            plt.plot(self.generator.sample_points,self.sample[i,:,0],'b',label='data')
            indices = np.random.choice(self.num_samples, size=self.batch_size)
            sample_x = self.x[i, indices, :]
            sample_y = self.sample[i, indices, :]
            plt.scatter(sample_x,sample_y,)
            sample_x, sample_y = torch.FloatTensor(sample_x).cuda(), torch.FloatTensor(
                sample_y).cuda()
            loss_rem = np.zeros((self.num_clusters,))
            for i in range(self.num_clusters):
                self.fast_net.copy_weights(self.base_net[i])
                loss, _ = self.fast_net.forward_pass(sample_x, sample_y)
                loss_rem[i] = loss.cpu().data.numpy()
            index = np.argmin(loss_rem)
            print(index)
            self.fast_net.copy_weights(self.base_net[index])
            test_points = torch.FloatTensor(self.generator.sample_points[:, np.newaxis]).cuda()
            output_ori = self.fast_net.forward_ori(test_points).cpu().data.numpy()
            plt.plot(self.generator.sample_points, output_ori, 'r', label='0 step')
            test_opt = torch.optim.SGD(self.fast_net.parameters(), lr=self.step_size)
            for i in range(10):
                loss,_ = self.fast_net.forward_pass(sample_x,sample_y)
                test_opt.zero_grad()
                loss.backward()
                test_opt.step()
                if i ==0:
                    output_1 = self.fast_net.forward_ori(test_points).cpu().data.numpy()
                    plt.plot(self.generator.sample_points, output_1, 'y', label='1 step')
                elif i ==9:
                    output_2 = self.fast_net.forward_ori(test_points).cpu().data.numpy()
                    plt.plot(self.generator.sample_points, output_2, 'g', label='10 steps')
            plt.legend()
            plt.title('train')

        plt.figure()
        for i in range(self.num_tasks * 2):
            plt.subplot(self.num_tasks*2, 1, i + 1)
            plt.plot(self.generator.sample_points, self.test_data[i, :, 0], 'b', label='data')
            indices = np.random.choice(self.num_samples, size=self.batch_size)
            sample_x = self.x[i, indices, :]
            sample_y = self.test_data[i, indices, :]
            plt.scatter(sample_x, sample_y, )
            sample_x, sample_y = torch.FloatTensor(sample_x).cuda(), torch.FloatTensor(
                sample_y).cuda()
            loss_rem = np.zeros((self.num_clusters,))
            for i in range(self.num_clusters):
                self.fast_net.copy_weights(self.base_net[i])
                loss, _ = self.fast_net.forward_pass(sample_x, sample_y)
                loss_rem[i] = loss.cpu().data.numpy()
            index = np.argmin(loss_rem)
            print(index)
            self.fast_net.copy_weights(self.base_net[index])
            test_points = torch.FloatTensor(self.generator.sample_points[:, np.newaxis]).cuda()
            output_ori = self.fast_net.forward_ori(test_points).cpu().data.numpy()
            plt.plot(self.generator.sample_points, output_ori, 'r', label='0 step')
            test_opt = torch.optim.SGD(self.fast_net.parameters(), lr=self.step_size)
            for i in range(10):
                loss, _ = self.fast_net.forward_pass(sample_x, sample_y)
                test_opt.zero_grad()
                loss.backward()
                test_opt.step()
                if i == 0:
                    output_1 = self.fast_net.forward_ori(test_points).cpu().data.numpy()
                    plt.plot(self.generator.sample_points, output_1, 'y', label='1 step')
                elif i == 9:
                    output_2 = self.fast_net.forward_ori(test_points).cpu().data.numpy()
                    plt.plot(self.generator.sample_points, output_2, 'g', label='10 steps')
            plt.legend()
            plt.title('test')


        plt.figure()
        test_points = torch.FloatTensor(self.generator.sample_points[:, np.newaxis]).cuda()
        for j in range(self.num_clusters):
            self.fast_net.copy_weights(self.base_net[j])
            output_ori = self.fast_net.forward_ori(test_points).cpu().data.numpy()
            plt.plot(self.generator.sample_points, output_ori, label='%i'%j)
        plt.legend()
        #plt.show()
        return

    def plt_test_plain(self):
        plt.figure()
        test_points = torch.FloatTensor(self.generator.sample_points[:,np.newaxis]).cuda()
        self.fast_net.copy_weights(self.base_net_plain)
        output_ori = self.fast_net.forward_ori(test_points).cpu().data.numpy()
        for i in range(self.num_tasks):
            plt.subplot(self.num_tasks, 1, i+1)
            plt.plot(self.generator.sample_points,self.sample[i,:,0],'b',label='data')
            plt.plot(self.generator.sample_points, output_ori, 'r',label='0 step')
            indices = np.random.choice(self.num_samples, size=self.batch_size)
            sample_x = self.x[i, indices, :]
            sample_y = self.sample[i, indices, :]
            plt.scatter(sample_x,sample_y,)
            sample_x, sample_y = torch.FloatTensor(sample_x).cuda(), torch.FloatTensor(
                sample_y).cuda()
            self.fast_net.copy_weights(self.base_net_plain)
            test_opt = torch.optim.Adam(self.fast_net.parameters(), lr=self.meta_step_size)
            for i in range(10):
                loss,_ = self.fast_net.forward_pass(sample_x,sample_y)
                test_opt.zero_grad()
                loss.backward()
                test_opt.step()
                if i ==0:
                    output_1 = self.fast_net.forward_ori(test_points).cpu().data.numpy()
                    plt.plot(self.generator.sample_points, output_1, 'y', label='1 step')
                elif i ==9:
                    output_2 = self.fast_net.forward_ori(test_points).cpu().data.numpy()
                    plt.plot(self.generator.sample_points, output_2, 'g', label='10 steps')
            plt.legend()
            plt.title('train(plain)')

        plt.figure()
        test_points = torch.FloatTensor(self.generator.sample_points[:, np.newaxis]).cuda()
        self.fast_net.copy_weights(self.base_net_plain)
        output_ori = self.fast_net.forward_ori(test_points).cpu().data.numpy()
        for i in range(self.num_tasks):
            plt.subplot(self.num_tasks, 1, i + 1)
            plt.plot(self.generator.sample_points, self.test_data[i, :, 0], 'b', label='data')
            plt.plot(self.generator.sample_points, output_ori, 'r', label='0 step')
            indices = np.random.choice(self.num_samples, size=self.batch_size)
            sample_x = self.x[i, indices, :]
            sample_y = self.test_data[i, indices, :]
            plt.scatter(sample_x, sample_y, )
            sample_x, sample_y = torch.FloatTensor(sample_x).cuda(), torch.FloatTensor(
                sample_y).cuda()
            self.fast_net.copy_weights(self.base_net_plain)
            test_opt = torch.optim.SGD(self.fast_net.parameters(), lr=self.meta_step_size)
            for i in range(10):
                loss, _ = self.fast_net.forward_pass(sample_x, sample_y)
                test_opt.zero_grad()
                loss.backward()
                test_opt.step()
                if i == 0:
                    output_1 = self.fast_net.forward_ori(test_points).cpu().data.numpy()
                    plt.plot(self.generator.sample_points, output_1, 'y', label='1 step')
                elif i == 9:
                    output_2 = self.fast_net.forward_ori(test_points).cpu().data.numpy()
                    plt.plot(self.generator.sample_points, output_2, 'g', label='10 steps')
            plt.legend()
            plt.title('test(plain)')

        #plt.show()
        return

    def test(self):
        total_loss = 0
        for i in range(self.num_tasks):
            indices = np.random.choice(self.num_samples, size=self.batch_size)
            indices_meta = np.random.choice(self.num_samples, size=self.meta_batch_size)
            sample_x = self.x[i, indices, :]
            sample_y = self.test_data[i, indices, :]
            sample_x_meta = self.x[i, indices_meta, :]
            sample_y_meta = self.test_data[i, indices_meta, :]
            sample_x, sample_y, sample_x_meta, sample_y_meta = torch.FloatTensor(sample_x).cuda(), torch.FloatTensor(
                sample_y
                ).cuda(), torch.FloatTensor(sample_x_meta).cuda(), torch.FloatTensor(sample_y_meta).cuda()
            loss_tem = np.zeros((self.num_clusters,))
            for j in range(self.num_clusters):
                self.fast_net.copy_weights(self.base_net[j])
                g, loss = self.fast_net.forward(sample_x, sample_y, sample_x_meta, sample_y_meta,self.num_updates)
                loss_tem[j] = loss.cpu().data.numpy()
            total_loss = total_loss + np.min(loss_tem)

        return total_loss

    def test_plain(self):
        total_loss = 0
        for i in range(self.num_tasks):
            indices = np.random.choice(self.num_samples, size=self.batch_size)
            sample_x = self.x[i, indices, :]
            sample_y = self.test_data[i, indices, :]
            sample_x, sample_y = torch.FloatTensor(sample_x).cuda(), torch.FloatTensor(
                sample_y).cuda()
            out = self.base_net_plain.forward_ori(sample_x)
            loss = self.base_net_plain.loss_fn(out, sample_y)
            total_loss = total_loss + loss
        return total_loss

if __name__=='__main__':
    g = sin_generator(100)
    data,x = g.generate(1,2)
    #print(x)
    t = trainer(num_samples=100,num_tasks=2,num_updates=5, step_size=1e-2,meta_step_size=1e-2,
                batch_size=16, meta_batch_size=16,num_iterations=1000,num_clusters=2)
    #plt.figure()
    #for i in range(4):
    #    plt.plot(t.x[i,:,0],t.sample[i,:,0])
    t.train()
    t.plt_test()
    #t.train_plain()
    #t.plt_test_plain()
    plt.show()