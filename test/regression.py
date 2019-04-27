import numpy as np
import torch
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from base_net import Base_net
from fast_net import Fast_net
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
    def __init__(self,num_samples,num_tasks,num_updates, step_size,meta_step_size, batch_size, meta_batch_size,num_iterations):
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.meta_batch_size = meta_batch_size
        self.generator = sin_generator(num_samples)
        self.num_tasks = num_tasks
        self.num_samples = num_samples
        self.sample = np.zeros((self.num_tasks,num_samples,1))
        self.test_data = np.zeros((self.num_tasks, num_samples, 1))
        self.x = np.zeros((self.num_tasks,num_samples,1))
        self.num_updates = num_updates
        self.step_size = step_size
        self.meta_step_size = meta_step_size
        amp = 0.5+np.random.rand(self.num_tasks*2) * 2.5
        phase = np.random.rand(self.num_tasks*2)*np.pi
        for i in range(self.num_tasks):
            self.sample[i,:,0],self.x[i,:,0] = self.generator.generate(amp[i],phase[i])
        for i in range(self.num_tasks):
            self.test_data[i,:,0],_ = self.generator.generate(amp[self.num_tasks+i],phase[self.num_tasks+i])
        self.base_net = Base_net(nn.MSELoss())
        self.base_net_plain = Base_net(nn.MSELoss())
        self.fast_net = Fast_net(nn.MSELoss(),num_updates, step_size, batch_size, meta_batch_size)
        self.base_net.cuda()
        self.fast_net.cuda()
        self.base_net_plain.cuda()
        self.opt = torch.optim.Adam(self.base_net.parameters(), lr=meta_step_size)
        self.opt_plain = torch.optim.Adam(self.base_net.parameters(), lr=meta_step_size)
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
        loss_rem = []
        loss_rem_2 = []
        for it in range(self.num_iterations):
            grads = []
            total_loss = 0
            for i in range(self.num_tasks):
                indices = np.random.choice(self.num_samples, size=self.batch_size)
                indices_meta = np.random.choice(self.num_samples, size=self.meta_batch_size)
                sample_x = self.x[i,indices,:]
                sample_y = self.sample[i,indices,:]
                sample_x_meta = self.x[i, indices_meta, :]
                sample_y_meta = self.sample[i, indices_meta, :]
                sample_x,sample_y,sample_x_meta,sample_y_meta = torch.FloatTensor(sample_x).cuda(),torch.FloatTensor(sample_y
                ).cuda(),torch.FloatTensor(sample_x_meta).cuda(),torch.FloatTensor(sample_y_meta).cuda()
                self.fast_net.copy_weights(self.base_net)
                g,loss = self.fast_net.forward(sample_x,sample_y,sample_x_meta,sample_y_meta,self.num_updates)
                total_loss = total_loss + loss
                grads.append(g)
            # Perform the meta update
            loss_rem.append(total_loss.data.cpu().numpy())
            if (it+1) %100==0:
                print('Meta update', it,'train_loss:',total_loss.data.cpu().numpy(),'test loss:',self.test().data.cpu().numpy())
                loss_rem_2.append(self.test().data.cpu().numpy())
            self.meta_update(sample_x,sample_y, grads)

        plt.figure()
        plt.plot(np.array(loss_rem))
        plt.title('train loss')
        #plt.show()
        plt.figure()
        plt.plot(np.array(loss_rem_2))
        plt.title('test loss')
        #plt.show()

    def meta_update(self, sample_x,sample_y, ls):
        out = self.base_net.forward_ori(sample_x,None)
        loss = self.base_net.loss_fn(out,sample_y)
        # Unpack the list of grad dicts
        gradients = {k: sum(d[k] for d in ls) for k in ls[0].keys()}
        # Register a hook on each parameter in the net that replaces the current dummy grad
        # with our grads accumulated across the meta-batch
        hooks = []
        for (k, v) in self.base_net.named_parameters():
            def get_closure():
                key = k
                def replace_grad(grad):
                    return gradients[key]
                return replace_grad
            hooks.append(v.register_hook(get_closure()))
        # Compute grads for current step, replace with summed gradients as defined by hook
        self.opt.zero_grad()
        loss.backward()
        # Update the net parameters with the accumulated gradient according to optimizer
        self.opt.step()
        # Remove the hooks before next training phase
        for h in hooks:
            h.remove()

    def plt_test(self):
        plt.figure()
        test_points = torch.FloatTensor(self.generator.sample_points[:,np.newaxis]).cuda()
        self.fast_net.copy_weights(self.base_net)
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
            self.fast_net.copy_weights(self.base_net)
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
        test_points = torch.FloatTensor(self.generator.sample_points[:, np.newaxis]).cuda()
        self.fast_net.copy_weights(self.base_net)
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
            self.fast_net.copy_weights(self.base_net)
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
            self.fast_net.copy_weights(self.base_net)
            g, loss = self.fast_net.forward(sample_x, sample_y, sample_x_meta, sample_y_meta,self.num_updates)
            total_loss = total_loss + loss

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
                batch_size=16, meta_batch_size=16,num_iterations=5000)
    t.train()
    t.plt_test()
    t.train_plain()
    t.plt_test_plain()
    plt.show()