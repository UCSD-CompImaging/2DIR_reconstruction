import numpy as np
import matplotlib.pyplot as plt
import cv2
from run_params import run_param, Parameter
#from model import dynamic_model
from model_torch import dynamic_model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--msg', default='noname')
parser.add_argument('--seed', default=5, type=int)
parser.add_argument('--idx1', default=10, type=int)
parser.add_argument('--idx2', default=50, type=int)
parser.add_argument('--use_noise', default=1, type=int)

if __name__ == '__main__':
    args = parser.parse_args()
    sample_num = 1

    for i in range(sample_num):

        print (i)

        # input section
        manual = 1 # choose whether to plot every single spectrum or not
        param = Parameter(args.seed) # 
        param = run_param(param)

        # vibrational lifetime will be used, and the relative peak amplitudes and frequencies will be assumed based on FTIR results.
        freqax = np.arange(1500, 1801)
        list_delays = np.arange(0, 2200, 200)
        systems_to_generate = 1
        t2_to_fit = list_delays

        # preallocate with zeros
        peak_volume_model = np.zeros(len(list_delays))
        inter_spec = np.zeros((freqax.shape[0], freqax.shape[0], list_delays.shape[0]))
        t1_len = param.aux.taxis.shape[0]
        meas_cube = np.zeros((t1_len, t1_len, 11)).astype(np.csingle) # should be flexible, temporarary version
        randomized_params = np.zeros((list_delays.shape[0], 18))
        print ('dbg: ', args.seed, param.packed_random)

        for n in range(8, 9):
            # run parameter file to generate random inputs
            randomized_params[n, :] = param.packed_random
            if list_delays[n] >= 0:
                if param.crosspeaks == 0:
                    pass # unbuilt
                elif param.crosspeaks == 1:
                    pass
                elif param.crosspeaks == 2:
                    int_spec, meas, spec = dynamic_model(param.packed_random, param.aux, t2_to_fit[n], freqax, args.msg, args.seed)
                    inter_spec[:, :, n] = int_spec
                    meas_cube[:, :, n] = meas
            
            peak_volume_model[n] =  np.sum(np.abs(inter_spec[:, :, n]))
            spec_min = np.amin(inter_spec[:, :, n])
            spec_max = np.amax(inter_spec[:, :, n])

            if manual == 1:
                plt.figure()
                #plt.imshow(inter_spec[:, :, n])
                plt.contourf(freqax, freqax, inter_spec[:, :, n], 30, cmap=plt.cm.jet)
                plt.xlabel('$\omega_1$ (cm-1)')
                plt.ylabel('$\omega_3$ (cm-1)')
                plt.xlim([1600, 1700])
                plt.ylim([1600, 1700])
                plt.colorbar()
                #plt.show()
                plt.title('t2={} fs'.format(t2_to_fit[n]))
                #plt.savefig('./fig/spec_{}_rand.png'.format(n))
                plt.savefig('./fig/RA_r_gt_freq_{}.png'.format(n))
                plt.close()

            #end

        #np.save('train_data/sample_{}.npy'.format(i), inter_spec)
        #np.save('train_data/spec_{}.npy'.format(i), spec)
        #np.save('train_data/meas_{}.npy'.format(i), meas_cube)
        #np.save('train_data/param_{}.npy'.format(i), param.packed_random)

