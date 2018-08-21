# i/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import random
import time
from scipy.stats import multivariate_normal
from scipy.stats import norm
import math
import os
from expdata import setexperimentdata
import sys

# An example of a class
class Network:
    def __init__(self, Topo, Train, Test, learn_rate):
        self.Top = Topo  # NN topology [input, hidden, output]
        self.TrainData = Train
        self.TestData = Test
        np.random.seed()
        self.lrate = learn_rate

        self.W1 = np.random.randn(self.Top[0], self.Top[1]) / np.sqrt(self.Top[0])
        self.B1 = np.random.randn(1, self.Top[1]) / np.sqrt(self.Top[1])  # bias first layer
        self.W2 = np.random.randn(self.Top[1], self.Top[2]) / np.sqrt(self.Top[1])
        self.B2 = np.random.randn(1, self.Top[2]) / np.sqrt(self.Top[1])  # bias second layer

        self.hidout = np.zeros((1, self.Top[1]))  # output of first hidden layer
        self.out = np.zeros((1, self.Top[2]))  # output last layer

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sampleEr(self, actualout):
        error = np.subtract(self.out, actualout)
        sqerror = np.sum(np.square(error)) / self.Top[2]
        return sqerror

    def ForwardPass(self, X):
        z1 = X.dot(self.W1) - self.B1
        self.hidout = self.sigmoid(z1)  # output of first hidden layer
        z2 = self.hidout.dot(self.W2) - self.B2
        self.out = self.sigmoid(z2)  # output second hidden layer

    def BackwardPass(self, Input, desired):
        out_delta = (desired - self.out) * (self.out * (1 - self.out))
        hid_delta = out_delta.dot(self.W2.T) * (self.hidout * (1 - self.hidout))

        #self.W2 += (self.hidout.T.dot(out_delta) * self.lrate)
        #self.B2 += (-1 * self.lrate * out_delta)
        #self.W1 += (Input.T.dot(hid_delta) * self.lrate)
        #self.B1 += (-1 * self.lrate * hid_delta)

        layer = 1  # hidden to output
        for x in xrange(0, self.Top[layer]):
            for y in xrange(0, self.Top[layer + 1]):
                self.W2[x, y] += self.lrate * out_delta[y] * self.hidout[x]
        for y in xrange(0, self.Top[layer + 1]):
            self.B2[y] += -1 * self.lrate * out_delta[y]

        layer = 0  # Input to Hidden
        for x in xrange(0, self.Top[layer]):
            for y in xrange(0, self.Top[layer + 1]):
                self.W1[x, y] += self.lrate * hid_delta[y] * Input[x]
        for y in xrange(0, self.Top[layer + 1]):
            self.B1[y] += -1 * self.lrate * hid_delta[y]

    def decode(self, w):
        w_layer1size = self.Top[0] * self.Top[1]
        w_layer2size = self.Top[1] * self.Top[2]

        w_layer1 = w[0:w_layer1size]
        self.W1 = np.reshape(w_layer1, (self.Top[0], self.Top[1]))

        w_layer2 = w[w_layer1size:w_layer1size + w_layer2size]
        self.W2 = np.reshape(w_layer2, (self.Top[1], self.Top[2]))
        self.B1 = w[w_layer1size + w_layer2size:w_layer1size + w_layer2size + self.Top[1]]
        self.B2 = w[w_layer1size + w_layer2size + self.Top[1]:w_layer1size + w_layer2size + self.Top[1] + self.Top[2]]


    def encode(self):
        w1 = self.W1.ravel()
        w2 = self.W2.ravel()
        w = np.concatenate([w1, w2, self.B1, self.B2])
        return w

    def langevin_gradient(self, data, w, depth):  # BP with SGD (Stocastic BP)

        self.decode(w)  # method to decode w into W1, W2, B1, B2.
        size = data.shape[0]

        Input = np.zeros((1, self.Top[0]))  # temp hold input
        Desired = np.zeros((1, self.Top[2]))
        fx = np.zeros(size)

        for i in xrange(0, depth):
            for i in xrange(0, size):
                pat = i
                Input = data[pat, 0:self.Top[0]]
                Desired = data[pat, self.Top[0]:]
                self.ForwardPass(Input)
                self.BackwardPass(Input, Desired)

        w_updated = self.encode()

        return  w_updated

    def evaluate_proposal(self, data, w ):  # BP with SGD (Stocastic BP)

        self.decode(w)  # method to decode w into W1, W2, B1, B2.
        size = data.shape[0]

        Input = np.zeros((1, self.Top[0]))  # temp hold input
        Desired = np.zeros((1, self.Top[2]))
        fx = np.zeros((size,self.Top[2]))

        for i in xrange(0, size):  # to see what fx is produced by your current weight update
            Input = data[i, 0:self.Top[0]]
            self.ForwardPass(Input)
            fx[i] = self.out

        return fx



# --------------------------------------------------------------------------

def covert_time(secs):
    if secs >= 60:
        mins = str(secs/60)
        secs = str(secs%60)
    else:
        secs = str(secs)
        mins = str(00)

    if len(mins) == 1:
        mins = '0'+mins

    if len(secs) == 1:
        secs = '0'+secs

    return [mins, secs]


# -------------------------------------------------------------------


class MCMC:
    def __init__(self, samples, traindata, testdata, topology):
        self.samples = samples  # NN topology [input, hidden, output]
        self.topology = topology  # max epocs
        self.traindata = traindata  #
        self.testdata = testdata
        # ----------------

    def rmse(self, predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

    def likelihood_func(self, neuralnet, data, w, tausq):
        y = data[:, self.topology[0]:]
        fx = neuralnet.evaluate_proposal(data, w)
        rmse = self.rmse(fx, y)
        loss = -0.5 * np.log(2 * math.pi * tausq) - 0.5 * np.square(y - fx) / tausq
        return [np.sum(loss), fx, rmse]

    def prior_likelihood(self, sigma_squared, nu_1, nu_2, w, tausq):
        h = self.topology[1]  # number hidden neurons
        d = self.topology[0]  # number input neurons
        part1 = -1 * ((d * h + h + 2) / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(w)))
        log_loss = part1 - part2  - (1 + nu_1) * np.log(tausq) - (nu_2 / tausq)
        return log_loss

    def sampler(self, w_limit, tau_limit, file):

        start = time.time()

        # ------------------- initialize MCMC
        testsize = self.testdata.shape[0]
        trainsize = self.traindata.shape[0]
        samples = self.samples


        self.sgd_depth = 1

        x_test = np.linspace(0, 1, num=testsize)
        x_train = np.linspace(0, 1, num=trainsize)

        netw = self.topology  # [input, hidden, output]
        y_test = self.testdata[:, netw[0]:]
        y_train = self.traindata[:, netw[0]:]
        # print y_train.shape
        # print y_test.shape

        w_size = (netw[0] * netw[1]) + (netw[1] * netw[2]) + netw[1] + netw[2]  # num of weights and bias

        pos_w = np.ones((samples, w_size))  # posterior of all weights and bias over all samples
        pos_tau = np.ones((samples, 1))

        fxtrain_samples = np.ones((samples, trainsize, netw[2]))  # fx of train data over all samples
        fxtest_samples = np.ones((samples, testsize, netw[2]))  # fx of test data over all samples
        rmse_train = np.zeros(samples)
        rmse_test = np.zeros(samples)

        w = np.random.randn(w_size)
        w_proposal = np.random.randn(w_size)

        #step_w = 0.05;  # defines how much variation you need in changes to w
        #step_eta = 0.2; # exp 0


        step_w = w_limit  # defines how much variation you need in changes to w
        step_eta = tau_limit #exp 1
        # --------------------- Declare FNN and initialize
        learn_rate = 0.5

        neuralnet = Network(self.topology, self.traindata, self.testdata, learn_rate)
        # print 'evaluate Initial w'

        pred_train = neuralnet.evaluate_proposal(self.traindata, w)
        pred_test = neuralnet.evaluate_proposal(self.testdata, w)



        eta = np.log(np.var(pred_train - y_train))
        tau_pro = np.exp(eta)

        sigma_squared = 25
        nu_1 = 0
        nu_2 = 0


        sigma_diagmat = np.zeros((w_size, w_size))  # for Equation 9 in Ref [Chandra_ICONIP2017]
        np.fill_diagonal(sigma_diagmat, step_w)

        delta_likelihood = 0.5 # an arbitrary position


        prior_current = self.prior_likelihood(sigma_squared, nu_1, nu_2, w, tau_pro)  # takes care of the gradients

        [likelihood, pred_train, rmsetrain] = self.likelihood_func(neuralnet, self.traindata, w, tau_pro)
        [likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(neuralnet, self.testdata, w, tau_pro)

        # print likelihood

        naccept = 0
        # print 'begin sampling using mcmc random walk'
        # plt.plot(x_train, y_train)
        # plt.plot(x_train, pred_train)
        # plt.title("Plot of Data vs Initial Fx")
        # plt.savefig('mcmcresults/begin.png')
        # plt.clf()

        #plt.plot(x_train, y_train)


        for i in range(samples - 1):
            w_gd = neuralnet.langevin_gradient(self.traindata, w.copy(), self.sgd_depth) # Eq 8
            # print(sum(w_gd))
            w_proposal = w_gd  + np.random.normal(0, step_w, w_size) # Eq 7

            w_prop_gd = neuralnet.langevin_gradient(self.traindata, w_proposal.copy(), self.sgd_depth)

            # print(multivariate_normal.pdf(w, w_prop_gd, sigma_diagmat),multivariate_normal.pdf(w_proposal, w_gd, sigma_diagmat))

            diff_prop =  np.log(multivariate_normal.pdf(w, w_prop_gd, sigma_diagmat))  - np.log(multivariate_normal.pdf(w_proposal, w_gd, sigma_diagmat))

            eta_pro = eta + np.random.normal(0, step_eta, 1)
            tau_pro = math.exp(eta_pro)

            [likelihood_proposal, pred_train, rmsetrain] = self.likelihood_func(neuralnet, self.traindata, w_proposal,
                                                                                tau_pro)
            [likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(neuralnet, self.testdata, w_proposal,
                                                                            tau_pro)

            # likelihood_ignore  refers to parameter that will not be used in the alg.

            prior_prop = self.prior_likelihood(sigma_squared, nu_1, nu_2, w_proposal,
                                               tau_pro)  # takes care of the gradients


            diff_prior = prior_prop - prior_current
            diff_likelihood = likelihood_proposal - likelihood
            diff = min(700, diff_prior + diff_likelihood + diff_prop)

            # print()
            # print(diff, i )
            mh_prob = min(1, math.exp(diff))
            # print(mh_prob)




            u = random.uniform(0, 1)

            if u < mh_prob:
                # Update position
                # print    i, ' is accepted sample'
                naccept += 1
                likelihood = likelihood_proposal
                prior_current = prior_prop
                w = w_proposal
                eta = eta_pro


                # print  likelihood, prior_current, diff_prop, rmsetrain, rmsetest, w, 'accepted'
                #print w_proposal, 'w_proposal'
                #print w_gd, 'w_gd'

                #print w_prop_gd, 'w_prop_gd'

                pos_w[i + 1,] = w_proposal
                pos_tau[i + 1,] = tau_pro
                fxtrain_samples[i + 1,] = pred_train
                fxtest_samples[i + 1,] = pred_test
                rmse_train[i + 1,] = rmsetrain
                rmse_test[i + 1,] = rmsetest

                #plt.plot(x_train, pred_train)


            else:
                pos_w[i + 1,] = pos_w[i,]
                pos_tau[i + 1,] = pos_tau[i,]
                fxtrain_samples[i + 1,] = fxtrain_samples[i,]
                fxtest_samples[i + 1,] = fxtest_samples[i,]
                rmse_train[i + 1,] = rmse_train[i,]
                rmse_test[i + 1,] = rmse_test[i,]

            elapsed_time = ":".join(covert_time(int(time.time()-start)))
            sys.stdout.write('\r' + file + ' : ' + str("{:.2f}".format(float(i) / (samples - 1) * 100)) + '% complete....'+" time elapsed: " + elapsed_time)

                # print i, 'rejected and retained'
        sys.stdout.write('\r' + file + ' : 100% ..... Total Time: ' + ":".join(covert_time(int(time.time()-start))))
        print naccept, ' num accepted'
        print naccept / (samples * 1.0), '% was accepted'
        accept_ratio = naccept / (samples * 1.0) * 100

        # plt.title("Plot of Accepted Proposals")
        # plt.savefig('mcmcresults/proposals.png')
        # plt.savefig('mcmcresults/proposals.svg', format='svg', dpi=600)
        # plt.clf()

        return (pos_w, pos_tau, fxtrain_samples, fxtest_samples, x_train, x_test, rmse_train, rmse_test, accept_ratio)


def main():
    filenames = ["Iris", "Wine", "Cancer", "Heart", "CreditApproval", "Baloon", "TicTac", "Ions", "Zoo",
                 "Lenses", "Balance"]
    problemlist = np.array(range(11))
    input = np.array([4, 13, 9, 13, 15, 4, 9, 34, 16, 4, 4])
    hidden = np.array([6, 6, 6, 16, 20, 5, 30, 8, 6, 5, 8])
    output = np.array([2, 3, 1, 1, 1, 1, 1, 1, 7, 3, 3])

    samplelist = [5000, 8000, 10000, 20000, 15000, 5000, 20000, 5000, 3000, 5000, 2000]
    x = 3

    filetrain = open('Results/train.txt', 'r')
    filetest = open('Results/test.txt', 'r')
    filestdtr = open('Results/std_tr.txt','r')
    filestdts = open('Results/std_ts.txt', 'r')

    train_accs = np.loadtxt(filetrain)
    test_accs = np.loadtxt(filetest)

    train_stds = np.loadtxt(filestdtr)
    test_stds = np.loadtxt(filestdts)

    filetrain.close()
    filetest.close()
    filestdtr.close()
    filestdts.close()



    if x == 3:
        w_limit =  0.02
        tau_limit = 0.2
    #if x == 4:
        #w_limit =  0.02
        #tau_limit = 0.1

    for problem in problemlist:

        #if os.path.isfile("Results/"+filenames[problem]+"_rmse.txt"):
        #    print filenames[problem]
        #    continur

        [traindata, testdata, baseNet] = setexperimentdata(problem)

        topology = [input[problem], hidden[problem], output[problem]]

        random.seed(time.time())

        numSamples = samplelist[problem]   # need to decide yourself

        mcmc = MCMC(numSamples, traindata, testdata, topology)  # declare class

        [pos_w, pos_tau, fx_train, fx_test, x_train, x_test, rmse_train, rmse_test, accept_ratio] = mcmc.sampler(w_limit, tau_limit, filenames[problem])
        print '\nsucessfully sampled: '+ str(accept_ratio)+ ' samples accepted'

        burnin = 0.1 * numSamples  # use post burn in samples

        pos_w = pos_w[int(burnin):, ]
        pos_tau = pos_tau[int(burnin):, ]

        print("fx shape:"+str(fx_test.shape))
        print("fx_train shape:"+ str(fx_train.shape))

        fx_mu = fx_test.mean(axis=0)
        fx_high = np.percentile(fx_test, 95, axis=0)
        fx_low = np.percentile(fx_test, 5, axis=0)

        fx_mu_tr = fx_train.mean(axis=0)
        fx_high_tr = np.percentile(fx_train, 95, axis=0)
        fx_low_tr = np.percentile(fx_train, 5, axis=0)

        pos_w_mean = pos_w.mean(axis=0)
        # np.savetxt(outpos_w, pos_w_mean, fmt='%1.5f')


        rmse_tr = np.mean(rmse_train[int(burnin):])
        rmsetr_std = np.std(rmse_train[int(burnin):])
        rmse_tes = np.mean(rmse_test[int(burnin):])
        rmsetest_std = np.std(rmse_test[int(burnin):])
        # print rmse_tr, rmsetr_std, rmse_tes, rmsetest_std
        # np.savetxt(outres, (rmse_tr, rmsetr_std, rmse_tes, rmsetest_std, accept_ratio), fmt='%1.5f')

        ytestdata = testdata[:, input[problem]:]
        ytraindata = traindata[:, input[problem]:]

        train_acc = []
        test_acc = []

        for fx in fx_train:
            count = 0
            for index in range(fx.shape[0]):
                if np.allclose(fx[index],ytraindata[index],atol = 0.2):
                    count += 1
            train_acc.append(float(count)/fx.shape[0]*100)

        for fx in fx_test:
            count = 0
            for index in range(fx.shape[0]):
                if np.allclose(fx[index],ytestdata[index],atol = 0.5):
                    count += 1
            test_acc.append(float(count)/fx.shape[0]*100)

        train_acc = np.array(train_acc[int(burnin):])
        train_std = np.std(train_acc[int(burnin):])

        test_acc = np.array(test_acc[int(burnin):])
        test_std = np.std(test_acc[int(burnin):])

        train_acc_mu = train_acc.mean()
        test_acc_mu = test_acc.mean()

        train_accs[problem] = train_acc_mu
        test_accs[problem] = test_acc_mu
        train_stds[problem] = train_std
        test_stds[problem] = test_std

        testResults = np.c_[ytestdata, fx_mu, fx_high, fx_low]

        trainResults = np.c_[ytraindata, fx_mu_tr, fx_high_tr, fx_low_tr]

        # Write RMSE to
        with open("Results/" +
                  filenames[problem] + "_rmse" + ".txt", 'w') as fil:
            rmse = [rmse_tr, rmsetr_std, rmse_tes, rmsetest_std]
            rmse = "\t".join(list(map(str, rmse))) + "\n"
            fil.write(rmse)

    n_groups = len(filenames)
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.2
    opacity = 0.8
    capsize = 3

    filetrain = open('Results/train.txt', 'w+')
    filetest = open('Results/test.txt', 'w+')
    filestdtr = open('Results/std_tr.txt','w+')
    filestdts = open('Results/std_ts.txt', 'w+')

    np.savetxt(filetrain, train_accs, fmt='%2.2f')
    np.savetxt(filestdtr, train_stds, fmt='%2.2f')
    np.savetxt(filetest, test_accs, fmt='%2.2f')
    np.savetxt(filestdts, test_stds, fmt='%2.2f')

    filetrain.close()
    filetest.close()
    filestdtr.close()
    filestdts.close()

    print(train_accs)
    plt.bar(index + float(bar_width)/2, train_accs, bar_width,
                    alpha = opacity,
                    error_kw = dict(elinewidth=1, ecolor='r'),
                    yerr = train_stds,
                    color = 'c',
                    label = 'train')

    plt.bar(index + float(bar_width)/2 + bar_width, test_accs, bar_width,
                    alpha = opacity,
                    error_kw = dict(elinewidth=1, ecolor='g'),
                    yerr = test_stds,
                    color = 'b',
                    label = 'test')
    plt.xlabel('Datasets')
    plt.ylabel('Accuracy')
    plt.xticks(index+bar_width, filenames, rotation=70)
    plt.legend()

    plt.tight_layout()
    plt.savefig('barplt.png')
    plt.show()

if __name__ == "__main__": main()
