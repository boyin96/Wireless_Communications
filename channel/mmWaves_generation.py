#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time      : 7/14/2023 11:22 AM
# Author    : Bo Yin
# Email     : bo.yin@ugent.be

r"""
mmWaves_generation.py: Generation of mmWaves channel.
MmWaveChan_class with MATLAB from Hamid Ramezani
mmWaveChann_class (https://www.mathworks.com/matlabcentral/fileexchange/64901-mmwavechann_class),
MATLAB Central File Exchange. Retrieved December 23, 2022.

Channel model refer to
O. E. Ayach, S. Rajagopal, S. Abu-Surra, Z. Pi and R. W. Heath, "Spatially Sparse Precoding in Millimeter Wave MIMO
Systems," in IEEE Transactions on Wireless Communications, vol. 13, no. 3, pp. 1499-1513, March 2014.
"""

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

# DEFINE
PI = np.pi


class MmWaveChan:
    """
    Mmwave channel generation class.

    Attributes:
        ChannelType
        CarrierFrequency
        NumberOfMainPaths
        NumberOfSubPathsPerMainPath
        TxNumberOfAntennasInEachColumns
        TxNumberOfAntennasInEachRows
        TxAzimuthMaxAngle
        TxAzimuthMinAngle
        TxAzimuthAngleSubPathStd
        TxElevationMaxAngle
        TxElevationMinAngle
        TxElevationAngleSubPathStd
        RxNumberOfAntennasInEachColumns
        RxNumberOfAntennasInEachRows
        RxAzimuthMaxAngle
        RxAzimuthMinAngle
        RxAzimuthAngleSubPathStd
        RxElevationMaxAngle
        RxElevationMinAngle
        RxElevationAngleSubPathStd
        d

    Examples:
        C = MmWaveChan(
        CarrierFrequency=28e9,
        NumberOfMainPaths=1,
        NumberOfSubPathsPerMainPath=4,
        TxNumberOfAntennasInEachRows=4,
        TxNumberOfAntennasInEachColumns=4,
        RxNumberOfAntennasInEachRows=3,
        RxNumberOfAntennasInEachColumns=3,
        )
        # get info of channel
        C.get_info()
        # generate channel coefficients
        H1, P1 = C.channel_generate()
        # plot channel coefficients
        C.plot(P1)
    """

    def __init__(self,
                 ChannelType="Saleh-Valenzuela",
                 CarrierFrequency=28e9,
                 NumberOfMainPaths=3,
                 NumberOfSubPathsPerMainPath=8,
                 TxNumberOfAntennasInEachColumns=4,
                 TxNumberOfAntennasInEachRows=4,
                 TxAzimuthMaxAngle=PI,
                 TxAzimuthMinAngle=-PI,
                 TxAzimuthAngleSubPathStd=PI / 128,
                 TxElevationMaxAngle=PI / 2,
                 TxElevationMinAngle=-PI / 2,
                 TxElevationAngleSubPathStd=PI / 128,
                 RxNumberOfAntennasInEachColumns=4,
                 RxNumberOfAntennasInEachRows=1,
                 RxAzimuthMaxAngle=PI,
                 RxAzimuthMinAngle=-PI,
                 RxAzimuthAngleSubPathStd=PI / 32,
                 RxElevationMaxAngle=PI / 2,
                 RxElevationMinAngle=-PI / 2,
                 RxElevationAngleSubPathStd=PI / 32,
                 dis=100,
                 ):
        self.ChannelType = ChannelType  # channel model: "Saleh-Valenzuela"
        self.CarrierFrequency = CarrierFrequency  # carrier frequency in Hz
        self.Lambda = 3e8 / self.CarrierFrequency  # wave length

        self.NumberOfMainPaths = NumberOfMainPaths  # number of main paths (how many clusters)
        self.NumberOfSubPathsPerMainPath = NumberOfSubPathsPerMainPath  # number of sub-paths per main paths

        # for transmitter
        # number of antenna columns # in a uniformly planar array Tx
        self.TxNumberOfAntennasInEachColumns = TxNumberOfAntennasInEachColumns
        # number of antenna rows in a uniformly planar array Tx
        self.TxNumberOfAntennasInEachRows = TxNumberOfAntennasInEachRows
        self.TxAzimuthMaxAngle = TxAzimuthMaxAngle  # for random angle generation
        self.TxAzimuthMinAngle = TxAzimuthMinAngle  # for random angle generation
        self.TxAzimuthAngleSubPathStd = TxAzimuthAngleSubPathStd  # for each cluster
        self.TxElevationMaxAngle = TxElevationMaxAngle  # for random angle generation
        self.TxElevationMinAngle = TxElevationMinAngle  # for random angle generation
        self.TxElevationAngleSubPathStd = TxElevationAngleSubPathStd  # for each cluster
        self.TxInterElementSpacing = self.Lambda / 2  # space between two adjacent antennas (now lambda/2)

        # for receiver
        # number of antenna columns in a uniformly planar array Rx
        self.RxNumberOfAntennasInEachColumns = RxNumberOfAntennasInEachColumns
        # number of antenna rows in a uniformly planar array Rx
        self.RxNumberOfAntennasInEachRows = RxNumberOfAntennasInEachRows
        self.RxAzimuthMaxAngle = RxAzimuthMaxAngle  # maximum value that an azimuth angle can get
        self.RxAzimuthMinAngle = RxAzimuthMinAngle  # minimum value that an azimuth angle can get
        self.RxAzimuthAngleSubPathStd = RxAzimuthAngleSubPathStd  # standard deviation of the scattered sub-path
        self.RxElevationMaxAngle = RxElevationMaxAngle  # maximum value that an elevation angle can get
        self.RxElevationMinAngle = RxElevationMinAngle  # minimum value that an elevation angle can get
        self.RxElevationAngleSubPathStd = RxElevationAngleSubPathStd  # standard deviation of the scattered sub-path
        self.RxInterElementSpacing = self.Lambda / 2  # space between two adjacent antennas (now lambda/2)

        # number of transmit antennas
        self.TxNumberOfAntennas = self.TxNumberOfAntennasInEachRows * self.TxNumberOfAntennasInEachColumns
        # number of receive antennas
        self.RxNumberOfAntennas = self.RxNumberOfAntennasInEachRows * self.RxNumberOfAntennasInEachColumns
        self.nTx = self.TxNumberOfAntennas  # number of transmit antennas same as TxNumberOfAntennas
        self.nRx = self.RxNumberOfAntennas  # number of transmit antennas same as RxNumberOfAntennas

        # distance
        self.dis = dis

    def channel_generate(self):
        # generate random angles for AoD (uniform distribution)
        txPhi = (np.random.rand(self.NumberOfMainPaths, 1) * (self.TxAzimuthMaxAngle - self.TxAzimuthMinAngle) +
                 self.TxAzimuthMinAngle)
        txTheta = (np.random.rand(self.NumberOfMainPaths, 1) * (self.TxElevationMaxAngle - self.TxElevationMinAngle) +
                   self.TxElevationMinAngle)
        rxPhi = (np.random.rand(self.NumberOfMainPaths, 1) * (self.RxAzimuthMaxAngle - self.RxAzimuthMinAngle) +
                 self.RxAzimuthMinAngle)
        rxTheta = (np.random.rand(self.NumberOfMainPaths, 1) * (self.RxElevationMaxAngle - self.RxElevationMinAngle) +
                   self.RxElevationMinAngle)

        # generate normalization factor
        gamma = np.sqrt((self.TxNumberOfAntennas * self.RxNumberOfAntennas) /
                        (self.NumberOfMainPaths * self.NumberOfSubPathsPerMainPath))

        # generate the complex channel gain (here los gain)
        sigma_los = self.los_pathloss()
        alpha = np.sqrt(1 / 2) * sigma_los * (np.random.randn(self.NumberOfMainPaths, self.NumberOfSubPathsPerMainPath)
                                              + 1j * np.random.randn(self.NumberOfMainPaths,
                                                                     self.NumberOfSubPathsPerMainPath))

        # coefficients of the subpaths
        txPhiSubPath = np.zeros((self.NumberOfMainPaths, self.NumberOfSubPathsPerMainPath))
        txThetaSubPath = np.zeros((self.NumberOfMainPaths, self.NumberOfSubPathsPerMainPath))
        rxPhiSubPath = np.zeros((self.NumberOfMainPaths, self.NumberOfSubPathsPerMainPath))
        rxThetaSubPath = np.zeros((self.NumberOfMainPaths, self.NumberOfSubPathsPerMainPath))

        # generate random azimuth and elevation angles
        for n in range(self.NumberOfMainPaths):
            txPhiSubPath[n, :] = (txPhi[n] + np.random.randn(self.NumberOfSubPathsPerMainPath, 1).T *
                                 self.TxAzimuthAngleSubPathStd)
            txThetaSubPath[n, :] = (txTheta[n] + np.random.randn(self.NumberOfSubPathsPerMainPath, 1).T *
                                   self.TxElevationAngleSubPathStd)
            rxPhiSubPath[n, :] = (rxPhi[n] + np.random.randn(self.NumberOfSubPathsPerMainPath, 1).T *
                                 self.RxAzimuthAngleSubPathStd)
            rxThetaSubPath[n, :] = (rxTheta[n] + np.random.randn(self.NumberOfSubPathsPerMainPath, 1).T *
                                   self.RxElevationAngleSubPathStd)

        # initialize H with zeros
        H = np.zeros((self.RxNumberOfAntennas, self.TxNumberOfAntennas)) * 1j
        try:
            if self.ChannelType == "Saleh-Valenzuela":
                for n in range(self.NumberOfMainPaths):
                    for m in range(self.NumberOfSubPathsPerMainPath):
                        aTx = self.txBuildArrayResponses(txPhiSubPath[n, m], txThetaSubPath[n, m])
                        aRx = self.rxBuildArrayResponses(rxPhiSubPath[n, m], rxThetaSubPath[n, m])
                        H = H + alpha[n, m] * aRx * aTx.T.conjugate()
            else:
                raise ValueError("Such channel type is not supported.")
        except ValueError as e:
            print("Error: ", repr(e))
        H = gamma * H

        # initialize channel info
        keys = ["txPhiSubPath", "txThetaSubPath", "rxPhiSubPath", "rxThetaSubPath", "alpha", "H"]
        values = [txPhiSubPath, txThetaSubPath, rxPhiSubPath, rxThetaSubPath, alpha, H]
        P = dict(zip(keys, values))
        return H, P

    def txBuildArrayResponses(self, phi, theta):
        # for planar case which is general
        TxRows = np.arange(0, self.TxNumberOfAntennasInEachRows)
        TxCols = np.arange(0, self.TxNumberOfAntennasInEachColumns)
        n, m = np.meshgrid(TxRows, TxCols)
        aTx = (1 / np.sqrt(self.TxNumberOfAntennas) * np.exp(
            2 * PI * self.TxInterElementSpacing / self.Lambda * 1j * (
                    m.reshape(-1, 1, order='F') * np.sin(phi) * np.sin(theta) +
                    n.reshape(-1, 1, order='F') * np.cos(theta))
            )
        )
        return aTx

    def rxBuildArrayResponses(self, phi, theta):
        # for planar case which is general
        RxRows = np.arange(0, self.RxNumberOfAntennasInEachRows)
        RxCols = np.arange(0, self.RxNumberOfAntennasInEachColumns)
        n, m = np.meshgrid(RxRows, RxCols)
        aRx = (1 / np.sqrt(self.RxNumberOfAntennas) * np.exp(
            2 * PI * self.RxInterElementSpacing / self.Lambda * 1j * (
                    m.reshape(-1, 1, order='F') * np.sin(phi) * np.sin(theta) +
                    n.reshape(-1, 1, order='F') * np.cos(theta))
            )
        )
        return aRx

    def los_pathloss(self):
        # los complex gain
        var_a = 61.4
        var_b = 2
        xi = 5.8
        var_xi = xi * np.random.randn()
        pl_los = var_a + 10 * var_b * np.log10(self.dis) + var_xi
        pl_los = np.sqrt(np.power(10, -0.1 * pl_los))
        return pl_los

    def nlos_pathloss(self):
        # nlos complex gain
        var_a = 72
        var_b = 2.92
        xi = 8.7
        var_xi = xi * np.random.randn()
        pl_nlos = var_a + 10 * var_b * np.log10(self.dis) + var_xi
        pl_nlos = np.sqrt(np.power(10, -0.1 * pl_nlos))
        return pl_nlos

    def get_info(self):
        print("The properties of mmWaveChan_Generate_clas: ")
        print('\n'.join(('{}: {}'.format(key, value) for key, value in self.__dict__.items())))

    def chann_plot(self, P):
        plt.figure()
        plt.subplot(2, 2, 1)
        for n in range(self.NumberOfMainPaths):
            # for transmitter
            plt.plot(np.cos(P['txPhiSubPath'][n, :]), np.sin(P['txPhiSubPath'][n, :]), 'o')
            plt.plot(np.cos(P['rxPhiSubPath'][n, :]), np.sin(P['rxPhiSubPath'][n, :]), 'x')
        x = np.linspace(-1, 1, 100)
        y = np.sqrt(1 - np.power(x, 2))
        plt.plot(np.concatenate((x, x[::-1]), axis=0), np.concatenate((y, -y[::-1]), axis=0), 'k:', linewidth=1)
        plt.xlabel(r"cos($\phi$)")
        plt.ylabel(r"sin($\phi$)")
        plt.legend([r"tx $\phi$", r"rx $\phi$"], loc='upper right')
        plt.title("Azimuth angles of each path from Tx to Rx")

        plt.subplot(2, 2, 2)
        for n in range(self.NumberOfMainPaths):
            # for receiver
            plt.plot(np.cos(P['txThetaSubPath'][n, :]), np.sin(P['txThetaSubPath'][n, :]), 'o')
            plt.plot(np.cos(P['rxThetaSubPath'][n, :]), np.sin(P['rxThetaSubPath'][n, :]), 'x')
        x = np.linspace(-1, 1, 100)
        y = np.sqrt(1 - np.power(x, 2))
        plt.plot(np.concatenate((x, x[::-1]), axis=0), np.concatenate((y, -y[::-1]), axis=0), 'k:', linewidth=1)
        plt.xlabel(r"cos($\theta$)")
        plt.ylabel(r"sin($\theta$)")
        plt.legend([r"tx $\theta$", r"rx $\theta$"], loc='upper right')
        plt.title("Elevation angles of each path from Tx to Rx")

        plt.subplot(2, 2, 3)
        x = np.arange(P['alpha'].shape[0])
        total_width = 0.4
        width = total_width / self.NumberOfSubPathsPerMainPath
        x = x - (total_width - width) / 2
        for i in range(P['alpha'].shape[1]):
            plt.bar(x + i * width, abs(P['alpha'][:, i]), width=width)
        plt.xlabel("path index")
        plt.ylabel("path amplitude")
        plt.title("Amplitude of each sub-path")

        plt.subplot(2, 2, 4)
        _, S, _ = np.linalg.svd(P['H'])
        plt.plot(S)
        plt.xlabel("singular index")
        plt.ylabel("singular value")
        plt.title("Singular values of the channel")

        plt.show()


if __name__ == "__main__":
    Channel = MmWaveChan(
        CarrierFrequency=28e9,
        NumberOfMainPaths=1,
        NumberOfSubPathsPerMainPath=4,
        TxNumberOfAntennasInEachRows=4,
        TxNumberOfAntennasInEachColumns=4,
        RxNumberOfAntennasInEachRows=3,
        RxNumberOfAntennasInEachColumns=3,
    )
    Channel.get_info()
    H1, P1 = Channel.channel_generate()
    Channel.chann_plot(P1)
    print(H1)
