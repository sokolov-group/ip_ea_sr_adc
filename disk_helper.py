/***********************************************************************************
 * Copyright (C) 2019 Samragni Banerjee & Alexander Yu. Sokolov -All Rights Reserved
 * This file is part of IP/EA-ADC.
 * Unauthorized copying of this file, via any medium is strictly prohibited
 ***********************************************************************************/
import os
import h5py
import tempfile


def empty_dataset(shape):
    _, fname = tempfile.mkstemp()
    f = h5py.File(fname, mode='w')
    return f.create_dataset('data', shape)


def dataset(data):
    _, fname = tempfile.mkstemp()
    f = h5py.File(fname, mode='w')
    return f.create_dataset('data', data=data)


def remove_dataset(dataset):
    os.remove(dataset.file.filename)
