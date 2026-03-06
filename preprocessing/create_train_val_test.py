import pickle
import glob
import argparse
import random
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--nb_train",type=int,default=10,
                    dest="nb_train",help="number of training files")
parser.add_argument("-v", "--nb_val",type=int,default=10,
                    dest="nb_val",help="number of validation files")
parser.add_argument("-e", "--nb_test",type=int,default=10,
                    dest="nb_test",help="number of test files")
parser.add_argument("-n", "--nb_split",type=int,default=1,
                    dest="nb_split",help="number of files to split into")
parser.add_argument("-i", "--input",type=str,default="/mnt/scratch/lehieu1/training_files/with_energy/",
                    dest="inp",help="path to directory of input pickle files")
parser.add_argument("-o", "--output",type=str,default="/mnt/scratch/lehieu1/training_files/processed/",
                    dest="out",help="path to directory of output pickle files")
parser.add_argument("--emin",type=float,default=0,
                    dest="emin",help="minimum energy")
parser.add_argument("--emax",type=float,default=float('inf'),
                    dest="emax",help="maximum energy")
parser.add_argument("--flat",type=float,default=1,
                    dest="flat",help="generate flat sample")
args = parser.parse_args()

# Masking non-coordinate features of feature array
def mask_features(data,feature_ind=(3,4,5)):
    for i in range(np.shape(data)[0]):
        data[i][:,feature_ind] = 0
    return data

# Masking coordinates of feature array (padding inactive DOMs with 0 for pulse data)
def mask_coordinates(data):
    coords = []
    batch_size = np.shape(data)[0]
    for i in range(batch_size):
        coords.append(data[i][:,0:3])
    coords_list = np.unique(np.concatenate(coords),axis=0)
    
    for i in range(batch_size):
        inactive_DOMs = np.array([x for x in set(tuple(x) for x in coords_list) ^ set(tuple(x) for x in coords[i])])
        data[i] = np.concatenate([np.concatenate([inactive_DOMs,np.zeros(np.shape(inactive_DOMs))],axis=1),data[i]])
    return data

# Get rid of enough data to get a sample with 50/50 label distribution
def create_equal_samples(data):
    data = np.asarray(data)
    label = data[1]
    nb_0 = np.sum(label==0)
    nb_1 = np.sum(label==1)
    if nb_0 < nb_1:
        less_ind = np.where(label==0)[0]
        more_ind = np.where(label==1)[0][0:nb_0]
    else:
        less_ind = np.where(label==1)[0]
        more_ind = np.where(label==0)[0][0:nb_1]
    data_less = data[:,less_ind]
    data_more_equal = data[:,more_ind]
    data = np.concatenate((data_less,data_more_equal),axis=1)

    # Shuffle along axis=1 or batch axis
    data = data[:, np.random.permutation(data.shape[1])]

    return data

def energy_cut(data, emin=0, emax=float('inf')):
    E = np.asarray(data[5])
    ind = (E > emin) & (E < emax)
    return np.asarray(data)[:, ind].tolist()

def cleaned_pulses_filter(X, nb_dom=3):
    index = []
    for i in range(len(X)):
        # Take note of all events with more than # hit DOMs
        if np.shape(X[i])[0] > nb_dom:
            index.append(i)
    return index

# Open pickled .i3 files one by one and concatenate all data into master arrays
def pickleList(fileList):
    first = True
    for fileName in fileList:
        try:
            with open(fileName,'rb') as f:
                X, y, weights, event_id, filenames, energy = pickle.load(f)
            if first == True:
                X_all = X
                y_all = y
                w_all = weights
                e_all = event_id
                f_all = filenames
                E_all = energy
                first = False
            else:
                X_all = np.concatenate((X_all,X))
                y_all = np.concatenate((y_all,y))
                w_all = np.concatenate((w_all,weights))
                e_all = np.concatenate((e_all,event_id))
                f_all = np.concatenate((f_all,filenames))
                E_all = np.concatenate((E_all,energy))
        except ValueError or EOFError as e:
            print("Error: file "+fileName+" failed to pickle correctly. Skipping file")
            print(e)
            continue
    ####### Masking weights
    #w_all = np.ones(np.shape(w_all))
    ####### Masking features
    # X_all = mask_features(X_all)
    ####### Masking coordinates
    # X_all = mask_coordinates(X_all)

    # Filtering cleaned pulses
    index = cleaned_pulses_filter(X)    
    data = [X_all[index], y_all[index], w_all[index], e_all[index], f_all[index], E_all[index]]

    print("Total number of events: ", np.shape(data[1])[0])
    if args.flat == False:
        print("(Non-flat) sample differential: ", np.shape(data[1])[0]-np.sum(data[1]))
        print("Tracks: ", np.sum(data[1]))
        print("Cascades: ", np.size(data[1])-np.sum(data[1]),'\n')
        ####### Creating flat sample
        #data = create_equal_samples(data)


    ####### Energy cuts
    if args.emin != 0 and args.emax !=float('inf'): 
        data = energy_cut(data, args.emin, args.emax)

    return data

# Shuffling ALL files in folder to make sure there's no systematic problem. Probably overkill.
nb_total = args.nb_train + args.nb_val + args.nb_test
total_file = glob.glob(args.inp + '/*.pkl')
print("Number of files used: ", nb_total)
print("Number of files total: ", len(total_file),'\n')
assert nb_total <= len(total_file), "Not enough files to create samples."
random.shuffle(total_file)

if args.nb_train != 0:
    print("Pickling training files...")
    split_ind = args.nb_train/args.nb_split
    for i in range(args.nb_split):
        train_file = total_file[int(i*split_ind):int((i+1)*split_ind)]
        with open(args.out + '/train_file_'+str(i+1)+'.pkl',"wb") as f:
            pickle.dump(pickleList(train_file),f)

if args.nb_val != 0:
    print("Pickling validation files...")
    val_file = total_file[args.nb_train:nb_total-args.nb_test]
    with open(args.out + '/val_file.pkl',"wb") as f:
        pickle.dump(pickleList(val_file),f)

if args.nb_test != 0:
    print("Pickling test files...")
    test_file = total_file[nb_total-args.nb_test:nb_total]
    with open(args.out + '/test_file.pkl',"wb") as f:
        pickle.dump(pickleList(test_file),f)
