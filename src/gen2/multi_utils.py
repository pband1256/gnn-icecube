import os
import csv
import argparse
import logging
import pickle
import yaml
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

import matplotlib
matplotlib.use('Agg') # no display on clusters
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import NullFormatter

import torch
from torch.autograd import Variable

import model

#####################
#     CONSTANTS     #
#####################
ARGS_NAME  = 'args.yml'
MODEL_NAME = 'model.pkl'
BEST_MODEL = 'best_model.pkl'
STATS_CSV  = 'training_stats.csv'
CURRENT_BASELINE = [1.44576*10**-6, 0.04302]

#####################################
#     EXPERIMENT INITIALIZATION     #
#####################################

def read_args():
  '''
  Parse stdin arguments
  '''

  parser = argparse.ArgumentParser(description=
                      'Arguments for GNN model and experiment')
  add_arg = parser.add_argument

  # Experiment
  add_arg('--name', help='Experiment reference name', required=True)
  add_arg('--run', help='Experiment run number', default=0)
  add_arg('--eval_tpr',help='FPR at which TPR will be evaluated', default=0.000003)
  add_arg('--evaluate', help='Perform evaluation on test set only',action='store_true')
  add_arg('--regr_mode', help='Regression quantity (energy vs. direction)',default='energy')

  # Training
  add_arg('--nb_epoch', help='Number of epochs to train', type=int, default=2)
  add_arg('--lrate', help='Initial learning rate', type=float, default = 0.005)
  add_arg('--batch_size', help='Size of each minibatch', type=int, default=4)
  add_arg('--patience',help='Patience for lrate scheduler', type=int, default=20)

  # Dataset
  add_arg('--train_file', help='List of paths to train pickle file',type=str,nargs='+',default=[None])
  add_arg('--val_file',   help='Path to val   pickle file',type=str,default=None)
  add_arg('--test_file',  help='Path to test  pickle file',type=str,default=None)
  add_arg('--nb_train', help='Number of training samples', type=int, default=10)
  add_arg('--nb_val', help='Number of validation samples', type=int, default=10)
  add_arg('--nb_test', help='Number of test samples', type=int, default=10)

  # Model Architecture
  add_arg('--nb_hidden', help='Number of hidden units per layer', type=int, default=32)
  add_arg('--nb_layer', help='Number of network grapn conv layers', type=int, default=6)

  return parser.parse_args()

def initialize_logger(experiment_dir):
  '''
  Logger prints to stdout and logfile
  '''
  logfile = os.path.join(experiment_dir, 'log.txt')
  logging.basicConfig(filename=logfile,format='%(message)s',level=logging.INFO)
  logging.getLogger().addHandler(logging.StreamHandler())

def get_experiment_dir(experiment_name, run_number):
  '''
  Saves all models within a 'models' directory where the experiment is run.
  Returns path to the specific experiment within the 'models' directory.
  '''
  current_dir = os.getcwd()
  save_dir = os.path.join(current_dir, 'models')
  if not os.path.exists(save_dir):
    os.mkdir(save_dir) # Create models dir which will contain experiment data
  return os.path.join(save_dir, experiment_name, str(run_number))

def initialize_experiment_if_needed(model_dir, evaluate_only):
  '''
  Check if experiment initialized and initialize if not.
  Perform evaluate safety check.
  '''
  if not os.path.exists(model_dir):
    initialize_experiment(model_dir)
    if evaluate_only:
      logging.warning("EVALUATING ON UNTRAINED NETWORK")

def initialize_experiment(experiment_dir):
  '''
  Create experiment directory and initiate csv where epoch info will be stored.
  '''
  print("Initializing experiment.")
  os.makedirs(experiment_dir)
  csv_path = os.path.join(experiment_dir, STATS_CSV)
  with open(csv_path, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Epoch', 'lrate', 'train_loss', 'val_loss', 'running_loss'])


###########################
#     MODEL UTILITIES     #
###########################
def create_or_restore_model(
                            experiment_dir,
                            nb_hidden,
                            nb_layer,
                            input_dim,
                            output_dim,
                            spat_dim
                            ):
  '''
  Checks if model exists and creates it if not.
  Returns model.
  '''
  model_file = os.path.join(experiment_dir, MODEL_NAME)
  if os.path.exists(model_file):
    logging.warning("Loading model...")
    m = load_model(model_file)
    logging.warning("Model restored.")
  else:
    logging.warning("Creating new model:")
    m = model.GNN(nb_hidden, nb_layer, input_dim, output_dim, spat_dim)
    logging.info(m)
    save_model(m, model_file)
    logging.warning("Initial model saved.")
  return m

def load_model(model_file):
  '''
  Load torch model.
  '''
  m = torch.load(model_file)
  return m

def load_best_model(experiment_dir):
  '''
  Load the model which performed best in training.
  '''
  best_model_path = os.path.join(experiment_dir, BEST_MODEL)
  return load_model(best_model_path)

def save_model(m, model_file):
  '''
  Save torch model.
  '''
  torch.save(m, model_file)

def save_best_model(experiment_dir, net):
  '''
  Called if current model performs best.
  '''
  model_path = os.path.join(experiment_dir, BEST_MODEL)
  save_model(net, model_path)
  logging.warning("Best model saved.")

def save_epoch_model(experiment_dir, net):
  '''
  Optionally called after each epoch to save current model.
  '''
  model_path = os.path.join(experiment_dir, MODEL_NAME)
  save_model(net, model_path)


def load_args(experiment_dir):
  '''
  Restore experiment arguments
  args contain e.g. nb_epochs_complete, lrate
  '''
  args_path = os.path.join(experiment_dir, ARGS_NAME)
  with open(args_path, 'r') as argfile:
    args = yaml.load(argfile)
  logging.warning("Model arguments restored.")
  return args

def save_args(experiment_dir, args):
  '''
  Save experiment arguments.
  args contain e.g. nb_epochs_complete, lrate
  '''
  args_path = os.path.join(experiment_dir, ARGS_NAME)
  with open(args_path, 'w') as argfile:
    yaml.dump(args, argfile, default_flow_style=False)

######################
#     EVALUATION     #
######################
def update_best_plots(experiment_dir):
  '''
  Rename .png plots to best when called.
  '''
  for f in os.listdir(experiment_dir):
    # Write over old best curves
    if f.endswith(".png") and not f.startswith("best"):
      old_name = os.path.join(experiment_dir, f)
      new_name = os.path.join(experiment_dir, "best_"+f)
      os.rename(old_name, new_name)

def plot_reg_hist(true_y, pred_y, experiment_dir, plot_name, regr_mode='energy'):
  # Plotting
  plt.clf()
  nullfmt = NullFormatter()

  # definitions for the axes
  left, width = 0.1, 0.6
  bottom, height = 0.1, 0.6
  bottom_h = left_h = left + width + 0.02
  rect_hist  = [left, bottom, width, height]
  rect_histx = [left, bottom_h, width, 0.2]
  rect_histy = [left_h, bottom, 0.2, height]
  # start with a rectangular Figure
  plt.figure(figsize=(8, 7))
  axHist  = plt.axes(rect_hist)
  axHistx = plt.axes(rect_histx)
  axHisty = plt.axes(rect_histy)
  # no labels
  axHistx.xaxis.set_major_formatter(nullfmt)
  axHisty.yaxis.set_major_formatter(nullfmt)

  # the scatter plot:
  hist_min = np.min(true_y)
  hist_max = np.max(true_y)

  if regr_mode == 'energy':
    bins = np.logspace(np.log10(hist_min), np.log10(hist_max), 100)
    im = axHist.hist2d(pred_y.squeeze(), true_y.squeeze(), norm=colors.LogNorm(), bins=(bins,bins))
  elif regr_mode in ['zenith', 'azimuth', 'x', 'y', 'z']:
    bins = np.linspace(hist_min, hist_max, 100)
    im = axHist.hist2d(pred_y.squeeze(), true_y.squeeze(), bins=(bins,bins))
  else:
    print('Regression mode unspecified.')


  axHist.plot(bins, bins, color='r')

  # now determine nice limits by hand:
  axHistx.hist(pred_y, bins=bins)
  axHisty.hist(true_y,  bins=bins, orientation='horizontal')
  axHistx.set_xlim(axHist.get_xlim())
  axHisty.set_ylim(axHist.get_ylim())

  # Style
  axHist.set_xlabel("Prediction")
  axHist.set_ylabel("Truth")
  axHistx.tick_params(labelbottom=False)
  axHisty.tick_params(labelleft=False)
  if regr_mode == 'energy':
    axHist.set_xscale('log')
    axHist.set_yscale('log')
    axHistx.set_xscale('log')
    axHistx.set_yscale('log')
    axHisty.set_xscale('log')
    axHisty.set_yscale('log')
  plt.suptitle("Prediction histogram ("+regr_mode+")")
  #cax = plt.axes([0.27, 0.8, 0.5, 0.05])
  #plt.colorbar(im, cax=cax)

  #Save
  plotfile = experiment_dir+'/test_hist.png'
  plt.savefig(plotfile)
  plt.clf()
  #Save
  plotfile = os.path.join(experiment_dir, '{}_{}.png'.format(plot_name, regr_mode))
  plt.savefig(plotfile)
  plt.clf()

def plot_energy_slices(truth, nn_reco, regr_mode='energy',\
                       use_fraction = False, use_old_reco = False, old_reco=None,\
                       bins=20,minenergy=3000.,maxenergy=300000.,\
                       save=False,savefolder=None):
    """Plots different energy slices vs each other (systematic set arrays)
    Receives:
        truth= array with truth labels
                (contents = [energy], shape = number of events)
        nn_reco = array that has NN predicted reco results
                    (contents = [energy], shape = number of events)
        use_fraction = bool, use fractional resolution instead of absolute, where (reco - truth)/truth
        use_old_reco = bool, True if you want to compare to another reconstruction (like pegleg)
        old_reco = optional, array of pegleg labels
                (contents = [energy], shape = number of events)
        bins = integer number of data points you want (range/bins = width)
        minenergy = minimum energy value to start cut at (default = 0.)
        maxenergy = maximum energy value to end cut at (default = 60.)
    Returns:
        Scatter plot with energy values on x axis (median of bin width)
        y axis has median of resolution with error bars containing 68% of resolution
    """
    if use_fraction:
        resolution = ((nn_reco-truth)/truth) # in fraction
    else:
        resolution = (nn_reco-truth)
    percentile_in_peak = 68.27
    left_tail_percentile  = (100.-percentile_in_peak)/2
    right_tail_percentile = 100.-left_tail_percentile
    energy_ranges  = np.linspace(minenergy,maxenergy, num=bins)
    energy_centers = (energy_ranges[1:] + energy_ranges[:-1])/2.
    medians  = np.zeros(len(energy_centers))
    err_from = np.zeros(len(energy_centers))
    err_to   = np.zeros(len(energy_centers))
    if use_old_reco:
        if use_fraction:
            resolution_reco = ((old_reco-truth)/truth)
        else:
            resolution_reco = (old_reco-truth)
        err_from_reco = np.zeros(len(energy_centers))
        err_to_reco   = np.zeros(len(energy_centers))
    for i in range(len(energy_ranges)-1):
        en_from = energy_ranges[i]
        en_to   = energy_ranges[i+1]
        cut = (truth >= en_from) & (truth < en_to)
        lower_lim = np.percentile(resolution[cut], left_tail_percentile)
        upper_lim = np.percentile(resolution[cut], right_tail_percentile)
        median = np.percentile(resolution[cut], 50.)
        medians[i] = median    
        err_from[i] = lower_lim
        err_to[i] = upper_lim
        if use_old_reco:
            lower_lim_reco = np.percentile(resolution_reco[cut], left_tail_percentile)
            upper_lim_reco = np.percentile(resolution_reco[cut], right_tail_percentile)
            median_reco = np.percentile(resolution_reco[cut], 50.)
            medians_reco[i] = median_reco
            err_from_reco[i] = lower_lim_reco
            err_to_reco[i] = upper_lim_reco
    plt.figure(figsize=(10,7))
    plt.errorbar(energy_centers, medians, yerr=[medians-err_from, err_to-medians], xerr=[ energy_centers-energy_ranges[:-1], energy_ranges[1:]-energy_centers ], capsize=5.0, fmt='o',label="NN Reco")
    if use_old_reco:
        plt.errorbar(energy_centers, medians_reco, yerr=[medians_reco-err_from_reco, err_to_reco-medians_reco], xerr=[ energy_centers-energy_ranges[:-1], energy_ranges[1:]-energy_centers ], capsize=5.0, fmt='o',label="Pegleg Reco")
        plt.legend(loc="upper center")
    plt.plot([minenergy,maxenergy], [0,0], color='k')
    plt.xlim(minenergy,maxenergy)
    plt.xlabel("Energy range (GeV)")
    if use_fraction:
        plt.ylabel("Fractional resolution ("+regr_mode+"): \n (reco - truth)/truth")
    else:
        plt.ylabel("Resolution ("+regr_mode+"): \n reco - truth (GeV)")
    plt.title("Resolution energy dependence")
    savename = "EnergyResolutionSlices_"+regr_mode
    if use_fraction:
        savename += "Frac"
    if use_old_reco:
        savename += "_CompareOldReco"
    if save == True:
        plt.savefig("%s%s.png"%(savefolder,savename))

def track_epoch_stats(epoch, lrate, train_loss, train_stats, val_stats, experiment_dir):
  '''
  Write loss information to .csv file in model directory.
  '''
  csv_path = os.path.join(experiment_dir, STATS_CSV)
  with open(csv_path, 'a') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow((epoch, lrate)+train_stats+val_stats+(train_loss,))

def save_preds(evt_id, f_name, energy, pred_y, true_y, experiment_dir):
  '''
  Save predicted outputs for predicted event id, filename.
  '''
  pred_file = os.path.join(experiment_dir, 'preds.csv')
  with open(pred_file, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['event_id', 'filename', 'energy', 'prediction', 'truth'])
    for e, f, y_p, y_t, en in zip(evt_id, f_name, energy, pred_y, true_y):
      writer.writerow((e, f, y_p, y_t, en))

def save_test_scores(nb_eval, epoch_loss, experiment_dir):
  test_scores = {'nb_eval':nb_eval,
                 'epoch_loss':epoch_loss}
  pred_file = os.path.join(experiment_dir, 'test_scores.yml')
  with open(pred_file, 'w') as f:
    yaml.dump(test_scores, f, default_flow_style=False)

def save_best_scores(epoch, epoch_loss, experiment_dir):
  best_scores = {'epoch':epoch,
                 'epoch_loss':epoch_loss}
  pred_file = os.path.join(experiment_dir, 'best_scores.yml')
  with open(pred_file, 'w') as f:
    yaml.dump(best_scores, f, default_flow_style=False)
