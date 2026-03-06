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
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

  def nullable_string(string):
    if string in ['None', '']:
      return None
    return string

  parser = argparse.ArgumentParser(description='Arguments for GNN model and experiment')
  add_arg = parser.add_argument

  # Experiment
  add_arg('--name', help='Experiment reference name', required=True)
  add_arg('--run', help='Experiment run number', default=0)
  #add_arg('--eval_tpr',help='FPR at which TPR will be evaluated', default=0.000003)
  add_arg('--evaluate', help='Model file name, perform evaluation on test set only',
          default=None, type=nullable_string)
  #action='store_true')
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
  add_arg('--old_reco_file', help='Path to old recon dict pickle file',type=nullable_string, default=None)
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
  os.makedirs(os.path.join(experiment_dir, 'plots'))
  os.makedirs(os.path.join(experiment_dir, 'models'))
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
  model_file = os.path.join(experiment_dir, "models",  MODEL_NAME)
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
  best_model_path = os.path.join(experiment_dir, "models",  BEST_MODEL)
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
  model_path = os.path.join(experiment_dir, "models", BEST_MODEL)
  save_model(net, model_path)
  logging.warning("Best model saved.")

def save_current_model(experiment_dir, net, epoch):
  '''
  Optionally called after each epoch to save current model.
  '''
  model_path = os.path.join(experiment_dir, "models", "model_epoch_{:02}.pkl".format(epoch))
  save_model(net, model_path)

def load_args(experiment_dir):
  '''
  Restore experiment arguments
  args contain e.g. nb_epochs_complete, lrate
  '''
  args_path = os.path.join(experiment_dir, ARGS_NAME)
  with open(args_path, 'r') as argfile:
    args = yaml.unsafe_load(argfile)
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
  experiment_dir = os.path.join(experiment_dir, "plots")
  for f in os.listdir(experiment_dir):
    # Write over old best curves
    if f.endswith(".png") and not f.startswith("best"):
      old_name = os.path.join(experiment_dir, f)
      new_name = os.path.join(experiment_dir, "best_"+f)
      os.rename(old_name, new_name)

def load_loss_data(filename, labels):
  data  = pd.read_csv(filename)
  arrays = []
  for label in labels:
    if pd.api.types.is_string_dtype(data[label]):
      symbols = "['!\"#$%&()*/:;<=>?@[\]^_`{|}~\n]"
      array = np.array(data[label].str.replace(symbols, ""))

      # Convert to ndarray
      array = np.array([np.fromstring(j, dtype=np.float, sep=' ') for j in array]).squeeze()
    else:
      array = np.array(data[label])
      arrays.append(array)
  return arrays

##########################
###                    ###
###    Brian's code    ###
###                    ###
##########################

def find_contours_2D(x_values,y_values,xbins,weights=None,c1=16,c2=84):   
        """
        Find upper and lower contours and median
        x_values = array, input for hist2d for x axis (typically truth)
        y_values = array, input for hist2d for y axis (typically reconstruction)
        xbins = values for the starting edge of the x bins (output from hist2d)
        c1 = percentage for lower contour bound (16% - 84% means a 68% band, so c1 = 16)
        c2 = percentage for upper contour bound (16% - 84% means a 68% band, so c2=84)
        Returns:
                x = values for xbins, repeated for plotting (i.e. [0,0,1,1,2,2,...]
                y_median = values for y value medians per bin, repeated for plotting (i.e. [40,40,20,20,50,50,...]
                y_lower = values for y value lower limits per bin, repeated for plotting (i.e. [30,30,10,10,20,20,...]
                y_upper = values for y value upper limits per bin, repeated for plotting (i.e. [50,50,40,40,60,60,...]
        """
        if weights is not None:
                import wquantiles as wq
        y_values = np.array(y_values)
        indices = np.digitize(x_values, xbins)
        r1_save = []
        r2_save = []
        median_save = []
        for i in range(1,len(xbins)):
                mask = indices==i
                if len(y_values[mask])>0:
                        if weights is None:
                                r1, m, r2 = np.nanpercentile(y_values[mask],[c1,50,c2])
                        else:
                                r1 = wq.quantile(y_values[mask],weights[mask],c1/100.)
                                r2 = wq.quantile(y_values[mask],weights[mask],c2/100.)
                                m = wq.median(y_values[mask],weights[mask])
                else:
                        #print(i,'empty bin')
                        r1 = np.nan
                        m = np.nan
                        r2 = np.nan
                median_save.append(m)
                r1_save.append(r1)
                r2_save.append(r2)
        median = np.array(median_save)
        lower = np.array(r1_save)
        upper = np.array(r2_save)

        # the first return with the [1:] and [:-1] is about locating the bin centers
        return (xbins[1:] + xbins[:-1])/2, median, lower, upper

def plot_reg_hist(x, y, experiment_dir, n_bins=100, plot_name='reg_hist', regr_mode='energy'):
  '''
  Regression: energy, zenith, azimuth
  Input:
    energy: log10(E)
    zenith: radians
    azimuth: radians
  '''
  font = 30
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
  plt.figure(figsize=(16, 14))
  axHist  = plt.axes(rect_hist)
  axHistx = plt.axes(rect_histx)
  axHisty = plt.axes(rect_histy)
  # no labels
  axHistx.xaxis.set_major_formatter(nullfmt)
  axHisty.yaxis.set_major_formatter(nullfmt)

  # the scatter plot:

  # 2D histogram of pred vs. truth
  if regr_mode == 'energy':
    hist_min = min(np.min(x), np.min(y))
    hist_max = max(np.max(x), np.max(y))
    hist_min = 2
    hist_max = 8
    x = 10**x
    y = 10**y
    bins = np.logspace(hist_min, hist_max, n_bins)
    im = axHist.hist2d(x, y, norm=colors.LogNorm(), bins=(bins,bins))
  elif regr_mode in ['zenith','azimuth']:
    if regr_mode == 'zenith':
      x = np.cos(x)
      y = np.cos(y)
    hist_min = min(np.min(x), np.min(y))
    hist_max = max(np.max(x), np.max(y))
    bins = np.linspace(hist_min, hist_max, n_bins)
    im = axHist.hist2d(x, y, bins=(bins,bins))
  else:
    print('Regression mode unspecified, no histogram generated.')
    return

  # Finding and plotting contours
  x_range, y_median, y_lower, y_upper = find_contours_2D(x, y, bins)
  axHist.plot(x_range, y_median, 'r', linewidth=4, label="median")
  axHist.plot(x_range, y_upper, 'r--', linewidth=4, label=r"$1\sigma$")
  axHist.plot(x_range, y_lower, 'r--', linewidth=4)
  axHist.plot(bins, bins, 'k', label='1-1', linewidth=4)
  axHist.legend(fontsize=font)

  # now determine nice limits by hand:
  axHistx.set_xlim(axHist.get_xlim())
  axHisty.set_ylim(axHist.get_ylim())

  axHistx.hist(x, bins=bins)
  axHisty.hist(y, bins=bins, orientation='horizontal')

  # Style
  if regr_mode == 'energy':
      axHist.set_ylabel(r"$E_{\mu, reco}$ [GeV]", fontsize=font)
      axHist.set_xlabel(r"$E_{\mu, truth}$ [GeV]", fontsize=font)
  elif regr_mode == 'zenith':
      axHist.set_ylabel(r"$\cos \theta_{\mu, reco}$", fontsize=font)
      axHist.set_xlabel(r"$\cos \theta_{\mu, truth}$", fontsize=font)
  elif regr_mode == 'azimuth':
      axHist.set_ylabel(r"$\phi_{\mu, reco}$ [rad]", fontsize=font)
      axHist.set_xlabel(r"$\phi_{\mu, truth}$ [rad]", fontsize=font)
  axHist.tick_params(axis='both', labelsize=font)
  axHistx.tick_params(axis='both', labelsize=font)
  axHisty.tick_params(axis='both', labelsize=font)
  axHistx.tick_params(labelbottom=False)
  axHisty.tick_params(labelleft=False)
  if regr_mode == 'energy':
    axHist.set_xscale('log')
    axHist.set_yscale('log')
    axHistx.set_xscale('log')
    axHistx.set_yscale('log')
    axHisty.set_xscale('log')
    axHisty.set_yscale('log')
  plt.suptitle("Prediction histogram ("+regr_mode+")", fontsize=font)

  divider = make_axes_locatable(axHisty)
  cax = divider.append_axes('right', size='10%', pad=0.1)
  cbar = plt.colorbar(im[3], cax=cax)
  cbar.ax.tick_params(labelsize=font)
  cbar.ax.get_yaxis().labelpad = 25
  cbar.ax.set_ylabel('Counts', rotation=270, fontsize=font)
  #plt.tight_layout()

  #Save
  plotfile = os.path.join(experiment_dir, "plots", '{}_{}.png'.format(plot_name, regr_mode))
  plt.savefig(plotfile, bbox_inches='tight', pad_inches=0.1)
  plt.clf()

def plot_loss(train_loss, val_loss, epoch, experiment_dir, plot_name="LossProgress"):
    font = 20
    plt.figure(figsize=(10,7))
    plt.plot(epoch, train_loss, label="Training")
    plt.plot(epoch, val_loss, label="Validation")
    plt.tick_params(axis='both', labelsize=font)

    plt.xlabel("Epoch", fontsize=font)
    plt.ylabel("Loss", fontsize=font)
    plt.legend(fontsize=font)
    plt.tight_layout()

    plt.savefig(os.path.join(experiment_dir, "plots", '{}.png'.format(plot_name)))
    plt.close()

###########################
###                     ###
###    Jessie's code    ###
###                     ###
###########################

def plot_energy_slices(truth, nn_reco, en, regr_mode='energy',\
                       use_fraction = False, old_reco=None, old_truth=None, old_en=None,\
                       bins=20,minenergy=3000.,maxenergy=300000.,\
                       savefolder=None, spec=None):

    font = 30
    if regr_mode == 'zenith':
        truth = np.cos(truth)
        nn_reco = np.cos(nn_reco)
        if old_reco is not None:
            old_reco = np.cos(old_reco)
            old_truth = np.cos(old_truth)

    # Setup
    percentile_in_peak = np.asarray([68.27, 95])
    left_tail_percentile  = (100.-percentile_in_peak)/2
    right_tail_percentile = 100.-left_tail_percentile
    energy_ranges  = np.linspace(minenergy,maxenergy, num=bins)
    energy_centers = (energy_ranges[1:] + energy_ranges[:-1])/2.

    def resolution_setup(reco, truth, regr_mode=regr_mode, use_fraction=use_fraction):
        if not use_fraction:
            resolution = reco-truth
        else:
            resolution = (reco-truth)/np.abs(truth)
        return resolution

    def find_percentile(resolution, en, energy_ranges=energy_ranges, left_tail_percentile=left_tail_percentile, right_tail_percentile=right_tail_percentile):
        medians  = np.zeros(len(energy_centers))
        err_from = np.zeros(len(energy_centers))
        err_to   = np.zeros(len(energy_centers))
        for i in range(len(energy_ranges)-1):
            en_from = energy_ranges[i]
            en_to   = energy_ranges[i+1]

            cut = (en >= en_from) & (en < en_to)
            if np.size(resolution[cut]) == 0:
                lower_lim = float('nan')
                upper_lim = float('nan')
                median = float('nan')
            else:
                lower_lim = np.nanpercentile(resolution[cut], left_tail_percentile)
                upper_lim = np.nanpercentile(resolution[cut], right_tail_percentile)
                median = np.nanpercentile(resolution[cut], 50.)

            medians[i] = median
            err_from[i] = lower_lim
            err_to[i] = upper_lim
        return medians, err_from, err_to

    nn_resolution = resolution_setup(nn_reco, truth)
    medians, err_from_1sigma, err_to_1sigma = find_percentile(nn_resolution, en, left_tail_percentile=left_tail_percentile[0], right_tail_percentile=right_tail_percentile[0])
    _, err_from_2sigma, err_to_2sigma = find_percentile(nn_resolution, en, left_tail_percentile=left_tail_percentile[1], right_tail_percentile=right_tail_percentile[1])

    if old_reco is not None:
        resolution_reco = resolution_setup(old_reco, old_truth)
        #########################################################################
        #upper_limit = np.nanpercentile(resolution_reco, 99.5)
        #lower_limit = np.nanpercentile(resolution_reco, 0.50)
        #index = (resolution_reco < upper_limit) & (resolution_reco > lower_limit)
        #resolution_reco = resolution_reco[index]
        #old_en = old_en[index]
        #########################################################################
        medians_reco, err_from_reco_1sigma, err_to_reco_1sigma = find_percentile(resolution_reco, old_en, left_tail_percentile=left_tail_percentile[0], right_tail_percentile=right_tail_percentile[0])
        _, err_from_reco_2sigma, err_to_reco_2sigma = find_percentile(resolution_reco, old_en, left_tail_percentile=left_tail_percentile[1], right_tail_percentile=right_tail_percentile[1])

    plt.figure(figsize=(15,10.5))
    plt.plot(energy_centers, medians, color='C0', alpha=1, label='GNN Reco')
    plt.fill_between(energy_centers, err_from_1sigma, err_to_1sigma, color='C0', alpha=0.4, label='GNN 68%')
    plt.fill_between(energy_centers, err_from_2sigma, err_to_2sigma, color='C0', alpha=0.2, label='GNN 95%')
    if old_reco is not None:
        plt.plot(energy_centers, medians_reco, color='C1', alpha=1, label='SplineMPE/MuEX Reco')
        plt.fill_between(energy_centers, err_from_reco_1sigma, err_to_reco_1sigma, color='C1', alpha=0.4, label='SplineMPE/MuEX 68%')
        plt.fill_between(energy_centers, err_from_reco_2sigma, err_to_reco_2sigma, color='C1', alpha=0.2, label='SplineMPE/MuEX 95%')
    plt.legend(fontsize=font)
    plt.plot([minenergy,maxenergy], [0,0], color='k')
    plt.xlim(minenergy,maxenergy)

    # Axis labels & ticks
    plt.xlabel(r"$\log_{10} E_\mu$ [GeV]", fontsize=font)
    if regr_mode == 'energy':
        if use_fraction:
            plt.ylabel("Fractional resolution ("+regr_mode+")\n (reco - truth)/truth ($\Delta E_{\mu}/E_{\mu}$)", fontsize=font)
        else:
            plt.ylabel(r"Resolution ("+regr_mode+")\n reco - truth ($\Delta \log_{10} E_{\mu}$ [GeV])", fontsize=font)
    elif regr_mode == 'zenith':
        if use_fraction:
            plt.ylabel("Fractional resolution ("+regr_mode+")\n (reco - truth)/truth ($\Delta \cos \\theta_{\mu}/|\cos \\theta_{\mu}|$)", fontsize=font)
        else:
            plt.ylabel(r"Resolution ("+regr_mode+")\n reco - truth ($\Delta \cos \\theta_{\mu}$)", fontsize=font)
    elif regr_mode == 'azimuth':
        if use_fraction:
            plt.ylabel("Fractional resolution ("+regr_mode+")\n (reco - truth)/truth ($\Delta \phi_{\mu}/\phi_{\mu}$)", fontsize=font)
        else:
            plt.ylabel(r"Resolution ("+regr_mode+")\n reco - truth ($\Delta \phi_{\mu}$ [rad])", fontsize=font)
    plt.tick_params(axis='both', labelsize=font)
    plt.title("Resolution energy dependence", y=1.08, fontsize=font)

    # Save name
    if spec is not None:
        regr_mode = regr_mode + "_" + spec
    savename = "EnergyResolutionSlices_" + regr_mode
    if use_fraction:
        savename += "_Frac"
    if old_reco is not None:
        savename += "_CompareOldReco"
    if savefolder != None:
        plt.tight_layout()
        plt.savefig(os.path.join(savefolder, "plots", '{}.png'.format(savename)))


def plot_res_hist(truth, nn_reco, old_truth, old_reco, regr_mode, experiment_dir=None):
    font = 30
    if regr_mode == 'zenith':
        truth = np.cos(truth)
        nn_reco = np.cos(nn_reco)
        old_truth = np.cos(old_truth)
        old_reco = np.cos(old_reco)

    resolution = nn_reco - truth
    reco_res = old_reco - old_truth

    '''
    def find_percentile(resolution):
        median = np.nanmedian(resolution)
        upper = np.nanpercentile(resolution, 84,1)
        lower = np.nanpercentile(resolution, 15.9)
        sigma = min(upper-median, median-lower)
        return median, sigma
    median, std = find_percentile(resolution)
    median_reco, std_reco = find_percentile(reco_res)
    '''

    #bins = np.linspace(np.min(resolution), np.max(resolution), 50)
    plt.figure(figsize=(15,10.5))
    plt.hist(resolution, bins=30, alpha=0.5, density=True, label=r'GNN')
    if len(old_truth) != 0:
        plt.hist(reco_res, bins=30, alpha=0.5, density=True, label=r'SplineMPE')
    plt.legend(fontsize=font)
    plt.xlabel('reco-truth', fontsize=font)
    plt.title('Residual histogram ({})'.format(regr_mode), fontsize=font)
    plt.tick_params(axis='both', labelsize=font)
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, "plots", '{}_{}.png'.format('ResHistTest', regr_mode)))
    plt.clf()


##########################
###                    ###
###     Debug code     ###
###                    ###
##########################

def track_epoch_stats(epoch, lrate, train_loss, train_stats, val_stats, experiment_dir):
  ''' 
  Write loss information to .csv file in model directory.
  '''
  csv_path = os.path.join(experiment_dir, STATS_CSV)
  with open(csv_path, 'a') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow((epoch, lrate)+(train_stats,val_stats)+(train_loss,))

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
