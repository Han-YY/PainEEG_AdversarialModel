# Load the libs
import framework_wandb as fw
import numpy as np
import wandb
import sys

data_sample = np.load('data_all.npz')
# Load the sample data for testing the framework
data_frames = data_sample.f.data_frames
frame_label = np.argmax(data_sample.f.frame_label, axis=1)
frame_subject = data_sample.f.frame_subject

# Select the frame labels
select_idx = select_idx = [i for i, e in enumerate(frame_label) if e == 0 or e == 2]
data_frames_s = data_frames[select_idx]
frame_label_s = frame_label[select_idx]
frame_subject_s = frame_subject[select_idx]

class_unique = np.unique(frame_label_s)
class_unique = class_unique.tolist()
class_count = len(class_unique)
class_labels_n = []
for fl in frame_label_s:
    class_labels_n.append(class_unique.index(fl))
class_labels_n = np.asarray(class_labels_n)

data_frames_s = np.reshape(data_frames_s, newshape=(data_frames_s.shape[0], data_frames_s.shape[1], data_frames_s.shape[2], 1))
# run_name = sys.argv[1]


# Create the object for training and testing the model
# Each participant will be excluded as the testing set in each epoch


main_losses = []
adv_losses = []
main_accs = []
adv_accs = []
cf_mats = []
test_accs = []

sweep_config = {'method': 'bayes'} # Sweeping the hyperparameters
metric = {'name': 'pred_acc', 'goal': 'maximize'}
sweep_config['metric'] = metric
parameters_dict = {
    'lam1': {
        'values': [0.01, 0.05, 0.075, 0.1, 0.5, 0.75, 1]
    },
    'lam2': {
        'values': [0.05, 0.09, 0.1, 0.5, 0.9, 1]
    },
    'epochs':{
        'values': [50, 100, 200]
    },
    'batch_size': {
        'values': [256, 512, 1024, 2048]
    }
}

# Set the sweep agent
sweep_config['parameters'] = parameters_dict
sub_id = int(sys.argv[1]) # The exluded participant's id in the encoded list
sweep_id = wandb.sweep(sweep_config, project="adversarial_model-OvsH_hyper_finding_" + str(sub_id))

ad_model = fw.AdversarialModel(data_samples = data_frames_s, class_label=class_labels_n, subject_label=frame_subject_s, exclude_idx=sub_id)   
wandb.agent(sweep_id, ad_model.train, count=64)



