# Load the libs
import framework
import numpy as np

data_sample = np.load('sample_data.npz')
# Load the sample data for testing the framework
data_frames = data_sample.f.data_frames
frame_label = data_sample.f.frame_label
frame_subject = data_sample.f.frame_subject

data_frames = np.reshape(data_frames, newshape=(data_frames.shape[0], data_frames.shape[1], data_frames.shape[2], 1))

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

# Create the object for training and testing the model
# Each participant will be excluded as the testing set in each epoch
lams = [2.0, 1.0, 0.5, 0.05, 0.0]
for l_id in range(5):
    main_losses = []
    adv_losses = []
    main_accs = []
    adv_accs = []
    cf_mats = []
    test_accs = []

    lam = lams[l_id]
    for sub_id in range(36):
        ad_model = framework.AdversarialModel(data_samples = data_frames_s, class_label=class_labels_n, subject_label=frame_subject_s, exclude_idx=sub_id, lam=lam)
        main_loss, adv_loss, main_acc, adv_acc, cf_matrix, test_acc = ad_model.train()

        main_losses.append(main_loss)
        adv_losses.append(adv_loss)
        main_accs.append(main_acc)
        adv_accs.append(adv_loss)
        cf_mats.append(cf_matrix)
        test_accs.append(test_acc)

        print('[%d/36]\tLoss_main: %.4f\tLoss_adv: %.4f\tLoss(x): %.4f\tAccuracy(Train):%.2f\tAccuracy(Adv):%.2f\tAccuracy(Test):%.2f'
                            % (sub_id + 1, main_loss[-1], adv_loss[-1], main_loss[-1] - adv_loss[-1], main_acc[-1], adv_acc[-2], test_acc))
    
    # Save the results
    np.savez("model_result_" + str(lam) + ".npz", main_losses=main_losses, adv_losses=adv_losses, main_accs=main_accs, adv_accs=adv_accs, cf_mats=cf_mats, test_accs=test_accs)