#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import os
import yaml
import sys
import numpy as np
import torch
from pathlib import Path


TRAIN_ROOT = str(Path(__file__).resolve().parents[2])
if TRAIN_ROOT not in sys.path:
    sys.path.insert(0, TRAIN_ROOT)

from tasks.semantic.modules.ioueval import iouEval
from common.laserscan import SemLaserScan

# possible splits
splits = ['train','valid','test']

FOUR_CLASS_NAMES = [
    "non-drivable",
    "drivable",
    "static obstacle",
    "dynamic object",
]


def map_to_four_classes(labels):
    """Map original SemanticKITTI IDs to the project's four classes."""
    labels = np.asarray(labels)
    mapped = np.zeros(labels.shape, dtype=np.int64)

    # 1 = drivable
    mapped[(labels == 40) | (labels == 44) | (labels == 60)] = 1

    # 2 = static obstacle
    mapped[
        (labels == 50) | (labels == 51) | (labels == 71) |
        (labels == 80) | (labels == 81)
    ] = 2

    # 3 = dynamic object, including moving-object SemanticKITTI IDs
    mapped[
        (labels == 10) | (labels == 11) | (labels == 15) |
        (labels == 18) | (labels == 20) | (labels == 30) |
        (labels == 31) | (labels == 32) | (labels == 252) |
        (labels == 253) | (labels == 254) | (labels == 255) |
        (labels == 256) | (labels == 257) | (labels == 258) |
        (labels == 259) | (labels == 13)
    ] = 3

    return mapped


def save_to_log(logdir,logfile,message):
    f = open(logdir+'/'+logfile, "a")
    f.write(message+'\n')
    f.close()
    return

def eval(test_sequences,splits,pred):
    # get scan paths
    scan_names = []
    for sequence in test_sequences:
        sequence = '{0:02d}'.format(int(sequence))
        scan_paths = os.path.join(FLAGS.dataset, "sequences",
                                  str(sequence), "velodyne")
        # populate the scan names
        seq_scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(scan_paths)) for f in fn if ".bin" in f]
        seq_scan_names.sort()
        scan_names.extend(seq_scan_names)
    # print(scan_names)

    # get label paths
    label_names = []
    for sequence in test_sequences:
        sequence = '{0:02d}'.format(int(sequence))
        label_paths = os.path.join(FLAGS.dataset, "sequences",
                                   str(sequence), "labels")
        # populate the label names
        seq_label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(label_paths)) for f in fn if ".label" in f]
        seq_label_names.sort()
        label_names.extend(seq_label_names)
    # print(label_names)

    # get predictions paths
    pred_names = []
    for sequence in test_sequences:
        sequence = '{0:02d}'.format(int(sequence))
        pred_paths = os.path.join(FLAGS.predictions, "sequences",
                                  sequence, "predictions")
        # populate the label names
        seq_pred_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(pred_paths)) for f in fn if ".label" in f]
        seq_pred_names.sort()
        pred_names.extend(seq_pred_names)
    # print(pred_names)

    # check that I have the same number of files
    # print("labels: ", len(label_names))
    # print("predictions: ", len(pred_names))
    assert (len(label_names) == len(scan_names) and
            len(label_names) == len(pred_names))

    print("Evaluating sequences: ")
    # open each file, get the tensor, and make the iou comparison
    for scan_file, label_file, pred_file in zip(scan_names, label_names, pred_names):
        print("evaluating label ", label_file, "with", pred_file)
        # open label
        label = SemLaserScan(project=False)
        label.open_scan(scan_file)
        label.open_label(label_file)
        u_label_sem = map_to_four_classes(label.sem_label)
        if FLAGS.limit is not None:
            u_label_sem = u_label_sem[:FLAGS.limit]

        # open prediction
        pred = SemLaserScan(project=False)
        pred.open_scan(scan_file)
        pred.open_label(pred_file)
        u_pred_sem = pred.sem_label.astype(np.int64)
        if not np.all(np.isin(u_pred_sem, [0, 1, 2, 3])):
            raise ValueError(
                f"Prediction {pred_file} contains labels outside 0..3"
            )
        if FLAGS.limit is not None:
            u_pred_sem = u_pred_sem[:FLAGS.limit]

        # add single scan to evaluation
        evaluator.addBatch(u_pred_sem, u_label_sem)

    # Compute four-class accuracy and IoU from the confusion matrix.
    confusion = evaluator.conf_matrix.cpu().numpy()
    true_positive = np.diag(confusion)
    ground_truth_count = confusion.sum(axis=0)
    prediction_count = confusion.sum(axis=1)

    overall_accuracy = true_positive.sum() / max(confusion.sum(), 1)
    per_class_accuracy = np.divide(
        true_positive,
        ground_truth_count,
        out=np.zeros(4, dtype=float),
        where=ground_truth_count != 0,
    )
    union = ground_truth_count + prediction_count - true_positive
    per_class_iou = np.divide(
        true_positive,
        union,
        out=np.zeros(4, dtype=float),
        where=union != 0,
    )
    mean_iou = per_class_iou.mean()

    summary = (
        f'{splits} set:\n'
        f'Overall accuracy: {overall_accuracy:.4f}\n'
        f'Mean IoU: {mean_iou:.4f}'
    )
    print(summary)
    save_to_log(FLAGS.predictions, 'pred.txt', summary)

    for class_id, class_name in enumerate(FOUR_CLASS_NAMES):
        class_summary = (
            f'{class_name}: accuracy={per_class_accuracy[class_id]:.4f}, '
            f'IoU={per_class_iou[class_id]:.4f}, '
            f'support={ground_truth_count[class_id]}'
        )
        print(class_summary)
        save_to_log(FLAGS.predictions, 'pred.txt', class_summary)

    # print for spreadsheet
    print("*" * 80)
    print("below can be copied straight for paper table")
    for i, class_iou in enumerate(per_class_iou):
        sys.stdout.write(f'{class_iou:.3f}')
        sys.stdout.write(",")
    sys.stdout.write(f'{mean_iou:.3f}')
    sys.stdout.write(",")
    sys.stdout.write(f'{overall_accuracy:.3f}')
    sys.stdout.write('\n')
    sys.stdout.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./evaluate_iou.py")
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='Dataset dir. No Default',
    )
    parser.add_argument(
        '--predictions', '-p',
        type=str,
        required=None,
        help='Prediction dir. Same organization as dataset, but predictions in'
             'each sequences "prediction" directory. No Default. If no option is set'
             ' we look for the labels in the same directory as dataset'
    )
    parser.add_argument(
        '--split', '-s',
        type=str,
        required=False,
        choices=["train", "valid", "test"],
        default=None,
        help='Split to evaluate on. One of ' +
             str(splits) + '. Defaults to %(default)s',
    )
    parser.add_argument(
        '--data_cfg', '-dc',
        type=str,
        required=False,
        default="config/labels/semantic-kitti.yaml",
        help='Dataset config file. Defaults to %(default)s',
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        required=False,
        default=None,
        help='Limit to the first "--limit" points of each scan. Useful for'
             ' evaluating single scan from aggregated pointcloud.'
             ' Defaults to %(default)s',
    )

    FLAGS, unparsed = parser.parse_known_args()

    # fill in real predictions dir
    if FLAGS.predictions is None:
        FLAGS.predictions = FLAGS.dataset

    # print summary of what we will do
    print("*" * 80)
    print("INTERFACE:")
    print("Data: ", FLAGS.dataset)
    print("Predictions: ", FLAGS.predictions)
    print("Split: ", FLAGS.split)
    print("Config: ", FLAGS.data_cfg)
    print("Limit: ", FLAGS.limit)
    print("*" * 80)

    # assert split
    assert (FLAGS.split in splits)

    # open data config file
    try:
        print("Opening data config file %s" % FLAGS.data_cfg)
        DATA = yaml.safe_load(open(FLAGS.data_cfg, 'r'))
    except Exception as e:
        print(e)
        print("Error opening data yaml file.")
        quit()

    # Evaluate in the four-class space. Non-drivable is a valid class, so
    # no class is ignored.
    nr_classes = 4
    ignore = []

    device = torch.device("cpu")
    evaluator = iouEval(nr_classes, device, ignore)
    evaluator.reset()

    # get test set
    if FLAGS.split is None:
        for splits in ('train','valid'):
            eval((DATA["split"][splits]),splits,FLAGS.predictions)
    else:
        eval(DATA["split"][FLAGS.split], FLAGS.split, FLAGS.predictions)

