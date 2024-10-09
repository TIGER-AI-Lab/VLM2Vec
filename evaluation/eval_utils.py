import numpy as np
import os
import json


def get_pred(qry_t, tgt_t, normalization=False):
    """
    Use L2 norms.
    """
    if normalization:
        qry_t_norm = np.linalg.norm(qry_t)
        tgt_t_norms = np.linalg.norm(tgt_t, axis=1)
        scores = np.dot(tgt_t, qry_t) / (tgt_t_norms * qry_t_norm)
    else:
        scores = np.dot(tgt_t, qry_t)
    pred = np.argmax(scores)
    return scores, pred

def save_results(results, model_args, data_args, train_args):
    save_file = model_args.model_name + "_" + (model_args.model_type if  model_args.model_type is not None else "") + "_" + data_args.embedding_type + "_results.json"
    with open(os.path.join(data_args.encode_output_path, save_file), "w") as json_file:
        json.dump(results, json_file, indent=4)

def print_results(results):
    for dataset, acc in results.items():
        print(dataset, ",", acc)
