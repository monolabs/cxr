import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
from typing import Union, List, Tuple

from sklearn.metrics import accuracy_score

import torch
from torch.utils.data import Dataset, DataLoader
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

from clip import load


class CXRTestDataset(Dataset):
    """
    H5 dataset.
    Params:
        img_path: Path to hdf5 file containing images.
        transform: PyTorch transform to apply to every data instance (default=None).
    """
    def __init__(
        self, 
        img_path: str, 
        transform = None, 
    ):
        super().__init__()
        self.img_dset = h5py.File(img_path, 'r')['cxr']
        self.transform = transform
            
    def __len__(self):
        return len(self.img_dset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img = self.img_dset[idx] # np array, (320, 320)
        img = np.expand_dims(img, axis=0)
        img = np.repeat(img, 3, axis=0)
        img = torch.from_numpy(img) # torch, (320, 320)
        
        if self.transform:
            img = self.transform(img)
            
        sample = {'img': img}
    
        return sample
    


def load_clip(model_path: str) -> torch.nn.Module: 
    """
    loads pretrained model from model_path
    """
    model, _ = load("ViT-B/32", device=device, jit=False) 
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def load_ensemble(model_paths: List[str]) -> List[torch.nn.Module]:
    """
    loads a list of models from model_paths
    """
    models = []
    for mpath in tqdm(model_paths, desc='loading models'):
        model = load_clip(mpath)
        models.append(model)
    return models


def tokenize(
        texts: Union[str, List[str]], 
        tokenizer, 
        context_length: int = 77
        ) -> torch.Tensor:
    """
    Returns the tokenized representation of given input string(s)

    Args:
        texts: Union[str, List[str]]: An input string or a list of input strings to tokenize
        tokenizer: an object with method 'encode'
        context_length: int: The context length to use; all CLIP models use 77 as the context length

    Returns:
        A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result



def generate_average_template_embeddings(
        classnames: List[str], 
        templates: List[str], 
        model: torch.nn.Module, 
        tokenizer, 
        context_length: int=77
        ) -> torch.Tensor:
    """
    generate class embeddings for the combination of classnames and templates. 
    Templates will be appended by classnames using 'format' function.
    Embeddings will be averages over templates, so templates must be picked harmoniously (representing similar semantics)
    E.g.
        classnames = ['atelectasis', 'lung opacity']
        templates = ['no {}']
    
    Args:
        classnames: A list of classes (pathologies) to query
        templates: A list of string template to be combined with classnames
        model: A CLIP model
        tokenizer: a text tokenizer with method 'encode'
        context_length: int: The context length to use; all CLIP models use 77 as the context length

    Returns:
        A tensor of embeddings (averaged over templates)
    """
    model = model.to(device)
    with torch.no_grad():
        average_embeddings = []
        # compute embedding through model for each class
        for classname in tqdm(classnames, desc='embedding templates for each class'):
            texts = [template.format(classname) for template in templates] # format with class
            texts = tokenize(texts, tokenizer, context_length=context_length) # tokenize
            texts = texts.to(device)
            class_embeddings = model.encode_text(texts) # embed with text encoder
            # normalize class_embeddings (magnitude of vector for each template's embedding is 1)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            # average over templates 
            avg_embedding = class_embeddings.mean(dim=0)
            # norm over new averaged templates
            avg_embedding /= avg_embedding.norm()
            average_embeddings.append(avg_embedding)
        average_embeddings = torch.stack(average_embeddings, dim=1)
    return average_embeddings



def predict(
        loader: DataLoader, 
        model: torch.nn.Module, 
        average_template_embeddings: torch.Tensor, 
        verbose: bool=False
        ) -> np.ndarray: 
    """
    This function runs the cxr images through the model 
    and computes the cosine similarities between the images
    and the text embeddings. 
    
    Args: 
        loader: PyTorch data loader, loads in cxr images
        model: PyTorch model, trained CLIP model 
        average_template_embeddings: PyTorch Tensor, outputs of text encoder for labels
        verbose (optional): bool, If True, will print out intermediate tensor values for debugging.
        
    Returns 
        numpy array, predictions on all test data samples. 
    """
    y_pred = []
    with torch.no_grad():
        for _, data in enumerate(tqdm(loader, desc='predicting')):
            images = data['img'].to(device)

            # predict
            image_features = model.encode_image(images) 
            image_features /= image_features.norm(dim=-1, keepdim=True) # (1, 768)

            # obtain logits
            logits = image_features @ average_template_embeddings # (1, num_classes)
            logits = np.squeeze(logits.cpu().numpy(), axis=0) # (num_classes,)
            
            y_pred.append(logits)
            
            # legacy verbose...
            if verbose: 
                plt.imshow(images[0][0])
                plt.show()
                print('images: ', images)
                print('images size: ', images.size())
                
                print('image_features size: ', image_features.size())
                print('logits: ', logits)
                print('logits size: ', logits.size())
         
    y_pred = np.array(y_pred)
    return np.array(y_pred)



def run_single_prediction(
        classnames: List[str], 
        template: str, 
        model: torch.nn.Module, 
        tokenizer, 
        loader: DataLoader, 
        context_length: int=77
        ) -> np.ndarray: 
    """
    run_single_predictions (via 'predict' function)
    This function will make probability predictions for a single template
    (i.e. "has {}"). 
    
    Args: 
        classanmes: list, labels for a specific zero-shot task. (i.e. ['Atelectasis',...])
        template: string, template to input into model. 
        model: PyTorch model, trained clip model
        tokenizer: a text tokenizer with method 'encode'
        loader: PyTorch data loader, loads in cxr images
        context_length (optional): int, max number of tokens of text inputted into the model.
        
    Returns
        predictions (logit/cosine similarity) from the given template. Shape: (num images, num class)
    """
    average_template_embeddings = generate_average_template_embeddings(classnames, [template], model, tokenizer, context_length=context_length)
    y_pred = predict(loader, model, average_template_embeddings)
    return y_pred



def run_softmax_eval(
        model: torch.nn.Module, 
        loader: DataLoader, 
        classnames: List[str], 
        pair_template: List[str], 
        tokenizer, 
        context_length: int = 77
        ) -> np.ndarray: 
    """
    Run softmax evaluation to obtain a single prediction from the model.

    Args: 
        classnames: list, labels for a specific zero-shot task. (i.e. ['Atelectasis',...])
        tokenizer: a text tokenizer with method 'encode'
        pair_template: string, template to input into model. A pair of positive prompt and negative prompt. 
        model: PyTorch model, trained clip model
        loader: PyTorch data loader, loads in cxr images
        context_length (optional): int, max number of tokens of text inputted into the model.
        
    Returns
        predictions (logit/cosine similarity) from the given template. Shape: (num images, num class)
    """
     # get pos and neg phrases
    pos = pair_template[0]
    neg = pair_template[1]

    # get pos and neg predictions, (num_samples, num_classes)
    pos_pred = run_single_prediction(classnames, pos, model, tokenizer, loader, context_length=context_length) 
    neg_pred = run_single_prediction(classnames, neg, model, tokenizer, loader, context_length=context_length) 

    # compute probabilities with softmax
    sum_pred = np.exp(pos_pred) + np.exp(neg_pred)
    y_pred = np.exp(pos_pred) / sum_pred
    return y_pred



def get_ensemble_predictions(
        models: torch.nn.Module, 
        loader: DataLoader, 
        classnames: List[str], 
        pair_template: List[str], 
        tokenizer, 
        context_length: int=77
        ) -> Tuple[np.ndarray]:
    """
    get average prediction (logit) from multiple models.

    Args: 
        classnames: list, labels for a specific zero-shot task. (i.e. ['Atelectasis',...])
        tokenizer: a text tokenizer with method 'encode'
        pair_template: string, template to input into model. A pair of positive prompt and negative prompt. 
        model: PyTorch model, trained clip model
        loader: PyTorch data loader, loads in cxr images
        context_length (optional): int, max number of tokens of text inputted into the model.

    Returns:
        A tuple of the models' individual predictions (shape: (num models, num images, num classes)) and
        average predictions over all models (shape: (num images, num classes))
    """
    predictions = []
    for i, model in enumerate(models):
        print(f"predicting with model {i+1}/{len(models)}")
        y_pred = run_softmax_eval(model, loader, classnames, pair_template, tokenizer, context_length=context_length)
        predictions.append(y_pred)
    y_pred_avg = np.mean(predictions, axis=0)
    return predictions, y_pred_avg



def make_true_labels(
    cxr_true_labels_path: str, 
    cxr_labels: List[str],
    cutlabels: bool = True
) -> np.ndarray: 
    """
    Create an array of labels from groundtruth csv dataset

    Args:
        cxr_true_labels_path: path to csv dataset
        cxr_labels: labelnames corresponding to dataset column names to use
        cutlabels - bool, if True, will keep columns of ground truth labels that correspond
            with the labels inputted through `cxr_labels`. Otherwise, drop the first column and keep remaining.

    Returns:
        numpy array (shape: (n samples, n labels))
    """
    # create ground truth labels
    full_labels = pd.read_csv(cxr_true_labels_path)
    if cutlabels: 
        full_labels = full_labels.loc[:, cxr_labels]
    else: 
        full_labels.drop(full_labels.columns[0], axis=1, inplace=True)

    y_true = full_labels.to_numpy()
    return y_true



def get_best_accuracy_threshold(true: np.ndarray, pred_score: np.ndarray) -> Tuple[float]:
    """
    iteratively look for best threshold maximizing accuracy

    Args:
        true: binary groundtruth (1D)
        pred_score: predicted probability score (1D)

    Returns:
        a tuple of best thresholds and best accuracy
    """
    best_accuracy = -np.inf
    best_thresh = None
    for thresh in pred_score:
        p_binary = (pred_score >= thresh).astype(int)
        acc = accuracy_score(true, p_binary)
        if acc > best_accuracy:
            best_accuracy = acc
            best_thresh = thresh
    return best_thresh, best_accuracy



def optimize_accuracy(true: np.ndarray, pred_score: np.ndarray) -> Tuple[List[float]]:
    """
    Args:
        true: binary groundtruth (2D, shape: (num samples, num classes))
        pred_score: predicted probability score (2D, shape: (num samples, num classes))

    Returns:
        an tuple of best thresholds and best accuracies
    """
    assert true.shape == pred_score.shape, "true labels must be of the same shape as prediction score"

    best_thresholds = []
    best_accuracies = []
    for i in range(true.shape[-1]):
        best_thresh, best_acc = get_best_accuracy_threshold(true[:, i], pred_score[:, i])
        best_thresholds.append(best_thresh)
        best_accuracies.append(best_acc)
    return best_thresholds, best_accuracies
    

