import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
from typing import Union, List

import torch
from torch.utils.data import Dataset
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

from clip import load


class CXRTestDataset(Dataset):
    """Represents an abstract HDF5 dataset.
    
    Input params:
        img_path: Path to hdf5 file containing images.
        label_path: Path to file containing labels (csv) 
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
    


def load_clip(model_path): 
    """
    load pretrained model from model_path
    """
    model, _ = load("ViT-B/32", device=device, jit=False) 
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def load_ensemble(model_paths):
    models = []
    for mpath in tqdm(model_paths):
        model = load_clip(mpath)
        models.append(model)
    return models


def tokenize(texts: Union[str, List[str]], tokenizer, context_length: int = 77) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    Returns
    -------
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



def generate_average_template_embeddings(classnames, templates, model, tokenizer, context_length=77):
    """
    generate class embeddings for the combination of classnames and templates. 
    Templates will be appended by classnames using 'format' function.
    Embeddings will be averages over templates, so templates must be picked harmoniously (representing similar semantics)
    E.g.
        classnames = ['atelectasis', 'lung opacity']
        templates = ['no {}']
    """
    model = model.to(device)
    with torch.no_grad():
        average_embeddings = []
        # compute embedding through model for each class
        for classname in tqdm(classnames):
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



def predict(loader, model, average_template_embeddings, verbose=0): 
    """
    FUNCTION: predict
    ---------------------------------
    This function runs the cxr images through the model 
    and computes the cosine similarities between the images
    and the text embeddings. 
    
    args: 
        * loader -  PyTorch data loader, loads in cxr images
        * model - PyTorch model, trained clip model 
        * average_text_embeddings - PyTorch Tensor, outputs of text encoder for labels
        * softmax_eval (optional) - Use +/- softmax method for evaluation 
        * verbose (optional) - bool, If True, will print out intermediate tensor values for debugging.
        
    Returns numpy array, predictions on all test data samples. 
    """
    y_pred = []
    with torch.no_grad():
        for _, data in enumerate(tqdm(loader)):
            images = data['img'].to(device)

            # predict
            image_features = model.encode_image(images) 
            image_features /= image_features.norm(dim=-1, keepdim=True) # (1, 768)

            # obtain logits
            logits = image_features @ average_template_embeddings # (1, num_classes)
            logits = np.squeeze(logits.cpu().numpy(), axis=0) # (num_classes,)
            
            y_pred.append(logits)
            
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



def run_single_prediction(classnames, template, model, tokenizer, loader, context_length=77): 
    """
    FUNCTION: run_single_prediction
    --------------------------------------
    This function will make probability predictions for a single template
    (i.e. "has {}"). 
    
    args: 
        * cxr_labels - list, labels for a specific zero-shot task. (i.e. ['Atelectasis',...])
        * template - string, template to input into model. 
        * model - PyTorch model, trained clip model
        * loader - PyTorch data loader, loads in cxr images
        * softmax_eval (optional) - Use +/- softmax method for evaluation 
        * context_length (optional) - int, max number of tokens of text inputted into the model.
        
    Returns list, predictions from the given template. 
    """
    average_template_embeddings = generate_average_template_embeddings(classnames, [template], model, tokenizer, context_length=context_length)
    y_pred = predict(loader, model, average_template_embeddings)
    return y_pred



def run_softmax_eval(model, loader, classnames: list, pair_template: tuple, tokenizer, context_length: int = 77): 
    """
    Run softmax evaluation to obtain a single prediction from the model.
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



def get_ensemble_predictions(models, loader, classnames, pair_template, tokenizer, context_length=77):
    """
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
): 
    """
    Loads in data containing the true binary labels
    for each pathology in `cxr_labels` for all samples. This
    is used for evaluation of model performance.

    args: 
        * cxr_true_labels_path - str, path to csv containing ground truth labels
        * cxr_labels - List[str], subset of label columns to select from ground truth df 
        * cutlabels - bool, if True, will keep columns of ground truth labels that correspond
            with the labels inputted through `cxr_labels`. Otherwise, drop the first column and keep remaining.

    Returns a numpy array of shape (# samples, # labels/pathologies)
        representing the binary ground truth labels for each pathology on each sample.
    """
    # create ground truth labels
    full_labels = pd.read_csv(cxr_true_labels_path)
    if cutlabels: 
        full_labels = full_labels.loc[:, cxr_labels]
    else: 
        full_labels.drop(full_labels.columns[0], axis=1, inplace=True)

    y_true = full_labels.to_numpy()
    return y_true