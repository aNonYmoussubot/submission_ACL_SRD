# SRD Method Code

This repository contains the implementation of the SRD (Seeking Rational Demonstrations for unsupervised cross-domain keyphrase generation), modified from [facebookresearch/DPR](https://github.com/facebookresearch/DPR). Below is a step-by-step guide on how to use the code, including data preparation, model training, inference, and evaluation.

## 1. Data Preparation

### 1.1 Dataset Download

The training set (KP20k) and out-of-domain test sets (StackExchange, DUC, KPTimes, OpenKP, and KPBiomed) can be downloaded from the following sources:

| Datasets   | Links                           |
|------------|---------------------------------|
| KP20k      | [memray/kp20k](https://huggingface.co/memray/kp20k) |
| StackExchange | [memray/stackexchange](https://huggingface.co/memray/stackexchange) |
| DUC        | [memray/duc](https://huggingface.co/memray/duc) |
| KPTimes    | [memray/kptimes](https://huggingface.co/memray/kptimes) |
| OpenKP     | [memray/openkp](https://huggingface.co/memray/openkp) |
| KPBiomed   | [taln-ls2n/kpbiomed](https://huggingface.co/taln-ls2n/kpbiomed) |

You can use the `huggingface-cli` command to download the datasets. For example:

```bash
huggingface-cli download --repo-type dataset --resume-download memray/stackexchange --local-dir stackexchange --local-dir-use-symlinks False
```

### 1.2 Data Preprocessing

Run the `process_code/preprocess_test_datasets.py` script to preprocess the five out-of-domain test sets.

## 2. Analyzing Cross-Domain Features

Navigate to the `analyse_distribution_shift` directory and run the following scripts:

- `convert_kp20k_clean.py` and `convert_kp20k_embeddings.py` to preprocess the KP20k training set and out-of-domain test sets.
- `compute_mmd.py` to calculate the MMD distance between datasets.

## 3. Constructing Positive and Negative Samples

Run the script `process_code/generate_retrieval_data.py` to construct positive and negative samples, including training and validation sets.

## 4. Training the Retrieval Model

Run `train_encoder.py` to train the retrieval model.

## 5. Generating Candidate Sets for Test Samples

- Run `generate_dense_embeddings.py` to generate dense embeddings.
- Run `dense_retriever.py` to retrieve the dense retrieval candidates.

## 6. Inference

### 6.1 Zero-Shot Prediction

Run `generate_zero_shot_deepseek.py` for zero-shot predictions.

### 6.2 SRD Method Prediction

Run `generate_deepseek.py` for SRD method predictions.

## 7. Evaluation

Run `evaluate_prediction.py` to evaluate the predictions.

## 8. Training and Evaluating the Evaluator model

First, use the K-means algorithm to split KP20k into pseudo target and pseudo source domains. Then, based on the different checkpoints from steps 5-7, obtain the corresponding model data (i.e., `X_src`, `X_ev`, `tgt_score`). Run `train_evaluator.py` and `inference_evaluator.py` to calculate scores for the candidate sets.

## Environment Setup

The following dependencies are required to run the code:

- `torch`: 1.12.1
- `transformers`: 4.46.3
- `faiss-cpu`: 1.8.0
- `clean-text`: 0.6.0
- `datasets`: 2.14.6
- `en-core-web-sm`: 3.6.0
- `h5py`: 3.11.0
- `numpy`: 3.8
- `nltk`: 3.8
- `spacy`: 3.6.0
- `scikit-learn`: 1.3.2



## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
