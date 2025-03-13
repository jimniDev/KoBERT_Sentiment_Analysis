# Korean Emotion Analysis with KoBERT

This repository contains a Korean text emotion classifier built using KoBERT (Korean BERT). The model can categorize Korean text into six different emotions: joy, anxiety, confusion, sadness, anger, and hurt.

## Overview

The project fine-tunes the pre-trained KoBERT model from SKT on a Korean emotional dialogue dataset to classify text into the following emotion categories:

- 기쁨 (Joy/Happiness)
- 불안 (Anxiety)
- 당황 (Embarrassment/Confusion)
- 슬픔 (Sadness)
- 분노 (Anger)
- 상처 (Hurt/Pain)

## Repository Structure

- `kobert_finetuning.ipynb`: Jupyter notebook for model training and fine-tuning
- `inference.ipynb`: Jupyter notebook for running inference with the trained model
- `Dataset/`: Directory containing training and validation datasets (not included in repo)
  - `감성대화말뭉치(최종데이터)_Training.xlsx`
  - `감성대화말뭉치(최종데이터)_Validation.xlsx`
- `SentimentAnalysisKOBert.pt`: Saved model file
- `SentimentAnalysisKOBert_StateDict.pt`: Saved model state dictionary

## Requirements

```
torch
transformers==4.10.0
gluonnlp
mxnet
pandas
sentencepiece
tqdm
scikit-learn
kobert_tokenizer
```

You can install the KoBERT tokenizer with:
```bash
pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'
```

## Model Architecture

The model consists of:
- KoBERT base model from SKT
- Dropout layer (p=0.5)
- Linear classification layer with 6 output nodes

## Training

The model was trained with the following parameters:
- Maximum sequence length: 64 tokens
- Batch size: 64
- Learning rate: 5e-5
- Number of epochs: 5
- Optimizer: AdamW with weight decay
- Learning rate schedule: Cosine with warmup
- Loss function: Cross-Entropy Loss

## Usage

### Training
To train the model, run the `kobert_finetuning.ipynb` notebook.

### Inference
To use the model for prediction:

```python
def predict(sentence):
    dataset = [[sentence, '0']]
    test = BERTDataset(dataset, 0, 1, tok, vocab, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size, num_workers=2)
    model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length = valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)

        emotion_label = ["기쁨", "불안", "당황", "슬픔", "분노", "상처"]
        
        for i in out:
            logits = i
            logits = logits.detach().cpu().numpy()
            probability = []
            logits = np.round(new_softmax(logits), 3).tolist()

            for idx, logit in enumerate(logits):
                probability.append([idx, emotion_label[idx], np.round(logit, 3)])

            probability.append(emotion_label[np.argmax(logits)])
            
    return probability

# Example
sentence = '오늘 회사 다녀왔는데 휴식이 필요하다.. 너무 피곤.'
result = predict(sentence)
```

## Example Output

For the input sentence "오늘 회사 다녀왔는데 휴식이 필요하다.. 너무 피곤." (I went to work today and need a rest.. Too tired.), the model returns probability scores for each emotion and the predicted emotion label.

## Acknowledgements

- [SKT Brain's KoBERT](https://github.com/SKTBrain/KoBERT)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)

## License

This project is available under [LICENSE NAME].
