# BERT Masked Word Predictor

## Overview

This project is a Streamlit application that leverages the BERT (Bidirectional Encoder Representations from Transformers) model to predict masked words in a sentence. By utilizing a pre-trained BERT model, the application provides top predictions for words that can replace a masked token represented as `[MASK]`.

## Features

- **Interactive Interface:** Users can input sentences with masked tokens and receive predictions.
- **Top Predictions:** Displays the top 5 predicted words for the masked token based on the BERT model's output.
- **Real-Time Feedback:** Instant predictions upon user interaction.

## Installation

To run this application, you'll need Python 3.6 or higher and the following Python packages:

- **Streamlit**
- **Transformers**
- **PyTorch**

Install the necessary packages using pip:

```bash
    pip install streamlit transformers torch
```

## Code Explanation
### Import Libraries
```bash
    import streamlit as st
    from transformers import BertTokenizer, BertForMaskedLM
    import torch
```
### Load Pre-trained BERT Model and Tokenizer
```bash
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)
```
### Predict Masked Words Function
```bash
def predict_masked_words(sentence):
    # Tokenize the input sentence
    tokenized_input = tokenizer.encode_plus(sentence, return_tensors="pt", add_special_tokens=True)

    # Get the position of the masked token
    masked_index = torch.where(tokenized_input["input_ids"] == tokenizer.mask_token_id)[1]

    # Predict the masked token
    with torch.no_grad():
        outputs = model(**tokenized_input)

    # Get the logits for the masked token
    predictions = outputs.logits[0, masked_index, :]

    # Get the top predictions
    top_indices = torch.topk(predictions, 5, dim=1).indices[0].tolist()

    # Convert token IDs to actual words
    predicted_tokens = [tokenizer.decode([index]).strip() for index in top_indices]

    return predicted_tokens
```
### Streamlit Application
```bash
# Streamlit app
st.title("BERT Masked Word Predictor")

st.write("Enter a sentence with a masked word, denoted by [MASK], and see BERT's top predictions.")

# Text input for the sentence
input_sentence = st.text_input("Input Sentence", "I want to go [MASK].")

if st.button("Predict"):
    if "[MASK]" not in input_sentence:
        st.error("The input sentence must contain the [MASK] token.")
    else:
        # Predict masked words
        predicted_words = predict_masked_words(input_sentence)
        st.write("Predicted words:")
        for i, word in enumerate(predicted_words):
            st.write(f"{i+1}. {word}")
```
## Running the Application
1. Save the provided code to a file named 'MLM.py'
2. Open your terminal and navigae to the directory containing 'app.py'
3. Run the Streamlit applciation using the following command:
    ```bash
    streamlit run app.py
    ```
4. The application will start, and a new browser window will open displaying the interface for predicting masked words.
