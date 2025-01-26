import torch
import nltk
import torch.nn as nn
from nltk.tokenize import word_tokenize
import numpy as np
import pickle
from flask import Flask, request, jsonify

nltk.download('punkt_tab')

# Define the Flask app
app = Flask(__name__)

# Define the LSTM model class
class HateSpeechLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout=0.5):
        super(HateSpeechLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
        self.batch_norm_lstm = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        self.batch_norm_fc = nn.BatchNorm1d(1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        hidden = hidden[-1]
        hidden = self.batch_norm_lstm(hidden)
        hidden = self.dropout(hidden)
        out = self.fc(hidden)
        out = self.batch_norm_fc(out)
        return self.sigmoid(out)

# Function to vectorize posts
def vectorize_post(tokens, embedding_matrix, max_len=10):
    vectors = [embedding_matrix[token] if token in embedding_matrix else np.zeros(embedding_dim) for token in tokens]
    if len(vectors) > max_len:
        vectors = vectors[:max_len]
    else:
        vectors.extend([np.zeros(embedding_dim)] * (max_len - len(vectors)))
    return np.array(vectors)

# Load embedding matrix
with open("embedding_matrix.pkl", "rb") as f:
    loaded_embedding_matrix = pickle.load(f)

# Initialize model and load weights
embedding_dim = 100
hidden_dim = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_model = HateSpeechLSTM(embedding_dim=embedding_dim, hidden_dim=hidden_dim).to(device)
best_model.load_state_dict(torch.load("best_model.pth"))
best_model.eval()

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get posts from the request JSON
        data = request.get_json()
        posts = data.get("posts", [])

        if not posts:
            return jsonify({"error": "No posts provided"}), 400

        # Vectorize and tokenize posts
        post_vectors = []
        for post in posts:
            tokens = word_tokenize(post)
            vector = vectorize_post(tokens, loaded_embedding_matrix, max_len=50)
            post_vectors.append(vector)

        # Convert vectors to tensor
        post_vectors = torch.tensor(post_vectors, dtype=torch.float32).to(device)

        # Make predictions
        with torch.no_grad():
            outputs = best_model(post_vectors).squeeze()
            test_preds = outputs.cpu().numpy()

        # Binarize predictions and format results
        predictions = [{
            "post": post,
            "hate_content_probability_percentage": f"{prob * 100:.2f}%",  # Convert to percentage
            "label": "Hate" if prob >= 0.5 else "Free"
        } for post, prob in zip(posts, test_preds)]

        return jsonify({"predictions": predictions})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
