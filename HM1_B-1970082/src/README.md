# Install dependencies
pip install -r requirements.txt

# Run baselines
python src/baselines.py

# Train LSTM model
python src/train.py

# Evaluate LSTM model on test sets
python src/evaluate.py
