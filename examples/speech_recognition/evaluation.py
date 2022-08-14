import torch, sys,csv
sys.path.insert(0, './huggingsound')
from huggingsound import SpeechRecognitionModel

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 1
model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-english", device=device)

with open('../../../audios/train.csv', newline='') as f:
    reader = csv.reader(f)
    data = [{"path": row[1], "transcription": row[0]} for row in reader if row[1] != 'path']



evaluation = model.evaluate(data, inference_batch_size=batch_size)

print(evaluation)