'''
Model server, acccept json data and return the analysis result.
Used to refine the model's parameters manually after training.
'''
import socket
import threading
import json
import numpy as np
from model import Embedding, Output, Decider
import torch

HOST = '127.0.0.1'
PORT = 65432
BUFFER_SIZE = 4096

class ModelServer:
    def __init__(self):
        self.embedding = Embedding()
        self.output = Output()
        self.decider = Decider()
        self.load_models()

    def load_models(self):
        self.embedding.load_state_dict(torch.load('embedding.pth'))
        self.output.load_state_dict(torch.load('output.pth'))
        self.decider.load_state_dict(torch.load('decider.pth'))

    def handle_client(self, conn, addr):
        print(f'Connected by {addr}')
        data = b''
        while True:
            packet = conn.recv(BUFFER_SIZE)
            if not packet:
                break
            data += packet
        response = self.process_data(data)
        conn.sendall(response)
        conn.close()

    def process_data(self, data: bytes) -> bytes:
        # Process the incoming data and return the result
        json_data = json.loads(data.decode())
        input_array = np.array(json_data)
        input_tensor = torch.tensor(input_array, dtype=torch.float32)

        with torch.no_grad():
            embedded = self.embedding(input_tensor)
            output = self.output(embedded)
            decision = self.decider(input_tensor)

        result = {
            'output': output.numpy().tolist(),
            'decision': decision.numpy().tolist()
        }
        return json.dumps(result).encode()

    def start_server(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((HOST, PORT))
            s.listen()
            print(f'Server listening on {HOST}:{PORT}')
            while True:
                conn, addr = s.accept()
                client_thread = threading.Thread(target=self.handle_client, args=(conn, addr))
                client_thread.start()