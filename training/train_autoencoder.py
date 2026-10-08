import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scapy.all import PcapReader, TCP, IP

# Constants matching CUDA memory stream feature normalization
C_MEAN = np.array([10.5, 3.1, 0.5, 0.0], dtype=np.float32)
C_STD  = np.array([2.1, 0.8, 1.2, 1.0], dtype=np.float32)

def calculate_entropy(payload_bytes):
    if not payload_bytes:
        return 0.0
    entropy = 0.0
    length = float(len(payload_bytes))
    for x in set(payload_bytes):
        p_x = payload_bytes.count(x) / length
        entropy -= p_x * math.log2(p_x)
    return float(entropy)

def parse_pcap(filepath):
    features = []
    last_ts = 0.0
    
    for pkt in PcapReader(filepath):
        if pkt.haslayer(TCP) and pkt.haslayer(IP):
            ts = float(pkt.time)
            iat = max(0.0, (ts - last_ts) * 1000.0) if last_ts > 0 else 0.0
            last_ts = ts
            
            flags = float(int(pkt[TCP].flags))
            entropy = calculate_entropy(bytes(pkt[TCP].payload))
            features.append([iat, entropy, flags, 0.0])
            
    return np.array(features, dtype=np.float32)

class OmniSightAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 64, bias=False)
        self.decoder = nn.Linear(64, 4, bias=False)
        
    def forward(self, x):
        return self.decoder(torch.relu(self.encoder(x)))

def train_and_export():
    pcap_file = "normal_traffic.pcapng"
    print(f"[+] Extracting packets from {pcap_file}...")
    data = parse_pcap(pcap_file)
    
    # Normalize identically to the CUDA inference engine
    data = (data - C_MEAN) / C_STD
    tensor_data = torch.tensor(data, dtype=torch.float32)
    
    model = OmniSightAutoencoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print(f"[+] Training on {len(data)} baseline packets...")
    for epoch in range(50):
        optimizer.zero_grad()
        loss = criterion(model(tensor_data), tensor_data)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/50 | MSE Loss: {loss.item():.4f}")
            
    print("[+] Exporting transposed weights for CUDA memory alignment...")
    # Transpose PyTorch (out, in) weights to match the CUDA kernel's expected layout
    w_enc = model.encoder.weight.detach().numpy().T  
    w_dec = model.decoder.weight.detach().numpy().T  
    
    os.makedirs("../data", exist_ok=True)
    with open("../data/weights.bin", "wb") as f:
        f.write(w_enc.tobytes())
        f.write(w_dec.tobytes())
        
    print("[+] Wrote 512 parameters to ../data/weights.bin successfully.")

if __name__ == "__main__":
    train_and_export()