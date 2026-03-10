class Config :

    # Model Architecture Parameters
    vocab_size=32000
    embedding_dim=256
    hidden_dim=256
    
    # Memory Parameters
    num_heads=4
    window_size=128
    memory_dim=256

    # Training Parameters
    batch_size=8
    seq_len=256
    lr=3e-4
    epochs=5

    device="cuda"