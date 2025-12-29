import re
import matplotlib.pyplot as plt

def parse_logs(log_file):
    epoch_data = {}
    
    with open(log_file, 'r') as f:
        # Read the file line by line to be memory efficient
        current_epoch_observed = None
        for line in f:
            # Update current observed epoch
            e_match = re.findall(r'Epoch (\d+)/\d+', line)
            if e_match:
                current_epoch_observed = int(e_match[-1])

            # Match Training Loss occurrences on this line
            # TQDM puts updates on the same line or in chunks
            train_matches = re.findall(r'Epoch (\d+)/\d+: 100%.*?loss=([\d\.]+)', line)
            for e_str, l_str in train_matches:
                epoch = int(e_str)
                train_loss = float(l_str)
                if epoch not in epoch_data:
                    epoch_data[epoch] = {}
                epoch_data[epoch]['train'] = train_loss
                
            # Match Validation Loss
            val_match = re.search(r'Validation Loss: ([\d\.]+)', line)
            if val_match:
                val_loss = float(val_match.group(1))
                if current_epoch_observed is not None:
                    if current_epoch_observed not in epoch_data:
                        epoch_data[current_epoch_observed] = {}
                    epoch_data[current_epoch_observed]['val'] = val_loss

    # Convert to sorted lists
    sorted_epochs = sorted(epoch_data.keys())
    final_epochs = []
    final_train = []
    final_val = []
    
    for e in sorted_epochs:
        t = epoch_data[e].get('train')
        v = epoch_data[e].get('val')
        if t is not None and v is not None:
            final_epochs.append(e)
            final_train.append(t)
            final_val.append(v)
            
    print(f"Found {len(final_epochs)} valid epochs with both train and val loss.")
    return final_epochs, final_train, final_val

def plot_losses(epochs, train_losses, val_losses, output_file):
    plt.style.use('bmh')
    plt.figure(figsize=(12, 7))
    
    plt.plot(epochs, train_losses, label='Training Loss', color='#1f77b4', linewidth=2, marker='o', markersize=3, alpha=0.8)
    plt.plot(epochs, val_losses, label='Validation Loss', color='#d62728', linewidth=2, marker='s', markersize=3, alpha=0.8)
    
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
    plt.title('v60 (CNN Stability Refinement) - Training Progress', fontsize=14, fontweight='bold', pad=20)
    
    plt.legend(frameon=True, fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if epochs:
        plt.xlim(left=max(0, min(epochs)-1))
        
    plt.tight_layout()
    plt.savefig(output_file, format='svg', dpi=300)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    log_path = 'training_logs.txt'
    output_path = 'v60_loss_curve.svg'
    epochs, train_losses, val_losses = parse_logs(log_path)
    if epochs:
        plot_losses(epochs, train_losses, val_losses, output_path)
    else:
        print("No paired loss data found in logs.")
