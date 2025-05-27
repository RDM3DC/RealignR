

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*NCCL support*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*cuBLAS*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*numba.generated_jit*")

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
import os
import psutil
import time

# Logging utilities for ARP/RealignR
from realignr_logging import log_arp_metrics, log_loss_slope, plot_rotor_memory_heatmap

# Imports from modularized files
from feature_extractors import CNNFeatureExtractor #, CMAExtractor, AdaptivePiExtractor
from spinor_layers import SpinorLayer #, blades, layout # Assuming blades/layout are handled within spinor_layers or passed
from arp_optimizer import ARPOptimizer # Or Adam, etc.

# Configuration Dictionary (Example)
config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "log_dir": "runs/gnc_experiment",
    "batch_size": 24576, # Increased from 12288 (doubled)
    "num_epochs": 20, # For initial testing, can be increased
    "learning_rate": 0.01, # Increased from 0.001
    "optimizer_type": "Adam", # "Adam" or "ARP"
    "feature_extractor_type": "CNN", # "CNN", "CMA", "AdaptivePi"
    "spinor_layers_params": {
        "num_layers": 4, # Increased from 2
        "num_classes": 100 # CIFAR-100
    },
    "arp_optimizer_params": { # Only used if optimizer_type is "ARP"
        "alpha": 1e-2,
        "mu": 1e-3,
        "weight_decay": 0.0,
        "clamp_G_min": 1e-4,
        "clamp_G_max": 10.0
    },
    "scheduler_step_size": 5,
    "scheduler_gamma": 0.5,
    "seed": 42,
    "checkpoint_dir": "checkpoints_experiment",
    "visualization_dir": "gnc_visualizations_experiment"
}

writer = SummaryWriter(config["log_dir"])

if __name__ == '__main__':
    # Set random seeds for reproducibility
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])

    # Device setup
    device = torch.device(config["device"])
    print(f"Using device: {device}")

    # TensorBoard setup
    os.makedirs(config["log_dir"], exist_ok=True)
    writer = SummaryWriter(config["log_dir"])

    # Create directories for checkpoints and visualizations
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["visualization_dir"], exist_ok=True)


    # CIFAR-100 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # CIFAR-100 means
    ])
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    # NOTE: num_workers > 0 can cause issues on Windows if not guarded by if __name__ == '__main__'
    # We are moving the main logic into this block, so num_workers=2 should be fine.
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config["batch_size"], shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config["batch_size"], shuffle=False, num_workers=2)


    # --- Model Components ---
    # Feature Extractor
    if config["feature_extractor_type"] == "CNN":
        feature_extractor = CNNFeatureExtractor().to(device)
    # elif config["feature_extractor_type"] == "CMA":
    #     feature_extractor = CMAExtractor().to(device) # Placeholder
    # elif config["feature_extractor_type"] == "AdaptivePi":
    #     feature_extractor = AdaptivePiExtractor().to(device) # Placeholder
    else:
        raise ValueError(f"Unsupported feature_extractor_type: {config['feature_extractor_type']}")

    # Class embedding (remains part of the main script for now, or could be modularized)
    class ClassEmbedding(nn.Module):
        def __init__(self, num_classes=100, embed_dim=5): # embed_dim matches feature_extractor output
            super(ClassEmbedding, self).__init__()
            self.embedding = nn.Embedding(num_classes, embed_dim)
        def forward(self, labels):
            return self.embedding(labels)

    class_embed = ClassEmbedding(num_classes=config["spinor_layers_params"]["num_classes"], embed_dim=feature_extractor.fc2.out_features).to(device)


    # Spinor Layer (conditionally used based on config, or always if it's central to the architecture)
    # For this example, we assume it's always used after the feature_extractor
    spinor_layer_module = SpinorLayer(
        num_layers=config["spinor_layers_params"]["num_layers"],
        num_classes=config["spinor_layers_params"]["num_classes"]
    ) # Note: SpinorLayer itself is not an nn.Module in the provided code, so no .to(device)

    # Classifier (takes output from SpinorLayer or directly from feature_extractor if no spinor layer)
    class SpinorClassifier(nn.Module): # Renaming for clarity if it always follows spinor layer
        def __init__(self, feature_dim=5, num_classes=100): # feature_dim matches spinor_layer output
            super(SpinorClassifier, self).__init__()
            self.fc = nn.Linear(feature_dim, num_classes)
        def forward(self, x):
            return self.fc(x)

    # Assuming feature_extractor outputs 5D, and spinor_layer also outputs 5D
    classifier = SpinorClassifier(feature_dim=feature_extractor.fc2.out_features, num_classes=config["spinor_layers_params"]["num_classes"]).to(device)

    # DataParallel Setup
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for DataParallel.")
        feature_extractor = nn.DataParallel(feature_extractor)
        class_embed = nn.DataParallel(class_embed)
        classifier = nn.DataParallel(classifier)
    
    # Parameters for optimizer
    model_parameters = list(feature_extractor.parameters()) + \
                       list(class_embed.parameters()) + \
                       list(classifier.parameters())
    # Note: SpinorLayer rotors are updated manually, not via PyTorch optimizer in the original script

    # Optimizer
    if config["optimizer_type"] == "Adam":
        optimizer = optim.Adam(model_parameters, lr=config["learning_rate"])
    elif config["optimizer_type"] == "ARP":
        optimizer = ARPOptimizer(model_parameters,
                                 lr=config["learning_rate"],
                                 alpha=config["arp_optimizer_params"]["alpha"],
                                 mu=config["arp_optimizer_params"]["mu"],
                                 weight_decay=config["arp_optimizer_params"]["weight_decay"],
                                 clamp_G_min=config["arp_optimizer_params"]["clamp_G_min"],
                                 clamp_G_max=config["arp_optimizer_params"]["clamp_G_max"])
    else:
        raise ValueError(f"Unsupported optimizer_type: {config['optimizer_type']}")

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["scheduler_step_size"], gamma=config["scheduler_gamma"])
    criterion = nn.CrossEntropyLoss()

    # 3D visualization (adapted)
    def plot_vectors(vectors, targets, epoch, batch_idx, save_dir):
        if vectors is None or targets is None or vectors.size(0) == 0 or targets.size(0) == 0:
            print(f"Skipping plot for epoch {epoch} batch {batch_idx} due to empty input.")
            return
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        # Detach and move to CPU before converting to numpy
        vectors_np = vectors.detach().cpu().numpy()[:5] # Plot first 5 for brevity
        targets_np = targets.detach().cpu().numpy()[:5]

        # Ensure vectors are at least 3D for plotting
        if vectors_np.shape[1] < 3 or targets_np.shape[1] < 3:
            print(f"Skipping plot for epoch {epoch} batch {batch_idx}: vectors not at least 3D.")
            plt.close(fig)
            return

        for v_idx in range(min(5, vectors_np.shape[0])): # Iterate up to 5 or available vectors
            ax.quiver(0, 0, 0, vectors_np[v_idx,0], vectors_np[v_idx,1], vectors_np[v_idx,2], color='b', alpha=0.5, label="Transformed" if v_idx==0 else "")
            ax.quiver(0, 0, 0, targets_np[v_idx,0], targets_np[v_idx,1], targets_np[v_idx,2], color='r', alpha=0.5, label="Target" if v_idx==0 else "")
        
        ax.set_xlim([-1.5, 1.5]) # Adjusted limits for better visibility
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.5, 1.5])
        ax.set_xlabel('e1')
        ax.set_ylabel('e2')
        ax.set_zlabel('e3')
        ax.legend()
        plt.title(f'Epoch {epoch}, Batch {batch_idx}')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Ensure the visualization directory exists (already created globally)
        # os.makedirs(save_dir, exist_ok=True) # Redundant if created at start
        save_path = os.path.join(save_dir, f'viz_epoch_{epoch}_batch_{batch_idx}_{timestamp}.png')
        try:
            plt.savefig(save_path)
        except Exception as e:
            print(f"Error saving plot {save_path}: {e}")
        plt.close(fig)

    # Training and evaluation loop
    global_step = 0
    for epoch in range(config["num_epochs"]):
        spinor_layer_module.memory = {i: [] for i in range(config["spinor_layers_params"]["num_classes"])} # Reset memory
        
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        epoch_start_time = time.time()

        feature_extractor.train()
        class_embed.train()
        classifier.train()

        for i, (inputs, labels) in enumerate(trainloader, 0):
            batch_start_time = time.time()
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()

            # Forward pass
            features = feature_extractor(inputs)       # (batch_size, 5)
            target_vectors = class_embed(labels)     # (batch_size, 5)

            # Spinor transformation (using the modularized SpinorLayer)
            # Ensure features and labels are correctly formatted for spinor_layer_module.forward
            transformed_features = spinor_layer_module.forward(features, labels) # (batch_size, 5)
            
            logits = classifier(transformed_features)  # (batch_size, num_classes)
            
            loss = criterion(logits, labels)
            loss.backward() # Computes gradients for parameters in optimizer
            optimizer.step()  # Updates parameters in optimizer

            # Update SpinorLayer rotors (manual update, not part of PyTorch autograd for rotors)
            # Ensure features, target_vectors, labels are correct for update_rotors
            spinor_layer_module.update_rotors(features.detach(), target_vectors.detach(), labels, learning_rate=0.05) # Example LR for rotors

            # --- ARP/RealignR Logging Integration ---
            if config["optimizer_type"] == "ARP":
                log_arp_metrics(writer, optimizer, global_step)
                if 'prev_loss' not in locals():
                    prev_loss = loss.item()
                prev_loss = log_loss_slope(writer, loss.item(), prev_loss, global_step)
            # Metrics
            running_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            # Logging
            if (i + 1) % 20 == 0: # Log every 20 batches
                current_loss = running_loss / 20
                current_train_acc = 100 * correct_train / total_train
                batch_time = time.time() - batch_start_time
                
                print(f'Epoch [{epoch+1}/{config["num_epochs"]}], Batch [{i+1}/{len(trainloader)}], '
                      f'Loss: {current_loss:.4f}, Acc: {current_train_acc:.2f}%, Batch Time: {batch_time:.2f}s, '
                      f'LR: {scheduler.get_last_lr()[0]:.6f}, Mem: {psutil.virtual_memory().percent}%')
                
                writer.add_scalar('Loss/train_batch', current_loss, global_step)
                writer.add_scalar('Accuracy/train_batch', current_train_acc, global_step)
                writer.add_scalar('LearningRate', scheduler.get_last_lr()[0], global_step)

                # Log ARP specific parameters if using ARP optimizer
                if config["optimizer_type"] == "ARP":
                    for grp_idx, group in enumerate(optimizer.param_groups):
                        if 'G' in optimizer.state[group['params'][0]]: # Check if G is initialized
                             # Log mean G for the first parameter in the group as a sample
                            writer.add_scalar(f'ARP/G_mean_group{grp_idx}', optimizer.state[group['params'][0]]['G'].mean().item(), global_step)
                        writer.add_scalar(f'ARP/alpha_group{grp_idx}', group['alpha'], global_step)
                        writer.add_scalar(f'ARP/mu_group{grp_idx}', group['mu'], global_step)
                
                # Log transformed features and target vectors for visualization
                if transformed_features.size(1) >=3 and target_vectors.size(1) >=3 : # Ensure they are at least 3D
                    plot_vectors(transformed_features.data, target_vectors.data, epoch + 1, i + 1, config["visualization_dir"])

                writer.flush() # Flush after batch logging

                running_loss = 0.0
                correct_train = 0 # Reset for next logging interval, or calculate epoch accuracy separately
                total_train = 0

            global_step += 1

        scheduler.step() # Step the LR scheduler at the end of each epoch

        # Epoch summary
        epoch_duration = time.time() - epoch_start_time
        # Calculate epoch accuracy properly if not reset above
        # For now, using last batch accuracy as a proxy or re-calculate on a subset.
        # A full pass over training data for epoch accuracy can be slow.
        print(f"Epoch {epoch+1} completed in {epoch_duration:.2f}s.")

        # Checkpoint at the end of each epoch
        checkpoint_path = os.path.join(config["checkpoint_dir"], f'model_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'feature_extractor_state_dict': feature_extractor.module.state_dict() if isinstance(feature_extractor, nn.DataParallel) else feature_extractor.state_dict(),
            'class_embed_state_dict': class_embed.module.state_dict() if isinstance(class_embed, nn.DataParallel) else class_embed.state_dict(),
            'classifier_state_dict': classifier.module.state_dict() if isinstance(classifier, nn.DataParallel) else classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            # 'spinor_layer_rotors': spinor_layer_module.rotors, # Save rotors if needed
            'loss': loss.item(), # Last batch loss
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")


        # --- Rotor Memory Heatmap Visualization ---
        plot_rotor_memory_heatmap(spinor_layer_module, epoch+1, config["visualization_dir"])

        # Evaluation at the end of each epoch
        feature_extractor.eval()
        class_embed.eval()
        classifier.eval()
        
        correct_test = 0
        total_test = 0
        test_loss = 0
        
        with torch.no_grad():
            for inputs_test, labels_test in testloader:
                inputs_test, labels_test = inputs_test.to(device), labels_test.to(device)
                
                features_test = feature_extractor(inputs_test)
                # target_vectors_test = class_embed(labels_test) # Not strictly needed for eval if not updating rotors
                
                transformed_features_test = spinor_layer_module.forward(features_test, labels_test) # Pass labels_test
                logits_test = classifier(transformed_features_test)
                
                loss_test_batch = criterion(logits_test, labels_test)
                test_loss += loss_test_batch.item()
                
                _, predicted_test = torch.max(logits_test.data, 1)
                total_test += labels_test.size(0)
                correct_test += (predicted_test == labels_test).sum().item()
                
        avg_test_loss = test_loss / len(testloader)
        test_accuracy = 100 * correct_test / total_test
        print(f'Epoch {epoch+1} Test Accuracy: {test_accuracy:.2f}%, Avg Test Loss: {avg_test_loss:.4f}')
        writer.add_scalar('Loss/test_epoch', avg_test_loss, epoch + 1) # Log per epoch
        writer.add_scalar('Accuracy/test_epoch', test_accuracy, epoch + 1) # Log per epoch
        writer.flush() # Flush after epoch test logging

    print('Finished Training and Evaluation')
    writer.close()

