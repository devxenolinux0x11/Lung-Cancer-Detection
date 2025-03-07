# Lung-Cancer-Detection-Using-Hybrid-Vision-Transformer-with-Clinical-Feature-Fusion
CliniScan Transformer is a deep learning framework that combines CT scan imaging and clinical patient data to predict lung cancer malignancy.

# Hybrid Lung Cancer Detection Project

Hey there! Welcome to my project on lung cancer detection. This is something I’ve been working on to combine the power of CT scan images and clinical patient data to predict whether a lung nodule might be cancerous or not. I built this using a mix of deep learning (think Vision Transformers and custom attention layers) and some solid data preprocessing to make it all work together. If you’re into medical imaging, machine learning, or just curious about how tech can help in healthcare, you’re in the right place!

## What’s This All About?

This project is a deep learning framework designed to classify lung CT scans as either "cancerous" or "non-cancerous." It doesn’t just stop at images, though—I’ve added clinical features like age, smoking history, and nodule size to give the model a fuller picture. The idea is to mimic how a doctor might look at both a scan and a patient’s history to make a call. The result? A hybrid model that’s pretty good at spotting patterns and making predictions.

Here’s the gist:
- **CT Scans**: Preprocessed and fed into a Vision Transformer (ViT) backbone.
- **Clinical Data**: Things like smoking habits and nodule density, normalized and paired with the images.
- **Model**: A custom `LungCancerNet` with an attention module to focus on what matters.
- **Goal**: Predict lung cancer with high accuracy and provide probabilities you can trust.

## Features

- **Data Generation**: Creates synthetic clinical data (e.g., `generate_clinical_data`) if you don’t have a dataset handy.
- **Preprocessing**: Handles CT scan-specific quirks like Hounsfield Units (HU) clipping and normalization.
- **Model Training**: Trains on both images and clinical features, with validation and test splits.
- **Prediction Tool**: A `LungCancerPredictor` class for real-world use—just give it an image and some patient info!
- **Performance**: Tracks metrics like accuracy, loss, and AUC to see how well it’s doing.

## Getting Started

Want to try it out? Here’s how to get it running on your machine.

### Prerequisites

You’ll need a few things installed:
- Python 3.8+
- PyTorch (`torch`, `torchvision`)
- MONAI (`monai.transforms` for medical imaging)
- OpenCV (`opencv-python`)
- Pandas, NumPy, Scikit-learn, and Pillow

Install them with:
```bash
pip install torch torchvision monai opencv-python pandas scikit-learn pillow
```

If you’re avoiding GPU conflicts, you might also want:
```bash
pip uninstall -y tensorflow && pip install tensorflow-cpu
```

### Setup

1. **Clone the repo**:
   ```bash
   git clone https://github.com/devxenolinux0x11/Lung-Cancer-Detection-Using-Hybrid-Vision-Transformer-with-Clinical-Feature-Fusion.git
   cd lung-cancer-detection
   ```

2. **Prepare your data**:
   - Unzip a `data.zip` file with CT scans in `data/cancerous` and `data/non-cancerous` folders, or let the code generate synthetic clinical data for you.
   - Run `create_project_structure()` to set up directories if you’re starting fresh.

3. **Train the model**:
   - Run the `main()` function in the first script to train and save the best model (`best_model.pth`).

4. **Make predictions**:
   - Use the second `main()` function with a trained model, a CT scan image, and clinical inputs to get a prediction.

### Example

Train the model:
```python
python lung_cancer_train.py
```

Predict with a sample:
```python
python lung_cancer_predict.py
# Follow the prompts for an image path and clinical data
```

## How It Works

1. **Data Prep**: CT scans are clipped to [-1000, 400] HU, normalized, and turned into RGB images. Clinical data gets its own normalization too.
2. **Model**: A Vision Transformer (ViT-B/16) processes images, a custom attention layer (`HybridAttentionModule`) sharpens the focus, and a clinical feature network merges everything for a final prediction.
3. **Training**: Uses AdamW optimizer, cosine annealing, and cross-entropy loss over 50 epochs (tweakable!).
4. **Prediction**: Outputs a "Cancerous" or "Non-cancerous" label with confidence scores.

## Main Method

Here’s a peek at the `main()` function that ties it all together. This is a simplified version from the training script—check the full code for all the details!

```python
def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Pick your device (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set up project folders
    create_project_structure()

    # Generate clinical data if we don’t have it
    if not os.path.exists('clinical_data.csv'):
        generate_clinical_data('data', 'clinical_data.csv')

    # Load and split the dataset
    dataset = CTScanDataset(
        data_dir='data',
        clinical_data_path='clinical_data.csv',
        transform=get_transforms('train')
    )
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

    # Set up the model
    model = LungCancerNet().to(device)

    # Train it!
    train_model(model, train_loader, val_loader, device, num_epochs=50)

    # Test the best model
    model.load_state_dict(torch.load('best_model.pth'))
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, test_auc = evaluate_model(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, Test AUC: {test_auc:.4f}')

if __name__ == '__main__':
    main()
```

This `main()` kicks off the whole process—data prep, training, and testing. For predictions, check out the separate `LungCancerPredictor` script!

## Results

After training, you’ll see metrics like:
- **Test Accuracy**: How often it gets it right.
- **AUC**: How well it distinguishes cancer from non-cancer (closer to 1 is better).
- **Loss**: How much it’s “off” during training and validation.

Check the console output for details after each run!

## Future Ideas

I’m thinking of adding:
- Real dataset integration (e.g., LIDC-IDRI).
- More advanced attention mechanisms.
- A web interface for easier predictions.

Got suggestions? Feel free to open an issue or send me a message!

## Contributing

Love to have you on board! Fork the repo, make your changes, and submit a pull request. Whether it’s a bug fix, a new feature, or just better docs, I’d appreciate the help.

## License

This project is under the MIT License—feel free to use it, tweak it, or share it. Check out the `LICENSE` file for details.

## Acknowledgments

Big thanks to the PyTorch team, the MONAI folks, and all the open-source contributors who make projects like this possible. Also, shoutout to coffee for keeping me going!

Happy coding, and let’s make healthcare tech better together!

---

### Notes
- The `main()` method here is a condensed version of your training script’s `main()` function, simplified for readability in the README. It assumes the user has the full code in a separate file (e.g., `lung_cancer_train.py`).
- I’ve kept the tone casual and welcoming, while ensuring the technical details are clear.
- If you want the prediction script’s `main()` included too, or any other tweaks, just let me know!

