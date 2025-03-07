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
### Setup

1. **Clone the repo**:
   ```bash
   git clone https://github.com/devxenolinux0x11/Lung-Cancer-Detection.git
   cd lung-cancer-detection
   ```

2. **Prepare your data**:
   - Unzip a `data.zip` file with CT scans in `data/cancerous` and `data/non-cancerous` folders, or let the code generate synthetic clinical data for you.
   - Run `create_project_structure()` to set up directories if you’re starting fresh.

3. **Train the model**:
   - Run the `main()` function in the first script to train and save the best model (`best_model.pth`).

4. **Make predictions**:
   - Use the second `main()` function with a trained model, a CT scan image, and clinical inputs to get a prediction.

## How It Works

1. **Data Prep**: CT scans are clipped to [-1000, 400] HU, normalized, and turned into RGB images. Clinical data gets its own normalization too.
2. **Model**: A Vision Transformer (ViT-B/16) processes images, a custom attention layer (`HybridAttentionModule`) sharpens the focus, and a clinical feature network merges everything for a final prediction.
3. **Training**: Uses AdamW optimizer, cosine annealing, and cross-entropy loss over 50 epochs (tweakable!).
4. **Prediction**: Outputs a "Cancerous" or "Non-cancerous" label with confidence scores.

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

Happy coding, and let’s make healthcare tech better together!
