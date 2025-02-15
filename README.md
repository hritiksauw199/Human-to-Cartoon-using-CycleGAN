# Learning from Images

# Human-to-Cartoon-using-CycleGAN 

## 📌 Motivation & Reasoning
We all have an innate desire to visualize ourselves as fictional characters, whether in movies, cartoons, or fantasy worlds. The concept of transforming real-life images into artistic representations intrigued us, pushing us to explore this creative domain. 

Initially, we set out to generate LEGO-style versions of images, but after facing multiple challenges and technical limitations in producing structured and realistic outputs, we had to reconsider our approach. 

Determined to find a better solution, we pivoted to cartoon-style transformations. This shift allowed us to refine our model, improve learning mechanisms, and achieve more meaningful results while still upholding the essence of artistic reimagination. 

## 📌 Understanding GANs and CycleGAN
Generative Adversarial Networks (GANs) are a class of deep learning models that consist of two neural networks—a generator and a discriminator—competing against each other. The generator attempts to create realistic images, while the discriminator evaluates them against real samples, leading to continuous improvements in quality. Traditional GANs require paired data, where input-output mappings are clearly defined. This necessity makes them ideal for applications where exact transformations between images are available.

CycleGAN, on the other hand, is designed to work with unpaired data, making it particularly useful for tasks where finding corresponding image pairs is impractical. Unlike a standard GAN, CycleGAN consists of two generator-discriminator pairs that learn bidirectional mappings between two domains. It introduces a cycle consistency loss, which ensures that an image transformed from domain A to domain B and back to A remains unchanged. This mechanism helps maintain realistic translations, making CycleGAN highly effective for style transfer applications such as converting real-world images into cartoons. 

In our project, we leverage CycleGAN to seamlessly translate input images into their artistic representations while maintaining essential features. This approach enables us to generate high-quality, visually appealing transformations that stay true to the original content. 

---

## 📌 Dataset

This project uses two datasets:
 [Book Recommendation System](https://github.com/ayushichawade/Software-Engineering-Project-Book-Recommendation-System-/blob/main/UI/UI.pdf)
1. [**CartoonSet**](https://google.github.io/cartoonset/): A collection of 2D cartoon avatars with different artistic styles.
   - 500 images for training
   - 10 images for testing
2. [**Human Dataset**](https://www.kaggle.com/datasets/prasoonkottarathil/face-mask-lite-dataset): **Face Mask Lite Dataset** from Kaggle, containing real human face images.
   - 500 images without masks for training
   - 10 images for testing

### Data Augmentation:

- **White Background Removal**: Ensures consistency with transparent cartoon backgrounds.
- **Glasses Removal**: Prevents inconsistencies in translation.
- **Front-Facing Image Selection**: Uses only front-facing images to improve accuracy.

## 📌 Architecture

### CycleGAN consists of:

- **Generator A (Human → Cartoon)**: Converts human images into cartoon-style images.
- **Generator B (Cartoon → Human)**: Converts cartoon images back into human-like images.
- **Discriminator A & B**: Distinguish real human/cartoon images from generated ones.
- **Cycle Consistency Loss**: Ensures that transformations are reversible while preserving identity.

### Model Enhancements:

- **Reduced Residual Blocks**: Uses 4 instead of 6 to improve training speed.
- **No Learning Rate Scheduler**: Simplifies model training.
- **Mixed-Precision Training**: Uses gradient scaling for efficient GPU memory utilization.

## 🎯 Training

- **Optimizer:** Adam (LR: 0.0002, Betas: (0.5, 0.999))
- **Loss Functions:**
  - **GAN Loss**: Measures the realism of generated images.
  - **Cycle Consistency Loss**: Ensures reversibility of transformations.
  - **Identity Loss**: Preserves key features in transformations.
- **Training Setup:**
  - Runs for **30 epochs** with a batch size of **1**.
  - Saves the best model checkpoint based on **generator loss**.

## 🔥 Results

- The model successfully generates **cartoonized human faces** while maintaining facial features.
- Below is a sample output:

  ![Sample Output](./output/final_30_nlr_6rb.png)

## 🛠 Installation & Usage

### Prerequisites:

- Python 3.8+
- PyTorch
- torchvision
- Pillow
- tqdm
- matplotlib

### Installation:

```bash
git clone https://github.com/hritiksauw199/Human-to-Cartoon-using-CycleGAN.git
cd Human-to-Cartoon-using-CycleGAN
pip install -r requirements.txt
```

### Running the Model:

```bash
python cyclegan.py
```

## ⚠️ Challenges & Limitations

- **Unpaired Data**: No direct mapping between human and cartoon images.
- **Preserving Details**: Balancing stylization while maintaining identity.
- **Training Time**: Takes **4-5 hours per 50 epochs** on a 4GB GPU.
- **Glasses & Accessories**: Faces with glasses sometimes cause inconsistencies.

## 📚 References

- [CycleGAN Paper](https://arxiv.org/pdf/1703.10593)
- [CycleGAN GitHub](https://github.com/junyanz/CycleGAN)
- [Face2Anime CycleGAN](https://github.com/lmtri1998/Face2Anime-using-CycleGAN)

## 📌 Authors

This project was developed by:

- **Hritik Sauw**
- **Zabihullah Azimy**
- **Ayushi Chawade**
