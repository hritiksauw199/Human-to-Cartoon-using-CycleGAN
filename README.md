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
CycleGAN operates through a cyclic transformation process, where images are converted between two domains—human and cartoon—without requiring paired datasets. It consists of two generators: one that translates human images into cartoon-style images and another that reverses the process, converting cartoons back into realistic human-like images. Alongside these generators, there are two discriminators, each assessing whether an image belongs to the real dataset or has been artificially generated. The adversarial training between these components refines the quality of outputs and ensures that generated images closely resemble the target domain.

A fundamental aspect of CycleGAN is its **cycle consistency loss**, which enforces the idea that an image should retain its essential features when transformed from one domain to another and then back again. This bidirectional consistency helps prevent mode collapse and unwanted distortions. The residual blocks within the generators play a crucial role in feature extraction by capturing high-level patterns in the input images. These blocks allow the model to preserve fine details and textures, leading to more realistic and visually coherent transformations.

To improve efficiency and performance, our implementation reduces the number of residual blocks from 6 to 4, speeding up training while maintaining quality. We also avoid using a learning rate scheduler to simplify training dynamics. Additionally, mixed-precision training with gradient scaling is employed to optimize GPU memory usage and accelerate computations. These enhancements contribute to a more stable and effective CycleGAN model, making it well-suited for high-quality image translation.

---
---

*(More sections to be added later, such as Implementation, Technologies Used, Dataset, etc.)* 📌


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
