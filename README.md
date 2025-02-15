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
## 📌 Our Approach 
In our approach, we modified the CycleGAN architecture to enhance training speed and reduce memory usage, while maintaining output quality.<br>
 
### Key differences include:
Residual Blocks: Reduced the number of residual blocks in the generator from 6 to 4 to simplify the model and speed up training.<br>

Learning Rate Scheduler: Omitted the learning rate scheduler from the original design to streamline the model.<br>

Mixed-Precision Training: Incorporated gradient scaling with mixed-precision training to improve efficiency, reduce memory usage, and handle larger models or datasets without exceeding GPU capacity.<br>

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


![anime](https://github.com/user-attachments/assets/6ad4396e-0308-4369-a821-752e3b7cc3c8)   ![human](https://github.com/user-attachments/assets/9ba72763-b53c-48c7-926e-9d7413ff74d2)
---
## 📌 Architecture
CycleGAN operates through a cyclic transformation process, where images are converted between two domains—human and cartoon—without requiring paired datasets. It consists of two generators: one that translates human images into cartoon-style images and another that reverses the process, converting cartoons back into realistic human-like images. Alongside these generators, there are two discriminators, each assessing whether an image belongs to the real dataset or has been artificially generated. The adversarial training between these components refines the quality of outputs and ensures that generated images closely resemble the target domain.

A fundamental aspect of CycleGAN is its **cycle consistency loss**, which enforces the idea that an image should retain its essential features when transformed from one domain to another and then back again. This bidirectional consistency helps prevent mode collapse and unwanted distortions. The residual blocks within the generators play a crucial role in feature extraction by capturing high-level patterns in the input images. These blocks allow the model to preserve fine details and textures, leading to more realistic and visually coherent transformations.

To improve efficiency and performance, our implementation reduces the number of residual blocks from 6 to 4, speeding up training while maintaining quality. We also avoid using a learning rate scheduler to simplify training dynamics. Additionally, mixed-precision training with gradient scaling is employed to optimize GPU memory usage and accelerate computations. These enhancements contribute to a more stable and effective CycleGAN model, making it well-suited for high-quality image translation.

## 📌 Generator
![generator png](https://github.com/user-attachments/assets/3e95ccab-a6e0-4d9f-9d7a-2ba695c407d9)


## Discriminator
![decoder png](https://github.com/user-attachments/assets/2ddb5aea-4d65-47ac-a45a-2b6d8a0a1b8e)

---

## 📌 Loss Function 

**Criterion for GAN Loss:** Using Mean Squared Error (MSE) loss to measure the difference between generated images and real images.<br>

**Cycle Consistency Loss:** L1 Loss ensures that when the model maps a cartoon back to a face, it should resemble the original face.<br>

**Identity Loss:** Another L1 Loss to ensure identity preservation between images (e.g., when a face image is passed through the cartoon generator, it should stay consistent)<br>
---
## 📌 Optimizer

**Adam Optimizer for Discriminators:** Optimizes both discriminators (D_cartoon and D_face) with learning rate 0.0002 and betas (0.5, 0.999) for smoother updates.

**Adam Optimizer for Generators:** Optimizes both generators (G_cartoon and G_face) with the same parameters as discriminators for balanced updates.

**Automatic Mixed Precision:**
**Grad Scalers for Generators and Discriminators:** Uses AMP for faster training with less memory usage on GPUs by scaling gradients to avoid overflow during backpropagation.
---
## 📌 Challenges
### with weights<br>
![WhatsApp Image 2025-02-13 at 14 45 15_c72aee96](https://github.com/user-attachments/assets/926184fb-f390-4e9b-bfaa-cb683f71d1a4)

### using linear schedular<br>
![WhatsApp Image 2025-02-13 at 14 45 15_c72aee96](https://github.com/user-attachments/assets/a7772bf7-6348-49ef-957b-ddbb1bd4527e)

### 80 epoch<br>

### Failed to match<br>

### with glasses<br>



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
