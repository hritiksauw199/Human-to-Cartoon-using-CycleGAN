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
![WhatsApp Image 2025-02-13 at 14 45 15_b01c840b](https://github.com/user-attachments/assets/6bf52d84-b0d2-4959-8f61-a2e39311eaa3)

### 80 epoch<br>
![WhatsApp Image 2025-02-13 at 14 45 14_cf5576e8](https://github.com/user-attachments/assets/a3a6d068-489a-4799-bbfc-f817a2b4c554)

### Failed to match<br>
![WhatsApp Image 2025-02-13 at 14 45 14_430475c2](https://github.com/user-attachments/assets/e37342f6-ec45-4e61-b88b-3c65ad84d9b2)

### with glasses<br>
![WhatsApp Image 2025-02-13 at 14 45 31_2a7beda5](https://github.com/user-attachments/assets/babda59c-e898-49a1-b4c8-86394cee6eb5)
---
## 📌 Results
### 4 Residual Blocks
![WhatsApp Image 2025-02-13 at 13 45 21_811b7056](https://github.com/user-attachments/assets/3cded4b5-46e3-4f92-b115-3a643946c9c7)


### 6 Residual Blocks
![WhatsApp Image 2025-02-13 at 13 45 22_d998ec46](https://github.com/user-attachments/assets/f067d1f0-db5e-4e62-b4e2-3ef4d6185316)


![WhatsApp Image 2025-02-13 at 13 48 59_5baa95e1](https://github.com/user-attachments/assets/64e95c47-15ac-486f-b9bb-e1a93093b5ec)
![WhatsApp Image 2025-02-13 at 13 48 59_bf649436](https://github.com/user-attachments/assets/9ef56779-7f17-4da8-87dd-1154c5e27dbf)

---
## 📌 Reference
https://www.reddit.com/r/node/comments/yd99nb/alternatives_to_ngrok/ (static domain)
https://ngrok.com/blog-post/free-static-domains-ngrok-users
https://stackoverflow.com/questions/63732353/error-could-not-build-wheels-for-opencv-python-which-use-pep-517-and-cannot-be
https://medium.com/imagescv/what-is-cyclegan-and-how-to-use-it-2bfc772e6195
https://abdulkaderhelwan.medium.com/how-to-train-a-deep-cyclegan-for-mobile-style-transfer-bd73a16bfc19
https://medium.com/@chilldenaya/cyclegan-introduction-pytorch-implementation-5b53913741ca
https://jonathan-hui.medium.com/gan-cyclegan-6a50e7600d7
https://www.youtube.com/watch?v=Gib_kiXgnvA
https://arxiv.org/pdf/1406.2661
https://jonathan-hui.medium.com/gan-whats-generative-adversarial-networks-and-its-application-f39ed278ef09
https://www.youtube.com/watch?v=5jziBapziYE
https://arxiv.org/pdf/1703.10593
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9587703
https://github.com/junyanz/CycleGAN?tab=readme-ov-file
https://github.com/rish-16/CycleGANsformer
https://github.com/lmtri1998/Face2Anime-using-CycleGAN
---
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


## 📌 Authors

This project was developed by:

- **Hritik Sauw**
- **Zabihullah Azimy**
- **Ayushi Chawade**
