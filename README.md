# Human Face to Cartoon Conversion using Reduced CycleGAN

## 📌 Motivation & Reasoning
We all have an innate desire to visualize ourselves as fictional characters, whether in movies, cartoons, or fantasy worlds. The concept of transforming real-life images into artistic representations intrigued us, pushing us to explore this creative domain. 

Initially, we set out to generate LEGO-style versions of images, but after facing multiple challenges and technical limitations in producing structured and realistic outputs, we had to reconsider our approach. 

Determined to find a better solution, we pivoted to cartoon-style transformations. This shift allowed us to refine our model, improve learning mechanisms, and achieve more meaningful results while still upholding the essence of artistic reimagination. 

## 📌 Understanding GANs and CycleGAN
Generative Adversarial Networks (GANs) are a class of deep learning models that consist of two neural networks—a generator and a discriminator—competing against each other. The generator attempts to create realistic images, while the discriminator evaluates them against real samples, leading to continuous improvements in quality. Traditional GANs require paired data, where input-output mappings are clearly defined. This necessity makes them ideal for applications where exact transformations between images are available.

![](images/gan1.png)
![](images/gan2.png)

CycleGAN, on the other hand, is designed to work with unpaired data, making it particularly useful for tasks where finding corresponding image pairs is impractical. Unlike a standard GAN, CycleGAN consists of two generator-discriminator pairs that learn bidirectional mappings between two domains. It introduces a cycle consistency loss, which ensures that an image transformed from domain A to domain B and back to A remains unchanged. This mechanism helps maintain realistic translations, making CycleGAN highly effective for style transfer applications such as converting real-world images into cartoons. 

In our project, we leverage CycleGAN to seamlessly translate input images into their artistic representations while maintaining essential features. This approach enables us to generate high-quality, visually appealing transformations that stay true to the original content. 

![](images/cyclegan.png)

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


![](images/data%20cartoon%20sample.png)   
![](images/data%20human%20sample.png)
---
## 📌 Architecture


## Our Approach 
In our approach, we modified the CycleGAN architecture to enhance training speed and reduce memory usage, while maintaining output quality.<br>

To improve efficiency and performance, our implementation reduces the number of residual blocks from 6 to 4, speeding up training while maintaining quality. We also avoid using a learning rate scheduler to simplify training dynamics. Additionally, mixed-precision training with gradient scaling is employed to optimize GPU memory usage and accelerate computations. These enhancements contribute to a more stable and effective CycleGAN model, making it well-suited for high-quality image translation.

## Generator
![](images/generator.png)

## Discriminator
![](images/discriminator.png)
 
### Key differences include:
1. Residual Blocks: Reduced the number of residual blocks in the generator from 6 to 4 to simplify the model and speed up training.<br>

2. Learning Rate Scheduler: Omitted the learning rate scheduler from the original design to streamline the model.<br>

3. Mixed-Precision Training: Incorporated gradient scaling with mixed-precision training to improve efficiency, reduce memory usage, and handle larger models or datasets without exceeding GPU capacity.<br>

4. Discriminator: Using an 8×8 PatchGAN discriminator reduces computational cost, avoids global artifacts, and improves training efficiency by evaluating realism at the patch level rather than the whole image.


---

## Loss Function 

- **Criterion for GAN Loss:** Using Mean Squared Error (MSE) loss to measure the difference between generated images and real images.

- **Cycle Consistency Loss:** L1 Loss ensures that when the model maps a cartoon back to a face, it should resemble the original face.

- **Identity Loss:** Another L1 Loss to ensure identity preservation between images (e.g., when a face image is passed through the cartoon generator, it should stay consistent).

---
##  Optimizer

- **Adam Optimizer for Discriminators:** Optimizes both discriminators (D_cartoon and D_face) with learning rate 0.0002 and betas (0.5, 0.999) for smoother updates.

- **Adam Optimizer for Generators:** Optimizes both generators (G_cartoon and G_face) with the same parameters as discriminators for balanced updates.

- **Grad Scalers for Generators and Discriminators:** Uses AMP for faster training with less memory usage on GPUs by scaling gradients to avoid overflow during backpropagation.

---

## 📌 Results
### 4 Residual Blocks
![](images/4%20residual%20block%20output.png)


### 6 Residual Blocks
![](images/6%20residual%20block%20output.png)

## 📌 Loss Functions

### 4 RB Loss/Residual Plot
![](images/loss_plot_4_Residual.png)

### 6 RB Loss/Residual Plot
![](images/loss_plot_6_Residual.png)

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
## 📌 References

https://www.reddit.com/r/node/comments/yd99nb/alternatives_to_ngrok/ (static domain) <br>
https://ngrok.com/blog-post/free-static-domains-ngrok-users <br>
https://stackoverflow.com/questions/63732353/error-could-not-build-wheels-for-opencv-python-which-use-pep-517-and-cannot-be <br>
https://medium.com/imagescv/what-is-cyclegan-and-how-to-use-it-2bfc772e6195 <br>
https://abdulkaderhelwan.medium.com/how-to-train-a-deep-cyclegan-for-mobile-style-transfer-bd73a16bfc19 <br>
https://medium.com/@chilldenaya/cyclegan-introduction-pytorch-implementation-5b53913741ca <br>
https://jonathan-hui.medium.com/gan-cyclegan-6a50e7600d7 <br>
https://www.youtube.com/watch?v=Gib_kiXgnvA <br>
https://arxiv.org/pdf/1406.2661 <br>
https://jonathan-hui.medium.com/gan-whats-generative-adversarial-networks-and-its-application-f39ed278ef09 <br>
https://www.youtube.com/watch?v=5jziBapziYE <br>
https://arxiv.org/pdf/1703.10593 <br>
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9587703 <br>
https://github.com/junyanz/CycleGAN?tab=readme-ov-file <br>
https://github.com/rish-16/CycleGANsformer <br>
https://github.com/lmtri1998/Face2Anime-using-CycleGAN <br>

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
