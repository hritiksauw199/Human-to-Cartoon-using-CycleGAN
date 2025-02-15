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
CycleGAN consists of two generators and two discriminators working together to enable image transformation between domains. The first generator, responsible for converting human images into cartoon-style images, ensures that the generated outputs resemble the target cartoon domain. The second generator performs the inverse operation, converting cartoon images back into realistic human images. 

Each generator is accompanied by a discriminator that evaluates whether an image belongs to the real dataset or is a generated sample. The adversarial loss between these components helps refine the output quality, making the transformations more believable. 

The transformation occurs in cycles: a human image is first converted into a cartoon image using Generator A. The generated cartoon image is then passed through Generator B to reconstruct the original human image. The same cycle applies in reverse, where a cartoon image is transformed into a human image and then reconstructed back into a cartoon. This cycle ensures that important visual features are retained, making the transformations more stable and realistic.

A key component of CycleGAN is **cycle consistency loss**, which ensures that when an image is transformed from one domain to another and back, it remains close to its original form. This bidirectional mapping improves stability and prevents unwanted distortions in the generated images. Additionally, **residual blocks** play a crucial role in feature extraction, helping the network preserve important structural and style-related information while performing domain translation. 

To enhance performance, our model incorporates optimizations such as reducing the number of residual blocks from 6 to 4 for faster training and omitting a learning rate scheduler for simplicity. Additionally, we employ mixed-precision training with gradient scaling to efficiently utilize GPU memory and speed up computations. 

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
