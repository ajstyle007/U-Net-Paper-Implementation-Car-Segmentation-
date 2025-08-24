# Car Image Segmentation using U-Net (PyTorch & Flask)

I have implemented the original [U-Net research paper](https://arxiv.org/pdf/1505.04597) using PyTorch and trained this model on a car dataset. Visit [Live App](https://musk12-car-segmentation-mask.hf.space/)

<img width="1000" height="500" alt="unet_arch" src="https://github.com/user-attachments/assets/0050f15d-3ce9-42b7-8fec-c141db5b1375" />

Initially, the original U-Net architecture without padding did not perform well on car images (originally designed for biomedical data). I trained the model on 5k car images and their masks, but results were suboptimal. To improve performance:

- I applied data augmentation, increasing the dataset to 25k images and masks.
- Added padding = 1 in the U-Net architecture to preserve spatial dimensions.

After these modifications, the model achieved excellent performance from scratch, without using any pretrained weights:

- Validation Dice Score: 0.9900
- Validation IoU: 0.9804

## Frontend Web App

I created a Flask-based web frontend using HTML and CSS to interact with the model.

<img width="1331" height="817" alt="Screenshot 2025-08-24 155157" src="https://github.com/user-attachments/assets/8fce91c5-abc8-432e-b550-54a8a07cc211" />

### Features:
- Upload a car image.
- Generate the segmentation mask of the car.
- Create an overlay showing the car mask on the original image.

### Usage:
1. Run the Flask app:
` python app.py `
2. Open [http://127.0.0.1:5000/](https://musk12-car-segmentation-mask.hf.space/) in your browser.
3. Upload an image and click the button to generate the mask and overlay.

### Key Learnings

This project helped me understand:

- How to implement a research paper from scratch.
- Training deep learning models for image segmentation.
- Handling dataset limitations and improving model performance with augmentation and architecture tweaks.
- Deploying a model with a user-friendly web interface.

### Time Spent Training the Model: ~6–7 hours

### Requirements

- Python 3.8+
- PyTorch
- Torchvision
- Flask
- Pillow
- NumPy

  This is a full learning implementation of U-Net trained from scratch on car images, demonstrating both deep learning and deployment skills.


