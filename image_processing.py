#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import imageio.v2 as imageio
import numpy as np


# In[2]:


import cv2
from PIL import Image
from IPython.display import display


# In[3]:


rgbImage = imageio.imread('shed1-small.jpg')


# In[4]:


(m,n,o) = rgbImage.shape
print(m,n,o)


# In[5]:


#normalize the image and display
from skimage import img_as_float
floatImage2 = img_as_float(rgbImage)
plt.imshow(floatImage2)
plt.show()
print(floatImage2[0:5, 0:5, 0])


# In[6]:


# Extract color channels.
R = rgbImage[:,:,0] # Red channel
G = rgbImage[:,:,1] # Green channel
B = rgbImage[:,:,2] # Blue channel


# In[7]:


# Create an all black channel.
allBlack = np.zeros((m, n), dtype=np.uint8)
# Create color versions of the individual color channels.
RC = np.stack((R, allBlack, allBlack), axis=2)
GC = np.stack((allBlack, G, allBlack),axis=2)
BC = np.stack((allBlack, allBlack, B),axis=2)


# In[8]:


# Recombine the individual color channels to create the original RGB image again.
recombinedRGBImage = np.stack(( R, G, B),axis=2)
plt.title('Recombined Image')
plt.imshow(recombinedRGBImage)
plt.axis('off')


# In[9]:


# Display RC, GC, BC
plt.figure(figsize=(14, 8))

plt.subplot(221)
plt.title('Red Channel')
plt.imshow(RC)
plt.axis('off')

plt.subplot(222)
plt.title('Green Channel')
plt.imshow(GC)
plt.axis('off')

plt.subplot(223)
plt.title('Blue Channel')
plt.imshow(BC)
plt.axis('off')


plt.tight_layout()
plt.show()


# In[10]:


# Calculate the gray-level image AG by averaging the RGB channels
AG = np.mean(rgbImage, axis=2).astype(np.uint8)
imageio.imsave('AG.jpg' , AG)


# Display AG
plt.imshow(AG, cmap='gray')
plt.title('Gray-Level Image AG')
plt.axis('off')
plt.show()


# In[11]:


# Compute histograms for RC, GC, and BC
hist_RC, bins_RC = np.histogram(R, bins=256, range=(0, 256))
hist_GC, bins_GC = np.histogram(G, bins=256, range=(0, 256))
hist_BC, bins_BC = np.histogram(B, bins=256, range=(0, 256))
hist_AG, bins_AG = np.histogram(AG, bins=256, range=(0, 256))

# Plot and display histograms
plt.figure(figsize=(12, 8))

plt.subplot(221)
plt.plot(hist_RC, color='red')
plt.title('Histogram of Red Channel')
plt.xlabel('Pixel Brightness')
plt.ylabel('Frequency')

plt.subplot(222)
plt.plot(hist_GC, color='green')
plt.title('Histogram of Green Channel')
plt.xlabel('Pixel Brightness')
plt.ylabel('Frequency')

plt.subplot(223)
plt.plot(hist_BC, color='blue')
plt.title('Histogram of Blue Channel')
plt.xlabel('Pixel Brightness')
plt.ylabel('Frequency')

plt.subplot(224)
plt.plot(hist_AG, color='gray')
plt.title('Histogram of AG Channel')
plt.xlabel('Pixel Brightness')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# In[12]:


# Load the gray-level image AG 
ag_image_path = 'AG.jpg'
AG = cv2.imread(ag_image_path, cv2.IMREAD_GRAYSCALE)

# Prompt the user to enter the threshold brightness (TB)
TB = int(input("Enter the threshold brightness (e.g., TB=100): "))

# Convert the image to a NumPy array
AG_array = np.array(AG)

# All pixels less than TB are assigned brightness 0 (black) and others are assigned brightness 255 (white)
AB_array = np.where(AG_array < TB, 0, 255)

# Create a new PIL image from the binarized NumPy array
AB = Image.fromarray(AB_array.astype(np.uint8))

# Displaying AB
plt.imshow(AB, cmap='gray')
plt.title('Binary Image AB')
plt.axis('off')
plt.show()


# In[13]:


# Loading the gray-level image AG 
ag_image_path = 'AG.jpg'
AG = cv2.imread(ag_image_path, cv2.IMREAD_GRAYSCALE)
AG = AG.astype(np.int64)  # Convert AG to int32

# Prompt the user to enter the threshold value TE
TE = int(input("Enter the threshold value (e.g., TE=15): "))

height, width = AG.shape

# Initializing arrays for Gx and Gy
Gx = np.zeros_like(AG, dtype="int64")
Gy = np.zeros_like(AG, dtype="int64")


# Calculating Gx manually and handle the edge case
for m in range(height):
    for n in range(width - 1):
        Gx[m, n] = AG[m, n + 1] - AG[m, n]
    # Set Gx to zero for the final pixel in each row
    Gx[m, width - 1] = 0

# Calculating Gy manually and handle the edge case
for m in range(height - 1):
    for n in range(width):
        Gy[m, n] = AG[m + 1, n] - AG[m, n]
    
# Set Gy to zero for the final pixel in each column
Gy[height - 1, :] = 0

# Calculating the gradient magnitude GM
GM = np.sqrt(Gx**2 + Gy**2)
GM = np.clip(GM, 0, 255).astype(np.uint8)

# Threshold GM to compute the edge image AE
AE = np.where(GM > TE, 255, 0).astype(np.uint8)

# Displaying AE
plt.imshow(AE, cmap='gray')
plt.title('Edge Image AE')
plt.axis('off')
plt.show()


# In[19]:


# Define a recursive function to build the image pyramid
def build_image_pyramid(image, levels):
    # Base case: If we've reached the desired number of levels, return the image
    if levels == 0:
        return [image]

    # Initialize an empty list to store downsampled images
    downsampled_images = []

    # Downsample the image by taking the average brightness of 2x2 blocks
    for i in range(0, image.shape[0], 2):
        row_avg = []
        for j in range(0, image.shape[1], 2):
            block = image[i:i+2, j:j+2]
            avg_brightness = np.mean(block)
            row_avg.append(avg_brightness)
        downsampled_row = np.array(row_avg, dtype=np.uint8)
        downsampled_images.append(downsampled_row)

    downsampled_image = np.array(downsampled_images, dtype=np.uint8)

    # Recursively build the rest of the pyramid
    lower_pyramid = build_image_pyramid(downsampled_image, levels - 1)

    # Combine the current level and the lower pyramid
    return [image] + lower_pyramid

# Load the gray-level image AG (replace 'AG.jpg' with your image file)
AG = cv2.imread('AG.jpg', cv2.IMREAD_GRAYSCALE)

# Define the number of pyramid levels
num_levels = 3

# Build the image pyramid
pyramid = build_image_pyramid(AG, num_levels)

# Create a figure to display the images
plt.figure(figsize=(16, 4))

# Display the first three levels of the image pyramid
for i in range(num_levels):
    plt.subplot(1, num_levels, i+1)
    plt.imshow(pyramid[i], cmap='gray', extent=[0, pyramid[i].shape[1], 0, pyramid[i].shape[0]])
    plt.title(f'Level {i + 1}')
    plt.axis('off')

# Assign AG2, AG4, and AG8 to the corresponding pyramid levels
AG2 = pyramid[1]
AG4 = pyramid[2]
AG8 = pyramid[3]

# Display the resulting images
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(AG2, cmap='gray', extent=[0, AG2.shape[1], 0, AG2.shape[0]])
plt.title('AG2 (1/2)')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(AG4, cmap='gray', extent=[0, AG4.shape[1], 0, AG4.shape[0]])
plt.title('AG4 (1/4)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(AG8, cmap='gray', extent=[0, AG8.shape[1], 0, AG8.shape[0]])
plt.title('AG8 (1/8)')
plt.axis('off')

# Adjust spacing between subplots
plt.tight_layout()
plt.show()


# In[ ]:




