# deep-image-prior
TensorFlow implementation for Deep Image Prior (https://dmitryulyanov.github.io/deep_image_prior)

The paper talks about using large neural networks like ResNet. Due to resource constraints, the networks we use are small. The network was tested for image inpainting, pattern inpainting, and image denoising. Image denoising required early stopping and hyperparameter tuning, and therefore is not perfect.

Pattern inpainting works well on simple patterns, like the following.

![Pattern with mask](images/1/input_image.png) ![Pattern inpainted](images/1/sample_5200.png)

On more complicated patterns like the one given below, it does not perform really well.

![Pattern with mask](images/2/input_image.png) ![Pattern inpainted](images/2/sample_4900.png)

Image inpainting requires larger networks. Results on a pair of images have been given below.
Ground truth and inpainted result.

![Ground truth](images/3/input_image.jpg) ![Image inpainted](images/3/sample_19900.png)

Deep image prior understands the overall "texture" of an image and creates a function mapping
from the noise space to the RGB space. To check this, an experiment was done where input noise
patterns were sampled according to a mean and some standard deviation. Prior to this, a network
was trained to take this mean noise vector as input and inpaint the above example (zebra in grass).
The results are shown below.

![sample 1](images/noise_samples/sample_96.png) ![sample 2](images/noise_samples/sample_97.png)
![sample 3](images/noise_samples/sample_98.png) ![sample 4](images/noise_samples/sample_99.png)

Removing overlay text is a form of inpainting. The result are given below.

![input](images/4/input_image.png) ![inpainted](images/4/sample_9900.png)

Results on Denoising: Gaussian random noise (stddev = 25) was added using np.random.normal().
The network was trained for 1700 optimizations (the original network was trained for 1800 steps).
The results are given below: 1. Input 2. Denoised (after 1700 iterations) 3. Averaged (last 5)

![input_image](images/8/input_image.png) ![sample 1700](images/8/sample_1700.png) ![average](images/8/average.png)

Deep image prior was also tried on depixelating, which is a form of SR. The results are given 2x depixelation and 4x depixelation.

![input](images/6/input_image.png) ![depixelated](images/6/sample_9900.png)
![input](images/7/input_image.png) ![depixelated](images/7/sample_9900.png)
