<h1> Methodology </h1>

<h2> Why PyTorch? </h2>

I ended up using PyTorch over Keras or TensorFlow (even though it would have made my life easier to use any of those two) to mostly comply with the academic requirements of the course.

PyTorch does allow you more room for any personal tweaks you want to add to a framework, though, so that's there.

<h2> Where's the Data From? </h2>

The Galaxy Zoo 2 dataset has been made available by the Galaxy Zoo project team (Willett et al. 2013). Still, for the sake of simplicity, the dataset was sourced from [Kaggle]([url](https://www.kaggle.com/datasets/jaimetrickz/galaxy-zoo-2-images)) so that it could be directly downloaded to the runtime, saving time and reducing human interaction while the programme is running. Similarly, the debiased data frame from Hart et al. (2016), along with a CSV file for mapping image names with asset IDs, was also downloaded. I didn't end up using anything else

<h2> Data Prep & Preprocessing </h2>

First, I merged the CSV files from Hart et al. (2016) and the filename mappings file, separately released by the Galaxy Zoo 2 team. I could see that while the filename mappings file had 355,990 entries, the Normal-depth sample with the new debiasing method file from Hart et al. (2016) had  239,694 entries. I found and removed 30,339 duplicates in the filename mappings file.

After conducting an inner join of the two data frames, I dropped 122 rows that did not have a corresponding image, resulting in the exclusion of 4,167 images from the image processing pipeline as they did not have a corresponding entry within the joined data frame. This procedure left behind 239,267 images. I then merged the super rare classes into one `class_reduced` column, and the joined data frame was saved to an SQLite database.

Lastly, to confirm the target classes, a feature distribution was plotted with a value threshold of greater than 0.5 within the columns. The resulting bar chart is depicted as follows:

<img width="1004" height="640" alt="image" src="https://github.com/user-attachments/assets/f6de4908-de12-4c24-bc3d-1e134aebe60b" />

Given the feature distribution, it can be determined that there are images that fit multiple class labels, making the problem a multi-label classification problem.

<h2> Data Processing </h2>

To process the filtered images, I found (can't remember where) and tweaked a function to apply several image augmentation techniques. It roughly looked like this after my changes:

•	Gaussian Blur: Smooth the image slightly to reduce noise.

•	Thresholding: Convert the image to binary to highlight key features.

•	Dilate: Expand the binary areas to close small gaps and better define the contours.

•	Contour Detection: Identify contours and find the one closest to the centre with an area above the specified threshold of 224*224 (keeping in line with the minimum dimensional requirements of ResNet50).

•	Bounding Box: Determine the bounding box around the selected contour.

•	Check Target Rectangles: Verify if the bounding box fits within predefined target rectangles.

•	Crop/Scale:
  o	Crop directly if the bounding box fits within the smallest target rectangle.
  o	Crop and then scale if it fits within the larger rectangle.
  o	Otherwise, scale the entire image.

The processing reduced the overall dataset’s dimensionality and the image size, bringing it down from 424*424 to 224*224, a reduction of about 47%. I then used thresholding, dilation, and contour detection to isolate the main object within the image, and then highlighted it using a bounding box. After verifying if the bounding box fit within the predefined target rectangles, I cropped/scaled the images, depending on whether the bounding box fit within the smallest target rectangle. The following is an example visual of the entire workflow of the image processing done here:

<img width="786" height="407" alt="image" src="https://github.com/user-attachments/assets/97bae952-64ae-45b9-86be-bd31933283b9" />

This workflow was applied to the entire dataset. Initially, I had a couple of other image transforms, like RandomHorizontalFlip and Rotate. I was aiming for a more robust dataset, but the processing took several hours (I used Colab and Google's TPU). I found a workaround: proactively apply the transforms during the training loop and process the images in grayscale. This took my average time to less than 2 minutes. 

After a quick sanity check (there are several of these in the code; my dissertation supervisor likes to be thorough), I stratified the data and ran the test-train split. I also moved the test and train images into two separate directories, but not before converting all images to PNG to further reduce image size. Finally, I loaded the images onto a NumPy array with a fixed size and defined the 'GalaxyDataset' dataset class, along with the data loaders for the models.
