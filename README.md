# statistical_methods_for_machine_learning_project
Image classification with Tensorflow

### Goal
Image classification with Neural Networks: Use Tensorflow 2 to train neural networks for the classification of fruit/vegetable types based on images from https://www.kaggle.com/moltean/fruits. Images must be transformed from JPG to RGB pixel values and scaled down (e.g., 32x32). Use fruit/vegetable types (as opposed to variety) as labels to predict and consider only the 10 most frequent types (apple, banana, plum, pepper, cherry, grape, tomato, potato, pear, peach). Experiment with different network architectures and training parameters documenting their influence of the final predictive performance. While the training loss can be chosen freely, the reported test errors must be measured according to the zero-one loss for multiclass classification.
If you are not familiar with neural networks for image classification, take one of the many tutorials available in Tensorflow.

### Experiment 1
Datasets are prepared using the images in the "fruits-360" folder and they are stored inside "dataset_folder". Then different models are trained and evaluated. The models are created by the script "models.py". Results are saved in csv and json formats inside the "results" folder. The experiments are run using python 3.8.

#### Reproduce the experiment 1
* Clone this repository
* Download the fruits dataset from https://www.kaggle.com/moltean/fruits
* Unzip the downloaded dataset
* Move the folder "fruits-360" at the root of this repository
* Run the experiment with `python experiment_1.py`

If you are on linux and you want to record stdout and stderr, you may want to run it with `python experiment_1.py |& tee output.txt`

Keep in min that this is a long experiment and can take up to several hours to complete.

### Dataset
Make sure to download the dataset from https://www.kaggle.com/moltean/fruits, unzip it and save the "fruits-360" folder at the root of this repository.
When the experiment is run. Datasets are automatically prepared and saved inside the "dataset_folder". This makes future runs faster since it is just a matter of loading pregenerated data.

### Results
Results of the experiment are saved inside the results folder.
The file "results.csv" contains the overall findings for all models.
Files named "{input_size}\_{type}\_{depth}\_{size}.json" contain the training history of a specific model.
