The train_and_validate.py program consists of 4 models:
1.	XGBoost
2.	K Nearest Neighbor (KNN)
3.	Random Forest (RF)
4.	Support Vector Machine (SVM)

Running the program in its current state will produce the run that gave me the highest Accuracy and F-score. I had two models virtually tie for the top spot, 
those being XGBoost and SVM. The program should take less than a minute to run and will output the models’ measurements in a text file named “console_output.txt” 
in the folder. 

I tried various methods and combinations of data input before getting my final models. The other methods that I used that didn’t preform as well are in my code 
and can be accessed by uncommenting some code or changing flags in the main method. Some examples of the other methods I used were grey scale feature extraction, 
creating an NDVI mask to cover up vegetation and body of water signals in the images, and exploring the different bands to see which ones produced the best model outputs.
I also commented out some k-fold validation metrics as well as they were overextending the run time of my code. Feel free to uncomment these metrics.   

I would improve on my model in a few different ways:
1.	I would find out more information about these types of power plants and use that information to improve my model. For example, I noticed on the excel sheet 
      that around 64 of the stations showed that some of the power plants were generating power at less than 5 percent of their capacity. I included these stations in the 
      "On” category, but I’d be interested in doing more research into this to see if there is a threshold where a station below a certain utilization point is considered 
       on standby or wouldn’t be producing vapor. 

2.	I would explore more advanced kinds of image feature extraction such as Color Channel Statistics, Harlick texture, or Histogram of Oriented Gradients (HOG). 

3.	I would try acquiring more training data and use neural networks such as the Convolutional Neural Network (CNN)) as it is known for being very accurate with image 
      classification and detection. 

4.	I would continue experimenting with what data was being put in the model. For instance, I think it would be interesting to find a way to incorporate wind direction 
      to see if a model could pick up on the direction of the vapor plume. 


Please reach out to me if you have any questions, and thanks again for the opportunity!