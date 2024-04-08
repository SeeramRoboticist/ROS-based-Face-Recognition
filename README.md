# ROS based Face Recognition

## Training

Training the Model is simple follow the below steps,

1. open the file train.py,

```py
DEFAULT_ENCODINGS_PATH = Path("/home/sriram54/face_reg/output/encodings.pkl")
```
2. Change the above line and Provide proper path for where you want to store the Model.

3. Create a folder and inside the folder store the data i.e images of the respective person in folder with there name. (verify the training folder for furter clarification).

4. Once done go to the below line 

``` py
    for filepath in Path("/home/sriram54/face_reg/training").glob("*/*"):
```
5. Change the path to the folder you have Saved the Training Data.

6. Build this repo as a package in a ROS WS, source the WS and run the node.

7. Now you should be having, topics named - /train_control, /train_result.

8. Publish True the the topic called Train control using Terminal.

9. As you have started the process, wait for the training to get completed, once completed you will get "completed" or "done" in /train_result and the termina you are runing the script.

10. Thats it Training is Completed.

# Recognition

