The main idea: a person can classify different programming languages just by looking at an instance of code, without going into the details of the implementation of some function or even understanding what that code does. He can do this by reading some visual characteristics inherent in this language. Therefore, it is possible to make a neural network that acts similarly to a person and recognizes a picture.

In connection with this idea, I made a dataset.
1) Downloaded the available code files from GitHub. For the “OTHER” class, I downloaded from Github, from Wikipedia and used datasets provided by Telegram.
2) Combined all files for each class into one large one.
3) Divided these files into small ones that the neural network can accept in a normal distribution. I made 2 neural networks. The first takes as input an image 32 by 32, that is, text of 1024 characters. This is the appropriate text length for the neural network to accurately predict the class. The second neural network, similar to it, takes a 16 by 16 image as input, since it is inconvenient to use a large model when the text is short. In this regard, I prepared two datasets for each neural network.
4) I expanded the resulting text files to lengths of 1024 and 256, respectively, by duplicating the text.
5) Converted these files into black and white pictures 32 by 32 and 16 by 16, respectively.


I took the mobile network as the basis for the neural network, where I changed only the first and last layers.



IF YOU WANT TO GET REAL SOLUTION NOT ONLY CODE CHECKOUT RELEASE WITH NAME SUBMISSION
