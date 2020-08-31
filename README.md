# Fight Recognition Using 3D ResNet

Using pre-trained models from [3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch) and training them on [Surveillance Camera Fight Dataset](https://github.com/sayibet/fight-detection-surv-dataset)

### Surveillance Camera Fight Dataset ([Link](https://github.com/sayibet/fight-detection-surv-dataset))
* There are 300 videos in total as 150 fight + 150 non-fight
* Videos are 2-second long
* Only the fight related parts are included in the samples

![Surveillance Camera Fight Dataset](/images/dataset_images.jpg)

---

## Results
Project results using 3D ResNet-50 

![Fight gif](/images/2f63530274bc.gif)
![noFight gif](/images/b99cfe94b103.gif)
![Fight gif](/images/3e953519843e.gif)

----

### How to run it

**First** download prerequisite files from [Google drive](https://drive.google.com/drive/folders/1WjIz8hxNFJ8o88f0j7zgecP69w7ks4U2?usp=sharing)

**Then**

1. Go to [Google Colab](https://colab.research.google.com) and sign in
2. Open "*Open Notebook*" then go to "*GitHub*" tab and then search för "*Linkanblomman*" and choose repository "*Linkanblomman/Fight_recognition*"
3. Pick a notebook and run it

**Or**

1. Choose a notebook from Github
2. Press the ![Colab button](/images/colab_button.jpg) button then run it

**NOTE!** 
In order to access downloaded [prerequisite files](https://drive.google.com/drive/folders/1WjIz8hxNFJ8o88f0j7zgecP69w7ks4U2?usp=sharing), Google drive need to be mounted ("*Mount Drive*") in Colab. Also the path to the downloaded files need to be changed (current example path in colab notebooks: "/content/drive/My Drive/Colab_Notebooks/fight_recognition/")

---
For more pre-trained models go to [3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch) and then download from [Google Drive](https://drive.google.com/drive/folders/1xbYbZ7rpyjftI_KCk6YuL-XrfQDz7Yd4)
