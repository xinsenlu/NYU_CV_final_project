# NYU_CV_final_project
final project of NYU CV course SSL
based on sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning


To run the code
1. First

```
You should download the flickr8k from website
meawhile, you also need to download the caption split notation file made by Andrej Karpathy
http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
After that, you need to configure the path to them in create_input_file.py
then run 
python create_input_file.py
```

2. Second
```
You also need to configure the path to dataset and caption splits file in train.py
You only need to do
python train.py
```
It will train the model for you

3. To generate caption file:
```
python caption.py --img='path/to/image.jpeg' --model='path/to/model' --word_map='path/to/WORDMAPjson' --beam_size=5
```
