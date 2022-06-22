# Leveraging Information Bottleneck for Scientific Document Summarization
EMNLP 2021 Findings paper: Leveraging Information Bottleneck for Scientific Document Summarization


## Requirements 
IBsumm is implemented using Python3.7 with dependencies specified in requirements.txt. Please also download nltk packages. 
```
!pip install -r requirements.txt
import nltk
nltk.download('punkt','stopwords','wordnet','omw-1.4')
```

For global labels computed using pre-trained Longformer, please download [here](https://drive.google.com/file/d/1itVpchvwZN3-lY5YCPWO3YIOGSzxPF_c/view?usp=sharing). Place the downloaded pickle file to the "models" directory.


## Extractive Summarization 

### Toy Example
A simple test based on the first 5 articles of COVID-19 dataset.
```
python main.py --input_path 'source' --input_file 'toy_test.txt' --output_path 'output' --output_file 'toy_outputs.txt' --model_path 'Longformer_global_label.pkl'
```


### COVID-19 
```
python main.py --input_path 'source' --input_file 'covid19_test.txt' --output_path 'output' --output_file 'covid19_outputs.txt' --model_path 'Longformer_global_label.pkl'
```

