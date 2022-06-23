# Leveraging Information Bottleneck for Scientific Document Summarization
EMNLP 2021 Findings paper: [Leveraging Information Bottleneck for Scientific Document Summarization](https://arxiv.org/pdf/2110.01280.pdf)


## Requirements 
IBsumm is implemented using Python3.7 with dependencies specified in requirements.txt. To download all the required packages, use the code block below: 
```
!pip install -r requirements.txt
!pip install transformers==3.4.0
!pip install pytorch-transformers
!pip install tensorflow==2.3.0
!pip install urllib3==1.25.10
!pip install fastai==1.0.61
!pip install --upgrade numpy scipy pandas
import nltk
nltk.download(['punkt','stopwords','wordnet','omw-1.4'])
```

For global labels computed using pre-trained Longformer, please download [here](https://drive.google.com/file/d/1itVpchvwZN3-lY5YCPWO3YIOGSzxPF_c/view?usp=sharing). Place the downloaded pickle file to the "models" directory.


## Extractive Summarization
Place the source document file in the source directory. The source document file should contain the source text in each line.
### Toy Example
A simple model run based on the first 5 articles of COVID-19 dataset.
```
python main.py --input_path 'source' --input_file 'toy_test.txt' --output_path 'output' --output_file 'toy_outputs.txt' --model_path 'Longformer_global_label.pkl'
```
### arXiv
```
python main.py --input_path 'source' --input_file 'arxiv_test.txt' --output_path 'output' --output_file 'arxiv_outputs.txt' --model_path 'Longformer_global_label.pkl'
```

### COVID-19 
```
python main.py --input_path 'source' --input_file 'covid19_test.txt' --output_path 'output' --output_file 'covid19_outputs.txt' --model_path 'Longformer_global_label.pkl'
```

### PubMed 
```
python main.py --input_path 'source' --input_file 'pubmed_test.txt' --output_path 'output' --output_file 'pubmed_outputs.txt' --model_path 'Longformer_global_label.pkl'
```

## Citation 
Please consider citing our work:
```
@inproceedings{ju2021leveraging,
  title={Leveraging Information Bottleneck for Scientific Document Summarization},
  author={Ju, Jiaxin and Liu, Ming and Koh, Huan Yee and Jin, Yuan and Du, Lan and Pan, Shirui},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2021},
  pages={4091--4098},
  year={2021}
}
```
