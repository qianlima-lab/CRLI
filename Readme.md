# Readme
It's a tensorflow implementation of AAAI2021 paper "Learning Representation for Incomplete Time-series Clustering".

Qianli Ma, Chuxin Chen, and Sen Li equally contributed to this work.

# To run your own model
```
python main.py --dataset_name xxx
```

## Parameter detail
The results are obtained by running grid search on following parameters:

- lambda_kmeans ∈ {1e-3,1e-6,1e-9}

- G_hiddensize ∈ {50,100,150}

- G_layer ∈ {1,2,3}

## Citation

If you find this repository useful in your research, please cite the following paper:

```
@inproceedings{ma2021learning,
  title={Learning Representations for Incomplete Time Series Clustering},
  author={Ma, Qianli and Chen, Chuxin and Li, Sen and Cottrell, Garrison W},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={10},
  pages={8837--8846},
  year={2021}
}
```

If you use any datasets provided in this repository, please cite their original paper as well:

+ Ali-v1~v3

  ```
  @article{alizadeh2000distinct,
    title={Distinct types of diffuse large B-cell lymphoma identified by gene expression profiling},
    author={Alizadeh, Ash A and Eisen, Michael B and Davis, R Eric and Ma, Chi and Lossos, Izidore S and Rosenwald, Andreas and Boldrick, Jennifer C and Sabet, Hajeer and Tran, Truc and Yu, Xin and others},
    journal={Nature},
    volume={403},
    number={6769},
    pages={503--511},
    year={2000},
    publisher={Nature Publishing Group}
  }
  ```

+ BloodSample

  ```
  @inproceedings{bianchi2017learning,
    title={Learning compressed representations of blood samples time series with missing data},
    author={Bianchi, Filippo Maria and Mikalsen, Karl {\O}yvind and Jenssen, Robert},
    booktitle={European Symposium on Artificial Neural Networks},
    year={2017}
  }
  ```

- Vote

  ```
  @misc{Dua:2019 ,
      author = "Dua, Dheeru and Graff, Casey",
      year = "2017",
      title = "{UCI} Machine Learning Repository",
      note = {\url{http://archive.ics.uci.edu/ml/}, last accessible on 2021/3/15},
      institution = "University of California, Irvine, School of Information and Computer Sciences" 
  }
  ```

- Chen

  ```
  @article{chen2002gene,
    title={Gene expression patterns in human liver cancers},
    author={Chen, Xin and Cheung, Siu Tim and So, Samuel and Fan, Sheung Tat and Barry, Christopher and Higgins, John and Lai, Kin-Man and Ji, Jiafu and Dudoit, Sandrine and Ng, Irene OL and others},
    journal={Molecular Biology of the Cell},
    volume={13},
    number={6},
    pages={1929--1939},
    year={2002},
    publisher={Am Soc Cell Biol}
  }
  ```

- Liang

  ```
  @article{liang2005gene,
    title={Gene expression profiling reveals molecularly and clinically distinct subtypes of glioblastoma multiforme},
    author={Liang, Yu and Diehn, Maximilian and Watson, Nathan and Bollen, Andrew W and Aldape, Ken D and Nicholas, M Kelly and Lamborn, Kathleen R and Berger, Mitchel S and Botstein, David and Brown, Patrick O and others},
    journal={Proceedings of the National Academy of Sciences},
    volume={102},
    number={16},
    pages={5814--5819},
    year={2005},
    publisher={National Acad Sciences}
  }
  ```

- Physionet

  ```
  @inproceedings{silva2012predicting,
    title={Predicting in-hospital mortality of icu patients: The physionet/computing in cardiology challenge 2012},
    author={Silva, Ikaro and Moody, George and Scott, Daniel J and Celi, Leo A and Mark, Roger G},
    booktitle={2012 Computing in Cardiology},
    pages={245--248},
    year={2012},
    organization={IEEE}
  }
  ```

  
