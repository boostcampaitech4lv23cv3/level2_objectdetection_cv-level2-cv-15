# πνλ‘μ νΈ κ°μ

<img src="./image/logo.png" alt="logo" style="zoom:80%;" />

λΆλ¦¬μκ±°λ μ¦κ°νλ μ°λ κΈ°μμ μ€μ¬ νκ²½ λΆλ΄μ μ€μΌ μ μλ λ°©λ² μ€ νλμλλ€. μ¬μ§μμ μ°λ κΈ°λ₯Ό Detection νλ λͺ¨λΈμ λ§λ€μ΄ λΆλ¦¬μκ±°λ₯Ό λ μ½κ² λμμ£Όμ΄ μ°λ κΈ° λ¬Έμ λ₯Ό ν΄κ²°νκ³ μ ν©λλ€.

<p align="center"><img src="./image/trash_image.png" alt="trash" width="40%" height="40%" /></p>



# πΎλ°μ΄ν°μ

- μ μ²΄ μ΄λ―Έμ§ κ°μ : 9754μ₯ (Training : 4883μ₯, Test : 4871μ₯)
- 10 class : General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
- μ΄λ―Έμ§ ν¬κΈ° : (1024, 1024)
- Annotation File (COCO format) : μ΄λ―Έμ§ λ΄ κ°μ²΄μ μμΉ λ° ν΄λμ€ μ λ³΄

<p align="center"><img src="./image/dataset.png" alt="trash" width="40%" height="40%" /></p>



# β νλ‘μ νΈ μν λ°©λ²

## Data Processing

- MisLabeled
- Mosaic
- StratifiedGroupKFold
- Heavy Augmentation



## Modeling

### 2-Stage
- Model
    - Cascade RCNN
    - Cascade Mask RCNN
    - ATSS
- Backbone
  - Swin Transformer - tiny, Small, Base, Large
  - ConvNext - Small

- Neck
  - FPN, BiFPN, NasFPN 




### 1-Stage

- YoLo
- EfficientDet



### Optimizer & Scheduler

- Adam, AdamW
- Cosine Annealing



## Training

- Pseudo Labeling
- Ensemble



# π νλ‘μ νΈ κ²°κ³Ό

- 2-Stage Ensemble

  <p align="center"><img src="./image/2stage.png" alt="trash" width="60%" height="60%" /></p>

- 1-Stage Ensemble

  <p align="center"><img src="./image/1stage.png" alt="trash" width="60%" height="60%" /></p>

  

- Score

  <p align="center"><img src="./image/score.png" alt="trash" width="100%" height="100%" /></p>



# π¨βπ¨βπ¦βπ¦ νμ μκ°


|         [ν©μμ](https://github.com/soonyoung-hwang)         |            [λ°μ©λ―Ό](https://github.com/yon-ninii)            |            [μμμ€](https://github.com/won-joon)             |              [μ΄νμ ](https://github.com/SS-hj)              |             [κΉλν](https://github.com/DHKim95)             |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![SY_image](https://avatars.githubusercontent.com/u/78343941?v=4) | ![YM_image](https://avatars.githubusercontent.com/u/87235003?v=4) | ![WJ_image](https://avatars.githubusercontent.com/u/59519591?v=4) | ![HJ_image](https://avatars.githubusercontent.com/u/54202082?v=4) | ![DH_image](https://avatars.githubusercontent.com/u/68861542?v=4) |

