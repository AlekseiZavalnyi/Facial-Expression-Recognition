# Facial-Expression-Recognition

The project is an attempt to create a model that will work equally well on different datasets for emotion recognition.
#
## Data preprocessing

The face in the photos must be cut out. Using the extreme reference points of the eyes, the face is aligned vertically. Then the eyes and mouth are cut out and concatenated.
The FER+, RAF-DB and MicroExpression datasets were cleaned and merged for one train dataset. 
#
![alt text](https://github.com/AlekseiZavalnyi/Facial-Expression-Recognition/blob/main/images/data_preprocessing.png)
#

## Models used
- My Net

Transfer learning:
- VGGFace
- ResNet50
- MobileNet v2

SOTA-models:
- [APViT](https://github.com/youqingxiaozhua/APViT)
- [DDAMNet](https://github.com/simon20010923/DDAMFN)

## Confusion matrix on test datasets
All models were tested on RAF-DB, CK+ and MicroExpression

#### VGGFace
![alt text](https://github.com/AlekseiZavalnyi/Facial-Expression-Recognition/blob/main/images/confm_vggface.png)

#### ResNet50
![alt text](https://github.com/AlekseiZavalnyi/Facial-Expression-Recognition/blob/main/images/confm_resnet50.png)

#### MobileNet v2
![alt text](https://github.com/AlekseiZavalnyi/Facial-Expression-Recognition/blob/main/images/confm_mobilenetv2.png)

#### My Net
![alt text](https://github.com/AlekseiZavalnyi/Facial-Expression-Recognition/blob/main/images/confm_mynet.png)

#### APViT
![alt text](https://github.com/AlekseiZavalnyi/Facial-Expression-Recognition/blob/main/images/confm_apvit.png)

#### DDAMNet
![alt text](https://github.com/AlekseiZavalnyi/Facial-Expression-Recognition/blob/main/images/confm_ddamnet.png)

## Test scores
![alt text](https://github.com/AlekseiZavalnyi/Facial-Expression-Recognition/blob/main/images/score.png)
