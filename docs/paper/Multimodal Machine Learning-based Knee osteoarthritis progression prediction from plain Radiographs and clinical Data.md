www.nature.com/scientificreports 



# **open Multimodal Machine Learningbased Knee osteoarthritis progression prediction from plain Radiographs and clinical Data** 

**Aleksei tiulpin**<sup>**1,8***</sup> **, Stefan Klein**<sup>**2**</sup> **, Sita M. A. Bierma-Zeinstra**<sup>**3,4**</sup> **, Jérôme thevenot**<sup>**1**</sup> **, esa Rahtu**<sup>**5**</sup> **, Joyce van Meurs**<sup>**6**</sup> **, Edwin H. G. oei**<sup>**7**</sup> **& Simo Saarakkala**<sup>**1,8**</sup> 

**K** **<mark>nee osteoarthritis (OA) is the most common musculoskeletal disease without a cure, and current treatment options are limited to symptomatic relief.</mark> Prediction of OA progression is a very** **<mark>challenging</mark> and** **<mark>timely issue,</mark> and it could, if resolved, accelerate the** **<mark>disease modifying drug development</mark> and ultimately help to prevent millions of total** **<mark>joint replacement surgeries</mark> performed annually. Here, we present a** **<mark>multi-modal machine learning-based OA progression prediction model</mark> that utilises** **<mark>raw radiographic data, clinical examination results</mark> and** **<mark>previous medical history</mark> of** **<mark>the patient.</mark> We validated this approach on an independent test set of 3** **<mark>,918 knee images from 2,129 subjects</mark> . Our method yielded area under the** **<mark>ROC curve (AUC)</mark> of 0.79 (0.78–0.81) and** **<mark>Average Precision (AP)</mark> of 0.68 (0.66–0.70). In contrast, a reference approach, based on** **<mark>logistic regression</mark> , yielded** **<mark>AUC</mark> of 0.75 (0.74–0.77) and** **<mark>AP</mark> of 0.62 (0.60–0.64). The proposed method could significantly improve the subject selection process for OA drug-development trials and help the development of personalised therapeutic plans.** 

K <mark>nee osteoarthritis (OA) is the most common musculoskeletal disorder causing signifcant disability for patients worldwide</mark><sup>1</sup> . OA is <mark>a degenerative disease</mark> and there is a lack of knowledge on the factors contributing to its progression. The overall etiology of OA is also not understood and there is no effective treatment, besides behavioral interventions. Furthermore, at the end stage of the disease, the only available treatment option is <mark>total knee replacement (TKR) surgery,</mark> which is <mark>highly invasive, costly and also strongly afects</mark> the patient’s quality of life. OA is a major burden for the public health care system and it is increasing further with the aging of the population. For ex <mark>ample, according to the statistics only in the United States, around 12% of the population sufer from OA and</mark> the annual rate of TKR for people 45–64 years of age has doubled since the year of 2000<sup>2</sup> . From the economical point of view, OA causes enormous costs for society and the costs of these surgeries are estimated to be over nine billion euros<sup>2</sup> . 

In primary health care, OA is <mark>currently diagnosed based on a combination of clinical history, physical examination, and X-ray imaging (radiography)</mark> if needed. However, the current widely available diagnostic modalities do not allow for effective OA prognosis assessment<sup>3</sup> , which is important for the planning of appropriate therapeutic interventions and also for recruitment to OA disease modifying drugs development trials<sup>4</sup> . A possible improvement would be to <mark>extend this diagnostic chain with Magnetic Resonance Imaging (MRI)</mark> , which is, <mark>however, costly, time-consuming</mark> , has limited availability and not applicable for wide use<sup>5</sup> . 

While being imperfect and lacking decision consistency, the current OA diagnostic tools can b <mark>e enhanced using computer-assisted methods.</mark> For example, it has been shown that th <mark>e gold clinical standard</mark> for OA severity assessment from radiographs, <mark>semi-quantitative Kellgren-Lawrence</mark> (KL)<sup>6</sup> system that highly suffers from 

1Research Unit of Medical Imaging, Physics and Technology, University of Oulu, Oulu, Finland. 2Biomedical Imaging Group Rotterdam, Depts. of Medical Informatics & Radiology, Erasmus MC, University Medical Center Rotterdam, Rotterdam, The Netherlands.<sup>3</sup> Department of General Practice, Erasmus MC, University Medical Center Rotterdam, Rotterdam, The Netherlands.<sup>4</sup> Department of Orthopedics, Erasmus MC, University Medical Center Rotterdam, Rotterdam, The Netherlands.<sup>5</sup> Department of Signal Processing, Tampere University of Technology, Tampere, Finland.<sup>6</sup> Department of Internal Medicine, Erasmus MC, University Medical Center Rotterdam, Rotterdam, The Netherlands.<sup>7</sup> Department of Radiology & Nuclear Medicine, University Medical Center Rotterdam, Rotterdam, The Netherlands.<sup>8</sup> Department of Diagnostic Radiology, Oulu University Hospital, Oulu, Finland. *email: aleksei.tiulpin@oulu.fi 

**Scientific RepoRtS** | _(2019) 9:20038_ | https://doi.org/10.1038/s41598-019-56527-3 

1 



<!-- Start of picture text -->
es |) re,<br>ee OY<br>@| | |<br><!-- End of picture text -->







<!-- Start of picture text -->
1.0 1.0<br>0g — Age, SEX, BMI, KL, Surg, Inj, WOMAC (0.62 [0.6, 0.64])<br>“ — Age, SEX, BMI, KL (0.61 [0.59, 0.63])<br>“ —— Age, SEX, BMI, Surg, Inj, WOMAC (0.56 [0.53, 0.58])<br>08 l 0.84 —— Age: SEX, BMI (0.53 (0.51, 0.55])<br>“ 4 A 2<br>£© 0.6 7“ 0.6 Wee=SS e a<br>a iy a /c a S = = Sy *<br>fo}a a4 0oO aX y<br>Vv 7“ a<br>20.4 + 0 .44.<br>EF Fa SS<br>“<br>“<br>“<br>0. 2 7" age, SEX, BMI, KL, Surg, Inj, WOMAC (0.75 [0.74, 0.77]) 0.2<br>—— Age, SEX, BMI, KL (0.75 [0.74, 0.77])<br>—— Age, SEX, BMI, Surg, Inj, WOMAC (0.68 [0.66, 0.69])<br>—— Age, SEX, BMI (0.65 [0.63, 0.67])<br>0.0 F 0.0<br>0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0<br>False positive rate Recall<br><!-- End of picture text -->







<!-- Start of picture text -->
1.0 FF 1.0<br>0.8 0.8 iyi<br>806oov c 0 6 SS 3<br>2 a SS<br>wn g2 [S} \<br>S 0.4 7 7 0.4 1 oS<br>f<br>j} ,“:Y<br>0.27 “4<br>__" age, SEX, BMI, KL, Surg, Inj, WOMAC (0.76 [0.75, 0.78]) 0.27 ___ age, SEX, BMI, KL, Surg, Inj, WOMAC (0.63 [0.61, 0.65])<br>—— Age, SEX, BMI, KL (0.76 [0.74, 0.77]) — Age, SEX, BMI, KL (0.61 [0.59, 0.63])<br>— Age, SEX, BMI, Surg, Inj, WOMAC (0.68 [0.66, 0.69]) — Age, SEX, BMI, Surg, Inj, WOMAC (0.56 [0.53, 0.58])<br>— Age, SEX, BMI (0.64 [0.63, 0.66]) — Age, SEX, BMI (0.51 [0.49, 0.54])<br>0.00.0 F 0.2+ 0.4 0.6 0.8 1.0 0.0.0.0 0.2 0.4 0.6 0.8t 1.0<br>False positive rate Recall<br><!-- End of picture text -->





<!-- Start of picture text -->
1.0 = 1.0 t t<br>Z “ — CNN (0.68 [0.66, 0.71])<br>“ —— GBM ref (0.63 [0.61, 0.65])<br>:“ —— LR ref. (0.62 [0.6, 0.64])<br>0.8 ie ra a“y / 0.8 Hy lieret o ~e<br>2 if “ Pe<br>©o 0.6 | 43H t < 0.6 SS<br>avo20.4 J aaZ a 0.4 4 NN<br>a fji Yo“yFa<br>0.2 f x“ | | 7 0.2 |<br>f<br>/ — CNN (0.79 [0.78, 0.8])<br>— GBM ref (0.76 [0.75, 0.78])<br>—— LR ref. (0.75 [0.74, 0.77])<br>0.0 0.0<br>0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0<br>False positive rate Recall<br>(a) (b) (c) (d)<br><!-- End of picture text -->





<!-- Start of picture text -->
1.0 oo 1.0<br>—s a“ — Stacking w. KL (0.7 [0.68, 0.721)<br>“ — Stacking w/o KL (0.68 [0.66, 0.7])<br>o “ 7" — GBM ref (0.63 [0.61, 0.65])<br>08 , +“ o s tni Vina _| — LR ref. (0.62 [0.6, 0.64]) i<br>8 o0.6 f :Ho’ra““ 0.6 ees ‘ eas vee~e ans = : .<br>Fa ’ @<br>So.4 : :<br>a fj J “’aa 0.44 NN<br>f “<br>i<br>0.2 fF “<br>Y — Stacking w. KL (0.81 [0.79, 0.82]) 0.2<br>jff 4 — Stacking w/o KL (0.79 [0.78, 0.81])<br>v: —— GBMLR ref.ref(0.75(0.76[0.74,[0.75,0.77])0.78])<br>0.0 F 0.0<br>0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0<br>False positive rate Recall<br><!-- End of picture text -->







www.nature.com/scientificreports 

www.nature.com/scientificreports/ 

|**Model #**|**Model**|**AUC**|**AP**|
|---|---|---|---|
|2|Age, Sex, BMI, Injury, Surgery, WOMAC, KL-grade (LR)|0.73 (0.70–0.75)|0.52 (0.49–0.55)|
|4|Age, Sex, BMI, Injury, Surgery, WOMAC, KL-grade (GBM)|0.75 (0.72–0.77)|0.54 (0.51–0.58)|
|5|CNN|0.78 (0.76–0.80)|0.58 (0.55–0.61)|
|6|CNN+Age, Sex, BMI, Injury, Surgery, WOMAC (GBM-based fusion)|0.78 (0.76–0.80)|0.58 (0.55–0.62)|
|7|CNN+Age, Sex, BMI, Injury, Surgery, WOMAC, KL-grade (GBM-based fusion)|0.80 (0.78–0.82)|0.62 (0.58–0.65)|



**Table 3.** Detailed comparison of the developed models for knees identified with Kellgren-Lawrence grade 0 or 1, which is considered as absence of osteoarthritis. The testing was done on the Multicenter Osteoarthritis Study dataset. 95% confidence intervals are reported in parentheses for each of the reported metric. KL-grade – KellgrenLawrence grade. CNN – Deep Convolutional Neural Network. BMI – Body-Mass Index. WOMAC – Western Ontario and McMaster Universities Arthritis Index. AUC – Area Under the Receiver Operating Characteristic Curve. AP – Average Precision. LR – Logistic regression. GBM – Gradient Boosting Machine. 

## **Discussion** 

In this study, we presented a <mark>patient-specifc machine learning-based method</mark> to predict structural knee OA progression from patient data acquired at a single clinical visit. The key difference of our method to the prior work is that it <mark>leverages the raw image of the patient</mark> ’ <mark>s knee instead of any measures derived by human observers</mark> (e.g. JSW, KL or bonebone texture descriptors). The results presented in this study demonstrate that our <mark>method</mark> yields <mark>signifcantly better prediction</mark> performance than the conventionally used reference methods. The major finding of this study is that it is possible to <mark>predict knee OA progression from a single knee radiograph complemented with clinical data in a fully automatic manner</mark> . Other findings of this study demonstrate that the <mark>knee X-ray image alone is already a very powerful source of data to predict</mark> whether a particular knee will have OA progression or not. Finally, one of the main results from a clinical point of view is that it is possible to predict progression for patients having KL-0 and KL-1 at baseline. 

To the best of our knowledge, this is the first study where <mark>CNNs were utilised to predict OA progression directly from radiograph</mark> s, and it is also one of the few studies in the field where an independent test set is used to robustly assess the results<sup>10,14,15</sup> . We believe that having such settings, where the test set remains unused until the final model’s validation, is crucial for further development of the OA progression prediction models. Another novelty of our approach is leveraging multi-modal patient data: plain radiographs <mark>(raw image data compared to KL-grades used previously</mark><sup><mark>10,15</mark></sup> <mark>or manually designed texture parameters</mark><sup><mark>11,12</mark></sup> ), <mark>symptomatic assessment, and patient</mark> ’ <mark>s injury and/or surgery history data for prediction</mark> . Our results highlight that a <mark>combination of all the data allows to make more accurate predictions</mark> . Furthermore, thanks to <mark>GBM, with this approach it was possible to use missing data without imputation.</mark> 

In principle, clinical application of the developed method is <mark>straightforward</mark> and <mark>makes it possible to detect OA progression</mark> at <mark>a low cost</mark> in primary health care with <mark>minimal modifcations</mark> to the current diagnostic chain. Our method can be utilized in a fully-automatic manner without a radiologist’s statement, and therefore, it could become <mark>available</mark> as an e.g. <mark>cloud service</mark> or <mark>sofware</mark> for <mark>physiotherapists</mark> to design behavioral interventions for the cases having high confidence of prediction. Compared to the other imaging modalities, such as <mark>MRI,</mark> the progression prediction methods developed just using radiographs and other easily obtainable data utilized in our study have potential to be the most accessible worldwide. 

Whil <mark>e machine learning-based approaches</mark> yiel <mark>d stronger prediction</mark> than <mark>conventional statistical models,</mark> ( _e_ . _g_ . LR), they are less transparent, which can lead to <mark>lack of trust</mark> from clinicians. To address this drawback, various methods have been developed to explain the decisions of “ <mark>black-box systems</mark> ”<sup>24,26,27</sup> . As such, we utilised the <mark>GradCAM approach</mark><sup>24</sup> that allowed us generating an <mark>attention map for each image</mark> at test timefor each image at test time, in order to <mark>highlight the zones where the CNN has paid its attention</mark> . While being attractive, this approach can also lead to wrong interpretations, i.e. there is <mark>no theoretical guarantee that the neural network identifes causal relationships between image features and the output variable</mark> . Therefore, a thorough <mark>analysis</mark> of the <mark>attention maps</mark> is required to assess t <mark>he signifcance of certain features</mark> an <mark>d anatomical zones</mark> picked-up by the model. Such analysis, however, could <mark>enable new possibilities for investigation</mark> of the <mark>visual features</mark> . For example, we observed interesting <mark>associations in the GradCAM-generated attention maps</mark> (Fig. 5), <mark>some of which are not captured by KL grading</mark> . As such, tibial spines (previously associated with OA progression<sup>28</sup> ) were highlighted in multiple attention maps. These associations, however, <mark>do not hold for all the progressors.</mark> 

From the <mark>attention maps</mark> , it can be seen that <mark>our model</mark> is hypothetically <mark>leveraging the information on JSW of the knee.</mark> We conducted multiple <mark>experiments solely on OAI dataset</mark> and verified whether our approach outperforms all our reference models that als <mark>o include explicit measurements of JSW at fxed locations</mark> (fJSW)<sup>29</sup> . We found that <mark>model 7 outperforms</mark> any <mark>GBM-based model</mark> that includes <mark>fJSW measurements</mark> . These results are presented in <mark>Supplementary Table S1</mark> and the detailed information regarding this result is presented in <mark>Supplementary Experiments.</mark> 

Although our study demonstrates a novel method, which outperforms various state-of-the art reference approaches, it also has <mark>several important limitations</mark> . Firstly, our model has <mark>not been tested in other populations than the ones from the United States.</mark> Testing the developed model on data from other populations would be a <mark>crucial step to bring the developed machine learning-based approach to primary healthcare</mark> . <mark>Secondly, we utilised</mark> only <mark>standardised radiographs acquired with a positioning frame,</mark> which is not used in all the hospitals worldwide. Therefore, a validation of our model using the images acquired without the positioning frame is still 

**Scientific RepoRtS** | _(2019) 9:20038_ | https://doi.org/10.1038/s41598-019-56527-3 

7 

www.nature.com/scientificreports 

www.nature.com/scientificreports/ 

needed. However, we tried to <mark>address this limitation by including data acquired under diferent beam angles to the test set.</mark> Thirdly, we relied only on th <mark>e KL-grading system to defne a progression outcome,</mark> and the symptomatic component of OA progression was completely ignored. This also needs to be addressed in the future studies. Fourthly, we use <mark>d imputation in the test set</mark> when evaluating LR models. This could <mark>potentially lower the performance of LR-based reference methods.</mark> In contrast, <mark>GBM-based approach</mark> allowed us to l <mark>everage all the samples with missing data without imputation</mark> total <mark>WOMAC score</mark> as a <mark>representation of patients</mark> . Fifthly, we would also like to mention the fact that we used ’ <mark>symptoms</mark> . While we think that this variable correlates with <mark>a symptoms check done in primary care</mark> , we also think tha <mark>t utilizing individual WOMAC components as features for GBM could potentially lead to a better performance of our method</mark> . Finally, our method is limited in terms of the requirements for training data. As such, in Supplementary Fig. S3 we demonstrated that the <mark>performance of our proposed CNN increases with the increased size of the train dataset</mark> (see Supplementary experiments for details). Consequently, future studies should consider enlarging the train dataset or improve the training techniques or CNN architecture. 

The results presented in this study show that, for subjects at risk, our proposed knee OA progression prediction model allows to identify the progressor cases on average <mark>6% more accurately than with the methods previously used in the OA literature.</mark> This study is an important step towards speeding up the OA disease modifying drug development process and also towards the development of better personalised treatment plans. 

## **Methods** 

**Data description and pre-processing.** We utilised <mark>OAI</mark> (https://data-archive.nimh.nih.gov/oai) and <mark>MOST</mark> (http://most.ucsf.edu)OAI (https://data-archive.nimh.nih.gov/oai) and <mark>MOST</mark> (http://most.ucsf. edu) follow-up cohorts. Both of theseof these datasets include clinical and imaging data from subjects at risk of developing <mark>OA 45–79 and 50–79 years old</mark> , from <mark>baseline</mark> to 96 (9 <mark>imaging follow-ups)</mark> and <mark>84 months (4 imaging follow-ups)</mark> , respectively. OAI dataset includes <mark>bilateral posterior-anterior knee</mark> images, acquired with <mark>™</mark> a <mark>Synafexer</mark> frame<sup>30</sup> and <mark>10 degrees beam angle,</mark> while the MOST dataset also has images acquired with 5- <mark>and 15-degrees beam angles. OAI and MOST studies</mark> were approved by th <mark>e institutional review board of the University of California San Francisco and also the data acquisition sites.</mark> The informed consent was obtained from all the subjects and all the data in both dataset is appropriately anonymised. All the protocols are available on the aforementioned web-sites for each of the cohorts. All the experiments with OAI and MOST datasets were performed in accordance with relevant guidelines and regulations. 

Our inclusion criteria were the following. Firstly, we <mark>excluded</mark> the knees that had <mark>TKA, end-stage OA (</mark> KL-4) or <mark>had a missing KL-data at the baseline.</mark> Subsequently, we <mark>excluded</mark> the <mark>knees which did not progress</mark> and were <mark>not examined at the last follow-up</mark> . This allowed us to ensure that the subjects in the train and test sets did not progress within 96 and 84 months, respectively. <mark>If the knee had any increase of the KL-grade</mark> during the follow-up, we a <mark>ssigned the class of the earliest noticed KL-grade increase,</mark> e.g. if the knee progressed at 30 months and 84 months, we used 30-months follow-up visit to define the fine-grained progression class. Data selection flowcharts for <mark>OAI and MOST da</mark> tasets are presented in <mark>Supplementary Figs. S1 and S2,</mark> respectively. The exact implementation of this selection process is also presented in the supplied source code (see Data Availability Statement). 

In our experiments, we utilized variables such as <mark>age, sex, BMI, injury history, surgery history and total WOMAC</mark> (Western Ontario and McMaster Universities Arthritis Index) score. Due to the presence of missing values, it would be impossible to train and test LR model without <mark>utilizing imputation techniques or removing the missing data</mark> . Therefore, during the training of LR, we excluded the knees with missing values. In the test dataset (MOST), we imputed the missing variables by utilizing mean value imputation strategy when testing the LR. When we <mark>trained GBM-based method</mark> , th <mark>e imputation strategies are not needed,</mark> thus we used the data extracted from OAI metadata as is. 

**Image pre-processing.** To pre-process the <mark>OAI and MOST DICOM</mark> images, for each knee <mark>we extracted a region of interest (ROI) of 140 × 140 mm using an ad-hoc script and BoneFinder sofware</mark><sup>31</sup> that enables accurate fully-automatic anatomicalfully-automatic anatomical landmark localization using regression voting approach. This was done in order to standardise the coordinate frame among the patients and the data acquisition centers. After localizing the bone landmarks, we <mark>rotated all the knee images</mark> so that the <mark>tibial plateau was horizontal.</mark> Subsequently, we performed a <mark>histogram clipping between</mark> 5<sup>_th_</sup> and 99<sup>_th_</sup> <mark>percentiles</mark> and used global contrast <mark>normalisation subtracting the image minimum and dividing all the image pixels by the maximum pixel value.</mark> Then, we converted the <mark>images to 8-bit depth multiplying them by 255</mark> . Finally, all the images were resised to <mark>310 × 310 pixels</mark> (new pixe <mark>l spacing of 0.45 mm</mark> ) and the left knee images were <mark>fipped horizontally to match the collateral (right) knee.</mark> 

In our initial experiments we tried to use <mark>16-bit data</mark> and this <mark>had no effect on performanc</mark> e, but rather i <mark>ncreased the size of the stored data.</mark> These experiments also included testing of different target pixel spacing, however, we eventually found that <mark>0.45 mm spacing yielded the best results on cross-validation.</mark> 

**Experimental setup and reference methods.** All experiments, including the hyper-parameter search, were carried out using <mark>the same 5-fold subject-wise cross-validation on OAI data.</mark> A stratified cross-validation was used to obtain the same distribution of progressed and non-progressed cases in both train and validation splits for each fold. To implement this validation scheme, we used the <mark>publicly available scikit-learn package</mark><sup>32</sup> . 

For building <mark>regularised LR models,</mark> we use <mark>d scikit-learn</mark> and for n <mark>on-regularised LR w</mark> e used the <mark>statsmodels</mark> package<sup>33</sup> . For <mark>GBM models,</mark> we utilised the <mark>LightGBM</mark><sup>34</sup> implementation. We built the CNN models using PyTorch 1.0<sup>35</sup> and trained them using three <mark>NVidia GTX 1080Ti cards.</mark> 

To find the best hyperparameters set for GBM, we used the <mark>Bayesian hyperparameters optimisation package hyperopt</mark><sup>36</sup> <mark>with 500 trials.</mark> Each trial maximised the A <mark>P on cross-validation</mark> . In the case of CNN, we also used 

8 

**Scientific RepoRtS** | _(2019) 9:20038_ | https://doi.org/10.1038/s41598-019-56527-3 

www.nature.com/scientificreports 

www.nature.com/scientificreports/ 

<mark>cross-validation and built 5 models</mark> . We used the snapshot of the model’s weights that yielded the maximum AP value on the validation set in each cross-validation split. The hyperparameters for CNN were found empirically. 

**Deep neural network’s implementation details.** We <mark>designed a multi-task CNN</mark> architecture to <mark>predict OA progression</mark> , and our model consisted of a <mark>convolutional (Conv)</mark> and <mark>two fully-connected (FC) blocks. One FC layer</mark> had <mark>three</mark> outputs corresponding to the <mark>three progression</mark> classes, and the other had <mark>5 outputs,</mark> corresponding to the prediction of the current – baseline KL grade. This is schematically illustrated in Fig. 1. To harmonize the size of the outputs after Conv layers and the inputs of the FC layers, we utilised a <mark>Global Average Pooling layer.</mark> 

We used the design of the <mark>Conv layers from se-resnext50_32x4d network</mark><sup>23</sup> . In the initial cross-validation experiments, we also evaluate <mark>d se-resnet50</mark> , <mark>inceptionv4, se-resnext101_32x4d;</mark> however, we did not obtain significantly better results than the ones reported in this study. To train the CNN, we utilised a <mark>transfer learning similarly</mark> to<sup>7</sup> and <mark>initialised the weights</mark> of all the Conv layers from a network trained on the ImageNet dataset<sup>37</sup> . The two FC layers wer <mark>e initialised from random noise.</mark> 

In contrast to the FC layers, the <mark>weights of the Conv layers were not trained during the frst 2 epochs</mark> (full passes through the training set) and then they were unfrozen <mark>. Subsequently, all the layers of the CNN were trained for 20 epochs.</mark> Such strategy ensured that the FC layers did not corrupt the pre-trained Conv weights during the first backpropagation passes. The CNN was trained with a learning rate of <mark>1</mark> _<mark>e</mark>_ <mark>− 3 (dropped at 15</mark><sup>_th_</sup> <mark>epoch),</mark> batch size of 64, <mark>weight decay of 1</mark> _<mark>e</mark>_ <mark>− 4 and Adam optimization method</mark><sup>38</sup> . We also placed a <mark>dropout layer</mark><sup>39</sup> <mark>with the rate of</mark> _<mark>p</mark>_ <mark>= 0 5. before each FC layer.</mark> During the training of the CNN, we used <mark>random noise addition,</mark> rando <mark>m rotation ±5 degrees, random cropping of the original 310 × 310 pixels image to 300 × 300 pixels</mark> (135 × 135 mm) and also random <mark>gamma correction.</mark> These data augmentations were performed randomly on-the-fly, with the aim to train our model to be invariant towards different data acquisition parameters. We used the <mark>SOLT</mark> package of version 0.1.3<sup>40</sup> in our experiments. 

**Inference pipeline.** At the <mark>test phase,</mark> we <mark>averaged the outputs of all the models trained in cross-validation.</mark> Additionally, for each CNN model here, we performed <mark>5-crop test-time augmentation (TTA).</mark> Specifically, we cropped 4 <mark>images</mark> of <mark>300</mark> <u>×</u> <mark>300 pixels</mark> from th <mark>e corners of the original image,</mark> and <mark>one same-sized crop from the centre of the image.</mark> The predictions for the 5 cropped images were e <mark>ventually averaged. S</mark> ubsequently, <mark>having the TTA prediction for each cross-validation model</mark> , we <mark>averaged</mark> their results as well. This approach allowed us to <mark>reduce the variance of the CNNs and boost the prediction accuracy.</mark> It is worth to mention that during the evaluation of CNN model alone, instead of using the <mark>fne-grained division into progression classes,</mark> we used the <mark>probability of progression</mark> _P_ ( _prog x_ | ) as a s <mark>um of</mark> _<mark>P</mark>_ <mark>(</mark> _<mark>y</mark>_ <mark>= 1|</mark> _<mark>x</mark>_ <mark>) and</mark> _P_ ( _y_ = 2| _x_ ). A similar technique was <mark>previously utilised in a skin cancer prediction study</mark><sup>41</sup> . 

**Interpreting neural network’s decisions.** In this study, we focused <mark>not only on producing the frst state-of-the-art model for knee OA progression prediction</mark> , but also developed an approach to <mark>examine the network</mark> ’ <mark>s decision to assess the radiological features detected by the network.</mark> Similar to our previous study<sup>7</sup> , we modified the <mark>GradCAM method</mark><sup>24</sup> <mark>to operate with TTA.</mark> The output of the <mark>GradCAM is an attention map, showing which region of the image positively correlates with the output of the network.</mark> 

In the previous section, we described a TTA-approach and it should be noted that <mark>all the operations including the sum of the progression probabilities are fully diferentiable</mark> , thus the application of the GradCAM here is fairly straightforward. 

**Model stacking:** **<mark>fusing heterogeneous data u</mark> sing tree gradient boosting.** We fused the predictions of the neural network <mark>– KL grade</mark> and <mark>progression probabilities</mark> _P_ ( _KL_ = _i x_ | ), _i_ ∈ {0, …, 4} and _P_ ( _y_ = _i x_ | ), _i_ ∈ {0, 1, 2} respectively – with other <mark>clinical measures such as patient’s age, sex, BMI, previous injury history, symptomatic assessments (WOMAC) and, optionally, a KL grade.</mark> Such fusion is challenging, prone to <mark>overftting</mark> and requires <mark>a robust cross-validation scheme.</mark> A <mark>stacked generalisation approach</mark> , proposed by Wolpert<sup>25</sup> allows <mark>to build multiple layers of models and handle these issues.</mark> 

Following our model inference strategy, we <mark>first trained the 5 CNN models corresponding to the 5 cross-validation train-validation splits.</mark> Subsequently, this allowed to perform the inference on each validation set in our cross-validation setup and, therefore, obtain CNN predictions for the whole training set. When building the s <mark>econd-level GBM,</mark> we utilised the <mark>same cross-validation split and used the predictions for each knee joint as input features, along with the other clinical measures.</mark> 

**Statistical analyses.** We utilise <mark>d Precision-Recall (PR) and ROC curves as the main methods to measure the performance of all the methods. PR</mark> curve can b <mark>e quantitatively summarised using the AP metric</mark> . The AP metric gives a general understanding on <mark>average positive predictive value (PPV)</mark> of the method. <mark>PPV indicates the probability of the object predicted as positive (progressor in the case of this study) actually being positive.</mark> The <mark>precision-recall curve has been shown to be more informative than the ROC curve</mark> when comparing classifiers on imbalanced datasets<sup>42</sup> . <mark>ROC curve can quantitatively be summarised using the AUC. ROC curve demonstrates a trade-of between the true positive rate (sensitivity) and the false positive rate (1 - specifcity) of the classifer.</mark> AUC represents the quality of ranking random positive examples over the random negative examples<sup>43</sup> . To compute the AUC and AP on the test set, we used <mark>stratifed bootstrapping with 2,000 iterations</mark> . The stratification allowed us to reliably assess the confidence intervals for both <mark>AUC and AP.</mark> We assessed the statistical . significance of the difference between the models using DeLong’s test<sup>44</sup> 

**Scientific RepoRtS** | _(2019) 9:20038_ | https://doi.org/10.1038/s41598-019-56527-3 

9 

www.nature.com/scientificreports 

www.nature.com/scientificreports/ 

## **Data availability** 

OAI and MOST datasets are publicly available datasets and can be requested at http://most.ucsf.edu/ and https:// oai.epi-ucsf.org/. The Dockerfile, source codes, pre-trained models and other relevant data are publicly available at https://github.com/MIPT-Oulu/OAProgression. 

Received: 23 May 2019; Accepted: 6 December 2019; Published: xx xx xxxx 



<!-- Start of picture text -->
Published: xx xx xxxx<br><!-- End of picture text -->

## **References** 

1. Arden, N. & Nevitt, M. C. Osteoarthritis: epidemiology. _Best practice & research Clinical rheumatology_ **20** , 3–25 (2006). 

2. Ferket, B. S. _et al_ . Impact of total knee replacement practice: cost effectiveness analysis of data from the osteoarthritis initiative. _bmj_ **356** , j1131 (2017). 

3. Bedson, J., Jordan, K. & Croft, P. The prevalence and history of knee osteoarthritis in general practice: a case–control study. _Family practice_ **22** , 103–108 (2005). 

4. Jamshidi, A., Pelletier, J.-P. & Martel-Pelletier, J. Machine-learning-based patient-specific prediction models for knee osteoarthritis. _Nature Reviews Rheumatology_ 1 (2018). 

5. van Oudenaarde, K. _et al_ . General practitioners referring adults to mr imaging for knee pain: a randomized controlled trial to assess cost-effectiveness. _Radiology_ **288** , 170–176 (2018). 

6. Kellgren, J. & Lawrence, J. Radiological assessment of osteo-arthrosis. _Annals of the rheumatic diseases_ **16** , 494 (1957). 

7. Tiulpin, A., Thevenot, J., Rahtu, E., Lehenkari, P. & Saarakkala, S. Automatic knee osteoarthritis diagnosis from plain radiographs: A deep learning-based approach. _Scientific reports_ **8** , 1727 (2018). 

8. Norman, B., Pedoia, V., Noworolski, A., Link, T. M. & Majumdar, S. Applying densely connected convolutional neural networks for staging osteoarthritis severity from plain radiographs. _Journal of digital imaging_ 1–7 (2018). 

9. Antony, J., McGuinness, K., O’Connor, N. E. & Moran, K. Quantifying radiographic knee osteoarthritis severity using deep convolutional neural networks. In _2016 23rd International Conference on Pattern Recognition (ICPR)_ , 1195–1200 (IEEE, 2016). 

10. Kerkhof, H. J. _et al_ . Prediction model for knee osteoarthritis incidence, including clinical, genetic and biochemical risk factors. _Annals of the rheumatic diseases_ **73** , 2116–2121 (2014). 

11. Janvier, T. _et al_ . Subchondral tibial bone texture analysis predicts knee osteoarthritis progression: data from the osteoarthritis initiative: tibial bone texture & knee oa progression. _Osteoarthritis and cartilage_ **25** , 259–266 (2017). 

12. Janvier, T., Jennane, R., Toumi, H. & Lespessailles, E. Subchondral tibial bone texture predicts the incidence of radiographic knee osteoarthritis: data from the osteoarthritis initiative. _Osteoarthritis and cartilage_ **25** , 2047–2054 (2017). 

13. Kraus, V. B. _et al_ . Trabecular morphometry by fractal signature analysis is a novel marker of osteoarthritis progression. _Arthritis & Rheumatism: Official Journal of the American College of Rheumatology_ **60** , 3711–3722 (2009). 

14. Yu, D. _et al_ . Development and validation of prediction models to estimate risk of primary total hip and knee replacements using data from the uk: two prospective open cohorts using the uk clinical practice research datalink. _Annals of the rheumatic diseases_ **78** , 91–99 (2019). 

15. Hosnijeh, F. S. _et al_ . Development of a prediction model for future risk of radiographic hip osteoarthritis. _Osteoarthritis and cartilage_ **26** , 540–546 (2018). 

16. Emrani, P. S. _et al_ . Joint space narrowing and kellgren–lawrence progression in knee osteoarthritis: an analytic literature synthesis. _Osteoarthritis and Cartilage_ **16** , 873–882 (2008). 

17. LaValley, M. P., McAlindon, T. E., Chaisson, C. E., Levy, D. & Felson, D. T. The validity of different definitions of radiographic worsening for longitudinal studies of knee osteoarthritis. _Journal of clinical epidemiology_ **54** , 30–39 (2001). 

18. Schmidhuber, J. Deep learning in neural networks: An overview. _Neural Networks_ **61** , 85–117, https://doi.org/10.1016/j. neunet.2014.09.003, Published online 2014; based on TR arXiv:1404.7828 [cs.NE] (2015). 

19. LeCun, Y., Bengio, Y. & Hinton, G. Deep learning. _Nature_ **521** , 436 (2015). 

20. Friedman, J. H. Greedy function approximation: a gradient boosting machine. _Annals of statistics_ 1189–1232 (2001). 

21. Friedman, J., Hastie, T. & Tibshirani, R. _The elements of statistical learning_ . (Springer series in statistics, New York, 2001). 

22. Bellamy, N., Buchanan, W. W., Goldsmith, C. H., Campbell, J. & Stitt, L. W. Validation study of womac: a health status instrument for measuring clinically important patient relevant outcomes to antirheumatic drug therapy in patients with osteoarthritis of the hip or knee. _The Journal of rheumatology_ **15** , 1833–1840 (1988). 

23. Hu, J., Shen, L. & Sun, G. Squeeze-and-excitation networks. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , 7132–7141 (2018). 

24. Selvaraju, R. R. _et al_ . Grad-cam: Visual explanations from deep networks via gradient-based localization. In _Proceedings of the IEEE International Conference on Computer Vision_ , 618–626 (2017). 

25. Wolpert, D. H. Stacked generalization. _Neural networks_ **5** , 241–259 (1992). 

26. Olah, C. _et al_ . The building blocks of interpretability. _Distill_ **3** , e10 (2018). 

27. Bach, S. _et al_ . On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation. _PLoS One_ **10** , e0130140 (2015). 

28. Kinds, M. B. _et al_ . Quantitative radiographic features of early knee osteoarthritis: development over 5 years and relationship with 

   - symptoms in the check cohort. _The Journal of rheumatology_ **40** , 58–65 (2013). 

29. Neumann, G. _et al_ . Location specific radiographic joint space width for osteoarthritis progression. _Osteoarthritis and cartilage_ **17** , 761–765 (2009). 

30. Kothari, M. _et al_ . Fixed-flexion radiography of the knee provides reproducible joint space width measurements in osteoarthritis. _European radiology_ **14** , 1568–1573 (2004). 

31. Lindner, C., Bromiley, P. A., Ionita, M. C. & Cootes, T. F. Robust and accurate shape model matching using random forest regressionvoting. _IEEE transactions on pattern analysis and machine intelligence_ **37** , 1862–1874 (2015). 

32. Pedregosa, F. _et al_ . Scikit-learn: Machine learning in python. _Journal of machine learning research_ **12** , 2825–2830 (2011). 

33. Seabold, S. & Perktold, J. Statsmodels: Econometric and statistical modeling with python. In _Proceedings of the 9th Python in Science Conference_ , vol. 57, 61 (Scipy, 2010). 

34. Ke, G. _et al_ . Lightgbm: A highly efficient gradient boosting decision tree. In _Advances in Neural Information Processing Systems_ , 3146–3154 (2017). 

35. Paszke, A. _et al_ . Automatic differentiation in pytorch. In _NIPS-W_ (2017). 

36. Bergstra, J., Yamins, D. & Cox, D. D. Hyperopt: A python library for optimizing the hyperparameters of machine learning algorithms. In _Proceedings of the 12th Python in science conference_ , 13–20 (Citeseer, 2013). 

37. Deng, J. _et al_ . Imagenet: A large-scale hierarchical image database. In _2009 IEEE conference on computer vision and pattern recognition_ , 248–255 (Ieee, 2009). 

38. Kingma, D. P. & Ba, J. Adam: A method for stochastic optimization. _arXiv preprint arXiv:1412_ . _6980_ (2014). 

39. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I. & Salakhutdinov, R. Dropout: a simple way to prevent neural networks from overfitting. _The Journal of Machine Learning Research_ **15** , 1929–1958 (2014). 

**Scientific RepoRtS** | _(2019) 9:20038_ | https://doi.org/10.1038/s41598-019-56527-3 

10 

www.nature.com/scientificreports 

www.nature.com/scientificreports/ 

40. Tiulpin, A. Solt: Streaming over lightweight transformations, https://github.com/MIPT-Oulu/solt (2019). 

41. Esteva, A. _et al_ . Dermatologist-level classification of skin cancer with deep neural networks. _Nature_ **542** , 115 (2017). 

42. Saito, T. & Rehmsmeier, M. The precision-recall plot is more informative than the roc plot when evaluating binary classifiers on imbalanced datasets. _PLoS One_ **10** , e0118432 (2015). 

43. Cortes, C. & Mohri, M. Auc optimization vs. error rate minimization. In _Advances in neural information processing systems_ , 313–320 (2004). 

44. DeLong, E. R., DeLong, D. M. & Clarke-Pearson, D. L. Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. _Biometrics_ **44** , 837–845 (1988). 

## **Acknowledgements** 

The OAI is a public-private partnership comprised of five contracts (N01-AR-2-2258; N01-AR-2-2259; N01-AR-2-2260; N01-AR-2-2261; N01-AR-2-2262) funded by the National Institutes of Health, a branch of the Department of Health and Human Services, and conducted by the OAI Study Investigators. Private funding partners include Merck Research Laboratories; Novartis Pharmaceuticals Corporation, GlaxoSmithKline; and Pfizer, Inc. Private sector funding for the OAI is managed by the Foundation for the National Institutes of Health. MOST is comprised of four cooperative grants (Felson - AG18820; Torner - AG18832; Lewis - AG18947; and Nevitt - AG19069) funded by the National Institutes of Health, a branch of the Department of Health and Human Services, and conducted by MOST study investigators. This manuscript was prepared using MOST data and does not necessarily reflect the opinions or views of MOST investigators. We would like to acknowledge the strategic funding of the University of Oulu, Infotech Oulu, KAUTE foundation and Sigrid Juselius Foundation for supporting this work. Dr. Claudia Lindner is acknowledged for providing BoneFinder and Egor Panfilov is acknowledged for proof-reading of the manuscript. 

## **Author contributions** 

A.T. and S.S. originated the idea of the study. A.T., S.S. and S.K. designed the study, A.T. performed the experiments and wrote the manuscript S.K., J.T. and E.R. provided the technical feedback. S.B., E.O. and J.M. provided the clinical feedback. All authors participated in the manuscript writing and editing. 

## **competing interests** 

Mr. Aleksei Tiulpin is a co-founder and a shareholder of Ailean Technologies Oy. Other authors declare no competing interests. 

## **Additional information** 

**Supplementary information** is available for this paper at https://doi.org/10.1038/s41598-019-56527-3. 

**Correspondence** and requests for materials should be addressed to A.T. 

**Reprints and permissions information** is available at www.nature.com/reprints. 

**Publisher’s note** Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations. 

**Open Access** This article is licensed under a Creative Commons Attribution 4.0 International License, which permits use, sharing, adaptation, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons license, and indicate if changes were made. The images or other third party material in this article are included in the article’s Creative Commons license, unless indicated otherwise in a credit line to the material. If material is not included in the article’s Creative Commons license and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/. 

© The Author(s) 2019 

**Scientific RepoRtS** | _(2019) 9:20038_ | https://doi.org/10.1038/s41598-019-56527-3 

11 

