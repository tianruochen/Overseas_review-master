###model path
+ **old model**: /home/changqing/workspace/Overseas_review-master/model/unpron_cla_4_epoch_28_acc_0.9309_auc_0.9910.pth  
+ **new model**: /home/changqing/workspaces/Overseas_classification-master/EfficientNet_Simple/model/unpron/unpron_cla_4_epoch_28_acc_0.9309_auc_0.9910.pth

### data path
- **cocofun_normal**: /data1/zhaoshiyu/cocofun_normal
- **cocofun_unnorm**: /data/wangruihao/serious_data/kill_image
###logits path
> cocofun_normal
> > * for old model: /summary/logits/cocofun_normal/best_accuracy_4_class_b4_accuracy_adl_0_380.pth_logits.txt
> > * for new model: /summary/logits/cocofun_normal/unpron_cla_4_epoch_28_acc_0.9309_auc_0.9910.pth_logits.txt

> cocofun_unnorm
> > * for old model: /summary/logits/cocofun_unnorm/best_accuracy_4_class_b4_accuracy_adl_0_380.pth_logits.txt
> > * for new model: /summary/logits/cocofun_unnorm/unpron_cla_4_epoch_28_acc_0.9309_auc_0.9910.pth_logits.txt

***


***
###result (two treshold:0.3 and 0.5)
>###############################cocofun normal############################
>####old:
> > + total imgs : 71565, error imgs : 15805, img acc : *0.7791518200237546*
> > + total invs : 4998,  error invs : 2430,  inv acc : *0.5138055222088835*
>
>####new:
> > + total imgs : 71565, error imgs : 20523, img acc : *0.7132257388388179*
> > + total invs : 4998,  error invs : 1989,  inv acc : *0.6020408163265306*
>
>####data aug
> > + total imgs : 71565, error imgs : 18208, img acc : *0.7455739537483407* 
> > + total invs : 4998,  error invs : 2259,  inv acc : *0.5480192076830732*
>
>####Load best model : Overseas_review-master/model/new/unporn_class_4_epoch_1_acc_0.9466_auc_0.9927.pth
> > + total imgs : 71565, error imgs : 16976, img acc : *0.7627890728708168* 
> > + total invs : 4998, error invs : 2313, inv acc :   *0.5372148859543817*
>
>####Load best model : Overseas_review-master/model/new/unporn_class_4_epoch_2_acc_0.9483_auc_0.9927.pth
> > + total imgs : 71565, error imgs : 17621, img acc : *0.7537762872912737* 
> > + total invs : 4998, error invs : 2250, inv acc : *0.5498199279711885*
>####Load best model :Overseas_review-master/model/t_max5/unporn_class_4_epoch_1_acc_0.9458_auc_0.9926.pth
> > + total imgs : 71565, error imgs : 17367, img acc : 0.7573255082791868 
> > + total invs : 4998, error invs : 2276, inv acc : 0.5446178471388555
>####Load best model : Overseas_review-master/model/t_max5/unporn_class_4_epoch_2_acc_0.9458_auc_0.9926.pth
> > + total imgs : 71565, error imgs : 17253, img acc : 0.7589184657304548 
> > + total invs : 4998, error invs : 2293, inv acc : 0.5412164865946378
>####Load best model : Overseas_review-master/model/t_max5/unporn_class_4_epoch_6_acc_0.9475_auc_0.9926.pth
> > + total imgs : 71565, error imgs : 17631, img acc : 0.7536365541815133 
> > + total invs : 4998, error invs : 2265, inv acc : 0.5468187274909964
>####Load best model : /Overseas_review-master/model/aug_lr_0.0001/unporn_class_4_epoch_3_acc_0.9483_auc_0.9927.pth
> > + total imgs : 71565, error imgs : 17406, img acc : 0.7567805491511214 
> > + total invs : 4998, error invs : 2283, inv acc : 0.5432172869147659
>####Load best model : /Overseas_review-master/model/aug_lr_0.001/unporn_class_4_epoch_20_acc_0.9407_auc_0.9923.pth
> > + total imgs : 71565, error imgs : 16828, img acc : 0.7648571228952701 
> > + total invs : 4998, error invs : 2393, inv acc : 0.5212084833933573
>
Load best model : /home/changqing/workspaces/Overseas_review-master/model/new/unporn_class_4_epoch_98_acc_0.9441_auc_0.9913.pth
Loaded done!
4998
100%|███████████████████████████████████████| 4998/4998 [48:49<00:00,  1.71it/s]
total imgs : 71565, error imgs : 20798, img acc : 0.709383078320408 
total invs : 4998, error invs : 2056, inv acc : 0.5886354541816726




>###############################skill images##############################
>####old:
> > + total imgs : 46604, error imgs : 23775, img acc : *0.48985065659600036*
> > + total invs : 3184,  error invs : 488,   inv acc : *0.8467336683417085*
>####new:
> > + total imgs : 46604, error imgs : 26239, img acc : *0.43697965839842073*
> > + total invs : 3184,  error invs : 410,   inv acc : *0.8712311557788944*
>####data aug
> > + total imgs : 46604, error imgs : 23630, img acc : *0.49296197751265985*
> > + total invs : 3184,  error invs : 517,   inv acc : *0.8376256281407035*
>####Overseas_review-master/model/new/unporn_class_4_epoch_1_acc_0.9466_auc_0.9927.pth
> > + total imgs : 46604, error imgs : 23203, img acc : 0.5021242811775813 
> > + total invs : 3184, error  invs : 483, inv acc : 0.8483040201005025
>####Overseas_review-master/model/new/unporn_class_4_epoch_2_acc_0.9483_auc_0.9927.pth
> > + total imgs : 46604, error imgs : 23574, img acc : 0.4941635911080594 
> > + total invs : 3184, error invs : 472, inv acc : 0.8517587939698492
>####Overseas_review-master/model/t_max5/unporn_class_4_epoch_1_acc_0.9458_auc_0.9926.pth
> > + total imgs : 46604, error imgs : 23385, img acc : 0.49821903699253284 
> > + total invs : 3184, error invs : 488, inv acc : 0.8467336683417085
>####Overseas_review-master/model/t_max5/unporn_class_4_epoch_2_acc_0.9458_auc_0.9926.pth
> > + total imgs : 46604, error imgs : 23102, img acc : 0.504291477126427 
> > + total invs : 3184, error invs : 494, inv acc : 0.8448492462311558
>####/Overseas_review-master/model/t_max5/unporn_class_4_epoch_6_acc_0.9475_auc_0.9926.pth
> > + total imgs : 46604, error imgs : 23323, img acc : 0.4995493949017252 
> > + total invs : 3184, error invs : 494, inv acc : 0.8448492462311558
>####Overseas_review-master/model/aug_lr_0.0001/unporn_class_4_epoch_3_acc_0.9483_auc_0.9927.pth
> > + total imgs : 46604, error imgs : 23357, img acc : 0.4988198437902326 
> > + total invs : 3184, error invs : 480, inv acc : 0.8492462311557789
>####Overseas_review-master/model/aug_lr_0.001/unporn_class_4_epoch_20_acc_0.9407_auc_0.9923.pth
> > + total imgs : 46604, error imgs : 22170, img acc : 0.5242897605355763 
> > + total invs : 3184, error invs : 622, inv acc : 0.8046482412060302
>Load best model : /home/changqing/workspaces/Overseas_review-master/model/new/unporn_class_4_epoch_98_acc_0.9441_auc_0.9913.pth
Loaded done!
3184
100%|███████████████████████████████████████| 3184/3184 [42:07<00:00,  1.26it/s]
total imgs : 46604, error imgs : 26396, img acc : 0.4336108488541756 
total invs : 3184, error invs : 359, inv acc : 0.887248743718593
>
Load best model : /home/changqing/workspaces/Overseas_review-master/model/mixup/unporn_class_4_epoch_16_acc_0.9288_auc_0.9935.pth
Loaded done!
3184
100%|███████████████████████████████████████| 3184/3184 [34:37<00:00,  1.53it/s]
total imgs : 46604, error imgs : 24432, img acc : 0.4757531542356879 
total invs : 3184, error invs : 475, inv acc : 0.8508165829145728






###result (one treshold )
#################################cocofun_norm####################################
###(image mode...):
####best_accuracy_4_class_b4_accuracy_adl_0_380 
under threshold 0.2, image recall: 0.8627820862153287--61745/71565
under threshold 0.3, image recall: 0.8458743799343255--60535/71565
under threshold 0.4, image recall: 0.8281981415496402--59270/71565
under threshold 0.5, image recall: 0.8129812058967373--58181/71565
under threshold 0.6, image recall: 0.7973450709145532--57062/71565
under threshold 0.7, image recall: 0.7791518200237546--55760/71565
under threshold 0.8, image recall: 0.7576189478096835--54219/71565
under threshold 0.9, image recall: 0.7212743659610145--51618/71565

####unporn_class_4_epoch_98_acc_0.9441_auc_0.9913.pth_logits.txt---new
under threshold 0.2, image recall: 0.8764899042828198--62726/71565
under threshold 0.3, image recall: 0.8451058478306435--60480/71565
under threshold 0.4, image recall: 0.8180115978481101--58541/71565
under threshold 0.5, image recall: 0.7895619367009012--56505/71565
under threshold 0.6, image recall: 0.7565010829316007--54139/71565
under threshold 0.7, image recall: 0.709383078320408--50767/71565
under threshold 0.8, image recall: 0.6325578145741634--45269/71565
under threshold 0.9, image recall: 0.3672605323831482--26283/71565

####unporn_class_4_epoch_15_acc_0.9415_auc_0.9933.pth_logits.txt:
under threshold 0.2, image recall: 0.8371829805072312--59913/71565
under threshold 0.3, image recall: 0.8196604485432823--58659/71565
under threshold 0.4, image recall: 0.802710822329351--57446/71565
under threshold 0.5, image recall: 0.7858729826032278--56241/71565
under threshold 0.6, image recall: 0.7674421854258366--54922/71565
under threshold 0.7, image recall: 0.7455739537483407--53357/71565
under threshold 0.8, image recall: 0.7166492000279466--51287/71565
under threshold 0.9, image recall: 0.6667365332215468--47715/71565
####unporn_class_4_epoch_1_acc_0.9458_auc_0.9926.pth_logits.txt:
under threshold 0.2, image recall: 0.8491301613917418--60768/71565
under threshold 0.3, image recall: 0.8297491790679802--59381/71565
under threshold 0.4, image recall: 0.8124641933906239--58144/71565
under threshold 0.5, image recall: 0.7938657164815203--56813/71565
under threshold 0.6, image recall: 0.7766366240480682--55580/71565
under threshold 0.7, image recall: 0.7573255082791868--54198/71565
under threshold 0.8, image recall: 0.7335010130650458--52493/71565
under threshold 0.9, image recall: 0.6938866764479843--49658/71565
####unporn_class_4_epoch_2_acc_0.9458_auc_0.9926.pth_logits.txt
under threshold 0.2, image recall: 0.8520505833857333--60977/71565
under threshold 0.3, image recall: 0.8313421365192483--59495/71565
under threshold 0.4, image recall: 0.8126877663662405--58160/71565
under threshold 0.5, image recall: 0.7948717948717948--56885/71565
under threshold 0.6, image recall: 0.777740515615175--55659/71565
under threshold 0.7, image recall: 0.7589184657304548--54312/71565
under threshold 0.8, image recall: 0.7336547194857822--52504/71565
under threshold 0.9, image recall: 0.6936211835394397--49639/71565
####unporn_class_4_epoch_6_acc_0.9475_auc_0.9926.pth_logits.txt
under threshold 0.2, image recall: 0.8468385383916719--60604/71565
under threshold 0.3, image recall: 0.8242716411653741--58989/71565
under threshold 0.4, image recall: 0.806665269335569--57729/71565
under threshold 0.5, image recall: 0.790595961713128--56579/71565
under threshold 0.6, image recall: 0.7727799902186823--55304/71565
under threshold 0.7, image recall: 0.7536365541815133--53934/71565
under threshold 0.8, image recall: 0.7275763292112066--52069/71565
under threshold 0.9, image recall: 0.6864668483197094--49127/71565
####unporn_class_4_epoch_1_acc_0.9466_auc_0.9927.pth_logits.txt
under threshold 0.2, image recall: 0.8539928736114022--61116/71565
under threshold 0.3, image recall: 0.8358694892754839--59819/71565
under threshold 0.4, image recall: 0.8189338363725285--58607/71565
under threshold 0.5, image recall: 0.8024593027317823--57428/71565
under threshold 0.6, image recall: 0.7832180535177811--56051/71565
under threshold 0.7, image recall: 0.7627890728708168--54589/71565
under threshold 0.8, image recall: 0.7379165793334731--52809/71565
under threshold 0.9, image recall: 0.6901558024173828--49391/71565
####unporn_class_4_epoch_2_acc_0.9483_auc_0.9927.pth_logits.txt
under threshold 0.2, image recall: 0.8466009921050793--60587/71565
under threshold 0.3, image recall: 0.8265492908544679--59152/71565
under threshold 0.4, image recall: 0.808593586250262--57867/71565
under threshold 0.5, image recall: 0.7900789492070146--56542/71565
under threshold 0.6, image recall: 0.772626283797946--55293/71565
under threshold 0.7, image recall: 0.7537762872912737--53944/71565
under threshold 0.8, image recall: 0.7285544609795291--52139/71565
under threshold 0.9, image recall: 0.6791168867463145--48601/71565
####unporn_class_4_epoch_15_acc_0.9415_auc_0.9933.pth_logits.txt
under threshold 0.2, image recall: 0.8371829805072312--59913/71565
under threshold 0.3, image recall: 0.8196604485432823--58659/71565
under threshold 0.4, image recall: 0.802710822329351--57446/71565
under threshold 0.5, image recall: 0.7858729826032278--56241/71565
under threshold 0.6, image recall: 0.7674421854258366--54922/71565
under threshold 0.7, image recall: 0.7455739537483407--53357/71565
under threshold 0.8, image recall: 0.7166492000279466--51287/71565
under threshold 0.9, image recall: 0.6667365332215468--47715/71565
####unporn_class_4_epoch_3_acc_0.9483_auc_0.9927.pth_logits.txt
under threshold 0.2, image recall: 0.8500803465381122--60836/71565
under threshold 0.3, image recall: 0.830098511842381--59406/71565
under threshold 0.4, image recall: 0.8129253126528331--58177/71565
under threshold 0.5, image recall: 0.794522462097394--56860/71565
under threshold 0.6, image recall: 0.775770278767554--55518/71565
under threshold 0.7, image recall: 0.7567805491511214--54159/71565
under threshold 0.8, image recall: 0.7318102424369455--52372/71565
under threshold 0.9, image recall: 0.6838538391671907--48940/71565




###(invitation mode...):
####best_accuracy_4_class_b4_accuracy_adl_0_380 
under threshold 0.2, invitation recall: 0.6262954684007316--3082/4921
under threshold 0.3, invitation recall: 0.5882950619792725--2895/4921
under threshold 0.4, invitation recall: 0.5521235521235521--2717/4921
under threshold 0.5, invitation recall: 0.5313960577118472--2615/4921
under threshold 0.6, invitation recall: 0.5057915057915058--2489/4921
under threshold 0.7, invitation recall: 0.4781548465758992--2353/4921
under threshold 0.8, invitation recall: 0.44767323714692137--2203/4921
under threshold 0.9, invitation recall: 0.40377971956919323--1987/4921
#####unporn_class_4_epoch_98_acc_0.9441_auc_0.9913.pth_logits.txt---new
under threshold 0.2, invitation recall: 0.6766917293233082--3330/4921
under threshold 0.3, invitation recall: 0.6033326559642349--2969/4921
under threshold 0.4, invitation recall: 0.5590327169274538--2751/4921
under threshold 0.5, invitation recall: 0.51371672424304--2528/4921
under threshold 0.6, invitation recall: 0.46271083113188377--2277/4921
under threshold 0.7, invitation recall: 0.4021540337329811--1979/4921
under threshold 0.8, invitation recall: 0.3161958951432636--1556/4921
under threshold 0.9, invitation recall: 0.13635439951229425--671/4921

####unporn_class_4_epoch_1_acc_0.9458_auc_0.9926.pth_logits.txt:
under threshold 0.2, invitation recall: 0.6084129242023979--2994/4921
under threshold 0.3, invitation recall: 0.5608616134931924--2760/4921
under threshold 0.4, invitation recall: 0.5252997358260516--2585/4921
under threshold 0.5, invitation recall: 0.5009144482828694--2465/4921
under threshold 0.6, invitation recall: 0.47551310709205447--2340/4921
under threshold 0.7, invitation recall: 0.4468603942288153--2199/4921
under threshold 0.8, invitation recall: 0.4167852062588905--2051/4921
under threshold 0.9, invitation recall: 0.37167242430400327--1829/4921
####unporn_class_4_epoch_2_acc_0.9458_auc_0.9926.pth_logits.txt
under threshold 0.2, invitation recall: 0.6138996138996139--3021/4921
under threshold 0.3, invitation recall: 0.5608616134931924--2760/4921
under threshold 0.4, invitation recall: 0.529770371875635--2607/4921
under threshold 0.5, invitation recall: 0.5031497663076611--2476/4921
under threshold 0.6, invitation recall: 0.4767323714692136--2346/4921
under threshold 0.7, invitation recall: 0.4503149766307661--2216/4921
under threshold 0.8, invitation recall: 0.41759804917699656--2055/4921
under threshold 0.9, invitation recall: 0.36760820971347286--1809/4921
####unporn_class_4_epoch_6_acc_0.9475_auc_0.9926.pth_logits.txt
under threshold 0.2, invitation recall: 0.5962202804308068--2934/4921
under threshold 0.3, invitation recall: 0.5511074984759196--2712/4921
under threshold 0.4, invitation recall: 0.5240804714488925--2579/4921
under threshold 0.5, invitation recall: 0.5025401341190815--2473/4921
under threshold 0.6, invitation recall: 0.47368421052631576--2331/4921
under threshold 0.7, invitation recall: 0.4446250762040236--2188/4921
under threshold 0.8, invitation recall: 0.41048567364356836--2020/4921
under threshold 0.9, invitation recall: 0.363340784393416--1788/4921
####unporn_class_4_epoch_1_acc_0.9466_auc_0.9927.pth_logits.txt
under threshold 0.2, invitation recall: 0.6242633611054663--3072/4921
under threshold 0.3, invitation recall: 0.5819955293639504--2864/4921
under threshold 0.4, invitation recall: 0.5441983336720179--2678/4921
under threshold 0.5, invitation recall: 0.5114814062182483--2517/4921
under threshold 0.6, invitation recall: 0.4767323714692136--2346/4921
under threshold 0.7, invitation recall: 0.4543791912212965--2236/4921
under threshold 0.8, invitation recall: 0.42267831741515954--2080/4921
under threshold 0.9, invitation recall: 0.3735013208697419--1838/4921
####unporn_class_4_epoch_2_acc_0.9483_auc_0.9927.pth_logits.txt
under threshold 0.2, invitation recall: 0.6000812842918106--2953/4921
under threshold 0.3, invitation recall: 0.5531396057711847--2722/4921
under threshold 0.4, invitation recall: 0.5202194675878886--2560/4921
under threshold 0.5, invitation recall: 0.4970534444218655--2446/4921
under threshold 0.6, invitation recall: 0.46799431009957326--2303/4921
under threshold 0.7, invitation recall: 0.4415769152611258--2173/4921
under threshold 0.8, invitation recall: 0.40987604145498885--2017/4921
under threshold 0.9, invitation recall: 0.3665921560658403--1804/4921
####unporn_class_4_epoch_15_acc_0.9415_auc_0.9933.pth_logits.txt
under threshold 0.2, invitation recall: 0.5939849624060151--2923/4921
under threshold 0.3, invitation recall: 0.5598455598455598--2755/4921
under threshold 0.4, invitation recall: 0.5322089006299533--2619/4921
under threshold 0.5, invitation recall: 0.5057915057915058--2489/4921
under threshold 0.6, invitation recall: 0.4767323714692136--2346/4921
under threshold 0.7, invitation recall: 0.44340581182686445--2182/4921
under threshold 0.8, invitation recall: 0.4086567770778297--2011/4921
under threshold 0.9, invitation recall: 0.3444421865474497--1695/4921
####unporn_class_4_epoch_3_acc_0.9483_auc_0.9927.pth_logits.txt
under threshold 0.2, invitation recall: 0.6110546636862426--3007/4921
under threshold 0.3, invitation recall: 0.561471245681772--2763/4921
under threshold 0.4, invitation recall: 0.5307864255232676--2612/4921
under threshold 0.5, invitation recall: 0.4984759195285511--2453/4921
under threshold 0.6, invitation recall: 0.4718553139605771--2322/4921
under threshold 0.7, invitation recall: 0.44828286933550093--2206/4921
under threshold 0.8, invitation recall: 0.41353383458646614--2035/4921
under threshold 0.9, invitation recall: 0.3710627921154237--1826/4921


###############################cocofun_unnorm##########################
/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_unnorm/best_accuracy_4_class_b4_accuracy_adl_0_380.pth_logits.txt
injudge: 0.1,  threshold 0.0003747180162463337, invitation outgassing rate recall: 0.26163818503241015--444/1697
injudge: 0.2,  threshold 0.012226594612002373, invitation outgassing rate recall: 0.11608721272834413--197/1697
injudge: 0.3,  threshold 0.1024278774857521, invitation outgassing rate recall: 0.0553918680023571--94/1697
injudge: 0.4,  threshold 0.26573309302330017, invitation outgassing rate recall: 0.03653506187389511--62/1697
injudge: 0.5,  threshold 0.6236545443534851, invitation outgassing rate recall: 0.014142604596346494--24/1697
injudge: 0.5218451534241008,  threshold 0.7000325322151184, invitation outgassing rate recall: 0.011785503830288745--20/1697
injudge: 0.6,  threshold 0.9046652317047119, invitation outgassing rate recall: 0.009428403064230996--16/1697
injudge: 0.7,  threshold 0.9856488704681396, invitation outgassing rate recall: 0.002357100766057749--4/1697
injudge: 0.8,  threshold 0.9989489912986755, invitation outgassing rate recall: 0.0011785503830288745--2/1697
injudge: 0.9,  threshold 0.9999810457229614, invitation outgassing rate recall: 0.0--0/1697

####new
/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_unnorm/unporn_class_4_epoch_98_acc_0.9441_auc_0.9913.pth_logits.txt
injudge: 0.1,  threshold 0.03970589116215706, invitation outgassing rate recall: 0.31408367707719503--533/1697
injudge: 0.2,  threshold 0.08720466494560242, invitation outgassing rate recall: 0.11314083677077195--192/1697
injudge: 0.3,  threshold 0.17782683670520782, invitation outgassing rate recall: 0.05774896876841485--98/1697
injudge: 0.4,  threshold 0.30580008029937744, invitation outgassing rate recall: 0.03535651149086624--60/1697
injudge: 0.5,  threshold 0.5273655652999878, invitation outgassing rate recall: 0.015910430170889805--27/1697
injudge: 0.5218451534241008,  threshold 0.5680807828903198, invitation outgassing rate recall: 0.011785503830288745--20/1697
injudge: 0.6,  threshold 0.702741801738739, invitation outgassing rate recall: 0.007071302298173247--12/1697
injudge: 0.7,  threshold 0.8121428489685059, invitation outgassing rate recall: 0.0035356511490866236--6/1697
injudge: 0.8,  threshold 0.8724690079689026, invitation outgassing rate recall: 0.0005892751915144372--1/1697
injudge: 0.9,  threshold 0.9129164814949036, invitation outgassing rate recall: 0.0--0/1697



/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_unnorm/unporn_class_4_epoch_15_acc_0.9415_auc_0.9933.pth_logits.txt
injudge: 0.1,  threshold 8.558211266063154e-05, invitation outgassing rate recall: 0.44372421921037125--753/1697
injudge: 0.2,  threshold 0.004656499717384577, invitation outgassing rate recall: 0.18385385975250443--312/1697
injudge: 0.3,  threshold 0.03373356908559799, invitation outgassing rate recall: 0.09428403064230996--160/1697
injudge: 0.4,  threshold 0.18091733753681183, invitation outgassing rate recall: 0.04832056570418385--82/1697
injudge: 0.5,  threshold 0.5178699493408203, invitation outgassing rate recall: 0.022392457277548614--38/1697
injudge: 0.5218451534241008,  threshold 0.5939021706581116, invitation outgassing rate recall: 0.018267530936947555--31/1697
injudge: 0.6,  threshold 0.8185585737228394, invitation outgassing rate recall: 0.012374779021803181--21/1697
injudge: 0.7,  threshold 0.9460698366165161, invitation outgassing rate recall: 0.008249852681202121--14/1697
injudge: 0.8,  threshold 0.9924302697181702, invitation outgassing rate recall: 0.0041249263406010605--7/1697
injudge: 0.9,  threshold 0.9995730519294739, invitation outgassing rate recall: 0.0011785503830288745--2/1697

/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_unnorm/unporn_class_4_epoch_1_acc_0.9458_auc_0.9926.pth_logits.txt
injudge: 0.1,  threshold 0.0002356839831918478, invitation outgassing rate recall: 0.27578078962875663--468/1697
injudge: 0.2,  threshold 0.007734728045761585, invitation outgassing rate recall: 0.13789039481437831--234/1697
injudge: 0.3,  threshold 0.06221406161785126, invitation outgassing rate recall: 0.06835592221567471--116/1697
injudge: 0.4,  threshold 0.22103212773799896, invitation outgassing rate recall: 0.03535651149086624--60/1697
injudge: 0.5,  threshold 0.5040110945701599, invitation outgassing rate recall: 0.024749558043606363--42/1697
injudge: 0.5218451534241008,  threshold 0.592111349105835, invitation outgassing rate recall: 0.02121390689451974--36/1697
injudge: 0.6,  threshold 0.8422273397445679, invitation outgassing rate recall: 0.01296405421331762--22/1697
injudge: 0.7,  threshold 0.9716606736183167, invitation outgassing rate recall: 0.005303476723629935--9/1697
injudge: 0.8,  threshold 0.9972276091575623, invitation outgassing rate recall: 0.0011785503830288745--2/1697
injudge: 0.9,  threshold 0.9999384880065918, invitation outgassing rate recall: 0.0005892751915144372--1/1697


injudge: 0.1,  threshold 0.0003415944520384073, invitation outgassing rate recall: 0.2869770182675309--487/1697
injudge: 0.2,  threshold 0.009462394751608372, invitation outgassing rate recall: 0.145550972304066--247/1697
injudge: 0.3,  threshold 0.07157686352729797, invitation outgassing rate recall: 0.0695344725987036--118/1697
injudge: 0.4,  threshold 0.2247319519519806, invitation outgassing rate recall: 0.040070713022981735--68/1697
injudge: 0.5,  threshold 0.5195491313934326, invitation outgassing rate recall: 0.026517383618149676--45/1697
injudge: 0.5218451534241008,  threshold 0.5960209369659424, invitation outgassing rate recall: 0.022392457277548614--38/1697
injudge: 0.6,  threshold 0.8404300212860107, invitation outgassing rate recall: 0.01296405421331762--22/1697
injudge: 0.7,  threshold 0.9723348617553711, invitation outgassing rate recall: 0.0076605774896876845--13/1697
injudge: 0.8,  threshold 0.9967570900917053, invitation outgassing rate recall: 0.0011785503830288745--2/1697
injudge: 0.9,  threshold 0.9999319314956665, invitation outgassing rate recall: 0.0005892751915144372--1/1697


/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_unnorm/unporn_class_4_epoch_6_acc_0.9475_auc_0.9926.pth_logits.txt
injudge: 0.1,  threshold 0.00036213966086506844, invitation outgassing rate recall: 0.2787271655863288--473/1697
injudge: 0.2,  threshold 0.010133186355233192, invitation outgassing rate recall: 0.14024749558043606--238/1697
injudge: 0.3,  threshold 0.07039403915405273, invitation outgassing rate recall: 0.07248084855627578--123/1697
injudge: 0.4,  threshold 0.19305504858493805, invitation outgassing rate recall: 0.04478491455509723--76/1697
injudge: 0.5,  threshold 0.5107679963111877, invitation outgassing rate recall: 0.025338833235120803--43/1697
injudge: 0.5218451534241008,  threshold 0.5834735631942749, invitation outgassing rate recall: 0.02357100766057749--40/1697
injudge: 0.6,  threshold 0.8280558586120605, invitation outgassing rate recall: 0.011785503830288745--20/1697
injudge: 0.7,  threshold 0.9697278141975403, invitation outgassing rate recall: 0.007071302298173247--12/1697
injudge: 0.8,  threshold 0.9962018132209778, invitation outgassing rate recall: 0.0017678255745433118--3/1697
injudge: 0.9,  threshold 0.9999088048934937, invitation outgassing rate recall: 0.0005892751915144372--1/1697


/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_unnorm/unporn_class_4_epoch_1_acc_0.9466_auc_0.9927.pth_logits.txt
injudge: 0.1,  threshold 0.0002692318521440029, invitation outgassing rate recall: 0.2846199175014732--483/1697
injudge: 0.2,  threshold 0.0075437892228364944, invitation outgassing rate recall: 0.145550972304066--247/1697
injudge: 0.3,  threshold 0.07093621790409088, invitation outgassing rate recall: 0.0648202710665881--110/1697
injudge: 0.4,  threshold 0.2460172325372696, invitation outgassing rate recall: 0.038892162639952856--66/1697
injudge: 0.5,  threshold 0.531326949596405, invitation outgassing rate recall: 0.027106658809664112--46/1697
injudge: 0.5218451534241008,  threshold 0.5920621156692505, invitation outgassing rate recall: 0.024749558043606363--42/1697
injudge: 0.6,  threshold 0.845272421836853, invitation outgassing rate recall: 0.01296405421331762--22/1697
injudge: 0.7,  threshold 0.9725262522697449, invitation outgassing rate recall: 0.005303476723629935--9/1697
injudge: 0.8,  threshold 0.9975792765617371, invitation outgassing rate recall: 0.0011785503830288745--2/1697
injudge: 0.9,  threshold 0.999945878982544, invitation outgassing rate recall: 0.0005892751915144372--1/1697


/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_unnorm/unporn_class_4_epoch_2_acc_0.9483_auc_0.9927.pth_logits.txt
injudge: 0.1,  threshold 0.00024105433840304613, invitation outgassing rate recall: 0.27342368886269885--464/1697
injudge: 0.2,  threshold 0.007441721856594086, invitation outgassing rate recall: 0.13847967000589276--235/1697
injudge: 0.3,  threshold 0.06367805600166321, invitation outgassing rate recall: 0.06364172068355922--108/1697
injudge: 0.4,  threshold 0.20070111751556396, invitation outgassing rate recall: 0.040070713022981735--68/1697
injudge: 0.5,  threshold 0.49033495783805847, invitation outgassing rate recall: 0.02592810842663524--44/1697
injudge: 0.5218451534241008,  threshold 0.5616140365600586, invitation outgassing rate recall: 0.022981732469063054--39/1697
injudge: 0.6,  threshold 0.8280057907104492, invitation outgassing rate recall: 0.01296405421331762--22/1697
injudge: 0.7,  threshold 0.9666893482208252, invitation outgassing rate recall: 0.005303476723629935--9/1697
injudge: 0.8,  threshold 0.9967689514160156, invitation outgassing rate recall: 0.0011785503830288745--2/1697
injudge: 0.9,  threshold 0.9999200105667114, invitation outgassing rate recall: 0.0005892751915144372--1/1697

/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_unnorm/unporn_class_4_epoch_15_acc_0.9415_auc_0.9933.pth_logits.txt
injudge: 0.1,  threshold 8.558211266063154e-05, invitation outgassing rate recall: 0.44372421921037125--753/1697
injudge: 0.2,  threshold 0.004656499717384577, invitation outgassing rate recall: 0.18385385975250443--312/1697
injudge: 0.3,  threshold 0.03373356908559799, invitation outgassing rate recall: 0.09428403064230996--160/1697
injudge: 0.4,  threshold 0.18091733753681183, invitation outgassing rate recall: 0.04832056570418385--82/1697
injudge: 0.5,  threshold 0.5178699493408203, invitation outgassing rate recall: 0.022392457277548614--38/1697
injudge: 0.5218451534241008,  threshold 0.5939021706581116, invitation outgassing rate recall: 0.018267530936947555--31/1697
injudge: 0.6,  threshold 0.8185585737228394, invitation outgassing rate recall: 0.012374779021803181--21/1697
injudge: 0.7,  threshold 0.9460698366165161, invitation outgassing rate recall: 0.008249852681202121--14/1697
injudge: 0.8,  threshold 0.9924302697181702, invitation outgassing rate recall: 0.0041249263406010605--7/1697
injudge: 0.9,  threshold 0.9995730519294739, invitation outgassing rate recall: 0.0011785503830288745--2/1697


/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_unnorm/unporn_class_4_epoch_3_acc_0.9483_auc_0.9927.pth_logits.txt
injudge: 0.1,  threshold 0.0002474015927873552, invitation outgassing rate recall: 0.2804949911608721--476/1697
injudge: 0.2,  threshold 0.007550599053502083, invitation outgassing rate recall: 0.13847967000589276--235/1697
injudge: 0.3,  threshold 0.0652749165892601, invitation outgassing rate recall: 0.06540954625810254--111/1697
injudge: 0.4,  threshold 0.22173309326171875, invitation outgassing rate recall: 0.03830288744843842--65/1697
injudge: 0.5,  threshold 0.4914308190345764, invitation outgassing rate recall: 0.026517383618149676--45/1697
injudge: 0.5218451534241008,  threshold 0.5730454325675964, invitation outgassing rate recall: 0.021803182086034177--37/1697
injudge: 0.6,  threshold 0.8327168226242065, invitation outgassing rate recall: 0.014142604596346494--24/1697
injudge: 0.7,  threshold 0.9704394936561584, invitation outgassing rate recall: 0.00648202710665881--11/1697
injudge: 0.8,  threshold 0.9972658157348633, invitation outgassing rate recall: 0.0011785503830288745--2/1697
injudge: 0.9,  threshold 0.9999428987503052, invitation outgassing rate recall: 0.0005892751915144372--1/1697

####new
/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_unnorm/unporn_class_4_epoch_98_acc_0.9441_auc_0.9913.pth_logits.txt
injudge: 0.1,  threshold 0.03970589116215706, invitation outgassing rate recall: 0.5653676932746701--1799/3182
injudge: 0.2,  threshold 0.08720466494560242, invitation outgassing rate recall: 0.3988057825267128--1269/3182
injudge: 0.3,  threshold 0.17782683670520782, invitation outgassing rate recall: 0.3032683846637335--965/3182
injudge: 0.4,  threshold 0.30580008029937744, invitation outgassing rate recall: 0.23538654934003772--749/3182
injudge: 0.5,  threshold 0.5273655652999878, invitation outgassing rate recall: 0.17064739157762412--543/3182
injudge: 0.5218451534241008,  threshold 0.5680807828903198, invitation outgassing rate recall: 0.1552482715273413--494/3182
injudge: 0.6,  threshold 0.702741801738739, invitation outgassing rate recall: 0.1109365179132621--353/3182
injudge: 0.7,  threshold 0.8121428489685059, invitation outgassing rate recall: 0.06976744186046512--222/3182
injudge: 0.8,  threshold 0.8724690079689026, invitation outgassing rate recall: 0.03802639849151477--121/3182
injudge: 0.9,  threshold 0.9129164814949036, invitation outgassing rate recall: 0.01225644248900063--39/3182

>####data_aug
injudge: 0.05,  threshold 3.585258673410863e-05, invitation outgassing rate recall: 0.5321154979375369--903/1697
injudge: 0.1,  threshold 8.558211266063154e-05, invitation outgassing rate recall: 0.44372421921037125--753/1697
injudge: 0.15,  threshold 0.0005521888961084187, invitation outgassing rate recall: 0.30288744843842075--514/1697
injudge: 0.2,  threshold 0.004656499717384577, invitation outgassing rate recall: 0.18385385975250443--312/1697
injudge: 0.25,  threshold 0.011758452281355858, invitation outgassing rate recall: 0.14319387153800825--243/1697
injudge: 0.3,  threshold 0.03373356908559799, invitation outgassing rate recall: 0.09428403064230996--160/1697
injudge: 0.35,  threshold 0.08696441352367401, invitation outgassing rate recall: 0.06894519740718916--117/1697
injudge: 0.4,  threshold 0.18091733753681183, invitation outgassing rate recall: 0.04832056570418385--82/1697
injudge: 0.45,  threshold 0.3315041661262512, invitation outgassing rate recall: 0.0347672362993518--59/1697
injudge: 0.5,  threshold 0.5178699493408203, invitation outgassing rate recall: 0.022392457277548614--38/1697
injudge: 0.521,  threshold 0.5898635387420654, invitation outgassing rate recall: 0.018267530936947555--31/1697
injudge: 0.55,  threshold 0.6831819415092468, invitation outgassing rate recall: 0.016499705362404242--28/1697
injudge: 0.6,  threshold 0.8185585737228394, invitation outgassing rate recall: 0.012374779021803181--21/1697
injudge: 0.65,  threshold 0.8898568749427795, invitation outgassing rate recall: 0.00883912787271656--15/1697
injudge: 0.7,  threshold 0.9460698366165161, invitation outgassing rate recall: 0.008249852681202121--14/1697
injudge: 0.75,  threshold 0.9769019484519958, invitation outgassing rate recall: 0.008249852681202121--14/1697
injudge: 0.8,  threshold 0.9924302697181702, invitation outgassing rate recall: 0.0041249263406010605--7/1697
injudge: 0.85,  threshold 0.9976536631584167, invitation outgassing rate recall: 0.0035356511490866236--6/1697
injudge: 0.9,  threshold 0.9995730519294739, invitation outgassing rate recall: 0.0011785503830288745--2/1697
injudge: 0.95,  threshold 0.9999822378158569, invitation outgassing rate recall: 0.0--0/1697


compare:
![avatar](compare.png)


image mode...
old
under threshold 0.5, image false recall: 0.5283452064200498--24623/46604
under threshold 0.6, image false recall: 0.5103424598746888--23784/46604
under threshold 0.7, image false recall: 0.48985065659600036--22829/46604
under threshold 0.8, image false recall: 0.46564672560295256--21701/46604
under threshold 0.9, image false recall: 0.42899751094326666--19993/46604

new
under threshold 0.5, image false recall: 0.5467771006780534--25482/46604
under threshold 0.6, image false recall: 0.4957728950304695--23105/46604
under threshold 0.7, image false recall: 0.43697965839842073--20365/46604
under threshold 0.8, image false recall: 0.3544974680284954--16521/46604
under threshold 0.9, image false recall: 0.21360827396789975--9955/46604



invitation mode...
old
under threshold 0.5, invitation false recall: 0.1876178504085481--597/3182
under threshold 0.6, invitation false recall: 0.16907605279698304--538/3182
under threshold 0.7, invitation false recall: 0.15273412947831552--486/3182
under threshold 0.8, invitation false recall: 0.13827781269641734--440/3182
under threshold 0.9, invitation false recall: 0.11627906976744186--370/3182

new
under threshold 0.5, invitation false recall: 0.2338152105593966--744/3182
under threshold 0.6, invitation false recall: 0.18824638592080453--599/3182
under threshold 0.7, invitation false recall: 0.12822124450031427--408/3182
under threshold 0.8, invitation false recall: 0.07291011942174733--232/3182
under threshold 0.9, invitation false recall: 0.021684475172847266--69/3182




under threshold 0.5337, invitation false recall: 0.21653048397234445--689/3182
under threshold 0.5655, invitation false recall: 0.20081709616593338--639/3182
under threshold 0.5984, invitation false recall: 0.18856065367693275--600/3182
under threshold 0.6334, invitation false recall: 0.1715901948460088--546/3182
under threshold 0.6844, invitation false recall: 0.13733500942803267--437/3182