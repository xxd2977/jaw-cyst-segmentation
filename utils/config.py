# This file is used to configure the training parameters for each task
class Config_US30K:
    # This dataset contain all the collected ultrasound dataset
    data_path = "../../dataset/SAMUS/"  
    save_path = "./checkpoints/SAMUS/"
    result_path = "./result/SAMUS/"
    tensorboard_path = "./tensorboard/SAMUS/"
    load_path = save_path + "/xxx.pth"
    save_path_code = "_"

    workers = 1                         # number of data loading workers (default: 8)
    epochs = 200                        # number of total epochs to run (default: 400)
    batch_size = 4                     # batch size (default: 4)
    learning_rate = 5e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                         # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    train_split = "train"               # the file name of training set
    val_split = "val"                   # the file name of testing set
    test_split = "test"                 # the file name of testing set
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "mask_slice"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

# -------------------------------------------------------------------------------------------------
class Config_Cyst:#定义了牙囊肿数据集类
    # data_path = "../../dataset/SAMUS/"
    data_path = "/root/xxd/xxd/SAMUS/dataset/SAMUS"

   
    # data_subpath = "../../dataset/SAMUS/Dental-Cyst/"
    data_subpath = "/root/xxd/xxd/SAMUS/dataset/SAMUS/Dental-Cyst/"

    # save_path = "./checkpoints/Dental-Cyst/"
    save_path = "/root/xxd/xxd/SAMUS/checkpoints/Cyst/checkpoint_DCSAM_test/"

    result_path = "./result/Dental-Cyst/test/"
    tensorboard_path = "./tensorboard/Dental-Cyst/"
    load_path = save_path + "SAMUS_12130229_175_0.9315398490112701.pth"#载入预训练模型
    load_path0 = "/root/xxd/xxd/SAMUS/checkpoints/sam_vit_b_01ec64.pth"
    save_path_code = "_"
    
    workers = 1                         # number of data loading workers (default: 8)
    epochs = 200                        # number of total epochs to run (default: 400)
    batch_size = 16                      # batch size (default: 4)
    learning_rate = 1e-4                # initial learning rate (default: 0.001)
    momentum = 0.9                      # momentum
    classes = 2                         # the number of classes (background + foreground)
    img_size = 256                      # the input size of model
    train_split = "train-Dental-Cyst"   # the file name of training set
    val_split = "val-Dental-Cyst"       # the file name of validation set
    test_split = "test-Dental-Cyst"     # the file name of testing set
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluating the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "mask_slice"            # the mode when evaluating the model, slice level or patient level
    pre_trained = True
    mode = "train"
    visual = False
    modelname = "SAM"

class Config_CVC:#定义了息肉数据������������������������������������������������������������������������类
    # data_path = "../../dataset/SAMUS/"
    data_path = "/root/xxd/xxd/SAMUS/dataset/SAMUS"

   
    # data_subpath = "../../dataset/SAMUS/CVC-ClinicDB/"
    data_subpath = "/root/xxd/xxd/SAMUS/dataset/SAMUS/CVC/"

    # save_path = "./checkpoints/Dental-Cyst/"
    save_path = "/root/xxd/xxd/SAMUS/checkpoints/CVC/checkpoint_DCSAM/"

    result_path = "./result/CVC/SwinUnet/"
    tensorboard_path = "./tensorboard/Dental-Cyst/"
    load_path = save_path + "SAMUS_01161326_34_0.8894974992556086.pth"#载入预训练模型
    load_path0 = "/root/xxd/xxd/SAMUS/checkpoints/sam_vit_b_01ec64.pth"
    save_path_code = "_"
    
    workers = 1                         # number of data loading workers (default: 8)
    epochs = 200                        # number of total epochs to run (default: 400)
    batch_size = 16                      # batch size (default: 4)
    learning_rate = 1e-4                # initial learning rate (default: 0.001)
    momentum = 0.9                      # momentum
    classes = 2                         # the number of classes (background + foreground)
    img_size = 256                      # the input size of model
    train_split = "train-CVC"   # the file name of training set
    val_split = "val-CVC"       # the file name of validation set
    test_split = "test-CVC"     # the file name of testing set
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluating the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "mask_slice"            # the mode when evaluating the model, slice level or patient level
    pre_trained = True
    mode = "train"
    visual = False
    modelname = "SAM"

class Config_ISIC:#定�����������������������������皮������病数据������������类
    # data_path = "../../dataset/SAMUS/"
    data_path = "/root/xxd/xxd/SAMUS/dataset/SAMUS"

   
    # data_subpath = "../../dataset/SAMUS/ISIC/"
    data_subpath = "/root/xxd/xxd/SAMUS/dataset/SAMUS/ISIC/"

    # save_path = "./checkpoints/ISIC/"
    save_path = "/root/xxd/xxd/SAMUS/checkpoints/ISIC/ISIC_JCSAM/"

    result_path = "./result/ISIC/SwinUnet"
    tensorboard_path = "./tensorboard/ISIC/"
    load_path = save_path + "SAMUS_04031231_13_0.8673150790463018.pth"#载入预训练模型
    load_path0 = "/root/xxd/xxd/SAMUS/checkpoints/sam_vit_b_01ec64.pth"
    save_path_code = "_"
    
    workers = 1                         # number of data loading workers (default: 8)
    epochs = 100                        # number of total epochs to run (default: 400)
    batch_size = 32                      # batch size (default: 4)
    learning_rate = 1e-4                # initial learning rate (default: 0.001)
    momentum = 0.9                      # momentum
    classes = 2                         # the number of classes (background + foreground)
    img_size = 256                      # the input size of model
    train_split = "train-ISIC"   # the file name of training set
    val_split = "val-ISIC"       # the file name of validation set
    test_split = "test-ISIC"     # the file name of testing set
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluating the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "mask_slice"            # the mode when evaluating the model, slice level or patient level
    pre_trained = True
    mode = "train"
    visual = False
    modelname = "SAM"

class Config_TN3K:
    data_path = "../../dataset/SAMUS/" 
    data_subpath = "../../dataset/SAMUS/ThyroidNodule-TN3K/" 
    save_path = "./checkpoints/TN3K/"
    result_path = "./result/TN3K/"
    tensorboard_path = "./tensorboard/TN3K/"
    load_path = save_path + "/xxx.pth"
    save_path_code = "_"

    workers = 1                  # number of data loading workers (default: 8)
    epochs = 400                 # number of total epochs to run (default: 400)
    batch_size = 8               # batch size (default: 4)
    learning_rate = 1e-4         # initial learning rate (default: 0.001)
    momentum = 0.9               # momentum
    classes = 2                  # the number of classes (background + foreground)
    img_size = 256               # the input size of model
    train_split = "train-ThyroidNodule-TN3K"  # the file name of training set
    val_split = "val-ThyroidNodule-TN3K"     # the file name of testing set
    test_split = "test-ThyroidNodule-TN3K"     # the file name of testing set
    crop = None                  # the cropped image size
    eval_freq = 1                # the frequency of evaluate the model
    save_freq = 2000               # the frequency of saving the model
    device = "cuda"              # training device, cpu or cuda
    cuda = "on"                  # switch on/off cuda option (default: off)
    gray = "yes"                 # the type of input image
    img_channel = 1              # the channel of input image
    eval_mode = "mask_slice"        # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

class Config_BUSI:
    # This dataset is for breast cancer segmentation
    data_path = "../../dataset/SAMUS/"
    data_subpath = "../../dataset/SAMUS/Breast-BUSI/"   
    save_path = "./checkpoints/BUSI/"
    result_path = "./result/BUSI/"
    tensorboard_path = "./tensorboard/BUSI/"
    load_path = save_path + "/xxx.pth"
    save_path_code = "_"

    workers = 1                         # number of data loading workers (default: 8)
    epochs = 400                        # number of total epochs to run (default: 400)
    batch_size = 8                     # batch size (default: 4)
    learning_rate = 1e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                         # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    train_split = "train-Breast-BUSI"   # the file name of training set
    val_split = "val-Breast-BUSI"       # the file name of testing set
    test_split = "test-Breast-BUSI"     # the file name of testing set
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "mask_slice"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

class Config_CAMUS:
    # This dataset is for breast cancer segmentation
    data_path = "../../dataset/SAMUS/"  # 
    data_subpath = "../../dataset/SAMUS/Echocardiography-CAMUS/" 
    save_path = "./checkpoints/CAMUS/"
    result_path = "./result/CAMUS/"
    tensorboard_path = "./tensorboard/CAMUS/"
    load_path = save_path + "/xxx.pth"
    save_path_code = "_"

    workers = 1                         # number of data loading workers (default: 8)
    epochs = 400                        # number of total epochs to run (default: 400)
    batch_size = 8                     # batch size (default: 4)
    learning_rate = 1e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 4                        # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    train_split = "train-EchocardiographyLA-CAMUS"   # the file name of training set
    val_split = "val-EchocardiographyLA-CAMUS"       # the file name of testing set
    test_split = "test-Echocardiography-CAMUS"     # the file name of testing set # HMCQU
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "camusmulti"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

# ==================================================================================================
def get_config(task="ISIC"):
    if task == "US30K":
        return Config_US30K()
    elif task == "TN3K":
        return Config_TN3K()
    elif task == "BUSI":
        return Config_BUSI()
    elif task == "CAMUS":
        return Config_CAMUS()
    elif task == "Cyst":
        return Config_Cyst()
    elif task == "ISIC":
        return Config_ISIC()
    elif task == "CVC":
        return Config_CVC()
    else:
        assert("We do not have the related dataset, please choose another task.")