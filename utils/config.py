# This file is used to configure the training parameters for each task

# -------------------------------------------------------------------------------------------------
class Config_Cyst:
    
    data_path = ""    
    data_subpath = ""
    save_path = ""
    result_path = ""
    tensorboard_path = ""
    load_path = save_path + ""
    load_path0 = ""
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

class Config_CVC:
    
    data_path = ""
    data_subpath = ""
    save_path = ""

    result_path = ""
    tensorboard_path = ""
    load_path = save_path + ""
    load_path0 = ""
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

class Config_ISIC:
    # data_path = ""
    data_path = ""

   
    # data_subpath = ""
    data_subpath = ""

    # save_path = ""
    save_path = ""

    result_path = ""
    tensorboard_path = ""
    load_path = save_path + ""
    load_path0 = ""
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

# ==================================================================================================
def get_config(task="ISIC"):
    if task == "Cyst":
        return Config_Cyst()
    elif task == "ISIC":
        return Config_ISIC()
    elif task == "CVC":
        return Config_CVC()
    else:
        assert("We do not have the related dataset, please choose another task.")
