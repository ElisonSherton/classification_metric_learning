from fastai.vision.all import *

def prepare_df():
    # Create the dataframe as we need it
    df = pd.read_csv("../data/final_explored_df_retrieval.csv")
    train_df = df[df.ImgType == "train"].reset_index(drop = True)
    count_map = train_df.Actual_Class.value_counts().to_dict()
    train_df["Class_Count"] = train_df.Actual_Class.apply(lambda x: count_map[x])
    train_df["Weight"] = train_df.Class_Count.apply(lambda x: 1 / (1 + x))
    train_df["ImgPath"] = train_df.ImgPath.apply(lambda x: Path(f"../data/{x}"))
    train_df["ImgType"] = ["train"] * len(train_df)
    return train_df

def prepare_valid_df():
    df = pd.read_csv("../data/retrieval_df.csv")
    df["ImgPath"] = df["ImgPath"].apply(lambda x: f"../data/{x}")
    df = df[df.ImgType != "train"].sort_values(by = ["ImgType"]).reset_index(drop = True)
    return df

def get_x(row): return row["ImgPath"]

def get_y(row): return row["Actual_Class"] 

def splitter(df):
    train_idxs = df[df.ImgType == "train"].index.tolist()
    # valid_idxs = df[df.ImgType == "valid"].index.tolist()
    return (train_idxs, random.sample(train_idxs, 384))

def get_item_tfms(size):
    return Resize(size, pad_mode = PadMode.Zeros, method = ResizeMethod.Pad)()

def get_aug_tfms():
    proba = 0.3
    h = Hue(max_hue = 0.3, p = proba, draw=None, batch=False)
    s = Saturation(max_lighting = 0.3, p = proba, draw=None, batch=False)
    ag_tfms = aug_transforms(mult = 1.00, do_flip = True, flip_vert = False, max_rotate = 5, 
                            min_zoom = 0.9, max_zoom = 1.1, max_lighting = 0.5, max_warp = 
                            0.05, p_affine = proba, p_lighting = proba, xtra_tfms = [h, s], 
                            size = 224, mode = 'bilinear', pad_mode = "zeros", align_corners = True, 
                            batch = False, min_scale = 0.75)
    return ag_tfms

def get_weights(df):
    return df[df.ImgType == "train"].Weight.tolist()

def sampler():
    df = pd.read_csv("../data/final_explored_df_retrieval.csv")#.sample(frac = 0.1)
    train_df = df[df.ImgType == "train"].reset_index(drop = True)
    
    class_to_index_map = defaultdict(lambda: [])
    for idx, cl in enumerate(train_df.Actual_Class.tolist()):
        class_to_index_map[cl].append(idx)
    
    indices = []
    while len(indices) < len(train_df):
        k = random.choice(list(class_to_index_map.keys()))
        try:
            idxs = random.sample(class_to_index_map[k], 3)
            for id_ in idxs:
                indices.append(id_)
        except Exception as e:
            pass
        
    # indices = list(range(len(train_df)))        
    return indices

def get_dls():
    BATCH_SIZE = 64
    train_df = prepare_df()
    datablock = DataBlock(blocks = (ImageBlock, CategoryBlock),
                          get_x = get_x,
                          get_y = get_y,
                          splitter = splitter,
                          item_tfms = Resize(size = 460),
                          batch_tfms = get_aug_tfms())

    dls = datablock.dataloaders(source=train_df, bs = BATCH_SIZE, drop_last = True)
    dls.train = dls.train.new(shuffle=False, get_idxs=sampler, drop_last = True)
    return dls

def get_test_dl(dls):
    df = pd.read_csv("../data/final_explored_df_retrieval.csv")
    df["ImgPath"] = df["ImgPath"].apply(lambda x: f"../data/{x}")
    df = df[df.ImgType != "train"].sort_values(by = ["ImgType"]).reset_index(drop = True)
    test_dl = dls.test_dl(test_items = df.ImgPath.tolist())
    classes = df.Actual_Class.tolist()
    gallery_idxs = df[df.ImgType == "gallery"].index.tolist()
    query_idxs = df[df.ImgType == "query"].index.tolist()
    return (test_dl, classes, gallery_idxs, query_idxs)