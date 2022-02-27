import torch
import natsort
import numpy as np
import pycocotools
from PIL import Image
import os, cv2, random
import torchvision.ops as ops
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

def get_masks(num_imgs):
    """
    Loops through all masks in mask directory and returns the masks as an array and the bounding boxes
    Arguments:
        num_images (int): number of images that are being used to train the model
    Returns:
        bboxes (array of size [N, 4]): bboxes where each bbox is (x1, y1, x2, y2)
        masks (array of size [N, H, W]): masks from directory as binary array
    """

    bboxes = []
    masks = []
    
    for i in range(num_imgs):
        masks_subdir = '/Users/jessicapetrochuk/Documents/School/UBC/2021-2022/Directed Studies/Code/Detectron_2/myDATASET/masks_with_rotations/section_masks_{}'.format(i) #Change to the directory masks are in 
        
        # Looping through all images in the images directory
        for mask in sorted(os.listdir(masks_subdir)):
            if not mask.startswith('.'):
                full_path = os.path.join(masks_subdir, mask)
                mask_img = Image.open(full_path).convert("1")
                mask_array = np.asarray(mask_img)
                mask_array_bin = np.where(mask_array > 0.5, 1, 0).astype(np.uint8)
                mask_tensor = torch.tensor(mask_array_bin).unsqueeze(0)
                bbox = ops.masks_to_boxes(mask_tensor)
                bbox_list = bbox.tolist()
                mask_array = pycocotools.mask.encode(np.asarray(mask_array, order="F"))
                masks.append(mask_array)
                bboxes.append(bbox_list[0])

        print(i, ':', masks_subdir)

    print('Done getting masks and bounding boxes')
    return bboxes, masks

def get_masks_dict(bboxes, masks):
    print('starting getting dataset dictionary')

    dataset_dicts = []
    images_path = "/Users/jessicapetrochuk/Documents/School/UBC/2021-2022/Directed Studies/Code/Detectron_2/myDATASET/images_with_rotations"
    image_files = os.listdir(images_path)
    image_files_sorted = natsort.natsorted(image_files,reverse=False)
    img_id = 0
    if img_id < 227:
        for image in image_files_sorted:
            record = {}
            if not image.startswith('.'):
                filename = os.path.join(images_path, image)
                
                height, width = cv2.imread(filename).shape[:2]
                record['file_name'] = filename
                record['image_id'] = img_id
                record['height'] = height
                record['width'] = width
                annotations = []
                # fix when there are multiple regions
                annotation_hippocampus = {}
                annotation_hippocampus['bbox'] = bboxes[img_id]
                annotation_hippocampus['bbox_mode'] = BoxMode.XYXY_ABS
                annotation_hippocampus['category_id'] = 0
                annotation_hippocampus['segmentation'] = masks[img_id]

                annotations.append(annotation_hippocampus)

                record['annotations'] = annotations

                dataset_dicts.append(record)
                img_id += 1

    return dataset_dicts

def visualize(dataset_dicts):
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=None, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        cv2.imshow('', out.get_image()[:, :, ::-1])

def train():
    TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
    CUDA_VERSION = torch.__version__.split("+")[-1]
    print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
    num_imgs = 227
    bboxes, masks = get_masks(num_imgs)
    dataset_dicts =  get_masks_dict(bboxes, masks)

    # for d in ["train", "val"]:
    for d in ["train"]:
        DatasetCatalog.register("brain_" + d, lambda d=d: get_masks_dict(bboxes, masks))
        MetadataCatalog.get("brain_" + d).set(thing_classes=["hippocampus"])
    brain_metadata = MetadataCatalog.get("brain_train")

    # DatasetCatalog.register("my_dataset", my_dataset_function)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.INPUT.MASK_FORMAT='bitmask'
    cfg.MODEL.DEVICE =  'cpu'
    cfg.DATASETS.TRAIN = ("brain_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    # os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # trainer = DefaultTrainer(cfg) 
    # trainer.resume_or_load(resume=False)
    # trainer.train()


    #Inference and evaluation
    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    for d in random.sample(dataset_dicts, 3):    
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                    metadata=brain_metadata, 
                    scale=0.5, 
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        print(out.get_image())
        cv2.imshow('', out.get_image()[:, :, ::-1])
        cv2.imwrite('hello.png', out.get_image()[:, :, ::-1])
    #setup_logger()


if __name__ == '__main__':
    train()