
import sys
import torch
import torchvision.transforms as transforms

import logging
import multiprocessing
from datetime import datetime

import test
import parser
import commons
from datasets.test_dataset import TestDataset
from eigenplaces_model import eigenplaces_network

def process_image_through_model(model, image, device):
    # Normalize the image (these values are typical for ImageNet models)
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Move image to the same device as model
    image = image.to(device)

    # Forward pass
    with torch.no_grad():
        output = model(image)

    # Print the shape of the output
    print(f"Output shape: {output.shape}")

if __name__=="__main__":

    torch.backends.cudnn.benchmark = True  # Provides a speedup

    args = parser.parse_arguments()
    start_time = datetime.now()
    output_folder = f"logs/{args.save_dir}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
    commons.make_deterministic(args.seed)
    commons.setup_logging(output_folder, console="info")
    logging.info(" ".join(sys.argv))
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {output_folder}")
    logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")

    #### Model
    if args.resume_model == "torchhub":
        model = torch.hub.load("gmberton/eigenplaces", "get_trained_model",
                            backbone=args.backbone, fc_output_dim=args.fc_output_dim)
    else:
        model = eigenplaces_network.GeoLocalizationNet_(args.backbone, args.fc_output_dim)
        
        if args.resume_model is not None:
            logging.info(f"Loading model_ from {args.resume_model}")
            model_state_dict = torch.load(args.resume_model)
            model.load_state_dict(model_state_dict)
        else:
            logging.info("WARNING: You didn't provide a path to resume the model_ (--resume_model parameter). " +
                        "Evaluation will be computed using randomly initialized weights.")

    model = model.to(args.device)
    image = torch.rand((3, 224, 224))
    process_image_through_model(model, image, args.device)



    # test_ds = TestDataset(args.test_dataset_folder, queries_folder="queries",
    #                       positive_dist_threshold=args.positive_dist_threshold)

    # recalls, recalls_str = test.test(args, test_ds, model)
    # logging.info(f"{test_ds}: {recalls_str}")

