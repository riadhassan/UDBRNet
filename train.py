import os
import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv
from DataLoader.data_loader import data_loaders
import argparse
from Network import Network_wrapper
from Metrics import Evaluation_metric as evaluate
import random
from monai.losses import DiceCELoss
from torch.nn.functional import one_hot


def conf():
    args = argparse.ArgumentParser()
    args.add_argument("--data_path", type=str,
                      default="C:\\Users\\IICT2\\Desktop\\Dataset_LCTSC")
    args.add_argument("--output_path", type=str,
                      default="C:\\Users\\IICT2\\Desktop\\output")
    args.add_argument("--dataset", type=str, default="LCTSC")
    args.add_argument("--model_name", type=str, default="UDBRNet")
    args.add_argument("--epoch_num", type=int, default=200)
    args = args.parse_args()
    return args

def out_directory_create(conf):
    if not os.path.exists(os.path.join(conf.output_path, conf.dataset)):
        os.mkdir(os.path.join(conf.output_path, conf.dataset))

    if not os.path.exists(os.path.join(conf.output_path, conf.dataset, conf.model_name)):
        os.mkdir(os.path.join(conf.output_path, conf.dataset, conf.model_name))

    if not os.path.exists(os.path.join(conf.output_path, conf.dataset, conf.model_name, "Model")):
        os.mkdir(os.path.join(conf.output_path, conf.dataset, conf.model_name, "Model"))

    if not os.path.exists(os.path.join(conf.output_path, conf.dataset, conf.model_name, "Image")):
        os.mkdir(os.path.join(conf.output_path, conf.dataset, conf.model_name, "Image"))

    return os.path.join(conf.output_path, conf.dataset, conf.model_name)

def main(conf):
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")

    loader_train, loader_valid = data_loaders(conf.data_path)
    loaders = {"train": loader_train, "valid": loader_valid}

    if conf.dataset == "LCTSC":
        out_channel = 6
        region = evaluate.get_LCTSC_regions()
    elif conf.dataset == "SegThor":
        out_channel = 5
        region = evaluate.get_SegThor_regions()

    model = Network_wrapper.model_wrapper(conf)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    dice_CE = DiceCELoss(softmax=True)

    output_directory = out_directory_create(conf)

    epoch_done = 0
    for epoch in tqdm(range(epoch_done, conf.epoch_num)):
        print("\n {epc} is running".format(epc=epoch))
        img_print = random.randint(500,600)

        for phase in ["train", 'valid']:
            if phase == "train":
                validation_predict = {}
                validation_true = {}
                model.train()
            else:
                model.eval()

            for i, data in enumerate(loaders[phase]):
                x, y_true, affine, patient_id, slice_id = data
                x, y_true = x.to(device), y_true.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    y_pred1, y_pred2, y_pred3, refined = model(x)
                    y_true_one_hot = one_hot(y_true, num_classes=out_channel).permute(0, 3, 1, 2)
                    loss1 = dice_CE(y_pred1, y_true_one_hot)
                    loss2 = dice_CE(y_pred2, y_true_one_hot)
                    loss3 = dice_CE(y_pred3, y_true_one_hot)
                    loss4 = dice_CE(refined, y_true_one_hot)
                    loss = loss1 + loss2 + loss3 + loss4

                    if(i%100 == 0):
                        print(f"Iteration: {i} Loss: {loss}")

                    if phase == "valid":
                        patient_id = int(patient_id[0])
                        slice_id = int(slice_id[0])

                        refined = refined.argmax(dim=1)

                        if patient_id not in validation_predict.keys():
                            validation_predict[patient_id] = refined
                            validation_true[patient_id] = y_true
                        else:
                            validation_predict[patient_id] = torch.cat((validation_predict[patient_id], refined))
                            validation_true[patient_id] = torch.cat((validation_true[patient_id], y_true))

                        if i % img_print == 0:
                            y_pred_np = refined.detach().cpu().numpy().squeeze()
                            y_true_np = y_true.detach().cpu().numpy().squeeze()
                            main_image_np = x.detach().cpu().numpy().squeeze()
                            plt.figure(i)
                            plt.subplot(1, 3, 1)
                            plt.imshow(main_image_np)
                            plt.subplot(1, 3, 2)
                            plt.imshow(y_true_np)
                            plt.subplot(1, 3, 3)
                            plt.imshow(y_pred_np)
                            plt.axis('off')
                            plt.savefig(os.path.join(output_directory, "Image", f"P_{patient_id}_S_{slice_id}.png"), bbox_inches='tight')
                            plt.close()

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

        all_dice = torch.Tensor().cuda()

        print(f"Epoch: {epoch} Evaluation of {conf.model_name} Validate ")

        for patient in validation_predict.keys():
            print(f"Evaluating Patient {patient}")
            val_patient_pred = validation_predict[patient].squeeze()
            val_patient_true = validation_true[patient].squeeze()
            dice = evaluate.evaluate_case(val_patient_pred, val_patient_true, out_channel)
            all_dice = torch.cat((all_dice,dice))

        organ_dice = torch.mean(torch.Tensor(all_dice), 0)

        if conf.dataset == "LCTSC":
            all = evaluate.print_LCTSC(organ_dice)
        if conf.dataset == "SegThor":
            all = evaluate.print_SegThor(organ_dice)
        with open(os.path.join(output_directory, f"{conf.model_name}_Validation_data.csv"), "a") as f:
            csv_writer = csv.writer(f, delimiter=',')
            csv_writer.writerow(all)


        if epoch < 1:
            prev_mean_dice = 0
        elif epoch == epoch_done:
            prev_mean_dice = 0

        curr_mean_dice = torch.mean(organ_dice)
        model_name = f"{conf.model_name}_epoch_{epoch}.pth"
        checkpoint_file = os.path.join(output_directory, "Model", model_name)
        if curr_mean_dice > prev_mean_dice:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, checkpoint_file)
            prev_mean_dice = curr_mean_dice
        else:
            continue

if __name__ == "__main__":
    main(conf())
