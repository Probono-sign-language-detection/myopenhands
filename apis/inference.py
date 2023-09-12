import torch
import pytorch_lightning as pl
from tqdm import tqdm
import time

from ..core.data import DataModule
from ..models.loader import get_model
from sklearn.metrics import confusion_matrix
import numpy as np

# merge with the corresponding modules in the future release.
class InferenceModel(pl.LightningModule):
    """
    This will be the general interface for running the inference across models.
    Args:
        cfg (dict): configuration set.

    """
    def __init__(self, cfg, stage="test"):
        super().__init__()
        self.cfg = cfg
        self.datamodule = DataModule(cfg.data)
        self.datamodule.setup(stage=stage)
        
        self.model = self.create_model(cfg.model)
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if stage == "test":
            self.model.to(self._device).eval()
    
    def create_model(self, cfg):
        """
        Creates and returns the model object based on the config.
        """
        return get_model(cfg, self.datamodule.in_channels, self.datamodule.num_class)
    
    def forward(self, x):
        """
        Forward propagates the inputs and returns the model output.
        """
        print("FORWARD: ", x.size())
        return self.model(x)
    
    def init_from_checkpoint_if_available(self, map_location=torch.device("cpu")):
        """
        Intializes the pretrained weights if the ``cfg`` has ``pretrained`` parameter.
        """
        if "pretrained" not in self.cfg.keys():
            return

        ckpt_path = self.cfg["pretrained"]
        print(f"Loading checkpoint from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=map_location)
        self.load_state_dict(ckpt["state_dict"], strict=False)
        print("Done!")
        del ckpt
    
    def test_inference(self):
        """
        Calculates the time taken for inference for all the batches in the test dataloader.
        """
        dataloader = self.datamodule.test_dataloader()
        total_time_taken, num_steps = 0.0, 0

        preds = []
        for batch in dataloader:
            start_time = time.time()

            """
            batch당 특정 Frame으로 Inference를 합니다.
            """
            # 프레임 단위로 쪼개기
            tot_frames = batch['frames'].size(2) # 입력 텐서의 프레임 수 (75)
            frame_size = 60 # 기준 프레임의 크기

            sliced_tensors = []
            for i in range(0, tot_frames, frame_size):
                start_frame = i
                end_frame = min(i + frame_size, tot_frames)
                sliced_tensor = batch['frames'][:, :, start_frame:end_frame]
                
                ## TODO: 여기 일단은 프레임으로 가득 안 채워줘도 돼서 그냥 실행 했는데.. 성능 봐가면서 PAdding 할지 말지 결정!
                # # 마지막 프레임으로 가득 채우기
                # if sliced_tensor.size(2) < frame_size:
                #     padding_frames = frame_size - sliced_tensor.size(2)
                #     last_frame = sliced_tensor[:, :, -1:, :]
                #     padding_last_frames = last_frame.repeat(1, 1, padding_frames, 1)
                #     sliced_tensor_padded = torch.cat([sliced_tensor, padding_last_frames], dim=2)
                #     sliced_tensors.append(sliced_tensor_padded)
                # else:
                #     sliced_tensors.append(sliced_tensor)
                sliced_tensors.append(sliced_tensor)
            # print("같음" if np.equal(sliced_tensors[0], sliced_tensors[1]) else "다름")

            for slice_frame in sliced_tensors:
                y_hat = self.model(slice_frame.to(self._device)).cpu()
                # print(y_hat)
                # print(len(y_hat))
                # print(y_hat.size())
                
                # print(torch.max(y_hat, dim=1))
                class_indices = torch.argmax(y_hat, dim=1)


                label = dataloader.dataset.id_to_gloss[class_indices.item()]

                    # filename = batch["files"][i]
                print(f"🤖 Prediction: {label}")
                preds.append(label)
                print(preds)

                total_time_taken += time.time() - start_time
                num_steps += 1
            
            print(f"Avg time per iteration: {total_time_taken*1000.0/num_steps} ms")
        return preds

    def compute_test_accuracy(self):
        """
        Computes the accuracy for the test dataloader.
        """
        # Ensure labels are loaded
        assert not self.datamodule.test_dataset.inference_mode
        # TODO: Write output to a csv
        dataloader = self.datamodule.test_dataloader()
        dataset_scores, class_scores = {}, {}
        for batch_idx, batch in tqdm(enumerate(dataloader), unit="batch"):
            y_hat = self.model(batch["frames"].to(self._device)).cpu()
            class_indices = torch.argmax(y_hat, dim=-1)
            for i, (pred_index, gt_index) in enumerate(zip(class_indices, batch["labels"])):

                dataset_name = batch["dataset_names"][i]
                score = pred_index == gt_index
                
                if dataset_name not in dataset_scores:
                    dataset_scores[dataset_name] = []
                dataset_scores[dataset_name].append(score)

                if gt_index not in class_scores:
                    class_scores[gt_index] = []
                class_scores[gt_index].append(score)
        
        
        for dataset_name, score_array in dataset_scores.items():
            dataset_accuracy = sum(score_array)/len(score_array)
            print(f"Accuracy for {len(score_array)} samples in {dataset_name}: {dataset_accuracy*100}%")


        classwise_accuracies = {class_index: sum(scores)/len(scores) for class_index, scores in class_scores.items()}
        avg_classwise_accuracies = sum(classwise_accuracies.values()) / len(classwise_accuracies)

        print(f"Average of class-wise accuracies: {avg_classwise_accuracies*100}%")
    
    def compute_test_avg_class_accuracy(self):
        """
        Computes the accuracy for the test dataloader.
        """
        #Ensure labels are loaded
        assert not self.datamodule.test_dataset.inference_mode
        # TODO: Write output to a csv
        dataloader = self.datamodule.test_dataloader()
        scores = []
        all_class_indices=[]
        all_batch_labels=[]
        for batch_idx, batch in tqdm(enumerate(dataloader),unit="batch"):
            y_hat = self.model(batch["frames"].to(self._device)).cpu()
            class_indices = torch.argmax(y_hat, dim=-1)

            for i in range(len(batch["labels"])):
                all_batch_labels.append(batch["labels"][i])
                all_class_indices.append(class_indices[i])
            for pred_index, gt_index in zip(class_indices, batch["labels"]):
                scores.append(pred_index == gt_index)
        cm = confusion_matrix(np.array(all_batch_labels), np.array(all_class_indices))
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(f"Average Class Accuracy for {len(all_batch_labels)} samples: {np.mean(cm.diagonal())*100}%")
